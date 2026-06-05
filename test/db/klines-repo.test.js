// klines-repo test - read/write + fallback + source lock
import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { initSchema } from '../../lib/db/schema.js';
import { saveKlinesSync, getExistingSource, checkSourceLock, clearExistingData } from '../../lib/db/klines-repo.js';

// 为了隔离测试，每个测试用独立的 :memory: DB
let testDb = null;

function setupDb() {
  testDb = new Database(':memory:');
  initSchema(testDb);
  return testDb;
}

// mock connection module to return testDb
import * as connection from '../../lib/db/connection.js';
const originalGetDb = connection.getDb;
let overrideDb = null;

function useDb(db) {
  overrideDb = db;
}

// Patch: 替换 getDb 返回测试数据库
// 注意：klines-repo.js 从 connection.js import getDb，
// ES module live binding 允许我们通过 connection.getDb 替换...
// 但 import 的绑定是只读的。这里只能用另一种方式：
// 直接测试 saveKlinesSync + 再通过 getDb 的原始机制...

// 实际上我们需要绕过 getDb。klines-repo 内部调 getDb()，
// 而 getDb 返回文件路径的数据库。测试应该用 :memory: 替代。
// 方案：用 DI —— 让 getKlines 接受 db 参数，或者临时改 connection。
// 简化方案：直接测 saveKlinesSync + 底层 SQL，不绕 getDb。

function makeSampleKlines(n = 5) {
  const klines = [];
  for (let i = 0; i < n; i++) {
    klines.push({
      date: `2026-${String(i + 1).padStart(2, '0')}`,
      open: 100 + i,
      close: 102 + i,
      high: 105 + i,
      low: 98 + i,
      volume: 1000000 * (i + 1),
      amount: 130000000 * (i + 1),
      amplitude: 5.5,
      changePercent: 2.0,
      change: 2,
      turnoverRate: 0.5,
    });
  }
  return klines;
}

test('saveKlinesSync 写入 + 读取验证', () => {
  const db = setupDb();
  const klines = makeSampleKlines(3);
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '贵州茅台', klines);

  const rows = db.prepare('SELECT * FROM monthly_klines WHERE code = ? ORDER BY date').all('600519');
  assert.equal(rows.length, 3);
  assert.equal(rows[0].date, '2026-01');
  assert.equal(rows[2].date, '2026-03');
});

test('saveKlinesSync 写入 stocks 表', () => {
  const db = setupDb();
  const klines = makeSampleKlines(1);
  saveKlinesSync(db, 'monthly_klines', '000001', '0', '平安银行', klines);

  const stock = db.prepare('SELECT * FROM stocks WHERE code = ?').get('000001');
  assert.equal(stock.name, '平安银行');
  assert.equal(stock.market, '0');
});

test('saveKlinesSync upsert 去重', () => {
  const db = setupDb();
  const klines1 = makeSampleKlines(2);
  const klines2 = [
    { date: '2026-02', open: 200, close: 210, high: 220, low: 190, volume: 2000000, amount: 260000000, amplitude: 15, changePercent: 5, change: 10, turnoverRate: 1 },
  ];
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', klines1);
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', klines2);

  const rows = db.prepare('SELECT * FROM monthly_klines WHERE code = ? ORDER BY date').all('600519');
  assert.equal(rows.length, 2);
  // 2026-02 被更新为新的 close=210
  const feb = rows.find((r) => r.date === '2026-02');
  assert.equal(feb.close, 210);
});

test('批量写入多周期', () => {
  const db = setupDb();
  const klines = makeSampleKlines(3);
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', klines);
  saveKlinesSync(db, 'weekly_klines', '600519', '1', '茅台', makeSampleKlines(5));
  saveKlinesSync(db, 'daily_klines', '600519', '1', '茅台', makeSampleKlines(10));

  assert.equal(db.prepare('SELECT count(*) as c FROM monthly_klines').get().c, 3);
  assert.equal(db.prepare('SELECT count(*) as c FROM weekly_klines').get().c, 5);
  assert.equal(db.prepare('SELECT count(*) as c FROM daily_klines').get().c, 10);
});

test('空 klines 不报错', () => {
  const db = setupDb();
  assert.doesNotThrow(() => {
    saveKlinesSync(db, 'monthly_klines', '600519', '1', '空', []);
  });
});

test('null 字段可写入', () => {
  const db = setupDb();
  const klines = [
    { date: '2026-01', open: 10, close: null, high: null, low: null, volume: null, amount: null, amplitude: null, changePercent: null, change: null, turnoverRate: null },
  ];
  saveKlinesSync(db, 'monthly_klines', '000001', '0', '测试', klines);
  const row = db.prepare('SELECT * FROM monthly_klines WHERE code = ?').get('000001');
  assert.equal(row.open, 10);
  assert.equal(row.close, null);
});

// ---- 源锁定测试 ----

test('source 列写入', () => {
  const db = setupDb();
  const klines = makeSampleKlines(2);
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', klines, 'baidu');
  const row = db.prepare('SELECT source FROM monthly_klines WHERE code = ? LIMIT 1').get('600519');
  assert.equal(row.source, 'baidu');
});

test('getExistingSource 返回正确源', () => {
  const db = setupDb();
  const klines = makeSampleKlines(1);
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', klines, 'sina');
  const source = getExistingSource(db, 'monthly_klines', '600519');
  assert.equal(source, 'sina');
});

test('getExistingSource 无数据返回 null', () => {
  const db = setupDb();
  const source = getExistingSource(db, 'monthly_klines', '999999');
  assert.equal(source, null);
});

test('同源写入不抛错', () => {
  const db = setupDb();
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', makeSampleKlines(2), 'baidu');
  assert.doesNotThrow(() => {
    saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', makeSampleKlines(3), 'baidu');
  });
});

test('跨源写入抛错', () => {
  const db = setupDb();
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', makeSampleKlines(2), 'baidu');
  assert.throws(() => {
    saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', makeSampleKlines(1), 'sina');
  }, /Cross-source contamination/);
});

test('不同周期可用不同源（不互锁）', () => {
  const db = setupDb();
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', makeSampleKlines(2), 'baidu');
  assert.doesNotThrow(() => {
    saveKlinesSync(db, 'weekly_klines', '600519', '1', '茅台', makeSampleKlines(2), 'sina');
  });
});

test('clearExistingData 清除后可用新源', () => {
  const db = setupDb();
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', makeSampleKlines(2), 'baidu');
  clearExistingData(db, 'monthly_klines', '600519');
  assert.doesNotThrow(() => {
    saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', makeSampleKlines(2), 'eastmoney');
  });
  const source = getExistingSource(db, 'monthly_klines', '600519');
  assert.equal(source, 'eastmoney');
});
