// db-update test - incremental update logic
import { test } from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { initSchema } from '../../lib/db/schema.js';
import { saveKlinesSync } from '../../lib/db/klines-repo.js';

test('INSERT OR REPLACE 增量更新', () => {
  const db = new Database(':memory:');
  initSchema(db);

  // 初始数据
  const initial = [
    { date: '2026-03', open: 100, close: 110, high: 115, low: 95, volume: 1e6, amount: 1e8, amplitude: 20, changePercent: 10, change: 10, turnoverRate: 0.5 },
  ];
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', initial);

  // 增量更新（新增 2026-04）
  const update = [
    { date: '2026-04', open: 110, close: 120, high: 125, low: 105, volume: 1.2e6, amount: 1.3e8, amplitude: 18, changePercent: 9, change: 10, turnoverRate: 0.6 },
  ];
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', update);

  const rows = db.prepare('SELECT date FROM monthly_klines WHERE code = ? ORDER BY date').all('600519');
  assert.equal(rows.length, 2);
  assert.equal(rows[0].date, '2026-03');
  assert.equal(rows[1].date, '2026-04');
});

test('增量更新替换已有月份', () => {
  const db = new Database(':memory:');
  initSchema(db);

  const v1 = [{ date: '2026-05', open: 100, close: 110, high: 115, low: 95, volume: 1e6, amount: 1e8, amplitude: 20, changePercent: 10, change: 10, turnoverRate: 0.5 }];
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', v1);

  // 同月数据更新（分红除权后复权因子变化）
  const v2 = [{ date: '2026-05', open: 100, close: 115, high: 115, low: 95, volume: 1e6, amount: 1e8, amplitude: 20, changePercent: 15, change: 15, turnoverRate: 0.5 }];
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', v2);

  const row = db.prepare('SELECT * FROM monthly_klines WHERE code = ? AND date = ?').get('600519', '2026-05');
  assert.equal(row.close, 115); // 被更新了
});

test('多只股票增量更新隔离', () => {
  const db = new Database(':memory:');
  initSchema(db);

  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台', [
    { date: '2026-04', open: 1300, close: 1320, high: 1350, low: 1280, volume: 1e6, amount: 1.3e9, amplitude: null, changePercent: null, change: null, turnoverRate: null },
  ]);
  saveKlinesSync(db, 'monthly_klines', '000001', '0', '平安', [
    { date: '2026-04', open: 10, close: 11, high: 12, low: 9, volume: 5e6, amount: 5e7, amplitude: null, changePercent: null, change: null, turnoverRate: null },
  ]);

  // 增量：平安加新月份
  saveKlinesSync(db, 'monthly_klines', '000001', '0', '平安', [
    { date: '2026-05', open: 11, close: 12, high: 13, low: 10, volume: 6e6, amount: 6e7, amplitude: null, changePercent: null, change: null, turnoverRate: null },
  ]);

  assert.equal(db.prepare('SELECT count(*) as c FROM monthly_klines WHERE code = ?').get('600519').c, 1);
  assert.equal(db.prepare('SELECT count(*) as c FROM monthly_klines WHERE code = ?').get('000001').c, 2);
});

test('股票名称再次写入更新（有 K 线数据时）', () => {
  const db = new Database(':memory:');
  initSchema(db);

  saveKlinesSync(db, 'monthly_klines', '600519', '1', '贵州茅台', [
    { date: '2026-04', open: 1300, close: 1320, high: 1350, low: 1280, volume: 1e6, amount: 1.3e9, amplitude: null, changePercent: null, change: null, turnoverRate: null },
  ]);

  // 更新名称需要至少 1 根 K 线，因 saveKlinesSync 在空 klines 时直接 return
  saveKlinesSync(db, 'monthly_klines', '600519', '1', '茅台更名', [
    { date: '2026-05', open: 1320, close: 1330, high: 1350, low: 1310, volume: 1e6, amount: 1.3e9, amplitude: null, changePercent: null, change: null, turnoverRate: null },
  ]);

  const stock = db.prepare('SELECT name FROM stocks WHERE code = ?').get('600519');
  assert.equal(stock.name, '茅台更名');
});
