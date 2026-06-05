// sector/build-sector-klines.js 测试 — 市值加权合成

import { test } from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { initSchema } from '../../lib/db/schema.js';
import { buildAllSectorKlines } from '../../lib/sector/build-sector-klines.js';

function setupDb() {
  const db = new Database(':memory:');
  initSchema(db);
  return db;
}

function seedMapping(db) {
  db.prepare('INSERT INTO industries (industry_code, industry_name) VALUES (?, ?)').run('801780.SI', '银行');
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, stock_name, market_cap) VALUES (?,?,?,?)').run('601398', '801780.SI', '工商银行', 25000);
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, stock_name, market_cap) VALUES (?,?,?,?)').run('601939', '801780.SI', '建设银行', 20000);
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, stock_name, market_cap) VALUES (?,?,?,?)').run('601288', '801780.SI', '农业银行', 18000);
}

function seedKlines(db) {
  const insert = db.prepare('INSERT OR REPLACE INTO monthly_klines (code, date, open, close, high, low, volume, amount, turnover_rate) VALUES (?,?,?,?,?,?,?,?,?)');
  // 3 只银行股，3 个月数据
  const data = [
    // code, date, open, close, high, low, volume, amount, turnover_rate
    ['601398', '2025-01', 5.0, 5.2, 5.3, 4.9, 1e8, 5.2e8, 1.2],
    ['601398', '2025-02', 5.2, 5.1, 5.4, 5.0, 1.1e8, 5.6e8, 1.3],
    ['601398', '2025-03', 5.1, 5.5, 5.6, 5.0, 1.2e8, 6.6e8, 1.5],
    ['601939', '2025-01', 6.0, 6.3, 6.4, 5.9, 9e7, 5.7e8, 1.0],
    ['601939', '2025-02', 6.3, 6.2, 6.5, 6.1, 8e7, 5.0e8, 0.9],
    ['601939', '2025-03', 6.2, 6.6, 6.7, 6.1, 1e8, 6.6e8, 1.1],
    ['601288', '2025-01', 4.0, 4.1, 4.2, 3.9, 1.5e8, 6.2e8, 1.5],
    ['601288', '2025-02', 4.1, 4.0, 4.3, 3.9, 1.4e8, 5.6e8, 1.4],
    ['601288', '2025-03', 4.0, 4.4, 4.5, 3.9, 1.6e8, 7.0e8, 1.6],
  ];
  for (const d of data) insert.run(...d);
}

test('buildAllSectorKlines: 市值加权合成与手工计算一致', () => {
  const db = setupDb();
  seedMapping(db);
  seedKlines(db);

  const result = buildAllSectorKlines(db, 'monthly', { force: true });
  assert.equal(result.built, 1);
  assert.equal(result.skipped, 0);
  assert.equal(result.errors.length, 0);

  // 验证 2025-01 close 值
  // 权重：601398=25000/63000=0.397, 601939=20000/63000=0.317, 601288=18000/63000=0.286
  // close = 0.397*5.2 + 0.317*6.3 + 0.286*4.1 = 2.064 + 1.997 + 1.173 = 5.234
  const row = db.prepare(
    "SELECT * FROM hs300_sector_klines WHERE sector_code='801780.SI' AND period='monthly' AND date='2025-01'"
  ).get();
  assert.ok(row);
  const expectedClose = 0.397 * 5.2 + 0.317 * 6.3 + 0.286 * 4.1;
  assert.ok(Math.abs(row.close - expectedClose) < 0.5, `close=${row.close}, expected=${expectedClose}`);
  assert.equal(row.member_count, 3);
  assert.equal(row.source, 'composite');
});

test('buildAllSectorKlines: 成分股不足3只时跳过', () => {
  const db = setupDb();
  // 只映射 1 只成分股
  db.prepare('INSERT INTO industries (industry_code, industry_name) VALUES (?, ?)').run('801210.SI', '社会服务');
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, market_cap) VALUES (?,?,?)').run('600888', '801210.SI', 3000);
  db.prepare('INSERT INTO monthly_klines (code, date, close) VALUES (?,?,?)').run('600888', '2025-01', 10);

  const result = buildAllSectorKlines(db, 'monthly', { force: true });
  assert.equal(result.built, 0);

  const count = db.prepare("SELECT COUNT(*) as n FROM hs300_sector_klines WHERE sector_code='801210.SI'").get();
  assert.equal(count.n, 0);
});

test('buildAllSectorKlines: 无 market_cap 成分股被跳过', () => {
  const db = setupDb();
  db.prepare('INSERT INTO industries (industry_code, industry_name) VALUES (?, ?)').run('801780.SI', '银行');
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, market_cap) VALUES (?,?,?)').run('601398', '801780.SI', null); // 无市值
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, market_cap) VALUES (?,?,?)').run('601939', '801780.SI', 20000);
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, market_cap) VALUES (?,?,?)').run('601288', '801780.SI', 18000);
  // 仅 2 只有效市值，<3 → 跳过
  db.prepare('INSERT INTO monthly_klines (code, date, close) VALUES (?,?,?)').run('601939', '2025-01', 6.0);
  db.prepare('INSERT INTO monthly_klines (code, date, close) VALUES (?,?,?)').run('601288', '2025-01', 4.0);

  const result = buildAllSectorKlines(db, 'monthly', { force: true });
  assert.equal(result.built, 0);
});

test('buildAllSectorKlines: 3 个周期均可合成', () => {
  const db = setupDb();
  seedMapping(db);

  // 准备 daily/weekly/monthly 数据
  const insertDaily = db.prepare('INSERT INTO daily_klines (code, date, close) VALUES (?,?,?)');
  const insertWeekly = db.prepare('INSERT INTO weekly_klines (code, date, close) VALUES (?,?,?)');
  const insertMonthly = db.prepare('INSERT INTO monthly_klines (code, date, close) VALUES (?,?,?)');

  for (const code of ['601398', '601939', '601288']) {
    insertDaily.run(code, '2025-05-10', 5.0 + Math.random());
    insertWeekly.run(code, '2025-W20', 5.0 + Math.random());
    insertMonthly.run(code, '2025-05', 5.0 + Math.random());
  }

  for (const period of ['monthly', 'weekly', 'daily']) {
    const result = buildAllSectorKlines(db, period, { force: true });
    assert.equal(result.built, 1, `${period} 应合成 1 行业`);
  }
});
