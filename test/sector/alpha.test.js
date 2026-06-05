// sector/alpha.js 测试 — 手工构造数据库验证 alpha 计算

import { test } from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { initSchema } from '../../lib/db/schema.js';
import { calcSectorAlpha, calcAllSectorAlpha } from '../../lib/sector/alpha.js';

function setupDb() {
  const db = new Database(':memory:');
  initSchema(db);
  return db;
}

function seedMapping(db) {
  // 食品饮料行业：3只成分股，不同市值
  db.prepare('INSERT INTO industries (industry_code, industry_name) VALUES (?, ?)').run('801120.SI', '食品饮料');
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, stock_name, market_cap) VALUES (?,?,?,?)').run('600519', '801120.SI', '贵州茅台', 25000);
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, stock_name, market_cap) VALUES (?,?,?,?)').run('000858', '801120.SI', '五粮液', 8000);
  db.prepare('INSERT INTO stock_industry_mapping (stock_code, industry_code, stock_name, market_cap) VALUES (?,?,?,?)').run('600809', '801120.SI', '山西汾酒', 5000);
}

function seedKlines(db) {
  // 个股月线：3只股票各 13 个月
  const stocks = [
    { code: '600519', closes: [1600, 1580, 1550, 1500, 1520, 1480, 1450, 1400, 1420, 1380, 1350, 1400, 1450] },
    { code: '000858', closes: [140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 92, 95] },
    { code: '600809', closes: [200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 155, 160] },
  ];
  const insert = db.prepare('INSERT OR REPLACE INTO monthly_klines (code, date, close) VALUES (?,?,?)');
  for (const s of stocks) {
    for (let i = 0; i < s.closes.length; i++) {
      const month = String(i + 1).padStart(2, '0');
      insert.run(s.code, `2025-${month}`, s.closes[i]);
    }
  }

  // 行业K线：市值加权合成（验证手工计算）
  // 权重：600519=25000/(25000+8000+5000)=0.658, 000858=0.211, 600809=0.132
  const sectorCloses = [
    0.658*1600 + 0.211*140 + 0.132*200, // = 1052.8 + 29.54 + 26.4 = 1108.74
    0.658*1580 + 0.211*135 + 0.132*195, // = 1039.64 + 28.485 + 25.74 = 1093.87
    0.658*1550 + 0.211*130 + 0.132*190, // = 1019.9 + 27.43 + 25.08 = 1072.41
    0.658*1500 + 0.211*125 + 0.132*185,
    0.658*1520 + 0.211*120 + 0.132*180,
    0.658*1480 + 0.211*115 + 0.132*175,
    0.658*1450 + 0.211*110 + 0.132*170,
    0.658*1400 + 0.211*105 + 0.132*165,
    0.658*1420 + 0.211*100 + 0.132*160,
    0.658*1380 + 0.211*95  + 0.132*155,
    0.658*1350 + 0.211*90  + 0.132*150,
    0.658*1400 + 0.211*92  + 0.132*155,
    0.658*1450 + 0.211*95  + 0.132*160,
  ];
  const insertS = db.prepare('INSERT OR REPLACE INTO hs300_sector_klines (sector_code, period, date, close) VALUES (?,?,?,?)');
  for (let i = 0; i < sectorCloses.length; i++) {
    const month = String(i + 1).padStart(2, '0');
    insertS.run('801120.SI', 'monthly', `2025-${month}`, sectorCloses[i]);
  }
}

test('calcSectorAlpha: 茅台 12 月 alpha 与手工计算结果一致', () => {
  const db = setupDb();
  seedMapping(db);
  seedKlines(db);

  const result = calcSectorAlpha(db, '600519', 'monthly', 12);
  assert.ok(result);
  assert.equal(result.code, '600519');
  assert.equal(result.sector_name, '食品饮料');
  assert.equal(result.sector_code, '801120.SI');

  // 茅台：首月 1600 → 末月 1450，涨幅 = (1450-1600)/1600*100 = -9.375%
  // 行业：首月 1108.74 → 末月 995.27，涨幅 = -10.23%
  // alpha = -9.38 - (-10.23) = +0.86pp
  assert.ok(Math.abs(result.hs300_sector_alpha - 0.86) < 0.5, `alpha=${result.hs300_sector_alpha}`);
  assert.equal(result.hs300_sector_total, 3);
  assert.equal(result.period, 'monthly');
  assert.equal(result.lookback, 12);
});

test('calcSectorAlpha: 无行业映射返回 null', () => {
  const db = setupDb();
  const result = calcSectorAlpha(db, '999999', 'monthly', 12);
  assert.equal(result, null);
});

test('calcSectorAlpha: 行业 K 线不足 2 根返回 null', () => {
  const db = setupDb();
  seedMapping(db);
  // 只写 1 根 K 线
  db.prepare('INSERT INTO monthly_klines (code, date, close) VALUES (?,?,?)').run('600519', '2025-01', 1600);
  db.prepare('INSERT INTO hs300_sector_klines (sector_code, period, date, close) VALUES (?,?,?,?)').run('801120.SI', 'monthly', '2025-01', 1100);

  const result = calcSectorAlpha(db, '600519', 'monthly', 12);
  assert.equal(result, null);
});

test('calcSectorAlpha: asOfDate 限制生效（walk-forward）', () => {
  const db = setupDb();
  seedMapping(db);
  seedKlines(db);

  // 限定到 2025-06，只看前 6 个月
  const result = calcSectorAlpha(db, '600519', 'monthly', 6, '2025-06');
  assert.ok(result);
  assert.equal(result.as_of_date, '2025-06');
});

test('calcAllSectorAlpha: 返回含排名和百分位', () => {
  const db = setupDb();
  seedMapping(db);
  seedKlines(db);

  const results = calcAllSectorAlpha(db, 'monthly', 12);
  assert.ok(results.length >= 2);

  // 验证有排名字段
  for (const r of results) {
    assert.ok(typeof r.hs300_sector_rank === 'number', `${r.code} 缺 rank`);
    assert.ok(typeof r.hs300_sector_percentile === 'number', `${r.code} 缺 percentile`);
  }
  // 按 alpha 降序排列
  for (let i = 1; i < results.length; i++) {
    assert.ok(results[i-1].hs300_sector_alpha >= results[i].hs300_sector_alpha);
  }
});
