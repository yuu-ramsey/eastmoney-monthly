// ema db status — 数据库统计信息

import { getDb, closeDb, getDbPath } from '../lib/db/connection.js';

export async function showDbStatus() {
  const db = getDb();

  console.log('=== 数据库状态 ===');
  console.log('文件:', getDbPath());

  // 股票数
  const stockCount = db.prepare('SELECT count(*) as c FROM stocks').get();
  console.log('股票数:', stockCount.c);

  // 各周期 K 线数 + 最早/最晚日期
  const periods = [
    { key: 'monthly', table: 'monthly_klines' },
    { key: 'weekly', table: 'weekly_klines' },
    { key: 'daily', table: 'daily_klines' },
    { key: '60min', table: 'kline_60min' },
  ];

  for (const p of periods) {
    const count = db.prepare(`SELECT count(*) as c FROM ${p.table}`).get();
    const range = db.prepare(`SELECT min(date) as earliest, max(date) as latest FROM ${p.table}`).get();
    console.log(`${p.key}: ${count.c} 根 | ${range.earliest || '-'} ~ ${range.latest || '-'}`);
  }

  // 最后更新时间（stocks 表）
  const lastUpdate = db.prepare('SELECT max(last_updated) as t FROM stocks').get();
  console.log('最后更新:', lastUpdate.t || '从未');

  // 月度缺失检查（月线最后日期距今 > 90 天的股票）
  const stale = db.prepare(`
    SELECT code FROM monthly_klines
    GROUP BY code HAVING max(date) < ?
  `).all(new Date().toISOString().slice(0, 7));
  if (stale.length > 0) {
    console.log(`数据可能过时的股票 (> 本月): ${stale.length} 只`);
    if (stale.length <= 10) {
      for (const s of stale) console.log(`  ${s.code}`);
    }
  }

  closeDb();
}
