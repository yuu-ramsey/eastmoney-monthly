// ema db status — database statistics

import { getDb, closeDb, getDbPath } from '../lib/db/connection.js';

export async function showDbStatus() {
  const db = getDb();

  console.log('=== Database Status ===');
  console.log('File:', getDbPath());

  // stock count
  const stockCount = db.prepare('SELECT count(*) as c FROM stocks').get();
  console.log('Stock count:', stockCount.c);

  // kline counts by period + earliest/latest dates
  const periods = [
    { key: 'monthly', table: 'monthly_klines' },
    { key: 'weekly', table: 'weekly_klines' },
    { key: 'daily', table: 'daily_klines' },
    { key: '60min', table: 'kline_60min' },
  ];

  for (const p of periods) {
    const count = db.prepare(`SELECT count(*) as c FROM ${p.table}`).get();
    const range = db.prepare(`SELECT min(date) as earliest, max(date) as latest FROM ${p.table}`).get();
    console.log(`${p.key}: ${count.c} bars | ${range.earliest || '-'} ~ ${range.latest || '-'}`);
  }

  // last update time (stocks table)
  const lastUpdate = db.prepare('SELECT max(last_updated) as t FROM stocks').get();
  console.log('Last update:', lastUpdate.t || 'never');

  // monthly staleness check (stocks whose last monthly bar > 90 days ago)
  const stale = db.prepare(`
    SELECT code FROM monthly_klines
    GROUP BY code HAVING max(date) < ?
  `).all(new Date().toISOString().slice(0, 7));
  if (stale.length > 0) {
    console.log(`Stocks with potentially stale data (> current month): ${stale.length}`);
    if (stale.length <= 10) {
      for (const s of stale) console.log(`  ${s.code}`);
    }
  }

  closeDb();
}
