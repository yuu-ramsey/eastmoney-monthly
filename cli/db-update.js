// ema db update — monthly incremental update (framework, logic to be completed later)

import { getDb, closeDb } from '../lib/db/connection.js';

export async function runDbUpdate(periods = ['monthly', 'weekly', 'daily']) {
  const db = getDb();
  console.log('=== Incremental Update ===');
  console.log('Periods:', periods.join(', '));

  // TODO: query each stock's last record date, fetch incremental klines after that
  console.log('db update framework ready, incremental logic to be completed');

  closeDb();
}
