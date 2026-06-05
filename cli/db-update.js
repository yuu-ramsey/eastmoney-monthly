// ema db update — 月度增量更新（框架，后续补全具体逻辑）

import { getDb, closeDb } from '../lib/db/connection.js';

export async function runDbUpdate(periods = ['monthly', 'weekly', 'daily']) {
  const db = getDb();
  console.log('=== 增量更新 ===');
  console.log('周期:', periods.join(', '));

  // TODO: 查每只股票最后一条记录的日期，fetch 之后的增量 K 线
  console.log('db update 框架已就绪，具体增量逻辑后续补全');

  closeDb();
}
