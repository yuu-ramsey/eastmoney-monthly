// Sector kline synthesis — market-cap-weighted aggregation of constituent stocks
// Synthesize hs300_sector_klines from stock_industry_mapping + daily/monthly/weekly_klines

/**
 * Synthesize K-lines for all industries
 * @param {object} db - better-sqlite3 instance
 * @param {string} period - 'monthly' | 'weekly' | 'daily'
 * @param {object} options - { force, onProgress }
 * @returns {{built: number, skipped: number, errors: string[]}}
 */
export function buildAllSectorKlines(db, period = 'monthly', options = {}) {
  const { force = false } = options;
  const table = periodTable(period);

  // Get all mapped industries
  const industries = db.prepare(`
    SELECT DISTINCT m.industry_code, i.industry_name
    FROM stock_industry_mapping m
    JOIN industries i ON i.industry_code = m.industry_code
    ORDER BY m.industry_code
  `).all();

  let built = 0;
  let skipped = 0;
  const errors = [];

  for (const ind of industries) {
    try {
      const result = buildOneSector(db, ind.industry_code, period, table, force);
      if (result.built) built++;
      else skipped++;
    } catch (err) {
      errors.push(`${ind.industry_code}: ${err.message}`);
    }
  }

  return { built, skipped, errors };
}

/**
 * Build klines for a single sector
 * @returns {{built: boolean, dates: number}}
 */
function buildOneSector(db, industryCode, period, table, force) {
  // Check if already exists
  const existing = db.prepare(
    'SELECT COUNT(*) as n FROM hs300_sector_klines WHERE sector_code = ? AND period = ?'
  ).get(industryCode, period);
  if (existing.n > 0 && !force) {
    return { built: false, dates: existing.n };
  }

  if (force) {
    db.prepare('DELETE FROM hs300_sector_klines WHERE sector_code = ? AND period = ?').run(industryCode, period);
  }

  // Get constituent stocks and market-cap weights
  const members = db.prepare(`
    SELECT stock_code, market_cap FROM stock_industry_mapping
    WHERE industry_code = ? AND market_cap IS NOT NULL
  `).all(industryCode);

  if (members.length < 3) {
    // Too few constituents, skip
    return { built: false, dates: 0 };
  }

  // Total weight
  const totalWeight = members.reduce((s, m) => s + m.market_cap, 0);
  if (totalWeight <= 0) return { built: false, dates: 0 };

  const weights = new Map(members.map(m => [m.stock_code, m.market_cap / totalWeight]));

  // Collect all constituent stock dates (union)
  const allDates = new Set();
  for (const m of members) {
    const rows = db.prepare(
      `SELECT date FROM ${table} WHERE code = ? ORDER BY date`
    ).all(m.stock_code);
    for (const r of rows) allDates.add(r.date);
  }

  if (allDates.size === 0) return { built: false, dates: 0 };

  const dates = [...allDates].sort();
  const klines = [];
  const insertStmt = db.prepare(`
    INSERT OR REPLACE INTO hs300_sector_klines
    (sector_code, period, date, open, close, high, low, volume, amount, amplitude, change_percent, change_amount, turnover_rate, member_count, source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'composite')
  `);

  const insertAll = db.transaction(() => {
    for (const date of dates) {
      let sumOpen = 0, sumClose = 0, sumHigh = 0, sumLow = 0;
      let sumVolume = 0, sumAmount = 0, sumTurnover = 0;
      let activeCount = 0;
      let totalW = 0;

      for (const m of members) {
        const row = db.prepare(
          `SELECT * FROM ${table} WHERE code = ? AND date = ?`
        ).get(m.stock_code, date);
        if (!row) continue;

        const w = weights.get(m.stock_code);
        sumOpen += (row.open || 0) * w;
        sumClose += (row.close || 0) * w;
        sumHigh += (row.high || 0) * w;
        sumLow += (row.low || 0) * w;
        sumVolume += (row.volume || 0);
        sumAmount += (row.amount || 0);
        sumTurnover += (row.turnover_rate || 0) * w;
        totalW += w;
        activeCount++;
      }

      // Only write when active constituents >= 3
      if (activeCount < 3) continue;

      // Normalize (totalW may be < 1 because some stocks lack data)
      const norm = totalW > 0 ? 1 / totalW : 1;
      const open = sumOpen * norm;
      const close = sumClose * norm;
      const high = sumHigh * norm;
      const low = sumLow * norm;

      // Calculate change percent (relative to previous kline)
      const prevClose = klines.length > 0 ? klines[klines.length - 1].close : close;
      const changePercent = prevClose !== 0 ? ((close - prevClose) / prevClose) * 100 : 0;
      const changeAmount = close - prevClose;
      const amplitude = low !== 0 ? ((high - low) / low) * 100 : 0;

      insertStmt.run(
        industryCode, period, date,
        open, close, high, low,
        sumVolume, sumAmount,
        amplitude, changePercent, changeAmount,
        sumTurnover * norm, activeCount
      );

      klines.push({ date, close });
    }
  });

  insertAll();
  return { built: true, dates: klines.length };
}

function periodTable(period) {
  const map = {
    monthly: 'monthly_klines',
    weekly: 'weekly_klines',
    daily: 'daily_klines',
  };
  const t = map[period];
  if (!t) throw new Error(`Unknown period: ${period}`);
  return t;
}
