// SW industry alpha calculation — individual stock excess return vs HS300 internal industry benchmark
// Strict walk-forward: data after asOfDate is invisible

/**
 * Compute excess return of a single stock vs its HS300 industry benchmark
 * @param {object} db - better-sqlite3 instance
 * @param {string} code - Stock code, e.g. "600519"
 * @param {string} period - 'monthly' | 'weekly' | 'daily'
 * @param {number} lookback - Lookback periods
 * @param {string|null} asOfDate - Cutoff date YYYY-MM-DD, null uses latest
 * @returns {object|null} alpha data, null if no industry mapping
 */
export function calcSectorAlpha(db, code, period = 'monthly', lookback = 12, asOfDate = null) {
  // Look up industry mapping
  const mapping = db.prepare(
    'SELECT industry_code FROM stock_industry_mapping WHERE stock_code = ?'
  ).get(code);
  if (!mapping) return null;

  const sectorCode = mapping.industry_code;

  // Industry name
  const indInfo = db.prepare(
    'SELECT industry_name FROM industries WHERE industry_code = ?'
  ).get(sectorCode);
  const sectorName = indInfo ? indInfo.industry_name : sectorCode;

  // Total HS300 constituents in this industry
  const sectorTotal = db.prepare(
    'SELECT COUNT(*) as n FROM stock_industry_mapping WHERE industry_code = ?'
  ).get(sectorCode).n;

  const klineTable = periodTable(period);

  // Sector klines (up to asOfDate)
  const sectorKlines = getKlines(db, 'hs300_sector_klines', sectorCode, period, lookback + 1, asOfDate);
  // Stock klines
  const stockKlines = getStockKlines(db, klineTable, code, lookback + 1, asOfDate);

  if (sectorKlines.length < 2 || stockKlines.length < 2) return null;

  // walk-forward: ensure last dates of both sides match (use earlier one)
  const sectorLast = sectorKlines[sectorKlines.length - 1];
  const stockLast = stockKlines[stockKlines.length - 1];
  const cutoffDate = sectorLast.date < stockLast.date ? sectorLast.date : stockLast.date;

  // Take lookback+1 bars before cutoff
  const sectorSlice = sectorKlines.filter(k => k.date <= cutoffDate).slice(-(lookback + 1));
  const stockSlice = stockKlines.filter(k => k.date <= cutoffDate).slice(-(lookback + 1));

  if (sectorSlice.length < 2 || stockSlice.length < 2) return null;

  // Calculate returns
  const sectorFirst = sectorSlice[0];
  const sectorLastEff = sectorSlice[sectorSlice.length - 1];
  const stockFirst = stockSlice[0];
  const stockLastEff = stockSlice[stockSlice.length - 1];

  const sectorReturn = sectorFirst.close !== 0
    ? ((sectorLastEff.close - sectorFirst.close) / sectorFirst.close) * 100
    : 0;
  const stockReturn = stockFirst.close !== 0
    ? ((stockLastEff.close - stockFirst.close) / stockFirst.close) * 100
    : 0;

  const alpha = stockReturn - sectorReturn;

  return {
    code,
    sector_name: sectorName,
    sector_code: sectorCode,
    hs300_sector_alpha: round2(alpha),
    hs300_sector_total: sectorTotal,
    stock_return: round2(stockReturn),
    sector_return: round2(sectorReturn),
    lookback: sectorSlice.length - 1,
    period,
    as_of_date: cutoffDate,
  };
}

/**
 * Batch-calculate alpha for all HS300 stocks
 * @returns {Array<object>} sorted by alpha descending, includes rank
 */
export function calcAllSectorAlpha(db, period = 'monthly', lookback = 12, asOfDate = null) {
  // Get all stocks with industry mapping
  const stocks = db.prepare(
    'SELECT DISTINCT stock_code FROM stock_industry_mapping'
  ).all();

  const results = [];
  for (const s of stocks) {
    const alpha = calcSectorAlpha(db, s.stock_code, period, lookback, asOfDate);
    if (alpha) results.push(alpha);
  }

  // Rank within industry
  const bySector = {};
  for (const r of results) {
    if (!bySector[r.sector_code]) bySector[r.sector_code] = [];
    bySector[r.sector_code].push(r);
  }

  for (const [sectorCode, items] of Object.entries(bySector)) {
    // Sort by alpha descending
    items.sort((a, b) => b.hs300_sector_alpha - a.hs300_sector_alpha);
    for (let i = 0; i < items.length; i++) {
      items[i].hs300_sector_rank = i + 1;
      items[i].hs300_sector_percentile = round2(((i + 1) / items.length) * 100);
    }
  }

  // Global sort by alpha descending
  results.sort((a, b) => b.hs300_sector_alpha - a.hs300_sector_alpha);
  return results;
}

function getKlines(db, table, code, period, limit, asOfDate) {
  let sql = `SELECT date, close FROM ${table} WHERE sector_code = ? AND period = ?`;
  const params = [code, period];
  if (asOfDate) {
    sql += ' AND date <= ?';
    params.push(asOfDate);
  }
  sql += ' ORDER BY date ASC';
  // Take last limit rows from tail
  const all = db.prepare(sql).all(...params);
  return all.slice(-limit);
}

function getStockKlines(db, table, code, limit, asOfDate) {
  let sql = `SELECT date, close FROM ${table} WHERE code = ?`;
  const params = [code];
  if (asOfDate) {
    sql += ' AND date <= ?';
    params.push(asOfDate);
  }
  sql += ' ORDER BY date ASC';
  const all = db.prepare(sql).all(...params);
  return all.slice(-limit);
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

function round2(n) {
  return Math.round(n * 100) / 100;
}
