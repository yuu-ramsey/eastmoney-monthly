// SW industry alpha calculation — individual stock excess return vs HS300 internal industry benchmark
// Strict walk-forward: data after asOfDate is invisible

/**
 * 计算单只股票相对其 HS300 内行业基准的超额收益
 * @param {object} db - better-sqlite3 实例
 * @param {string} code - 个股代码，如 "600519"
 * @param {string} period - 'monthly' | 'weekly' | 'daily'
 * @param {number} lookback - 回顾期数
 * @param {string|null} asOfDate - 截止日期 YYYY-MM-DD，null 用最新
 * @returns {object|null} alpha 数据，无行业映射时返回 null
 */
export function calcSectorAlpha(db, code, period = 'monthly', lookback = 12, asOfDate = null) {
  // 查行业映射
  const mapping = db.prepare(
    'SELECT industry_code FROM stock_industry_mapping WHERE stock_code = ?'
  ).get(code);
  if (!mapping) return null;

  const sectorCode = mapping.industry_code;

  // 行业名称
  const indInfo = db.prepare(
    'SELECT industry_name FROM industries WHERE industry_code = ?'
  ).get(sectorCode);
  const sectorName = indInfo ? indInfo.industry_name : sectorCode;

  // 行业内 HS300 成分股总数
  const sectorTotal = db.prepare(
    'SELECT COUNT(*) as n FROM stock_industry_mapping WHERE industry_code = ?'
  ).get(sectorCode).n;

  const klineTable = periodTable(period);

  // 行业 K 线（截止 asOfDate）
  const sectorKlines = getKlines(db, 'hs300_sector_klines', sectorCode, period, lookback + 1, asOfDate);
  // 个股 K 线
  const stockKlines = getStockKlines(db, klineTable, code, lookback + 1, asOfDate);

  if (sectorKlines.length < 2 || stockKlines.length < 2) return null;

  // walk-forward：确保两端数据的最后日期一致（取较早的）
  const sectorLast = sectorKlines[sectorKlines.length - 1];
  const stockLast = stockKlines[stockKlines.length - 1];
  const cutoffDate = sectorLast.date < stockLast.date ? sectorLast.date : stockLast.date;

  // 在 cutoff 之前取 lookback+1 根
  const sectorSlice = sectorKlines.filter(k => k.date <= cutoffDate).slice(-(lookback + 1));
  const stockSlice = stockKlines.filter(k => k.date <= cutoffDate).slice(-(lookback + 1));

  if (sectorSlice.length < 2 || stockSlice.length < 2) return null;

  // 计算涨跌幅
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
 * 批量计算所有 HS300 股票的 alpha
 * @returns {Array<object>} 按 alpha 降序排列，含排名
 */
export function calcAllSectorAlpha(db, period = 'monthly', lookback = 12, asOfDate = null) {
  // 获取所有有行业映射的股票
  const stocks = db.prepare(
    'SELECT DISTINCT stock_code FROM stock_industry_mapping'
  ).all();

  const results = [];
  for (const s of stocks) {
    const alpha = calcSectorAlpha(db, s.stock_code, period, lookback, asOfDate);
    if (alpha) results.push(alpha);
  }

  // 按行业内排名
  const bySector = {};
  for (const r of results) {
    if (!bySector[r.sector_code]) bySector[r.sector_code] = [];
    bySector[r.sector_code].push(r);
  }

  for (const [sectorCode, items] of Object.entries(bySector)) {
    // 按 alpha 降序
    items.sort((a, b) => b.hs300_sector_alpha - a.hs300_sector_alpha);
    for (let i = 0; i < items.length; i++) {
      items[i].hs300_sector_rank = i + 1;
      items[i].hs300_sector_percentile = round2(((i + 1) / items.length) * 100);
    }
  }

  // 全局按 alpha 降序
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
  // 从尾部取 limit 条
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
  if (!t) throw new Error(`未知周期: ${period}`);
  return t;
}

function round2(n) {
  return Math.round(n * 100) / 100;
}
