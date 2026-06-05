// Sector Alpha Predictive Power Matrix — zero LLM calls, pure SQLite offline analysis
// 4 lookback × 4 holding = 16 combos
// Metrics: IC (Spearman) / Hit Rate / Long-Short Sharpe
// Strict walk-forward
import { getDb } from '../lib/db/connection.js';

const db = getDb();

// ============================================================
// 1. Data preparation
// ============================================================

// get all stocks with industry mapping
const stockList = db.prepare(`
  SELECT DISTINCT m.stock_code, i.industry_code
  FROM stock_industry_mapping m
  JOIN industries i ON m.industry_code = i.industry_code
`).all();
console.log(`Stocks with industry mapping: ${stockList.length}`);

// get industry list
const industryList = db.prepare(`SELECT industry_code, industry_name FROM industries`).all();
console.log(`Industries: ${industryList.length}`);

// build stock → industry map
const stockIndustry = new Map();
for (const s of stockList) {
  stockIndustry.set(s.stock_code, s.industry_code);
}

// read all needed monthly klines (from 2016, enough lookback)
const allKlines = db.prepare(`
  SELECT code, date, close FROM monthly_klines
  WHERE code IN (${stockList.map(() => '?').join(',')})
  AND date >= '2016-01'
  ORDER BY code, date
`).all(...stockList.map(s => s.stock_code));
console.log(`Stock monthly records: ${allKlines.length}`);

// read all sector klines (monthly)
const allSectorKlines = db.prepare(`
  SELECT sector_code, date, close FROM hs300_sector_klines
  WHERE period = 'monthly' AND date >= '2016-01'
  ORDER BY sector_code, date
`).all();
console.log(`Sector kline records: ${allSectorKlines.length}`);

// organize as Map<code, Map<date, close>>
const stockPrices = new Map();
for (const k of allKlines) {
  if (!stockPrices.has(k.code)) stockPrices.set(k.code, new Map());
  stockPrices.get(k.code).set(k.date, k.close);
}

const sectorPrices = new Map();
for (const k of allSectorKlines) {
  if (!sectorPrices.has(k.sector_code)) sectorPrices.set(k.sector_code, new Map());
  sectorPrices.get(k.sector_code).set(k.date, k.close);
}

// get eval month list (2018-01 to 2024-12, enough room for holding periods)
const evalMonths = [];
const allDates = [...new Set(allKlines.map(k => k.date))].sort();
for (const d of allDates) {
  if (d >= '2018-01' && d <= '2024-12') evalMonths.push(d);
}
console.log(`Eval months: ${evalMonths.length} (${evalMonths[0]} ~ ${evalMonths[evalMonths.length-1]})`);

// ============================================================
// 2. Utility functions
// ============================================================

// get price N months before a given date (looking back)
function getPriceBefore(prices, date, monthsBack) {
  const allDates = [...prices.keys()].sort();
  const idx = allDates.findIndex(d => d > date);
  const beforeDates = idx >= 0 ? allDates.slice(0, idx) : allDates;
  const targetIdx = beforeDates.length - 1 - monthsBack;
  if (targetIdx < 0) return null;
  return prices.get(beforeDates[targetIdx]);
}

// get price at or before date from ordered date list
function getPriceAtOrBefore(prices, date) {
  const allDates = [...prices.keys()].sort();
  for (let i = allDates.length - 1; i >= 0; i--) {
    if (allDates[i] <= date) return prices.get(allDates[i]);
  }
  return null;
}

// get price N months after a given date (holding period)
function getForwardPrice(prices, date, monthsForward) {
  const allDates = [...prices.keys()].sort();
  const startIdx = allDates.findIndex(d => d > date);
  if (startIdx < 0) return null;
  const targetIdx = startIdx + monthsForward - 1;
  if (targetIdx >= allDates.length) return null;
  return prices.get(allDates[targetIdx]);
}

// compute holding-period return
function getForwardReturn(prices, asOfDate, holdingMonths) {
  const startPrice = getPriceAtOrBefore(prices, asOfDate);
  const endDate = getForwardPrice(prices, asOfDate, holdingMonths);
  if (!startPrice || !endDate || startPrice <= 0) return null;
  return (endDate - startPrice) / startPrice;
}

// compute lookback-period return
function getLookbackReturn(prices, asOfDate, lookbackMonths) {
  // closest price before asOfDate
  const allDates = [...prices.keys()].sort();
  const idx = allDates.findIndex(d => d > asOfDate);
  const beforeDates = idx >= 0 ? allDates.slice(0, idx) : allDates;
  if (beforeDates.length < lookbackMonths + 1) return null;

  const endPrice = prices.get(beforeDates[beforeDates.length - 1]); // closest to asOfDate
  const startPrice = prices.get(beforeDates[beforeDates.length - 1 - lookbackMonths]);
  if (!startPrice || !endPrice || startPrice <= 0) return null;
  return (endPrice - startPrice) / startPrice;
}

// ============================================================
// 3. Build observation data
// ============================================================

const LOOKBACKS = [3, 6, 12, 24];
const HOLDINGS = [1, 3, 6, 12];

// observations: [{ stockCode, industryCode, asOfDate, lookback, alpha, forwardReturn(holding) }]
const observations = [];

let processed = 0;
for (const asOfDate of evalMonths) {
  for (const stock of stockList) {
    const code = stock.stock_code;
    const industry = stock.industry_code;

    const stockP = stockPrices.get(code);
    const sectorP = sectorPrices.get(industry);
    if (!stockP || !sectorP) continue;

    // check asOfDate has data
    if (!getPriceAtOrBefore(stockP, asOfDate)) continue;
    if (!getPriceAtOrBefore(sectorP, asOfDate)) continue;

    for (const lookback of LOOKBACKS) {
      const stockRet = getLookbackReturn(stockP, asOfDate, lookback);
      const sectorRet = getLookbackReturn(sectorP, asOfDate, lookback);
      if (stockRet == null || sectorRet == null) continue;

      const alpha = stockRet - sectorRet;

      // compute forward returns for each holding period
      const fwdRets = {};
      for (const hold of HOLDINGS) {
        const fr = getForwardReturn(stockP, asOfDate, hold);
        if (fr != null) fwdRets[hold] = fr;
      }

      if (Object.keys(fwdRets).length === 0) continue;

      observations.push({
        stockCode: code,
        industryCode: industry,
        asOfDate,
        lookback,
        alpha,
        forwardReturns: fwdRets,
      });
    }
  }
  processed++;
  if (processed % 12 === 0) {
    console.log(`  Progress: ${processed}/${evalMonths.length} months, obs=${observations.length}`);
  }
}

console.log(`Total observations: ${observations.length}`);

// ============================================================
// 4. Compute metrics
// ============================================================

// Spearman rank correlation
function spearmanR(xs, ys) {
  if (xs.length < 10) return null;
  const n = xs.length;

  // Rank x
  const xRanks = new Array(n);
  const xSorted = xs.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  for (let r = 0; r < n; r++) xRanks[xSorted[r].i] = r + 1;

  // Rank y
  const yRanks = new Array(n);
  const ySorted = ys.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  for (let r = 0; r < n; r++) yRanks[ySorted[r].i] = r + 1;

  // Correlation of ranks
  const xMean = (n + 1) / 2;
  const yMean = (n + 1) / 2;
  let num = 0, den1 = 0, den2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xRanks[i] - xMean;
    const dy = yRanks[i] - yMean;
    num += dx * dy;
    den1 += dx * dx;
    den2 += dy * dy;
  }
  if (den1 === 0 || den2 === 0) return null;
  return num / Math.sqrt(den1 * den2);
}

// 4×4 matrix
console.log('\n' + '='.repeat(80));
console.log('=== Sector Alpha Predictive Power Matrix ===');
console.log('='.repeat(80));

const matrix = {};

for (const lookback of LOOKBACKS) {
  for (const holding of HOLDINGS) {
    // filter observations for this combo
    const obs = observations.filter(o => o.lookback === lookback && o.forwardReturns[holding] != null);
    if (obs.length < 50) continue;

    const alphas = obs.map(o => o.alpha);
    const fwdRets = obs.map(o => o.forwardReturns[holding]);

    // IC = Spearman(alpha, forward return)
    const ic = spearmanR(alphas, fwdRets);

    // Hit Rate = P(forward return > median | alpha > +5pp)
    const allFwd = fwdRets.slice().sort((a, b) => a - b);
    const medianFwd = allFwd[Math.floor(allFwd.length / 2)];
    const strongAlphaObs = obs.filter(o => o.alpha > 0.05); // alpha > +5pp
    const hitCount = strongAlphaObs.filter(o => o.forwardReturns[holding] > medianFwd).length;
    const hitRate = strongAlphaObs.length > 10 ? hitCount / strongAlphaObs.length : null;

    // Long-Short Sharpe: long top 30% alpha, short bottom 30% alpha
    const sorted = [...obs].sort((a, b) => a.alpha - b.alpha);
    const cut30 = Math.floor(sorted.length * 0.3);
    const longStocks = sorted.slice(-cut30);
    const shortStocks = sorted.slice(0, cut30);

    const longRets = longStocks.map(o => o.forwardReturns[holding]);
    const shortRets = shortStocks.map(o => o.forwardReturns[holding]);

    // equal-weight portfolio: long - short
    const spreadRets = [];
    for (let i = 0; i < Math.min(longRets.length, shortRets.length); i++) {
      // group by month to compute average spread
      // simplified: directly compute mean and std of all spreads
    }
    // simplified calculation: mean(long returns) - mean(short returns), divided by spread std
    const longMean = longRets.reduce((a, b) => a + b, 0) / longRets.length;
    const shortMean = shortRets.reduce((a, b) => a + b, 0) / shortRets.length;
    const spreadMean = longMean - shortMean;

    // spread std (pooled)
    const allSpreads = longRets.map((lr, i) => i < shortRets.length ? lr - shortRets[i] : null).filter(v => v != null);
    const spreadStd = Math.sqrt(allSpreads.reduce((s, v) => s + (v - spreadMean) ** 2, 0) / (allSpreads.length - 1 || 1));
    // annualized Sharpe (monthly → annual: × sqrt(12/holding))
    const annualFactor = Math.sqrt(12 / holding);
    const sharpe = spreadStd > 0 ? spreadMean / spreadStd * annualFactor : null;

    const key = `${lookback}m_${holding}m`;
    matrix[key] = { lookback, holding, n: obs.length, ic, hitRate, sharpe };
    console.log(`  ${key}: n=${obs.length} IC=${ic != null ? ic.toFixed(4) : 'N/A'} HR=${hitRate != null ? (hitRate*100).toFixed(1)+'%' : 'N/A'} Sharpe=${sharpe != null ? sharpe.toFixed(3) : 'N/A'}`);
  }
}

// ============================================================
// 5. Formatted matrix output
// ============================================================

console.log('\n' + '='.repeat(80));
console.log('=== IC Matrix (Spearman) ===');
console.log('='.repeat(80));
let header = '| lookback \\ holding |';
for (const h of HOLDINGS) header += ` ${h}m |`;
console.log(header);
console.log('|' + header.replace(/[^|]/g, '-').replace(/\|/g, '|'));
for (const lb of LOOKBACKS) {
  let row = `| ${lb}m |`;
  for (const h of HOLDINGS) {
    const key = `${lb}m_${h}m`;
    const ic = matrix[key]?.ic;
    row += ` ${ic != null ? ic.toFixed(4) : '  N/A '} |`;
  }
  console.log(row);
}

console.log('\n' + '='.repeat(80));
console.log('=== Hit Rate Matrix (alpha > +5pp → forward > median) ===');
console.log('='.repeat(80));
header = '| lookback \\ holding |';
for (const h of HOLDINGS) header += ` ${h}m |`;
console.log(header);
console.log('|' + header.replace(/[^|]/g, '-').replace(/\|/g, '|'));
for (const lb of LOOKBACKS) {
  let row = `| ${lb}m |`;
  for (const h of HOLDINGS) {
    const key = `${lb}m_${h}m`;
    const hr = matrix[key]?.hitRate;
    row += ` ${hr != null ? (hr*100).toFixed(1)+'%' : ' N/A '} |`;
  }
  console.log(row);
}

console.log('\n' + '='.repeat(80));
console.log('=== Long-Short Sharpe (top30% - bottom30% alpha, annualized) ===');
console.log('='.repeat(80));
header = '| lookback \\ holding |';
for (const h of HOLDINGS) header += ` ${h}m |`;
console.log(header);
console.log('|' + header.replace(/[^|]/g, '-').replace(/\|/g, '|'));
for (const lb of LOOKBACKS) {
  let row = `| ${lb}m |`;
  for (const h of HOLDINGS) {
    const key = `${lb}m_${h}m`;
    const sr = matrix[key]?.sharpe;
    row += ` ${sr != null ? sr.toFixed(3) : '  N/A '} |`;
  }
  console.log(row);
}

// best combos
console.log('\n' + '='.repeat(80));
console.log('=== Best Combos ===');
console.log('='.repeat(80));
const validEntries = Object.values(matrix).filter(m => m.ic != null);
validEntries.sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic));
console.log('\nTop 5 by |IC|:');
for (const m of validEntries.slice(0, 5)) {
  console.log(`  lookback=${m.lookback}m holding=${m.holding}m IC=${m.ic.toFixed(4)} HR=${(m.hitRate*100).toFixed(1)}% Sharpe=${m.sharpe.toFixed(3)} n=${m.n}`);
}

validEntries.sort((a, b) => b.sharpe - a.sharpe);
console.log('\nTop 5 by Sharpe:');
for (const m of validEntries.slice(0, 5)) {
  console.log(`  lookback=${m.lookback}m holding=${m.holding}m IC=${m.ic.toFixed(4)} HR=${(m.hitRate*100).toFixed(1)}% Sharpe=${m.sharpe.toFixed(3)} n=${m.n}`);
}

// conclusion
const bestIC = validEntries.reduce((best, m) => Math.abs(m.ic) > Math.abs(best.ic) ? m : best, validEntries[0]);
if (bestIC.ic > 0.05) {
  console.log(`\nConclusion A: predictive combo found (lookback=${bestIC.lookback}m, holding=${bestIC.holding}m, IC=${bestIC.ic.toFixed(4)})`);
} else if (bestIC.ic < -0.05) {
  console.log(`\nConclusion C: significant negative prediction (lookback=${bestIC.lookback}m, holding=${bestIC.holding}m, IC=${bestIC.ic.toFixed(4)})`);
} else {
  console.log(`\nConclusion B: all combos IC in [-0.05, +0.05], sector alpha has no predictive power. Best: lookback=${bestIC.lookback}m holding=${bestIC.holding}m IC=${bestIC.ic.toFixed(4)}`);
}
