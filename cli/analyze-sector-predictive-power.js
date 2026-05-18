// Sector Alpha 预测力矩阵 — 零 LLM 调用，纯 SQLite 离线分析
// 4 lookback × 4 holding = 16 组合
// 指标: IC (Spearman) / Hit Rate / Long-Short Sharpe
// 严格 walk-forward
import { getDb } from '../lib/db/connection.js';

const db = getDb();

// ============================================================
// 1. 数据准备
// ============================================================

// 获取所有有行业映射的股票
const stockList = db.prepare(`
  SELECT DISTINCT m.stock_code, i.industry_code
  FROM stock_industry_mapping m
  JOIN industries i ON m.industry_code = i.industry_code
`).all();
console.log(`有行业映射的股票: ${stockList.length}`);

// 获取行业列表
const industryList = db.prepare(`SELECT industry_code, industry_name FROM industries`).all();
console.log(`行业数: ${industryList.length}`);

// 建立 stock → industry 映射
const stockIndustry = new Map();
for (const s of stockList) {
  stockIndustry.set(s.stock_code, s.industry_code);
}

// 读取所有需要的月度 K 线 (2016 年起，留足 lookback)
const allKlines = db.prepare(`
  SELECT code, date, close FROM monthly_klines
  WHERE code IN (${stockList.map(() => '?').join(',')})
  AND date >= '2016-01'
  ORDER BY code, date
`).all(...stockList.map(s => s.stock_code));
console.log(`个股月线记录: ${allKlines.length}`);

// 读取所有 sector klines (monthly)
const allSectorKlines = db.prepare(`
  SELECT sector_code, date, close FROM hs300_sector_klines
  WHERE period = 'monthly' AND date >= '2016-01'
  ORDER BY sector_code, date
`).all();
console.log(`行业 K 线记录: ${allSectorKlines.length}`);

// 组织为 Map<code, Map<date, close>>
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

// 获取评估月份列表 (2018-01 至 2024-12, 留足 holding)
const evalMonths = [];
const allDates = [...new Set(allKlines.map(k => k.date))].sort();
for (const d of allDates) {
  if (d >= '2018-01' && d <= '2024-12') evalMonths.push(d);
}
console.log(`评估月份: ${evalMonths.length} (${evalMonths[0]} ~ ${evalMonths[evalMonths.length-1]})`);

// ============================================================
// 2. 辅助函数
// ============================================================

// 找某日期之前第 N 个月的价格 (往前找)
function getPriceBefore(prices, date, monthsBack) {
  const allDates = [...prices.keys()].sort();
  const idx = allDates.findIndex(d => d > date);
  const beforeDates = idx >= 0 ? allDates.slice(0, idx) : allDates;
  const targetIdx = beforeDates.length - 1 - monthsBack;
  if (targetIdx < 0) return null;
  return prices.get(beforeDates[targetIdx]);
}

// 从有序日期列表中，取 date 之前 dates 的 price
function getPriceAtOrBefore(prices, date) {
  const allDates = [...prices.keys()].sort();
  for (let i = allDates.length - 1; i >= 0; i--) {
    if (allDates[i] <= date) return prices.get(allDates[i]);
  }
  return null;
}

// 找某日期之后第 N 个月的价格 (holding period)
function getForwardPrice(prices, date, monthsForward) {
  const allDates = [...prices.keys()].sort();
  const startIdx = allDates.findIndex(d => d > date);
  if (startIdx < 0) return null;
  const targetIdx = startIdx + monthsForward - 1;
  if (targetIdx >= allDates.length) return null;
  return prices.get(allDates[targetIdx]);
}

// 计算持有期收益率
function getForwardReturn(prices, asOfDate, holdingMonths) {
  const startPrice = getPriceAtOrBefore(prices, asOfDate);
  const endDate = getForwardPrice(prices, asOfDate, holdingMonths);
  if (!startPrice || !endDate || startPrice <= 0) return null;
  return (endDate - startPrice) / startPrice;
}

// 计算回顾期收益率
function getLookbackReturn(prices, asOfDate, lookbackMonths) {
  // asOfDate 之前的最接近价格
  const allDates = [...prices.keys()].sort();
  const idx = allDates.findIndex(d => d > asOfDate);
  const beforeDates = idx >= 0 ? allDates.slice(0, idx) : allDates;
  if (beforeDates.length < lookbackMonths + 1) return null;

  const endPrice = prices.get(beforeDates[beforeDates.length - 1]); // 最接近 asOfDate 的
  const startPrice = prices.get(beforeDates[beforeDates.length - 1 - lookbackMonths]);
  if (!startPrice || !endPrice || startPrice <= 0) return null;
  return (endPrice - startPrice) / startPrice;
}

// ============================================================
// 3. 构建观察数据
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

    // 检查 asOfDate 有数据
    if (!getPriceAtOrBefore(stockP, asOfDate)) continue;
    if (!getPriceAtOrBefore(sectorP, asOfDate)) continue;

    for (const lookback of LOOKBACKS) {
      const stockRet = getLookbackReturn(stockP, asOfDate, lookback);
      const sectorRet = getLookbackReturn(sectorP, asOfDate, lookback);
      if (stockRet == null || sectorRet == null) continue;

      const alpha = stockRet - sectorRet;

      // 计算各持有期的 forward returns
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
    console.log(`  进度: ${processed}/${evalMonths.length} 月, obs=${observations.length}`);
  }
}

console.log(`总观察数: ${observations.length}`);

// ============================================================
// 4. 计算指标
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

// 4×4 矩阵
console.log('\n' + '='.repeat(80));
console.log('=== Sector Alpha 预测力矩阵 ===');
console.log('='.repeat(80));

const matrix = {};

for (const lookback of LOOKBACKS) {
  for (const holding of HOLDINGS) {
    // 筛选该组合的观察
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

    // 等权组合: long - short
    const spreadRets = [];
    for (let i = 0; i < Math.min(longRets.length, shortRets.length); i++) {
      // 按月分组求平均 spread
      // 简化: 直接算所有 spread 的均值和标准差
    }
    // 简化计算: 所有长期收益的均值 - 短期收益的均值, 除以 spread 的标准差
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
// 5. 格式化输出矩阵
// ============================================================

console.log('\n' + '='.repeat(80));
console.log('=== IC 矩阵 (Spearman) ===');
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
console.log('=== Hit Rate 矩阵 (alpha > +5pp → forward > median) ===');
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

// 最佳组合
console.log('\n' + '='.repeat(80));
console.log('=== 最佳组合 ===');
console.log('='.repeat(80));
const validEntries = Object.values(matrix).filter(m => m.ic != null);
validEntries.sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic));
console.log('\n按 |IC| 排序前 5:');
for (const m of validEntries.slice(0, 5)) {
  console.log(`  lookback=${m.lookback}m holding=${m.holding}m IC=${m.ic.toFixed(4)} HR=${(m.hitRate*100).toFixed(1)}% Sharpe=${m.sharpe.toFixed(3)} n=${m.n}`);
}

validEntries.sort((a, b) => b.sharpe - a.sharpe);
console.log('\n按 Sharpe 排序前 5:');
for (const m of validEntries.slice(0, 5)) {
  console.log(`  lookback=${m.lookback}m holding=${m.holding}m IC=${m.ic.toFixed(4)} HR=${(m.hitRate*100).toFixed(1)}% Sharpe=${m.sharpe.toFixed(3)} n=${m.n}`);
}

// 结论
const bestIC = validEntries.reduce((best, m) => Math.abs(m.ic) > Math.abs(best.ic) ? m : best, validEntries[0]);
if (bestIC.ic > 0.05) {
  console.log(`\n结论 A: 存在预测力组合 (lookback=${bestIC.lookback}m, holding=${bestIC.holding}m, IC=${bestIC.ic.toFixed(4)})`);
} else if (bestIC.ic < -0.05) {
  console.log(`\n结论 C: 存在显著反向预测 (lookback=${bestIC.lookback}m, holding=${bestIC.holding}m, IC=${bestIC.ic.toFixed(4)})`);
} else {
  console.log(`\n结论 B: 所有组合 IC ∈ [-0.05, +0.05], 行业 alpha 无预测力。最强组合 lookback=${bestIC.lookback}m holding=${bestIC.holding}m IC=${bestIC.ic.toFixed(4)}`);
}
