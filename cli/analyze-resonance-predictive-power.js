// Phase 11 共振约束预测力矩阵
// 4 resonance levels × 4 holding periods = 16 cells
// 指标: avg forward return / Hit Rate / Long-Short Sharpe
// 严格 walk-forward, 零 LLM 调用
// 使用 lib/multi-period/ 完全相同的方向判断逻辑

import { getDb } from '../lib/db/connection.js';
import { calculateAll, tailIndicators } from '../lib/indicators/calculate.js';

const db = getDb();

// ============================================================
// 1. 方向判断 — 完全复刻 lib/multi-period/direction.js
// ============================================================

function slope(series, n) {
  const valid = series.filter(v => v != null);
  if (valid.length < n) return 0;
  const data = valid.slice(-n);
  const m = data.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < m; i++) {
    sumX += i; sumY += data[i]; sumXY += i * data[i]; sumX2 += i * i;
  }
  const denom = (m * sumX2 - sumX * sumX);
  return denom === 0 ? 0 : (m * sumXY - sumX * sumY) / denom;
}

function calcMonthlyDirection(klines, indicators) {
  const n = klines.length;
  if (n < 60) return 'neutral';
  const last = n - 1;
  const close = klines[last].close;
  const ma60 = indicators.ma60[last];
  if (close == null || ma60 == null) return 'neutral';
  const ma20Slope = slope(indicators.ma20, 5);
  const macdVal = indicators.macd.dif[last];
  if (close > ma60 && ma20Slope > 0 && macdVal != null && macdVal > 0) return 'bull';
  if (close < ma60 && ma20Slope < 0 && macdVal != null && macdVal < 0) return 'bear';
  return 'neutral';
}

function calcWeeklyDirection(klines, indicators) {
  const n = klines.length;
  if (n < 20) return 'neutral';
  const last = n - 1;
  const close = klines[last].close;
  const ma20 = indicators.ma20[last];
  if (close == null || ma20 == null) return 'neutral';
  const ma20Slope = slope(indicators.ma20, 5);
  if (close > ma20 && ma20Slope > 0) return 'bull';
  if (close < ma20 && ma20Slope < 0) return 'bear';
  return 'neutral';
}

function calcDailyDirection(klines, indicators) {
  const n = klines.length;
  if (n < 20) return 'neutral';
  const last = n - 1;
  const close = klines[last].close;
  const ma5 = indicators.ma5[last];
  const ma20 = indicators.ma20[last];
  const rsi = indicators.rsi14[last];
  if (close == null || ma20 == null || ma5 == null || rsi == null) return 'neutral';
  if (close > ma20 && ma5 > ma20 && rsi > 30 && rsi < 70) return 'bull';
  if (close < ma20 && ma5 < ma20 && rsi > 30 && rsi < 70) return 'bear';
  return 'neutral';
}

function calcResonance(monthly, weekly, daily) {
  const dirs = [monthly, weekly, daily];
  const counts = { bull: 0, bear: 0, neutral: 0 };
  dirs.forEach(d => { counts[d]++; });

  let dominant = 'neutral';
  if (counts.bull > counts.bear && counts.bull > counts.neutral) dominant = 'bull';
  else if (counts.bear > counts.bull && counts.bear > counts.neutral) dominant = 'bear';

  let resonanceLevel = 'divergent';
  if (counts[dominant] === 3) resonanceLevel = 'strong';
  else if (counts[dominant] === 2) resonanceLevel = 'partial';

  return {
    monthly, weekly, daily,
    resonanceLevel,
    resonanceCount: counts[dominant],
    dominant,
    allBull: monthly === 'bull' && weekly === 'bull' && daily === 'bull',
    allBear: monthly === 'bear' && weekly === 'bear' && daily === 'bear',
  };
}

// 组合信号: e.g. "strong_bull" = strong resonance + dominant bull
// "mild_bull" = partial resonance + dominant bull
// "strong_bear" = strong resonance + dominant bear
// "mild_bear" = partial resonance + dominant bear
function getSignal(r) {
  if (r.resonanceLevel === 'strong' && r.dominant === 'bull') return 'strong_bull';
  if (r.resonanceLevel === 'strong' && r.dominant === 'bear') return 'strong_bear';
  if (r.resonanceLevel === 'partial' && r.dominant === 'bull') return 'mild_bull';
  if (r.resonanceLevel === 'partial' && r.dominant === 'bear') return 'mild_bear';
  if (r.dominant === 'bull') return 'mild_bull'; // divergent but majority bull
  if (r.dominant === 'bear') return 'mild_bear';
  return 'neutral';
}

// ============================================================
// 2. 数据加载
// ============================================================

const stockList = db.prepare(`
  SELECT DISTINCT stock_code FROM stock_industry_mapping
`).all().map(r => r.stock_code);
console.log(`HS300 股票: ${stockList.length}`);

// 加载所有月线 K 线
const monthlyData = new Map();
const mRows = db.prepare(`
  SELECT code, date, open, close, high, low, volume FROM monthly_klines
  WHERE code IN (${stockList.map(() => '?').join(',')})
  AND date >= '2016-01'
  ORDER BY code, date
`).all(...stockList);
for (const r of mRows) {
  if (!monthlyData.has(r.code)) monthlyData.set(r.code, []);
  monthlyData.get(r.code).push(r);
}
console.log(`月线: ${mRows.length} 条`);

// 加载所有周线 K 线
const weeklyData = new Map();
const wRows = db.prepare(`
  SELECT code, date, open, close, high, low, volume FROM weekly_klines
  WHERE code IN (${stockList.map(() => '?').join(',')})
  AND date >= '2016-01'
  ORDER BY code, date
`).all(...stockList);
for (const r of wRows) {
  if (!weeklyData.has(r.code)) weeklyData.set(r.code, []);
  weeklyData.get(r.code).push(r);
}
console.log(`周线: ${wRows.length} 条`);

// 加载所有日线 K 线 (最近 2 年足够，因为日线只看方向)
const dailyData = new Map();
const dRows = db.prepare(`
  SELECT code, date, open, close, high, low, volume FROM daily_klines
  WHERE code IN (${stockList.map(() => '?').join(',')})
  AND date >= '2017-01'
  ORDER BY code, date
`).all(...stockList);
for (const r of dRows) {
  if (!dailyData.has(r.code)) dailyData.set(r.code, []);
  dailyData.get(r.code).push(r);
}
console.log(`日线: ${dRows.length} 条`);

// ============================================================
// 3. 每个评估点计算共振信号
// ============================================================

function sliceKlinesUpTo(klines, cutoffDate) {
  // 只取 ≤ cutoffDate 的 K 线
  const sliced = [];
  for (const k of klines) {
    if (k.date > cutoffDate) break;
    sliced.push(k);
  }
  return sliced;
}

function getForwardReturnMonthly(klines, asOfDate, holdingMonths) {
  // asOfDate 是 "YYYY-MM" 格式
  // 找 asOfDate 在 klines 中的索引
  let idx = -1;
  for (let i = 0; i < klines.length; i++) {
    if (klines[i].date > asOfDate) { idx = i - 1; break; }
  }
  if (idx < 0) idx = klines.length - 1;
  if (idx < 0) return null;

  const startPrice = klines[idx].close;
  const endIdx = idx + holdingMonths;
  if (endIdx >= klines.length) return null;
  const endPrice = klines[endIdx].close;
  if (!startPrice || !endPrice || startPrice <= 0) return null;
  return (endPrice - startPrice) / startPrice;
}

// 预计算每个 stock 每个月的方向
console.log('\n预计算方向...');
const directionCache = new Map(); // stockCode → Map<dateStr, resonance>

const evalMonths = [];
for (const d of [...new Set(mRows.map(r => r.date))].sort()) {
  if (d >= '2018-01' && d <= '2024-12') evalMonths.push(d);
}
console.log(`评估月份: ${evalMonths.length}`);

let stockProcessed = 0;
for (const code of stockList) {
  const mKlines = monthlyData.get(code) || [];
  const wKlines = weeklyData.get(code) || [];
  const dKlines = dailyData.get(code) || [];

  const stockDirs = new Map();

  // 为每个评估月计算共振
  for (const month of evalMonths) {
    const mSlice = sliceKlinesUpTo(mKlines, month);
    const wSlice = sliceKlinesUpTo(wKlines, month);
    const dSlice = sliceKlinesUpTo(dKlines, month);

    if (mSlice.length < 60 || wSlice.length < 20 || dSlice.length < 20) {
      stockDirs.set(month, null);
      continue;
    }

    const mInd = calculateAll(mSlice);
    const wInd = calculateAll(wSlice);
    const dInd = calculateAll(dSlice);

    const monthly = calcMonthlyDirection(mSlice, mInd);
    const weekly = calcWeeklyDirection(wSlice, wInd);
    const daily = calcDailyDirection(dSlice, dInd);

    stockDirs.set(month, calcResonance(monthly, weekly, daily));
  }

  directionCache.set(code, stockDirs);
  stockProcessed++;
  if (stockProcessed % 50 === 0) console.log(`  方向: ${stockProcessed}/${stockList.length}`);
}

// ============================================================
// 4. 构建观察数据
// ============================================================

console.log('\n构建观察...');
const HOLDINGS = [1, 3, 6, 12];
const observations = [];

for (const code of stockList) {
  const mKlines = monthlyData.get(code) || [];
  const dirs = directionCache.get(code);
  if (!dirs) continue;

  for (const month of evalMonths) {
    const r = dirs.get(month);
    if (!r) continue;

    const fwdRets = {};
    for (const h of HOLDINGS) {
      const fr = getForwardReturnMonthly(mKlines, month, h);
      if (fr != null) fwdRets[h] = fr;
    }
    if (Object.keys(fwdRets).length === 0) continue;

    observations.push({
      stockCode: code,
      asOfDate: month,
      ...r,
      signal: getSignal(r),
      forwardReturns: fwdRets,
    });
  }
}

console.log(`观察数: ${observations.length}`);

// ============================================================
// 5. 指标计算
// ============================================================

function mean(arr) { return arr.reduce((a,b)=>a+b,0)/arr.length; }
function std(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/(arr.length-1||1));
}

const SIGNALS = ['strong_bull', 'mild_bull', 'strong_bear', 'mild_bear'];

console.log('\n' + '='.repeat(80));
console.log('=== 共振预测力矩阵 ===');
console.log('='.repeat(80));

const matrix = {};
const allObs = observations.filter(o => o.forwardReturns[6] != null);
const allFwd = allObs.map(o => o.forwardReturns[6]);
const medianAll = [...allFwd].sort((a,b)=>a-b)[Math.floor(allFwd.length/2)];

for (const signal of SIGNALS) {
  for (const h of HOLDINGS) {
    const obs = observations.filter(o => o.signal === signal && o.forwardReturns[h] != null);
    if (obs.length < 20) continue;

    const fwdRets = obs.map(o => o.forwardReturns[h]);
    const avgRet = mean(fwdRets);
    const retStd = std(fwdRets);
    const hitCount = fwdRets.filter(r => r > medianAll).length;
    const hitRate = hitCount / fwdRets.length;
    const sharpe = retStd > 0 ? avgRet / retStd * Math.sqrt(12/h) : null;

    const key = `${signal}_${h}m`;
    matrix[key] = { signal, holding: h, n: obs.length, avgRet, hitRate, sharpe };
    console.log(`  ${key}: n=${obs.length} avgRet=${(avgRet*100).toFixed(2)}% HR=${(hitRate*100).toFixed(1)}% Sharpe=${sharpe != null ? sharpe.toFixed(3) : 'N/A'}`);
  }
}

// Long-Short: strong_bull vs strong_bear
console.log('\n' + '='.repeat(80));
console.log('=== Long-Short (strong_bull - strong_bear) ===');
console.log('='.repeat(80));
for (const h of HOLDINGS) {
  const bullObs = observations.filter(o => o.signal === 'strong_bull' && o.forwardReturns[h] != null);
  const bearObs = observations.filter(o => o.signal === 'strong_bear' && o.forwardReturns[h] != null);
  if (bullObs.length < 20 || bearObs.length < 20) continue;

  const bullRets = bullObs.map(o => o.forwardReturns[h]);
  const bearRets = bearObs.map(o => o.forwardReturns[h]);
  const bullMean = mean(bullRets);
  const bearMean = mean(bearRets);
  const spread = bullMean - bearMean;

  // Aligned random pairing for Sharpe
  const n = Math.min(bullRets.length, bearRets.length);
  const spreads = [];
  for (let i = 0; i < n; i++) spreads.push(bullRets[i] - bearRets[i]);
  const spreadStd = std(spreads);
  const lsSharpe = spreadStd > 0 ? spread / spreadStd * Math.sqrt(12/h) : null;

  console.log(`  hold=${h}m: spread=${(spread*100).toFixed(2)}% Sharpe=${lsSharpe != null ? lsSharpe.toFixed(3) : 'N/A'} (bull=${bullObs.length}, bear=${bearObs.length})`);
}

// 矩阵表格
console.log('\n' + '='.repeat(80));
console.log('=== Avg Forward Return (%) ===');
console.log('='.repeat(80));
let header = '| signal \\ holding |';
for (const h of HOLDINGS) header += ` ${h}m |`;
console.log(header);
console.log('|' + header.replace(/[^|]/g, '-').replace(/\|/g, '|'));
for (const sig of SIGNALS) {
  let row = `| ${sig} |`;
  for (const h of HOLDINGS) {
    const key = `${sig}_${h}m`;
    const m = matrix[key];
    row += ` ${m ? (m.avgRet*100).toFixed(2)+'%' : ' N/A '} |`;
  }
  console.log(row);
}

console.log('\n' + '='.repeat(80));
console.log('=== Hit Rate (forward > median) ===');
console.log('='.repeat(80));
header = '| signal \\ holding |';
for (const h of HOLDINGS) header += ` ${h}m |`;
console.log(header);
console.log('|' + header.replace(/[^|]/g, '-').replace(/\|/g, '|'));
for (const sig of SIGNALS) {
  let row = `| ${sig} |`;
  for (const h of HOLDINGS) {
    const key = `${sig}_${h}m`;
    const m = matrix[key];
    row += ` ${m ? (m.hitRate*100).toFixed(1)+'%' : ' N/A '} |`;
  }
  console.log(row);
}

console.log('\n' + '='.repeat(80));
console.log('=== Annualized Sharpe ===');
console.log('='.repeat(80));
header = '| signal \\ holding |';
for (const h of HOLDINGS) header += ` ${h}m |`;
console.log(header);
console.log('|' + header.replace(/[^|]/g, '-').replace(/\|/g, '|'));
for (const sig of SIGNALS) {
  let row = `| ${sig} |`;
  for (const h of HOLDINGS) {
    const key = `${sig}_${h}m`;
    const m = matrix[key];
    row += ` ${m && m.sharpe != null ? m.sharpe.toFixed(3) : ' N/A '} |`;
  }
  console.log(row);
}

// ============================================================
// 6. 最终对比 Phase 12 + 结论
// ============================================================
console.log('\n' + '='.repeat(80));
console.log('=== Phase 11 vs Phase 12 对比 ===');
console.log('='.repeat(80));
console.log('Phase 12 sector alpha: |IC| max = 0.074 (反向), Hit Rate 47.8-51.4%, all Sharpe negative');
console.log('Phase 11 resonance: see above matrix');

// 找最强信号
const best = Object.values(matrix)
  .filter(m => m.sharpe != null)
  .sort((a, b) => Math.abs(b.sharpe) - Math.abs(a.sharpe));
if (best.length > 0) {
  console.log(`\n共振最强组合: ${best[0].signal}_${best[0].holding}m Sharpe=${best[0].sharpe.toFixed(3)} avgRet=${(best[0].avgRet*100).toFixed(2)}% HR=${(best[0].hitRate*100).toFixed(1)}%`);
}

console.log('\n完成。');
