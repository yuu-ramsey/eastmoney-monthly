// Phase 11 共振矩阵可信度验证
// a. 样本量矩阵
// b. HS300 等权基准对照 (alpha = stock_return - benchmark_return)
// c. mild_bear 时间分布
// d. 修正 Long-Short Sharpe (扣除基准 beta)

import { getDb } from '../lib/db/connection.js';
import { calculateAll } from '../lib/indicators/calculate.js';

const db = getDb();

// 方向判断 — 完全复刻 lib/multi-period/direction.js
function slope(series, n) {
  const valid = series.filter(v => v != null);
  if (valid.length < n) return 0;
  const data = valid.slice(-n);
  const m = data.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < m; i++) { sumX += i; sumY += data[i]; sumXY += i * data[i]; sumX2 += i * i; }
  const denom = (m * sumX2 - sumX * sumX);
  return denom === 0 ? 0 : (m * sumXY - sumX * sumY) / denom;
}
function calcMonthlyDir(klines, ind) {
  if (klines.length < 60) return 'neutral';
  const last = klines.length - 1;
  const c = klines[last].close, m60 = ind.ma60[last];
  if (c == null || m60 == null) return 'neutral';
  const s = slope(ind.ma20, 5), macd = ind.macd.dif[last];
  if (c > m60 && s > 0 && macd != null && macd > 0) return 'bull';
  if (c < m60 && s < 0 && macd != null && macd < 0) return 'bear';
  return 'neutral';
}
function calcWeeklyDir(klines, ind) {
  if (klines.length < 20) return 'neutral';
  const last = klines.length - 1;
  const c = klines[last].close, m20 = ind.ma20[last];
  if (c == null || m20 == null) return 'neutral';
  const s = slope(ind.ma20, 5);
  if (c > m20 && s > 0) return 'bull';
  if (c < m20 && s < 0) return 'bear';
  return 'neutral';
}
function calcDailyDir(klines, ind) {
  if (klines.length < 20) return 'neutral';
  const last = klines.length - 1;
  const c = klines[last].close, m5 = ind.ma5[last], m20 = ind.ma20[last], rsi = ind.rsi14[last];
  if (c == null || m20 == null || m5 == null || rsi == null) return 'neutral';
  if (c > m20 && m5 > m20 && rsi > 30 && rsi < 70) return 'bull';
  if (c < m20 && m5 < m20 && rsi > 30 && rsi < 70) return 'bear';
  return 'neutral';
}
function calcResonance(m, w, d) {
  const dirs = [m, w, d], counts = { bull: 0, bear: 0, neutral: 0 };
  dirs.forEach(d => { counts[d]++; });
  let dom = 'neutral';
  if (counts.bull > counts.bear && counts.bull > counts.neutral) dom = 'bull';
  else if (counts.bear > counts.bull && counts.bear > counts.neutral) dom = 'bear';
  let lvl = 'divergent';
  if (counts[dom] === 3) lvl = 'strong';
  else if (counts[dom] === 2) lvl = 'partial';
  return { monthly: m, weekly: w, daily: d, resonanceLevel: lvl, resonanceCount: counts[dom], dominant: dom };
}
function getSignal(r) {
  if (r.resonanceLevel === 'strong' && r.dominant === 'bull') return 'strong_bull';
  if (r.resonanceLevel === 'strong' && r.dominant === 'bear') return 'strong_bear';
  if (r.resonanceLevel === 'partial' && r.dominant === 'bull') return 'mild_bull';
  if (r.resonanceLevel === 'partial' && r.dominant === 'bear') return 'mild_bear';
  if (r.dominant === 'bull') return 'mild_bull';
  if (r.dominant === 'bear') return 'mild_bear';
  return 'neutral';
}

// 数据加载 (与主脚本相同)
const stockList = db.prepare("SELECT DISTINCT stock_code FROM stock_industry_mapping").all().map(r => r.stock_code);
console.log(`Stock: ${stockList.length}`);

const loadKlines = (table, dateCol) => {
  const data = new Map();
  const rows = db.prepare(`SELECT code, date, open, close, high, low, volume FROM ${table} WHERE code IN (${stockList.map(()=>'?').join(',')}) AND date >= '2016-01' ORDER BY code, date`).all(...stockList);
  for (const r of rows) { if (!data.has(r.code)) data.set(r.code, []); data.get(r.code).push(r); }
  return data;
};

console.log('Loading klines...');
const monthlyData = loadKlines('monthly_klines');
const weeklyData = loadKlines('weekly_klines');
const dailyData = loadKlines('daily_klines');
console.log(`M:${[...monthlyData.values()].reduce((s,v)=>s+v.length,0)} W:${[...weeklyData.values()].reduce((s,v)=>s+v.length,0)} D:${[...dailyData.values()].reduce((s,v)=>s+v.length,0)}`);

// HS300 等权基准: 所有 HS300 股票的平均月收益
const evalMonths = [...new Set([...monthlyData.values()].flat().map(k => k.date))].filter(d => d >= '2018-01' && d <= '2024-12').sort();
console.log(`Eval months: ${evalMonths.length}`);

// 为每个月计算 HS300 等权基准收益
function getPriceAtOrBefore(klines, date) {
  let last = null;
  for (const k of klines) { if (k.date > date) break; last = k; }
  return last;
}
function getForwardReturn(klines, asOfDate, holdingMonths) {
  const allDates = [...new Set(klines.map(k => k.date))].sort();
  let idx = allDates.findIndex(d => d > asOfDate);
  if (idx < 0) idx = allDates.length;
  const startIdx = idx - 1;
  if (startIdx < 0) return null;
  const endIdx = startIdx + holdingMonths;
  if (endIdx >= allDates.length) return null;
  // For benchmark: use average of all stocks' close at start/end
  return null; // placeholder — computed below
}

// 计算每月 HS300 等权基准 forward return
const benchmarkReturns = new Map(); // asOfDate → {1m, 3m, 6m, 12m}
for (const asOf of evalMonths) {
  // 收集所有股票在 asOf 的收盘价，然后在 horizon 后的收盘价
  const bm = {};
  for (const h of [1, 3, 6, 12]) {
    let sumStart = 0, cntStart = 0, sumEnd = 0, cntEnd = 0;
    for (const code of stockList) {
      const klines = monthlyData.get(code);
      if (!klines) continue;
      const allDates = [...new Set(klines.map(k => k.date))].sort();
      let idx = allDates.findIndex(d => d > asOf);
      if (idx < 0) idx = allDates.length;
      const startI = idx - 1;
      if (startI < 0) continue;
      const startPrice = klines.find(k => k.date === allDates[startI])?.close;
      if (!startPrice || startPrice <= 0) continue;
      const endI = startI + h;
      if (endI >= allDates.length) continue;
      const endPrice = klines.find(k => k.date === allDates[endI])?.close;
      if (!endPrice || endPrice <= 0) continue;
      sumStart += startPrice; cntStart++;
      sumEnd += endPrice; cntEnd++;
    }
    if (cntStart > 0 && cntEnd > 0 && cntStart === cntEnd) {
      bm[h] = (sumEnd / cntEnd - sumStart / cntStart) / (sumStart / cntStart);
    }
  }
  benchmarkReturns.set(asOf, bm);
}

// 加载共振数据 (从之前的脚本缓存...不,重新计算)
// 因为需要 sample counts、time distribution、benchmark alpha
// 简化: 重新计算方向但这次收集全量数据

console.log('\nComputing directions...');
const SIGNALS = ['strong_bull', 'mild_bull', 'strong_bear', 'mild_bear'];
const HOLDINGS = [1, 3, 6, 12];

// 按信号分组收集 forward returns
const signalObs = {}; // signal_holding → { fwdRets:[], stockFwdRets:[], benchmarkRets:[], dates:[] }
for (const sig of SIGNALS) for (const h of HOLDINGS) signalObs[`${sig}_${h}`] = [];

let processed = 0;
for (const code of stockList) {
  const mKlines = monthlyData.get(code) || [];
  const wKlines = weeklyData.get(code) || [];
  const dKlines = dailyData.get(code) || [];

  for (const asOf of evalMonths) {
    // Slice
    const mSlice = [], wSlice = [], dSlice = [];
    for (const k of mKlines) { if (k.date > asOf) break; mSlice.push(k); }
    for (const k of wKlines) { if (k.date > asOf) break; wSlice.push(k); }
    for (const k of dKlines) { if (k.date > asOf) break; dSlice.push(k); }
    if (mSlice.length < 60 || wSlice.length < 20 || dSlice.length < 20) continue;

    const mInd = calculateAll(mSlice), wInd = calculateAll(wSlice), dInd = calculateAll(dSlice);
    const r = calcResonance(
      calcMonthlyDir(mSlice, mInd), calcWeeklyDir(wSlice, wInd), calcDailyDir(dSlice, dInd)
    );
    const sig = getSignal(r);
    if (sig === 'neutral') continue;

    // Forward returns
    const allDates = [...new Set(mKlines.map(k => k.date))].sort();
    let idx = allDates.findIndex(d => d > asOf);
    if (idx < 0) idx = allDates.length;
    const startI = idx - 1;
    if (startI < 0) continue;
    const startPrice = mKlines.find(k => k.date === allDates[startI])?.close;
    if (!startPrice || startPrice <= 0) continue;

    const bm = benchmarkReturns.get(asOf) || {};

    for (const h of HOLDINGS) {
      const endI = startI + h;
      if (endI >= allDates.length) continue;
      const endPrice = mKlines.find(k => k.date === allDates[endI])?.close;
      if (!endPrice || endPrice <= 0) continue;

      const stockRet = (endPrice - startPrice) / startPrice;
      const benchRet = bm[h] || 0;
      const alpha = stockRet - benchRet;

      signalObs[`${sig}_${h}`].push({
        stockCode: code,
        asOfDate: asOf,
        stockRet, benchRet, alpha,
      });
    }
  }
  processed++;
  if (processed % 50 === 0) console.log(`  ${processed}/${stockList.length}`);
}
console.log(`Done.`);

// ============================================================
// a. 样本量矩阵 + avg return + alpha
// ============================================================
function mean(arr) { return arr.reduce((a,b)=>a+b,0)/(arr.length||1); }
function std(arr) { const m = mean(arr); return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/((arr.length-1)||1)); }

console.log('\n' + '='.repeat(100));
console.log('=== a. 样本量 + avg stock return + avg alpha (扣除HS300等权基准) ===');
console.log('='.repeat(100));
console.log('| signal \\ holding | n | stock ret | alpha | hit rate(raw) | hit rate(alpha) |');
console.log('|---|---|---:|---:|---:|---:|');

const allStocksFwd = [];
for (const sig of SIGNALS) for (const h of HOLDINGS) {
  for (const obs of signalObs[`${sig}_${h}`]) allStocksFwd.push(obs.stockRet);
}
const medianAll = [...allStocksFwd].sort((a,b)=>a-b)[Math.floor(allStocksFwd.length/2)];
const allAlphas = [];
for (const sig of SIGNALS) for (const h of HOLDINGS) {
  for (const obs of signalObs[`${sig}_${h}`]) allAlphas.push(obs.alpha);
}
const medianAlpha = [...allAlphas].sort((a,b)=>a-b)[Math.floor(allAlphas.length/2)];

for (const sig of SIGNALS) {
  for (const h of HOLDINGS) {
    const obs = signalObs[`${sig}_${h}`];
    if (obs.length < 20) continue;
    const stockRets = obs.map(o => o.stockRet);
    const alphas = obs.map(o => o.alpha);
    const avgRet = mean(stockRets);
    const avgAlpha = mean(alphas);
    const hitRaw = stockRets.filter(r => r > medianAll).length / stockRets.length;
    const hitAlpha = alphas.filter(a => a > medianAlpha).length / alphas.length;
    console.log(`| ${sig} ${h}m | ${obs.length} | ${(avgRet*100).toFixed(2)}% | ${(avgAlpha*100).toFixed(2)}% | ${(hitRaw*100).toFixed(1)}% | ${(hitAlpha*100).toFixed(1)}% |`);
  }
}

// ============================================================
// b. Long-Short 扣除基准
// ============================================================
console.log('\n' + '='.repeat(100));
console.log('=== b. Long-Short (strong_bull - strong_bear) alpha ===');
console.log('='.repeat(100));
for (const h of HOLDINGS) {
  const bullAlpha = signalObs[`strong_bull_${h}`].map(o => o.alpha);
  const bearAlpha = signalObs[`strong_bear_${h}`].map(o => o.alpha);
  if (bullAlpha.length < 20 || bearAlpha.length < 20) continue;
  const bullMean = mean(bullAlpha);
  const bearMean = mean(bearAlpha);
  const spread = bullMean - bearMean;
  const n = Math.min(bullAlpha.length, bearAlpha.length);
  const spreads = [];
  for (let i = 0; i < n; i++) spreads.push(bullAlpha[i] - bearAlpha[i]);
  const spreadStd = std(spreads);
  const lsSharpe = spreadStd > 0 ? spread / spreadStd * Math.sqrt(12/h) : null;
  console.log(`hold=${h}m: spread_alpha=${(spread*100).toFixed(2)}% Sharpe=${lsSharpe?.toFixed(3) || 'N/A'} (n=${n})`);
}

// ============================================================
// c. mild_bear 时间分布
// ============================================================
console.log('\n' + '='.repeat(100));
console.log('=== c. mild_bear 6m 时间分布 (by year) ===');
console.log('='.repeat(100));
const mildBear6 = signalObs['mild_bear_6'];
const byYear = {};
for (const obs of mildBear6) {
  const year = obs.asOfDate.substring(0, 4);
  if (!byYear[year]) byYear[year] = [];
  byYear[year].push(obs);
}
console.log('| year | n | avg stock ret | avg alpha |');
console.log('|------|---|-------------|----------|');
for (const y of Object.keys(byYear).sort()) {
  const rets = byYear[y].map(o => o.stockRet);
  const alphas = byYear[y].map(o => o.alpha);
  console.log(`| ${y} | ${rets.length} | ${(mean(rets)*100).toFixed(2)}% | ${(mean(alphas)*100).toFixed(2)}% |`);
}

// 检查 strong_bull / strong_bear 的方向性是否受 mild_bear 异常影响
console.log('\n' + '='.repeat(100));
console.log('=== d. strong_bull vs strong_bear 核心验证 ===');
console.log('='.repeat(100));
for (const h of HOLDINGS) {
  const sbBull = signalObs[`strong_bull_${h}`];
  const sbBear = signalObs[`strong_bear_${h}`];
  console.log(`hold=${h}m: strong_bull n=${sbBull.length} ret=${(mean(sbBull.map(o=>o.stockRet))*100).toFixed(2)}% alpha=${(mean(sbBull.map(o=>o.alpha))*100).toFixed(2)}%`);
  console.log(`           strong_bear n=${sbBear.length} ret=${(mean(sbBear.map(o=>o.stockRet))*100).toFixed(2)}% alpha=${(mean(sbBear.map(o=>o.alpha))*100).toFixed(2)}%`);
  console.log(`           bear - bull alpha=${((mean(sbBear.map(o=>o.alpha))-mean(sbBull.map(o=>o.alpha)))*100).toFixed(2)}%`);
}

console.log('\nDone.');
