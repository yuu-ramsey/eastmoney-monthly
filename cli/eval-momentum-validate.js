// V1+V2+V3 动量因子验证 (零LLM)
// 用法: node cli/eval-momentum-validate.js
import { readFileSync } from 'fs';
import Database from 'better-sqlite3';
import * as path from 'path';
import { fileURLToPath } from 'node:url';
import { rulerRobustCenter, blockBootstrap, bootstrapCI, winsorize } from '../lib/eval/rulers.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const ds = JSON.parse(readFileSync(path.join(PROJECT_DIR, 'data', 'frozen-eval-lowpos-v1.json'), 'utf-8'));
const db = new Database(path.join(PROJECT_DIR, '.eastmoney-ai', 'db', 'klines-v2.sqlite'), { readonly: true });

const TRAIN = new Set(['2018-06', '2018-12', '2019-06', '2020-09', '2021-06', '2022-06']);
const TEST = new Set(['2020-03', '2021-12', '2022-10', '2023-06', '2024-02', '2024-10']);

// Build momentum pairs
const allPairs = [];
for (const tp of ds.testPoints) {
  if (tp.alpha == null) continue;
  const rows = db.prepare('SELECT * FROM monthly_klines WHERE code=? AND date<=? ORDER BY date').all(tp.stockCode, tp.cutoffDate);
  if (rows.length < 60) continue;
  const ci = rows.findIndex(r => String(r.date).startsWith(tp.cutoffDate));
  if (ci < 14) continue;
  const c = rows[ci].close; if (!c || c <= 0.01) continue;
  const t6 = ci >= 6 && rows[ci - 6].close > 0.01 ? (c - rows[ci - 6].close) / rows[ci - 6].close : null;
  const t12 = ci >= 12 && rows[ci - 12].close > 0.01 ? (c - rows[ci - 12].close) / rows[ci - 12].close : null;
  let ms = 0, mn = 0;
  for (let i = ci; i >= 0 && mn < 60; i--) { if (rows[i].close > 0.01) { ms += rows[i].close; mn++; } }
  allPairs.push({
    stockCode: tp.stockCode, cutoffDate: tp.cutoffDate, alpha: tp.alpha, t6, t12,
    d60: mn >= 60 ? (c - ms / mn) / (ms / mn) : null,
    isTrain: TRAIN.has(tp.cutoffDate), isTest: TEST.has(tp.cutoffDate),
  });
}

// Z-score composite
const keys = ['t6', 't12', 'd60'];
const ref = { mean: {}, std: {} };
for (const k of keys) {
  const v = allPairs.map(p => p[k]).filter(v => v != null);
  ref.mean[k] = v.reduce((a, b) => a + b, 0) / v.length;
  ref.std[k] = Math.sqrt(v.reduce((a, b) => a + Math.pow(b - ref.mean[k], 2), 0) / v.length);
}
for (const p of allPairs) {
  let z = 0, nz = 0;
  for (const k of keys) if (p[k] != null && ref.std[k] > 1e-8) { z += (p[k] - ref.mean[k]) / ref.std[k]; nz++; }
  p.momZ = nz > 0 ? z / nz : 0;
}

function assignSignals(pairs) {
  const s = [...pairs].sort((a, b) => b.momZ - a.momZ);
  const n20 = Math.floor(pairs.length * 0.2);
  for (let i = 0; i < pairs.length; i++) s[i].sig = i < n20 ? 'bull' : (i >= pairs.length - n20 ? 'bear' : 'neutral');
}

function momStats(pairs, label) {
  assignSignals(pairs);
  const r = rulerRobustCenter(pairs, 'sig', 'alpha');
  const boot = blockBootstrap(pairs, p => p.stockCode + '|' + p.cutoffDate, s => rulerRobustCenter(s, 'sig', 'alpha').spread);
  const ci = bootstrapCI(boot.values);
  const u = new Set(pairs.map(p => p.stockCode + '|' + p.cutoffDate)).size;
  const months = {};
  pairs.forEach(p => { months[p.cutoffDate] = (months[p.cutoffDate] || 0) + 1; });
  const thin = Object.entries(months).filter(([, n]) => n < 30).map(([m]) => m);
  return { label, n: pairs.length, nUnique: u, spread: r.spread, ci, thin, bootN: boot.nBlocks };
}

// ====== Full ======
const full = momStats(allPairs, 'Full');
console.log(`Full: n=${full.n} u=${full.nUnique} spread=${full.spread.toFixed(2)}% CI[${full.ci.lo.toFixed(1)},${full.ci.hi.toFixed(1)}]`);

// ====== V1 ======
console.log('\n=== V1 Hold-out ===');
const train = momStats(allPairs.filter(p => p.isTrain), 'Train');
const test = momStats(allPairs.filter(p => p.isTest), 'Test');
console.log(`Train: n=${train.n} u=${train.nUnique} spread=${train.spread.toFixed(2)}% CI[${train.ci.lo.toFixed(1)},${train.ci.hi.toFixed(1)}] thin=[${train.thin}]`);
console.log(`Test:  n=${test.n} u=${test.nUnique} spread=${test.spread.toFixed(2)}% CI[${test.ci.lo.toFixed(1)},${test.ci.hi.toFixed(1)}] thin=[${test.thin}]`);

const v1Ok = test.ci.lo > 0 && train.ci.lo > 0;
const v1Stable = Math.abs(test.spread - train.spread) < Math.abs(train.spread) * 0.5;
console.log(`V1: ${v1Ok ? (v1Stable ? '稳健 ✓' : '显著但量级差异>50%') : 'test CI含0 → 过拟合风险'}`);

// ====== V2 ======
console.log('\n=== V2 Regime分层 ===');
const hs300 = db.prepare("SELECT date,close FROM monthly_klines WHERE code='000300' ORDER BY date").all();
const regimeMap = {};

for (const t of [...new Set(allPairs.map(p => p.cutoffDate))]) {
  const idx = hs300.findIndex(r => String(r.date).startsWith(t));
  if (idx < 0) { regimeMap[t] = '?'; continue; }
  let fi = -1;
  const td = new Date(t + '-01');
  for (let i = idx + 1; i < hs300.length; i++) {
    const fd = new Date(hs300[i].date + '-01');
    if ((fd.getFullYear() - td.getFullYear()) * 12 + (fd.getMonth() - td.getMonth()) >= 6) { fi = i; break; }
  }
  if (fi < 0) fi = hs300.length - 1;
  const ret = hs300[idx].close > 0.01 ? (hs300[fi].close - hs300[idx].close) / hs300[idx].close * 100 : 0;
  regimeMap[t] = ret > 5 ? 'up' : ret < -5 ? 'down' : 'sideways';
  console.log(`  ${t}: CSI300 6m=${ret.toFixed(1)}% → ${regimeMap[t]}`);
}

for (const reg of ['up', 'down', 'sideways']) {
  const subset = allPairs.filter(p => regimeMap[p.cutoffDate] === reg);
  if (subset.length === 0) { console.log(`${reg}: n=0`); continue; }
  const r = momStats(subset, reg);
  console.log(`${reg.padEnd(9)}: n=${r.n} u=${r.nUnique} spread=${r.spread.toFixed(2)}% CI[${r.ci.lo.toFixed(1)},${r.ci.hi.toFixed(1)}] thin=[${r.thin}]`);
}

// ====== V3 ======
console.log('\n=== V3 可交易性 ===');
assignSignals(allPairs);
const sorted = [...allPairs].sort((a, b) => b.momZ - a.momZ);
const n20 = Math.floor(allPairs.length * 0.2);
const top20 = sorted.slice(0, n20);
const bot20 = sorted.slice(-n20);

const allMean = allPairs.reduce((s, p) => s + p.alpha, 0) / allPairs.length;
const top20Mean = top20.reduce((s, p) => s + p.alpha, 0) / top20.length;
const lOnlyExcess = top20Mean - allMean;
console.log(`全样本均值: ${allMean.toFixed(2)}%  top20%: ${top20Mean.toFixed(2)}%  只做多超额: ${lOnlyExcess.toFixed(2)}%`);

const origSpread = top20.reduce((s, p) => s + p.alpha, 0) / top20.length - bot20.reduce((s, p) => s + p.alpha, 0) / bot20.length;
const wTop = winsorize(top20.map(p => p.alpha));
const wBot = winsorize(bot20.map(p => p.alpha));
const wSpread = wTop.reduce((a, b) => a + b, 0) / wTop.length - wBot.reduce((a, b) => a + b, 0) / wBot.length;
console.log(`原始spread: ${origSpread.toFixed(2)}%  Winsorize: ${wSpread.toFixed(2)}%  极值贡献: ${(origSpread - wSpread).toFixed(2)}pp`);

// Turnover
let toSum = 0, toN = 0;
const tpMonths = [...new Set(allPairs.map(p => p.cutoffDate))].sort();
for (let i = 0; i < tpMonths.length - 1; i++) {
  const t1 = tpMonths[i], t2 = tpMonths[i + 1];
  const p1 = allPairs.filter(p => p.cutoffDate === t1).sort((a, b) => b.momZ - a.momZ);
  const p2 = allPairs.filter(p => p.cutoffDate === t2).sort((a, b) => b.momZ - a.momZ);
  const s1 = new Set(p1.slice(0, Math.floor(p1.length * 0.2)).map(p => p.stockCode));
  const s2 = new Set(p2.slice(0, Math.floor(p2.length * 0.2)).map(p => p.stockCode));
  const overlap = new Set([...s1].filter(x => s2.has(x))).size;
  const tot = Math.max(s1.size, s2.size);
  if (tot > 0) { toSum += 1 - overlap / tot; toN++; }
}
const avgTO = toN > 0 ? toSum / toN : 0;
console.log(`\n相邻时点换手(≈): ${(avgTO * 100).toFixed(0)}%/period`);

for (const cost of [0.2, 0.3]) {
  const costPct = avgTO * cost * 2;
  const netSpread = wSpread - costPct;
  const netLong = lOnlyExcess - costPct;
  console.log(`扣${cost.toFixed(1)}%成本: 净spread=${netSpread.toFixed(2)}%  净多头超额=${netLong.toFixed(2)}%`);
}

db.close();
