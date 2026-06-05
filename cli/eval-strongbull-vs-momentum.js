// strong_bull vs naive momentum baseline — verify whether LLM's sole edge has independent value
// Usage: node cli/eval-strongbull-vs-momentum.js
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');

const V4_PATH = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs', 'v4-signals-2026-05-17-00-41.jsonl');
const DATASET_PATH = path.join(PROJECT_DIR, 'data', 'frozen-eval-dataset-v1.json');
const N_BOOTSTRAP = 10000;

function loadJsonl(fp) {
  if (!fs.existsSync(fp)) { console.error(`Not found: ${fp}`); return []; }
  return fs.readFileSync(fp, 'utf-8').trim().split('\n').filter(Boolean).map(l => {
    try { return JSON.parse(l); } catch (_) { return null; }
  }).filter(r => r !== null && !r.error);
}

function percentile(arr, p) {
  const s = [...arr].sort((a, b) => a - b);
  const idx = p * (s.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

function blockBootstrap(records, keyFn, statFn) {
  const blocks = new Map();
  for (const r of records) {
    const bk = keyFn(r);
    if (!blocks.has(bk)) blocks.set(bk, []);
    blocks.get(bk).push(r);
  }
  const blockList = [...blocks.values()];
  const values = [];
  for (let b = 0; b < N_BOOTSTRAP; b++) {
    const sample = [];
    for (let i = 0; i < blockList.length; i++)
      sample.push(...blockList[Math.floor(Math.random() * blockList.length)]);
    const v = statFn(sample);
    if (v !== null) values.push(v);
  }
  return { values, nBlocks: blockList.length };
}

function bootstrapCI(values, alpha = 0.05) {
  const s = [...values].sort((a, b) => a - b);
  return {
    lo: s[Math.floor(s.length * alpha / 2)],
    hi: s[Math.floor(s.length * (1 - alpha / 2))],
    mean: s.reduce((a, v) => a + v, 0) / s.length,
  };
}

console.log('=== strong_bull vs Naive Momentum Baseline ===\n');

const dataset = JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'));
const v4Records = loadJsonl(V4_PATH);
if (v4Records.length === 0) { console.error('v4 data is empty'); process.exit(1); }

const alphaMap = new Map();
for (const tp of dataset.testPoints) {
  alphaMap.set(`${tp.stockCode}|${tp.cutoffDate}`, {
    alpha: tp.alpha, actualReturn: tp.actualReturn,
    indexReturn: tp.indexReturn, groundTruth: tp.groundTruth,
  });
}

const withAlpha = [];
for (const r of v4Records) {
  const key = `${r.code}|${r.cutoffDate}`;
  const label = alphaMap.get(key);
  if (!label) continue;
  withAlpha.push({ ...r, ...label });
}

// unique (stock, cutoff) pairs
const pairs = new Map();
for (const r of withAlpha) {
  const key = `${r.code}|${r.cutoffDate}`;
  if (!pairs.has(key)) pairs.set(key, r);
}

// ---- DB ----
console.log('Loading monthly K-lines...');
const { getDb } = await import('../lib/db/connection.js');
const db = getDb();
const klinesCache = new Map();
for (const code of new Set(withAlpha.map(r => r.code)))
  klinesCache.set(code, db.prepare('SELECT * FROM monthly_klines WHERE code=? ORDER BY date').all(code));
console.log(`K-lines: ${klinesCache.size} stocks`);

// ---- Momentum signal ----
function computeMomentum(klines, cutoffDate) {
  let cutoffIdx = -1;
  for (let i = 0; i < klines.length; i++) {
    if (klines[i].date && String(klines[i].date).startsWith(cutoffDate)) { cutoffIdx = i; break; }
  }
  if (cutoffIdx < 11) return null;
  const close = klines[cutoffIdx].close;
  if (!close || close <= 0) return null;
  const t6 = cutoffIdx >= 6 ? (close - klines[cutoffIdx - 6].close) / Math.max(klines[cutoffIdx - 6].close, 0.01) : null;
  const t12 = cutoffIdx >= 12 ? (close - klines[cutoffIdx - 12].close) / Math.max(klines[cutoffIdx - 12].close, 0.01) : null;
  let ma60 = 0;
  if (cutoffIdx >= 59) { for (let i = cutoffIdx - 59; i <= cutoffIdx; i++) ma60 += klines[i].close; ma60 /= 60; }
  const ma60Pos = ma60 > 0 ? (close - ma60) / ma60 : null;
  return { t6, t12, ma60Pos };
}

const pairMom = [];
for (const [key, rec] of pairs) {
  const kl = klinesCache.get(rec.code);
  if (!kl) continue;
  const m = computeMomentum(kl, rec.cutoffDate);
  if (!m) continue;
  pairMom.push({ code: rec.code, cutoffDate: rec.cutoffDate, alpha: rec.alpha, ...m });
}

// Z-score normalize each momentum component
const allT6 = pairMom.map(p => p.t6).filter(v => v != null);
const allT12 = pairMom.map(p => p.t12).filter(v => v != null);
const allMa60 = pairMom.map(p => p.ma60Pos).filter(v => v != null);
const [m6, s6] = [allT6.reduce((a,b)=>a+b,0)/allT6.length, Math.sqrt(allT6.reduce((a,b)=>a+Math.pow(b-allT6.reduce((a,b)=>a+b,0)/allT6.length,2),0)/allT6.length)];
const [m12, s12] = [allT12.reduce((a,b)=>a+b,0)/allT12.length, Math.sqrt(allT12.reduce((a,b)=>a+Math.pow(b-allT12.reduce((a,b)=>a+b,0)/allT12.length,2),0)/allT12.length)];
const [mma, sma] = [allMa60.reduce((a,b)=>a+b,0)/allMa60.length, Math.sqrt(allMa60.reduce((a,b)=>a+Math.pow(b-allMa60.reduce((a,b)=>a+b,0)/allMa60.length,2),0)/allMa60.length)];

for (const p of pairMom) {
  p.comp = ((p.t6 != null ? (p.t6 - m6) / Math.max(s6, 1e-8) : 0)
    + (p.t12 != null ? (p.t12 - m12) / Math.max(s12, 1e-8) : 0)
    + (p.ma60Pos != null ? (p.ma60Pos - mma) / Math.max(sma, 1e-8) : 0)) / 3;
}

// strong_bull unique pairs
const pairSig = new Map();
for (const r of withAlpha) {
  const k = `${r.code}|${r.cutoffDate}`;
  if (!pairSig.has(k)) pairSig.set(k, []);
  pairSig.get(k).push(r.signal || r.predictedSignal || '');
}
const sbKeys = new Set();
for (const [k, sigs] of pairSig) { if (sigs.some(s => s === 'strong_bull')) sbKeys.add(k); }

const allPairs = [...pairs.values()].filter(r => {
  const kl = klinesCache.get(r.code);
  return kl && computeMomentum(kl, r.cutoffDate) !== null;
});
const sbPairs = allPairs.filter(r => sbKeys.has(`${r.code}|${r.cutoffDate}`));
const k = sbKeys.size;
const topMom = [...pairMom].sort((a, b) => b.comp - a.comp).slice(0, k);
const momKeys = new Set(topMom.map(p => `${p.code}|${p.cutoffDate}`));
const momPairs = allPairs.filter(r => momKeys.has(`${r.code}|${r.cutoffDate}`));

const uMean = allPairs.reduce((s, r) => s + r.alpha, 0) / allPairs.length;

// ---- Output ----
console.log(`\nStock selection: all HS300, ${klinesCache.size} stocks`);
console.log(`Valid unique (stock, timepoint): ${allPairs.length}`);
console.log(`Unconditional alpha mean: ${uMean.toFixed(2)}%`);
console.log(`strong_bull unique: ${sbPairs.length} (${(sbPairs.length/allPairs.length*100).toFixed(1)}%)`);
console.log(`Momentum top-${k} unique: ${momPairs.length}`);

// 1. strong_bull CI
console.log(`\n=== 1. strong_bull Significance ===`);
const sbAlphas = sbPairs.map(r => r.alpha);
const sbMean = sbAlphas.reduce((s,a)=>s+a,0)/sbAlphas.length;
console.log(`alpha mean: ${sbMean.toFixed(2)}% (unconditional: ${uMean.toFixed(2)}%)`);
const sbCI = bootstrapCI(blockBootstrap(sbPairs, r => `${r.code}|${r.cutoffDate}`, s => s.reduce((a,b)=>a+b.alpha,0)/s.length).values);
console.log(`95% CI: [${sbCI.lo.toFixed(2)}%, ${sbCI.hi.toFixed(2)}%]`);
console.log(sbCI.lo > uMean ? `→ CI lower > unconditional → significant ✓` : `→ CI includes/below unconditional → not significant ✗`);

// 2. Momentum baseline
console.log(`\n=== 2. Momentum Baseline ===`);
const momAlphas = momPairs.map(r => r.alpha);
const momMean = momAlphas.reduce((s,a)=>s+a,0)/momAlphas.length;
console.log(`alpha mean: ${momMean.toFixed(2)}%`);
const momCI = bootstrapCI(blockBootstrap(momPairs, r => `${r.code}|${r.cutoffDate}`, s => s.reduce((a,b)=>a+b.alpha,0)/s.length).values);
console.log(`95% CI: [${momCI.lo.toFixed(2)}%, ${momCI.hi.toFixed(2)}%]`);

// 3. Overlap
console.log(`\n=== 3. Overlap ===`);
const sbSet = new Set(sbPairs.map(r => `${r.code}|${r.cutoffDate}`));
const mSet = new Set(momPairs.map(r => `${r.code}|${r.cutoffDate}`));
const inter = new Set([...sbSet].filter(x => mSet.has(x)));
const jac = inter.size / new Set([...sbSet, ...mSet]).size;
console.log(`LLM: ${sbSet.size}  Momentum: ${mSet.size}  Intersection: ${inter.size}  Jaccard: ${jac.toFixed(3)}`);

// 4. Difference
console.log(`\n=== 4. Difference (sb − mom) ===`);
const diff = sbMean - momMean;
// paired bootstrap
const sbBl = new Map(), momBl = new Map();
for (const r of sbPairs) { const bk = `${r.code}|${r.cutoffDate}`; if (!sbBl.has(bk)) sbBl.set(bk, []); sbBl.get(bk).push(r); }
for (const r of momPairs) { const bk = `${r.code}|${r.cutoffDate}`; if (!momBl.has(bk)) momBl.set(bk, []); momBl.get(bk).push(r); }
const allBk = [...new Set([...sbBl.keys(), ...momBl.keys()])];
const diffVals = [];
for (let b = 0; b < N_BOOTSTRAP; b++) {
  const sSamp = [], mSamp = [];
  for (let i = 0; i < allBk.length; i++) {
    const bk = allBk[Math.floor(Math.random() * allBk.length)];
    if (sbBl.has(bk)) sSamp.push(...sbBl.get(bk));
    if (momBl.has(bk)) mSamp.push(...momBl.get(bk));
  }
  if (sSamp.length > 0 && mSamp.length > 0)
    diffVals.push(sSamp.reduce((a,b)=>a+b.alpha,0)/sSamp.length - mSamp.reduce((a,b)=>a+b.alpha,0)/mSamp.length);
}
const diffCI = bootstrapCI(diffVals);
console.log(`Difference: ${diff.toFixed(2)}%`);
console.log(`95% CI: [${diffCI.lo.toFixed(2)}%, ${diffCI.hi.toFixed(2)}%]`);

// Interpretation
console.log(`\n=== Interpretation ===`);
const sbSig = sbCI.lo > uMean;
const diffSig = diffCI.lo > 0;
if (sbSig && diffSig) console.log(`→ A) strong_bull significant + beats momentum → LLM has independent edge`);
else if (sbSig && !diffSig) console.log(`→ B) strong_bull significant but ≈ momentum (overlap=${jac.toFixed(2)}) → LLM is expensive momentum proxy`);
else console.log(`→ C) strong_bull itself not significant → expand sample before concluding`);
