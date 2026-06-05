// Power analysis: compute sample size needed to detect strong_bull effect
// Usage: node cli/eval-power-analysis.js
import { readFileSync } from 'fs';

const PROJECT_DIR = process.cwd();

const ds = JSON.parse(readFileSync(PROJECT_DIR + '/data/frozen-eval-dataset-v1.json', 'utf-8'));
const v4Path = PROJECT_DIR + '/.eastmoney-ai/eval/runs/v4-signals-2026-05-17-00-41.jsonl';
const v4 = readFileSync(v4Path, 'utf-8')
  .trim().split('\n').filter(Boolean).map(l => JSON.parse(l)).filter(r => !r.error);
if (v4.length === 0) { console.error('v4 empty, path:', v4Path); process.exit(1); }

// unique (stock, cutoffDate) pairs
const pairs = new Map();
for (const tp of ds.testPoints) {
  const k = tp.stockCode + '|' + tp.cutoffDate;
  if (!pairs.has(k)) pairs.set(k, tp.alpha);
}
const alphas = [...pairs.values()];
const alphaMean = alphas.reduce((s, a) => s + a, 0) / alphas.length;
const alphaStd = Math.sqrt(alphas.reduce((s, a) => s + Math.pow(a - alphaMean, 2), 0) / alphas.length);

// strong_bull unique pairs
const sbSet = new Set();
v4.filter(r => (r.signal || r.predictedSignal) === 'strong_bull').forEach(r => sbSet.add(r.code + '|' + r.cutoffDate));
const sbAlphas = [...sbSet].map(k => pairs.get(k)).filter(a => a != null);
const sbMean = sbAlphas.reduce((s, a) => s + a, 0) / sbAlphas.length;
const sbN = sbAlphas.length;

const avgRecs = v4.length / [...new Set(v4.map(r => r.code + '|' + r.cutoffDate))].size;
console.log('// DEBUG v4.length:', v4.length, 'unique:', new Set(v4.map(r => r.code + '|' + r.cutoffDate)).size);
const ICC = 0.3;
const deff = 1 + (avgRecs - 1) * ICC;

function requiredNPerGroup(effectSize, stdDev, de) {
  const d = effectSize / stdDev;
  if (d <= 0) return Infinity;
  return Math.ceil(2 * Math.pow(1.96 + 0.84, 2) / (d * d) * de);
}

function mdeForN(nPerGroup, stdDev, de) {
  if (nPerGroup <= 0) return Infinity;
  return (1.96 + 0.84) * stdDev * Math.sqrt(2 * de / nPerGroup);
}

function cdLabel(d) {
  const ad = Math.abs(d);
  if (ad < 0.2) return 'negligible';
  if (ad < 0.5) return 'small';
  if (ad < 0.8) return 'medium';
  return 'large';
}

console.log('=== Power Analysis: Sample Size to Detect strong_bull Effect ===\n');

console.log('Data basis:');
console.log(`  40 stocks (all HS300, manually selected for long history & high liquidity)`);
console.log(`  160 unique (stock, timepoint) pairs, strong_bull=${sbN}`);
console.log(`  alpha cross-sectional std: ${alphaStd.toFixed(1)}%, mean: ${alphaMean.toFixed(1)}%`);
console.log(`  design effect deff=${deff.toFixed(2)} (ICC=${ICC}, ${avgRecs.toFixed(0)} records/pair)\n`);

console.log('1. MDE curve (power=0.8, α=0.05, two independent group mean difference)\n');
const nLevels = [34, 60, 100, 160, 240, 360, 600];
console.log('  n_sb     total_pairs    MDE(%)      d');
console.log('  ───────────────────────────────────────');
for (const n of nLevels) {
  const mde = mdeForN(n, alphaStd, deff);
  const totalN = Math.round(n / 0.2);
  console.log(`  ${String(n).padStart(4)}      ~${String(totalN).padStart(4)}         ${mde.toFixed(1).padStart(5)}%      ${cdLabel(mde/alphaStd)}`);
}

console.log('\n2. Given effect → required sample size\n');
const effects = [
  { label: 'current point est 12.5%', delta: 12.5 },
  { label: 'half 6.0%', delta: 6.0 },
  { label: '1/3 4.2%', delta: 4.2 },
  { label: 'conservative 3.0%', delta: 3.0 },
];
console.log('  effect           n_sb     total_pairs    40stocks×?t  or  100stocks×?t');
console.log('  ───────────────────────────────────────────────────────────────────');
for (const { label, delta } of effects) {
  const n = requiredNPerGroup(delta, alphaStd, deff);
  const total = Math.round(n / 0.2);
  console.log(`  ${label.padEnd(14)} ${String(n).padStart(4)}    ~${String(total).padStart(5)}         ${Math.ceil(total/4).toString().padStart(2)}×4t (40st)     ${Math.ceil(total/4).toString().padStart(2)}×?t`);
}

console.log('\n3. After Winsorize (trim p1/p99, std≈20%)\n');
const wStd = 20;
console.log('  effect           n_sb     total_pairs    40stocks×?t');
console.log('  ─────────────────────────────────────────────');
for (const { label, delta } of effects) {
  const n = requiredNPerGroup(delta, wStd, deff);
  const total = Math.round(n / 0.2);
  console.log(`  ${label.padEnd(14)} ${String(n).padStart(4)}    ~${String(total).padStart(5)}         ${Math.ceil(total/4).toString().padStart(2)}×4t`);
}

console.log('\n4. One-line recommendation\n');
console.log(`  alpha std=${alphaStd.toFixed(0)}% is the bottleneck. 6 outliers (alpha≤-30% or ≥150%) contribute ~60% of variance.`);
console.log(`  Recommend Winsorize p1/p99 to compress std to ~20%, target effect ≥ 6% (half point estimate):`);
const needSb = requiredNPerGroup(6.0, 20, deff);
console.log(`    → ${needSb} strong_bull, ~${Math.round(needSb/0.2)} total pairs`);
const t6 = Math.ceil(Math.round(needSb / 0.2) / 6);
const t12 = Math.ceil(Math.round(needSb / 0.2) / 12);
console.log(`    → Plan A: ${t6} stocks × 6 timepoints (50% bull market, 50% bear market)`);
console.log(`    → Plan B: ${Math.max(40, t12*2)} stocks × 4 timepoints (conservative, wider cross-section)`);
