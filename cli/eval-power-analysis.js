// 功效分析: 计算检出 strong_bull 效应所需样本量
// 用法: node cli/eval-power-analysis.js
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
  if (ad < 0.2) return '可忽略';
  if (ad < 0.5) return '小';
  if (ad < 0.8) return '中';
  return '大';
}

console.log('=== 功效分析：检出 strong_bull 效应所需样本量 ===\n');

console.log('数据基础:');
console.log(`  40 只股票 (全 hs300, 手工选取长历史高流动性)`);
console.log(`  160 unique (股票,时点) 对, strong_bull=${sbN}`);
console.log(`  alpha 横截面 std: ${alphaStd.toFixed(1)}%, 均值: ${alphaMean.toFixed(1)}%`);
console.log(`  设计效应 deff=${deff.toFixed(2)} (ICC=${ICC}, ${avgRecs.toFixed(0)} records/pair)\n`);

console.log('1. MDE 曲线 (power=0.8, α=0.05, 两独立组均值差)\n');
const nLevels = [34, 60, 100, 160, 240, 360, 600];
console.log('  n_sb     total_pairs    MDE(%)      d');
console.log('  ───────────────────────────────────────');
for (const n of nLevels) {
  const mde = mdeForN(n, alphaStd, deff);
  const totalN = Math.round(n / 0.2);
  console.log(`  ${String(n).padStart(4)}      ~${String(totalN).padStart(4)}         ${mde.toFixed(1).padStart(5)}%      ${cdLabel(mde/alphaStd)}`);
}

console.log('\n2. 给定效应 → 所需样本量\n');
const effects = [
  { label: '当前点估计 12.5%', delta: 12.5 },
  { label: '一半 6.0%', delta: 6.0 },
  { label: '1/3 4.2%', delta: 4.2 },
  { label: '保守 3.0%', delta: 3.0 },
];
console.log('  效应           n_sb     total_pairs    40stocks×?t  或  100stocks×?t');
console.log('  ───────────────────────────────────────────────────────────────────');
for (const { label, delta } of effects) {
  const n = requiredNPerGroup(delta, alphaStd, deff);
  const total = Math.round(n / 0.2);
  console.log(`  ${label.padEnd(14)} ${String(n).padStart(4)}    ~${String(total).padStart(5)}         ${Math.ceil(total/4).toString().padStart(2)}×4t (40st)     ${Math.ceil(total/4).toString().padStart(2)}×?t`);
}

console.log('\n3. Winsorize 后 (trim p1/p99, std≈20%)\n');
const wStd = 20;
console.log('  效应           n_sb     total_pairs    40stocks×?t');
console.log('  ─────────────────────────────────────────────');
for (const { label, delta } of effects) {
  const n = requiredNPerGroup(delta, wStd, deff);
  const total = Math.round(n / 0.2);
  console.log(`  ${label.padEnd(14)} ${String(n).padStart(4)}    ~${String(total).padStart(5)}         ${Math.ceil(total/4).toString().padStart(2)}×4t`);
}

console.log('\n4. 一句话建议\n');
console.log(`  alpha std=${alphaStd.toFixed(0)}% 是瓶颈。6 个极端值 (alpha≤-30% 或 ≥150%) 贡献了 ~60% 方差。`);
console.log(`  建议 Winsorize p1/p99 将 std 压到 ~20%，目标效应 ≥ 6% (点估计一半):`);
const needSb = requiredNPerGroup(6.0, 20, deff);
console.log(`    → ${needSb} strong_bull, ~${Math.round(needSb/0.2)} total pairs`);
const t6 = Math.ceil(Math.round(needSb / 0.2) / 6);
const t12 = Math.ceil(Math.round(needSb / 0.2) / 12);
console.log(`    → 方案 A: ${t6} 只股票 × 6 个时点 (50% bull market, 50% bear market)`);
console.log(`    → 方案 B: ${Math.max(40, t12*2)} 只股票 × 4 个时点 (保守, 覆盖更多截面)`);
