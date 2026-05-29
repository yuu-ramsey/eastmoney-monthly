// 条件信号重分析 — 不依赖评分矩阵，只用真实 alpha 评估预测信息量
// 用法: node cli/eval-reanalyze-conditional.js
// 输入: v4-signals jsonl + frozen-baseline jsonl + frozen-eval-dataset-v1.json
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const RUNS_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs');
const DATA_DIR = path.join(PROJECT_DIR, 'data');

const V4_PATH = path.join(RUNS_DIR, 'v4-signals-2026-05-17-00-41.jsonl');
const BASELINE_RUN_ID = 'runA-no-sector-2026-05-18-05-12-20';
const BASELINE_PATH = path.join(RUNS_DIR, `${BASELINE_RUN_ID}.jsonl`);
const DATASET_PATH = path.join(DATA_DIR, 'frozen-eval-dataset-v1.json');
const N_BOOTSTRAP = 10000;

// ---- 加载 ----

function loadJsonl(filePath) {
  if (!fs.existsSync(filePath)) {
    console.error(`文件不存在: ${filePath}`);
    return [];
  }
  return fs.readFileSync(filePath, 'utf-8').trim().split('\n').filter(Boolean).map(l => {
    try { return JSON.parse(l); } catch (_) { return null; }
  }).filter(r => r !== null && !r.error);
}

function loadDataset() {
  if (!fs.existsSync(DATASET_PATH)) throw new Error(`Dataset 不存在: ${DATASET_PATH}`);
  return JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'));
}

// ---- 重建 alpha label ----

function buildAlphaLookup(dataset) {
  const map = new Map();
  for (const tp of dataset.testPoints) {
    const key = `${tp.stockCode}|${tp.cutoffDate}`;
    map.set(key, {
      alpha: tp.alpha,
      actualReturn: tp.actualReturn,
      indexReturn: tp.indexReturn,
      groundTruth: tp.groundTruth,
    });
  }
  return map;
}

// ---- Bootstrap ----

function blockBootstrap(records, keyFn, nBootstrap) {
  const blocks = new Map();
  for (const r of records) {
    const bk = keyFn(r);
    if (!blocks.has(bk)) blocks.set(bk, []);
    blocks.get(bk).push(r);
  }
  const blockList = [...blocks.values()];
  const nBlocks = blockList.length;

  const results = [];
  for (let b = 0; b < nBootstrap; b++) {
    const sample = [];
    for (let i = 0; i < nBlocks; i++) {
      const idx = Math.floor(Math.random() * nBlocks);
      sample.push(...blockList[idx]);
    }
    results.push(sample);
  }
  return { results, nBlocks };
}

function percentileCI(bootstrapValues, alpha = 0.05) {
  const sorted = [...bootstrapValues].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * alpha / 2)];
  const hi = sorted[Math.floor(sorted.length * (1 - alpha / 2))];
  const mean = sorted.reduce((s, v) => s + v, 0) / sorted.length;
  return { lo, hi, mean, n: bootstrapValues.length };
}

// ---- 分析 ----

function analyze(records, label) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`  ${label}`);
  console.log(`${'='.repeat(70)}`);

  const dataset = loadDataset();
  const alphaLookup = buildAlphaLookup(dataset);

  // 关联 alpha
  const withAlpha = [];
  for (const r of records) {
    const code = r.code || r.stockCode;
    const key = `${code}|${r.cutoffDate}`;
    const labelData = alphaLookup.get(key);
    if (!labelData) continue;
    withAlpha.push({ ...r, ...labelData });
  }

  console.log(`记录: ${records.length} → 关联 alpha: ${withAlpha.length}`);

  // ====== 1. 条件实现收益 ======
  console.log(`\n--- 1. 条件实现收益 ---`);

  const allAlphas = withAlpha.map(r => r.alpha);
  const unconditionalMean = allAlphas.reduce((s, a) => s + a, 0) / allAlphas.length;
  const unconditionalMedian = percentile(allAlphas, 0.5);
  const unconditionalPosPct = allAlphas.filter(a => a > 0).length / allAlphas.length * 100;

  console.log(`全样本 (n=${allAlphas.length}): alpha均值=${unconditionalMean.toFixed(2)}%  中位数=${unconditionalMedian.toFixed(2)}%  alpha>0占比=${unconditionalPosPct.toFixed(1)}%`);

  const buckets = ['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear'];
  console.log(`\n预测桶            n    alpha均值  alpha中位  alpha>0%   vs无条件Δ`);
  console.log('-'.repeat(60));

  const bucketStats = {};
  for (const b of buckets) {
    const subset = withAlpha.filter(r => (r.signal || r.predictedSignal || 'parse_failed') === b);
    if (subset.length === 0) { bucketStats[b] = null; continue; }
    const alphas = subset.map(r => r.alpha);
    const mean = alphas.reduce((s, a) => s + a, 0) / alphas.length;
    const median = percentile(alphas, 0.5);
    const posPct = alphas.filter(a => a > 0).length / alphas.length * 100;
    const delta = mean - unconditionalMean;
    bucketStats[b] = { n: subset.length, mean, median, posPct, delta, alphas };
    console.log(`${b.padEnd(14)} ${String(subset.length).padStart(5)} ${mean.toFixed(2).padStart(8)}% ${median.toFixed(2).padStart(8)}% ${posPct.toFixed(1).padStart(7)}% ${delta >= 0 ? '+' : ''}${delta.toFixed(2).padStart(8)}%`);
  }

  // 判读
  const sb = bucketStats.strong_bull;
  const bu = bucketStats.bull;
  if (sb && bu) {
    const bullAlphas = (sb.alphas || []).concat(bu.alphas || []);
    const bullMean = bullAlphas.length > 0 ? bullAlphas.reduce((s, a) => s + a, 0) / bullAlphas.length : 0;
    console.log(`\n判读: bullish合并均值=${bullMean.toFixed(2)}%, 无条件均值=${unconditionalMean.toFixed(2)}%`);
    console.log(`  strong_bull桶: ${sb.mean >= unconditionalMean ? '✓ 高于无条件' : '✗ 不高于无条件'}`);
    console.log(`  bull桶:        ${bu.mean >= unconditionalMean ? '✓ 高于无条件' : '✗ 不高于无条件'}`);
  }

  // ====== 2. 方向区分度 (spread) ======
  console.log(`\n--- 2. 方向区分度 (spread) ---`);

  const bullish = withAlpha.filter(r => {
    const s = r.signal || r.predictedSignal || '';
    return s === 'strong_bull' || s === 'bull';
  });
  const bearish = withAlpha.filter(r => {
    const s = r.signal || r.predictedSignal || '';
    return s === 'strong_bear' || s === 'bear';
  });
  const neutralRecs = withAlpha.filter(r => {
    const s = r.signal || r.predictedSignal || '';
    return s === 'neutral';
  });

  const bullishMean = bullish.length > 0 ? bullish.reduce((s, r) => s + r.alpha, 0) / bullish.length : null;
  const bearishMean = bearish.length > 0 ? bearish.reduce((s, r) => s + r.alpha, 0) / bearish.length : null;
  const neutralMean = neutralRecs.length > 0 ? neutralRecs.reduce((s, r) => s + r.alpha, 0) / neutralRecs.length : null;
  const spread = bullishMean !== null && bearishMean !== null ? bullishMean - bearishMean : null;

  console.log(`bullish (n=${bullish.length}): alpha均值=${bullishMean?.toFixed(2)}%`);
  console.log(`bearish (n=${bearish.length}): alpha均值=${bearishMean?.toFixed(2)}%`);
  console.log(`neutral (n=${neutralRecs.length}): alpha均值=${neutralMean?.toFixed(2)}%`);
  console.log(`spread (bullish − bearish): ${spread?.toFixed(2)}%`);

  // Block bootstrap CI for spread
  const blockKey = r => `${r.code || r.stockCode}|${r.cutoffDate}`;
  const { results: bootSamples, nBlocks } = blockBootstrap(withAlpha, blockKey, N_BOOTSTRAP);

  const bootSpreads = bootSamples.map(sample => {
    const bl = sample.filter(r => {
      const s = r.signal || r.predictedSignal || '';
      return s === 'strong_bull' || s === 'bull';
    });
    const br = sample.filter(r => {
      const s = r.signal || r.predictedSignal || '';
      return s === 'strong_bear' || s === 'bear';
    });
    if (bl.length === 0 || br.length === 0) return null;
    const blM = bl.reduce((s, r) => s + r.alpha, 0) / bl.length;
    const brM = br.reduce((s, r) => s + r.alpha, 0) / br.length;
    return blM - brM;
  }).filter(s => s !== null);

  const spreadCI = percentileCI(bootSpreads);

  console.log(`\nBlock Bootstrap (${nBlocks} blocks, ${N_BOOTSTRAP} resamples):`);
  console.log(`  spread 95% CI: [${spreadCI.lo.toFixed(2)}%, ${spreadCI.hi.toFixed(2)}%]`);
  console.log(`  spread bootstrap mean: ${spreadCI.mean.toFixed(2)}%`);
  console.log(`  n_eff ≈ ${nBlocks} (blocks)`);
  if (spreadCI.lo > 0) {
    console.log(`  → ✓ CI 不含 0 → spread 显著 > 0，预测有方向区分力`);
  } else if (spreadCI.hi < 0) {
    console.log(`  → ✗ CI 不含 0 但 spread < 0 → 预测方向与真实收益反向`);
  } else {
    console.log(`  → ✗ CI 含 0 → 无法拒绝 spread=0，预测无显著方向区分力`);
  }

  // ====== 3. 方向命中率 ======
  console.log(`\n--- 3. 方向命中率 ---`);

  const directional = withAlpha.filter(r => {
    const s = r.signal || r.predictedSignal || '';
    return s === 'strong_bull' || s === 'bull' || s === 'strong_bear' || s === 'bear';
  });

  const dirHit = directional.filter(r => {
    const s = r.signal || r.predictedSignal || '';
    const isBull = s === 'strong_bull' || s === 'bull';
    return (isBull && r.alpha > 0) || (!isBull && r.alpha < 0);
  });

  const hitRate = directional.length > 0 ? dirHit.length / directional.length * 100 : 0;
  const alwaysBullHitRate = directional.length > 0 ? directional.filter(r => r.alpha > 0).length / directional.length * 100 : 0;

  console.log(`方向性预测 (n=${directional.length}): 命中=${dirHit.length} (${hitRate.toFixed(1)}%)`);
  console.log(`Always-bullish 命中率对照: ${alwaysBullHitRate.toFixed(1)}% (反映 GT 的 bullish 先验)`);
  console.log(`模型超额: ${hitRate >= alwaysBullHitRate ? '+' : ''}${(hitRate - alwaysBullHitRate).toFixed(1)}pp`);

  // 分桶细分
  console.log(`\n分桶命中率:`);
  for (const b of ['strong_bull', 'bull']) {
    const subset = directional.filter(r => (r.signal || r.predictedSignal) === b);
    if (subset.length === 0) continue;
    const hit = subset.filter(r => r.alpha > 0).length;
    console.log(`  ${b.padEnd(14)} n=${String(subset.length).padStart(4)}  命中=${hit} (${(hit / subset.length * 100).toFixed(1)}%)`);
  }
  for (const b of ['bear', 'strong_bear']) {
    const subset = directional.filter(r => (r.signal || r.predictedSignal) === b);
    if (subset.length === 0) continue;
    const hit = subset.filter(r => r.alpha < 0).length;
    console.log(`  ${b.padEnd(14)} n=${String(subset.length).padStart(4)}  命中=${hit} (${(hit / subset.length * 100).toFixed(1)}%)`);
  }

  return {
    label,
    n: withAlpha.length,
    nBlocks,
    unconditionalMean,
    bucketStats,
    bullishMean,
    bearishMean,
    spread,
    spreadCI,
    hitRate,
    alwaysBullHitRate,
  };
}

function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = p * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

// ---- 主流程 ----

console.log('=== 条件信号重分析：基于真实 alpha 的预测信息量 ===\n');

const v4Records = loadJsonl(V4_PATH);
const baselineRecords = loadJsonl(BASELINE_PATH);

const results = [];
if (v4Records.length > 0) {
  results.push(analyze(v4Records, 'v4-signals (score=0.187 exclPF)'));
} else {
  console.error(`v4 数据不存在: ${V4_PATH}`);
}

if (baselineRecords.length > 0) {
  results.push(analyze(baselineRecords, `frozen-baseline (score=0.1966, ${BASELINE_RUN_ID})`));
} else {
  console.error(`Baseline 数据不存在: ${BASELINE_PATH}`);
}

// 终表
if (results.length === 2) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`  对比总结`);
  console.log(`${'='.repeat(70)}`);
  console.log(`指标                          v4(0.187)        baseline(0.197)`);
  console.log('-'.repeat(60));
  for (let i = 0; i < 2; i++) {
    const r = results[i];
    const tag = i === 0 ? 'v4' : 'bl';
    console.log(`${tag} spread                    ${r.spread?.toFixed(2)}%`);
    console.log(`${tag} spread 95% CI              [${r.spreadCI.lo.toFixed(2)}, ${r.spreadCI.hi.toFixed(2)}]`);
    console.log(`${tag} 方向命中率                 ${r.hitRate.toFixed(1)}%`);
    console.log(`${tag} n_eff (blocks)             ${r.nBlocks}`);
    console.log('');
  }
}
