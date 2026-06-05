// Conditional signal re-analysis — using real alpha to evaluate prediction information content
// Usage: node cli/eval-reanalyze-conditional.js
// Input: v4-signals jsonl + frozen-baseline jsonl + frozen-eval-dataset-v1.json
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

// ---- Load ----

function loadJsonl(filePath) {
  if (!fs.existsSync(filePath)) {
    console.error(`File not found: ${filePath}`);
    return [];
  }
  return fs.readFileSync(filePath, 'utf-8').trim().split('\n').filter(Boolean).map(l => {
    try { return JSON.parse(l); } catch (_) { return null; }
  }).filter(r => r !== null && !r.error);
}

function loadDataset() {
  if (!fs.existsSync(DATASET_PATH)) throw new Error(`Dataset not found: ${DATASET_PATH}`);
  return JSON.parse(fs.readFileSync(DATASET_PATH, 'utf-8'));
}

// ---- Rebuild alpha label ----

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

// ---- Analysis ----

function analyze(records, label) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`  ${label}`);
  console.log(`${'='.repeat(70)}`);

  const dataset = loadDataset();
  const alphaLookup = buildAlphaLookup(dataset);

  // Link alpha
  const withAlpha = [];
  for (const r of records) {
    const code = r.code || r.stockCode;
    const key = `${code}|${r.cutoffDate}`;
    const labelData = alphaLookup.get(key);
    if (!labelData) continue;
    withAlpha.push({ ...r, ...labelData });
  }

  console.log(`Records: ${records.length} → linked alpha: ${withAlpha.length}`);

  // ====== 1. Conditional realized return ======
  console.log(`\n--- 1. Conditional Realized Return ---`);

  const allAlphas = withAlpha.map(r => r.alpha);
  const unconditionalMean = allAlphas.reduce((s, a) => s + a, 0) / allAlphas.length;
  const unconditionalMedian = percentile(allAlphas, 0.5);
  const unconditionalPosPct = allAlphas.filter(a => a > 0).length / allAlphas.length * 100;

  console.log(`All (n=${allAlphas.length}): alpha mean=${unconditionalMean.toFixed(2)}%  median=${unconditionalMedian.toFixed(2)}%  alpha>0 pct=${unconditionalPosPct.toFixed(1)}%`);

  const buckets = ['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear'];
  console.log(`\nPred Bucket        n    alpha mean  alpha med  alpha>0%   vs unc Δ`);
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

  // Interpretation
  const sb = bucketStats.strong_bull;
  const bu = bucketStats.bull;
  if (sb && bu) {
    const bullAlphas = (sb.alphas || []).concat(bu.alphas || []);
    const bullMean = bullAlphas.length > 0 ? bullAlphas.reduce((s, a) => s + a, 0) / bullAlphas.length : 0;
    console.log(`\nInterpretation: bullish combined mean=${bullMean.toFixed(2)}%, unconditional mean=${unconditionalMean.toFixed(2)}%`);
    console.log(`  strong_bull bucket: ${sb.mean >= unconditionalMean ? '✓ above unconditional' : '✗ not above unconditional'}`);
    console.log(`  bull bucket:        ${bu.mean >= unconditionalMean ? '✓ above unconditional' : '✗ not above unconditional'}`);
  }

  // ====== 2. Directional discrimination (spread) ======
  console.log(`\n--- 2. Directional Discrimination (spread) ---`);

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

  console.log(`bullish (n=${bullish.length}): alpha mean=${bullishMean?.toFixed(2)}%`);
  console.log(`bearish (n=${bearish.length}): alpha mean=${bearishMean?.toFixed(2)}%`);
  console.log(`neutral (n=${neutralRecs.length}): alpha mean=${neutralMean?.toFixed(2)}%`);
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
    console.log(`  → ✓ CI excludes 0 → spread significantly > 0, prediction has directional discrimination`);
  } else if (spreadCI.hi < 0) {
    console.log(`  → ✗ CI excludes 0 but spread < 0 → prediction direction contrary to true returns`);
  } else {
    console.log(`  → ✗ CI includes 0 → cannot reject spread=0, no significant directional discrimination`);
  }

  // ====== 3. Directional hit rate ======
  console.log(`\n--- 3. Directional Hit Rate ---`);

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

  console.log(`Directional predictions (n=${directional.length}): hit=${dirHit.length} (${hitRate.toFixed(1)}%)`);
  console.log(`Always-bullish hit rate baseline: ${alwaysBullHitRate.toFixed(1)}% (reflects GT bullish prior)`);
  console.log(`Model excess: ${hitRate >= alwaysBullHitRate ? '+' : ''}${(hitRate - alwaysBullHitRate).toFixed(1)}pp`);

  // Per-bucket detail
  console.log(`\nPer-bucket hit rate:`);
  for (const b of ['strong_bull', 'bull']) {
    const subset = directional.filter(r => (r.signal || r.predictedSignal) === b);
    if (subset.length === 0) continue;
    const hit = subset.filter(r => r.alpha > 0).length;
    console.log(`  ${b.padEnd(14)} n=${String(subset.length).padStart(4)}  hit=${hit} (${(hit / subset.length * 100).toFixed(1)}%)`);
  }
  for (const b of ['bear', 'strong_bear']) {
    const subset = directional.filter(r => (r.signal || r.predictedSignal) === b);
    if (subset.length === 0) continue;
    const hit = subset.filter(r => r.alpha < 0).length;
    console.log(`  ${b.padEnd(14)} n=${String(subset.length).padStart(4)}  hit=${hit} (${(hit / subset.length * 100).toFixed(1)}%)`);
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

// ---- Main flow ----

console.log('=== Conditional Signal Re-analysis: Prediction Information via Realized Alpha ===\n');

const v4Records = loadJsonl(V4_PATH);
const baselineRecords = loadJsonl(BASELINE_PATH);

const results = [];
if (v4Records.length > 0) {
  results.push(analyze(v4Records, 'v4-signals (score=0.187 exclPF)'));
} else {
  console.error(`v4 data not found: ${V4_PATH}`);
}

if (baselineRecords.length > 0) {
  results.push(analyze(baselineRecords, `frozen-baseline (score=0.1966, ${BASELINE_RUN_ID})`));
} else {
  console.error(`Baseline data not found: ${BASELINE_PATH}`);
}

// Final table
if (results.length === 2) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`  Comparison Summary`);
  console.log(`${'='.repeat(70)}`);
  console.log(`Metric                          v4(0.187)        baseline(0.197)`);
  console.log('-'.repeat(60));
  for (let i = 0; i < 2; i++) {
    const r = results[i];
    const tag = i === 0 ? 'v4' : 'bl';
    console.log(`${tag} spread                    ${r.spread?.toFixed(2)}%`);
    console.log(`${tag} spread 95% CI              [${r.spreadCI.lo.toFixed(2)}, ${r.spreadCI.hi.toFixed(2)}]`);
    console.log(`${tag} directional hit rate      ${r.hitRate.toFixed(1)}%`);
    console.log(`${tag} n_eff (blocks)             ${r.nBlocks}`);
    console.log('');
  }
}
