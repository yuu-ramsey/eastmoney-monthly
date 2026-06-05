// v6-sector eval — strict comparison: Run A (no sector) vs Run B (with sector)
// 6 safeguards: dry-run / checkpoint resume / retry / rate limit / independent jsonl / progress log
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { buildPromptByTemplate } from '../lib/build-prompt.js';
import { computeMA } from '../lib/compute-ma.js';
import { computeMACD } from '../lib/compute-macd.js';
import { scorePrediction, computeScoreTransparent, formatScoreComparison } from '../lib/eval/compute-score.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const EVAL_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval');
const RUNS_DIR = path.join(EVAL_DIR, 'runs');

const EVAL_RUNNER_VERSION = '1.0.0';
const PARSER_VERSION = '1.0.0';
const PROMPT_VERSION = 'v6-sector';
const EVAL_MAX_TOKENS = 4000;

// ---- Guard 4: rate limit config ----
const LLM_CONCURRENCY = 2;
const LLM_DELAY_MS = 500;

// ---- Guard 3: retry config ----
const MAX_RETRIES = 3;
const RETRY_DELAYS_MS = [1000, 4000, 16000];

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ---- LLM call ----
async function callDeepSeek(prompt, apiKey, model = 'deepseek-chat') {
  const resp = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      max_tokens: EVAL_MAX_TOKENS,
      temperature: 0.0,
    }),
  });
  if (!resp.ok) {
    const errText = await resp.text().catch(() => '');
    throw new Error(`DeepSeek HTTP ${resp.status}: ${errText.slice(0, 200)}`);
  }
  const data = await resp.json();
  const text = data.choices?.[0]?.message?.content || '';
  const usage = data.usage || {};
  return { text, usage: { inputTokens: usage.prompt_tokens || 0, outputTokens: usage.completion_tokens || 0 } };
}

// ---- Guard 3: LLM call with retry ----
async function callWithRetry(prompt, apiKey, model) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const result = await callDeepSeek(prompt, apiKey, model);
      return { result, retries: attempt };
    } catch (err) {
      lastErr = err;
      if (attempt < MAX_RETRIES) {
        const delay = RETRY_DELAYS_MS[attempt] || 16000;
        await sleep(delay);
      }
    }
  }
  throw lastErr;
}

// ---- Parse LLM output ----
function parseSignal(rawResponse) {
  if (!rawResponse) return { predictedSignal: 'parse_failed', scoreData: null };
  let scoreData = null;
  try {
    const m = rawResponse.match(/```json\s*([\s\S]*?)```/);
    if (m) scoreData = JSON.parse(m[1].trim());
  } catch (_) {}
  const predictedSignal = scoreData?.signal || 'parse_failed';
  return { predictedSignal, scoreData };
}

// ---- Guard 2: checkpoint resume — read existing results ----
function loadCompleted(filePath) {
  if (!fs.existsSync(filePath)) return new Set();
  const content = fs.readFileSync(filePath, 'utf-8').trim();
  if (!content) return new Set();
  const completed = new Set();
  for (const line of content.split('\n').filter(Boolean)) {
    try {
      const r = JSON.parse(line);
      if (r.error) continue; // failed records also count as incomplete
      completed.add(`${r.stockCode}_${r.template}`);
    } catch (_) {}
  }
  return completed;
}

// ---- Guard 2: append to jsonl ----
function appendResult(filePath, record) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.appendFileSync(filePath, JSON.stringify(record) + '\n', 'utf-8');
}

// ---- Main flow ----
async function runOneMode({ mode, runMode, dataset, apiKey, model, db, calcSectorAlpha, resonanceCache, outPath }) {
  const { testPoints, stocks } = dataset;
  const stockMap = new Map(stocks.map(s => [s.code, s]));

  // checkpoint resume
  const completed = loadCompleted(outPath);
  const pending = [];
  for (const tp of testPoints) {
    for (const tpl of ['technical', 'trend', 'valuation', 'sentiment']) {
      const key = `${tp.stockCode}_${tpl}`;
      if (!completed.has(key)) {
        pending.push({ tp, tpl, key });
      }
    }
  }

  const total = testPoints.length * 4;
  const skipped = total - pending.length;
  console.log(`\n[${runMode}] total ${total} calls, completed ${skipped}, pending ${pending.length}`);

  if (pending.length === 0) {
    console.log(`[${runMode}] all completed, skip`);
    return { total, succeeded: total - skipped, failed: 0, retryStats: { firstTry: total - skipped, retrySuccess: 0, finalFail: 0 } };
  }

  // Preload all K-lines from DB
  console.log(`[${runMode}] preloading K-lines...`);
  const klinesCache = new Map();
  const uniqueCodes = [...new Set(pending.map(j => j.tp.stockCode))];
  for (const code of uniqueCodes) {
    const rows = db.prepare('SELECT * FROM monthly_klines WHERE code=? ORDER BY date').all(code);
    if (rows.length >= 24) klinesCache.set(code, rows);
  }
  console.log(`[${runMode}] K-line cache: ${klinesCache.size}/${uniqueCodes.size} stocks`);

  // Record start time
  const startTime = Date.now();
  let completed_count = skipped;
  let succeeded = skipped;
  let failed = 0;
  let totalCost = 0;
  const retryStats = { firstTry: skipped, retrySuccess: 0, finalFail: 0 };

  // Process each pending task
  const jobs = [...pending];
  let jobIdx = 0;

  // Guard 4: serial main loop, LLM_CONCURRENCY concurrent per batch
  while (jobIdx < jobs.length) {
    const batch = jobs.slice(jobIdx, jobIdx + LLM_CONCURRENCY);
    const batchPromises = batch.map(async ({ tp, tpl, key }) => {
      const stock = stockMap.get(tp.stockCode);
      const klines = klinesCache.get(tp.stockCode);
      if (!stock || !klines) {
        const record = makeErrorRecord(tp, tpl, runMode, 'K-line cache missing');
        appendResult(outPath, record);
        failed++;
        completed_count++;
        return { key, record, ok: false };
      }

      // Truncate by cutoffIndex
      if (tp.cutoffIndex >= klines.length || tp.cutoffIndex < 12) {
        const record = makeErrorRecord(tp, tpl, runMode, `cutoffIndex=${tp.cutoffIndex} exceeds K-line range ${klines.length}`);
        appendResult(outPath, record);
        failed++;
        completed_count++;
        return { key, record, ok: false };
      }

      const cutoffKlines = klines.slice(0, tp.cutoffIndex + 1);

      // Compute indicators
      const closes = cutoffKlines.map(k => k.close);
      const ma5 = computeMA(closes, 5);
      const ma20 = computeMA(closes, 20);
      const ma60 = computeMA(closes, 60);
      const { dif, dea, hist } = computeMACD(closes);

      const kwi = cutoffKlines.map((k, i) => ({
        date: k.date,
        open: k.open, close: k.close, high: k.high, low: k.low,
        volume: k.volume,
        changePercent: k.change_percent,
        turnoverRate: k.turnover_rate,
        ma5: ma5[i], ma20: ma20[i], ma60: ma60[i],
        dif: dif[i], dea: dea[i], hist: hist[i],
      }));

      // sector alpha
      let sectorAlphaData = null;
      if (mode === 'with_sector' && calcSectorAlpha) {
        try {
          sectorAlphaData = calcSectorAlpha(db, tp.stockCode, 'monthly', 12, tp.cutoffDate);
        } catch (_) {}
      }

      // resonance (Run B reverse_resonance only)
      let resonance = null;
      if (mode === 'reverse_resonance' && resonanceCache) {
        resonance = resonanceCache.get(`${tp.stockCode}_${tp.cutoffDate}`) || null;
      }

      // Build prompt
      const prompt = await buildPromptByTemplate({
        templateKey: tpl,
        name: stock.name || tp.stockCode,
        code: tp.stockCode,
        klines: kwi,
        period: 'monthly',
        provider: 'deepseek',
        decisionMode: false,
        sectorAlphaData,
        resonance,
      });

      // Call LLM (with retry)
      try {
        const { result, retries } = await callWithRetry(prompt, apiKey, model);
        if (retries > 0) retryStats.retrySuccess++;
        else retryStats.firstTry++;

        const cost = result.usage
          ? (result.usage.inputTokens || 0) / 1e6 * 1 + (result.usage.outputTokens || 0) / 1e6 * 4
          : 0.02;
        totalCost += cost;

        const { predictedSignal } = parseSignal(result.text);
        const score = scorePrediction(predictedSignal, tp.groundTruth);
        const truncated = (result.usage?.outputTokens || 0) >= EVAL_MAX_TOKENS;

        const record = {
          testPointId: tp.id,
          stockCode: tp.stockCode,
          stockName: tp.stockName,
          cutoffDate: tp.cutoffDate,
          cutoffIndex: tp.cutoffIndex,
          category: tp.category,
          template: tpl,
          model,
          promptVersion: PROMPT_VERSION,
          runMode,
          hasSectorAlpha: mode === 'with_sector',
          hasResonance: mode === 'reverse_resonance',
          hasReverseResonance: mode === 'reverse_resonance',
          predictedSignal,
          groundTruth: tp.groundTruth,
          score,
          cost,
          alpha: tp.alpha,
          rawResponse: result.text,
          promptUsed: prompt,
          tokensUsed: result.usage?.inputTokens
            ? { input: result.usage.inputTokens, output: result.usage.outputTokens }
            : null,
          truncated,
          retries,
          parserVersion: PARSER_VERSION,
          evalRunnerVersion: EVAL_RUNNER_VERSION,
          timestamp: new Date().toISOString(),
        };

        appendResult(outPath, record);
        succeeded++;
        completed_count++;
        return { key, record, ok: true, cost };
      } catch (err) {
        retryStats.finalFail++;
        const record = makeErrorRecord(tp, tpl, runMode, err.message);
        appendResult(outPath, record);
        failed++;
        completed_count++;
        return { key, record, ok: false };
      }
    });

    await Promise.allSettled(batchPromises);

    // Guard 6: progress log
    if (completed_count % 20 === 0 || completed_count === total) {
      const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
      const pct = (completed_count / total * 100).toFixed(1);
      const eta = Math.max(0, (parseFloat(elapsed) / completed_count * total - elapsed)).toFixed(0);
      console.log(`[${runMode}] ${completed_count}/${total} (${pct}%) ok=${succeeded} pf=${failed} ¥${totalCost.toFixed(2)} ${elapsed}min ETA${eta}min`);
    }

    jobIdx += LLM_CONCURRENCY;
    if (jobIdx < jobs.length) await sleep(LLM_DELAY_MS);
  }

  const elapsedMin = ((Date.now() - startTime) / 60000).toFixed(1);
  console.log(`\n[${runMode}] done: ${succeeded}/${total} ok, ${fail} fail, ${elapsedMin}min, ¥${totalCost.toFixed(4)}`);
  console.log(`[${runMode}] retry stats: first-try=${retryStats.firstTry} retry-ok=${retryStats.retrySuccess} final-fail=${retryStats.finalFail}`);

  return { total, succeeded, failed, totalCost, retryStats, elapsedMin, outPath };
}

function makeErrorRecord(tp, tpl, runMode, error) {
  return {
    testPointId: tp.id,
    stockCode: tp.stockCode,
    cutoffDate: tp.cutoffDate,
    template: tpl,
    runMode,
    hasSectorAlpha: runMode === 'B_with_sector',
    hasResonance: runMode === 'B_reverse_resonance',
    hasReverseResonance: runMode === 'B_reverse_resonance',
    predictedSignal: 'parse_failed',
    groundTruth: tp.groundTruth,
    score: null,
    error,
    parserVersion: PARSER_VERSION,
    evalRunnerVersion: EVAL_RUNNER_VERSION,
    timestamp: new Date().toISOString(),
  };
}

// ---- Statistics output ----
function printComparisonTable(resultsA, resultsB) {
  const readResults = (filePath) => {
    if (!fs.existsSync(filePath)) return [];
    return fs.readFileSync(filePath, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
  };

  const recsA = readResults(resultsA.outPath);
  const recsB = readResults(resultsB.outPath);

  const statsA = computeScoreTransparent(recsA);
  const statsB = computeScoreTransparent(recsB);

  const n = recsA.length; // assume Run A and Run B have same sample size

  function ci95(p, total) {
    if (total === 0) return '±?';
    const se = 1.96 * Math.sqrt(p * (1 - p) / total);
    return `±${se.toFixed(3)}`;
  }

  function sigTest(statA, statB, total) {
    if (!statA || !statB) return '—';
    const diff = statB - statA;
    const se = Math.sqrt((statA * (1 - statA) + statB * (1 - statB)) / total);
    const ci = 1.96 * se;
    return Math.abs(diff) > ci ? '✓' : '✗';
  }

  const fA = statsA.full;
  const fB = statsB.full;
  const pfA = statsA.exclPf;
  const pfB = statsB.exclPf;

  const strongBullA = fA.signalDistribution.strong_bull / fA.denominator;
  const strongBullB = fB.signalDistribution.strong_bull / fB.denominator;
  const strongBearA = fA.signalDistribution.strong_bear / fA.denominator;
  const strongBearB = fB.signalDistribution.strong_bear / fB.denominator;

  // strong_bull false positive
  const strongBullRecsA = recsA.filter(r => r.predictedSignal === 'strong_bull');
  const strongBullRecsB = recsB.filter(r => r.predictedSignal === 'strong_bull');
  const strongBullFPA = strongBullRecsA.length > 0 ? strongBullRecsA.filter(r => r.groundTruth !== 'strong_bull').length / strongBullRecsA.length : 0;
  const strongBullFPB = strongBullRecsB.length > 0 ? strongBullRecsB.filter(r => r.groundTruth !== 'strong_bull').length / strongBullRecsB.length : 0;
  const strongBearFPA = fA.signalDistribution.strong_bear > 0 ? recsA.filter(r => r.predictedSignal === 'strong_bear' && r.groundTruth !== 'strong_bear').length / fA.signalDistribution.strong_bear : 0;
  const strongBearFPB = fB.signalDistribution.strong_bear > 0 ? recsB.filter(r => r.predictedSignal === 'strong_bear' && r.groundTruth !== 'strong_bear').length / fB.signalDistribution.strong_bear : 0;

  const signalCoverageA = (fA.denominator - fA.signalDistribution.parse_failed) / fA.denominator;
  const signalCoverageB = (fB.denominator - fB.signalDistribution.parse_failed) / fB.denominator;

  const avgTokensA = recsA.filter(r => r.tokensUsed).reduce((s, r) => s + (r.tokensUsed?.input || 0) + (r.tokensUsed?.output || 0), 0) / Math.max(1, recsA.filter(r => r.tokensUsed).length);
  const avgTokensB = recsB.filter(r => r.tokensUsed).reduce((s, r) => s + (r.tokensUsed?.input || 0) + (r.tokensUsed?.output || 0), 0) / Math.max(1, recsB.filter(r => r.tokensUsed).length);

  console.log('\n' + '='.repeat(80));
  console.log('=== Run A (no sector) vs Run B (with sector) Comparison ===');
  console.log('='.repeat(80));

  const rows = [
    ['weighted score (all)', fA.weightedScore, fB.weightedScore, (fB.weightedScore - fA.weightedScore).toFixed(4), ci95(fA.weightedScore, n), sigTest(fA.weightedScore, fB.weightedScore, n)],
    ['score_excl_pf', pfA.weightedScore, pfB.weightedScore, (pfB.weightedScore - pfA.weightedScore).toFixed(4), '—', '—'],
    ['strong_bull %', fmtPct(strongBullA), fmtPct(strongBullB), fmtDelta(strongBullB - strongBullA), ci95(strongBullA, n), sigTest(strongBullA, strongBullB, n)],
    ['strong_bull FP', fmtPct(strongBullFPA), fmtPct(strongBullFPB), fmtDelta(strongBullFPB - strongBullFPA), '—', '—'],
    ['strong_bear %', fmtPct(strongBearA), fmtPct(strongBearB), fmtDelta(strongBearB - strongBearA), ci95(strongBearA, n), sigTest(strongBearA, strongBearB, n)],
    ['strong_bear FP', fmtPct(strongBearFPA), fmtPct(strongBearFPB), fmtDelta(strongBearFPB - strongBearFPA), '—', '—'],
    ['bear %', fmtPct(fA.signalDistribution.bear / fA.denominator), fmtPct(fB.signalDistribution.bear / fB.denominator), fmtDelta(fB.signalDistribution.bear / fB.denominator - fA.signalDistribution.bear / fA.denominator), '—', '—'],
    ['neutral %', fmtPct(fA.signalDistribution.neutral / fA.denominator), fmtPct(fB.signalDistribution.neutral / fB.denominator), fmtDelta(fB.signalDistribution.neutral / fB.denominator - fA.signalDistribution.neutral / fA.denominator), '—', '—'],
    ['parse_failed', fA.signalDistribution.parse_failed, fB.signalDistribution.parse_failed, fB.signalDistribution.parse_failed - fA.signalDistribution.parse_failed, '—', '—'],
    ['signal coverage', fmtPct(signalCoverageA), fmtPct(signalCoverageB), fmtDelta(signalCoverageB - signalCoverageA), '—', '—'],
    ['avg tokens', Math.round(avgTokensA), Math.round(avgTokensB), Math.round(avgTokensB - avgTokensA), '—', '—'],
    ['total cost', `¥${resultsA.totalCost?.toFixed(2) || '?'}`, `¥${resultsB.totalCost?.toFixed(2) || '?'}`, '—', '—', '—'],
  ];

  // Print table
  console.log('| Metric | Run A | Run B | Δ | 95% CI | Sig? |');
  console.log('|------|-------|-------|---|--------|-------|');
  for (const row of rows) {
    console.log(`| ${row[0]} | ${row[1]} | ${row[2]} | ${row[3]} | ${row[4]} | ${row[5]} |`);
  }

  return { statsA, statsB };
}

function fmtPct(v) { return (v * 100).toFixed(1) + '%'; }
function fmtDelta(v) { return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%'; }

// ================================================================
// CLI entry
// ================================================================
async function main() {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('Please set DEEPSEEK_API_KEY environment variable');

  const dryRun = process.argv.includes('--dry-run');
  const runAOnly = process.argv.includes('--run-a-only');
  const runBOnly = process.argv.includes('--run-b-only');
  const subsetArg = process.argv.find(a => a.startsWith('--subset='));
  const subsetN = subsetArg ? parseInt(subsetArg.split('=')[1]) : (dryRun ? 5 : null);
  const seedArg = process.argv.find(a => a.startsWith('--seed='));
  const subsetSeed = seedArg ? parseInt(seedArg.split('=')[1]) : 42;

  // Load from frozen dataset
  const { loadFrozenDataset } = await import('../lib/eval/load-frozen-dataset.js');
  const dataset = loadFrozenDataset({ version: 'v1', subsetStocks: subsetN, seed: subsetSeed });
  const testPoints = dataset.testPoints;

  console.log('=== eval-v6-sector ===');
  console.log(`Mode: ${dryRun ? 'DRY-RUN (subset=' + subsetN + ' stocks)' : 'FULL (frozen-v1, ' + dataset.stocks.length + ' stocks)'}`);
  if (dataset.subsetInfo) console.log(`Random seed: ${dataset.subsetInfo.seed}, sample: ${dataset.subsetInfo.nStocks}/${dataset.subsetInfo.totalStocks}`);
  console.log(`EVAL_RUNNER_VERSION=${EVAL_RUNNER_VERSION} PARSER_VERSION=${PARSER_VERSION} PROMPT_VERSION=${PROMPT_VERSION}`);
  console.log(`frozen baseline score: ${dataset.baseline?.score || 'N/A'}`);
  console.log('');

  console.log(`testPoints: ${testPoints.length} (total: ${dataset.testPoints.length})`);
  console.log(`stocks: ${dataset.stocks.length}`);
  console.log(`expected LLM calls: ${testPoints.length * 4} / mode`);

  // Lazy init DB
  const { getDb } = await import('../lib/db/connection.js');
  const db = getDb();

  const { calcSectorAlpha } = await import('../lib/sector/alpha.js');

  const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  let resultA = null, resultB = null;

  // Precompute resonance data (once per unique stockCode + cutoffDate)
  console.log('\nPrecomputing resonance...');
  const { getResonanceAsOf } = await import('../lib/multi-period/resonance.js');
  const resonanceCache = new Map();
  const uniqueTps = [...new Map(testPoints.map(tp => [`${tp.stockCode}_${tp.cutoffDate}`, tp])).values()];
  for (const tp of uniqueTps) {
    try {
      const r = await getResonanceAsOf(tp.stockCode, tp.cutoffDate);
      resonanceCache.set(`${tp.stockCode}_${tp.cutoffDate}`, r);
    } catch (_) {
      resonanceCache.set(`${tp.stockCode}_${tp.cutoffDate}`, null);
    }
  }
  const validResonance = [...resonanceCache.values()].filter(r => r && r.resonanceLevel !== 'divergent').length;
  console.log(`Resonance cache: ${resonanceCache.size} entries, valid=${validResonance}`);

  // --- Run A: no resonance ---
  if (!runBOnly) {
    const outPathA = path.join(RUNS_DIR, `runA-no-resonance-${timestamp}.jsonl`);
    console.log(`\nRun A output: ${path.basename(outPathA)}`);
    resultA = await runOneMode({
      mode: 'no_resonance',
      runMode: 'A_no_resonance',
      dataset: { ...dataset, testPoints },
      apiKey,
      model: 'deepseek-chat',
      db,
      calcSectorAlpha: null,
      resonanceCache: null,
      outPath: outPathA,
    });
  }

  // --- Run B: reverse resonance ---
  if (!runAOnly) {
    const outPathB = path.join(RUNS_DIR, `runB-reverse-resonance-${timestamp}.jsonl`);
    console.log(`\nRun B output: ${path.basename(outPathB)}`);
    resultB = await runOneMode({
      mode: 'reverse_resonance',
      runMode: 'B_reverse_resonance',
      dataset: { ...dataset, testPoints },
      apiKey,
      model: 'deepseek-chat',
      db,
      calcSectorAlpha: null,
      resonanceCache,
      outPath: outPathB,
    });
  }

  // Comparison table
  if (resultA && resultB) {
    printComparisonTable(
      { outPath: resultA.outPath || path.join(RUNS_DIR, `runA-no-resonance-${timestamp}.jsonl`), totalCost: resultA.totalCost },
      { outPath: resultB.outPath || path.join(RUNS_DIR, `runB-reverse-resonance-${timestamp}.jsonl`), totalCost: resultB.totalCost },
    );
  }
}

main().catch(err => { console.error(err); process.exit(1); });
