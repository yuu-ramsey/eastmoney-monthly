// Evaluation runner v2 — kline cache + multi-source fallback + concurrency control + distribution stats
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { KlinesCache, fetchWithRetry } from './klines-cache.js';
import { buildPromptByTemplate } from '../build-prompt.js';
import { computeMA } from '../compute-ma.js';
import { computeMACD } from '../compute-macd.js';

// Schema version — increment when runner output format changes
const EVAL_RUNNER_VERSION = '1.0.0';
// Used to track parser version (synced with JSON parse logic in runner.js)
const PARSER_VERSION = '1.0.0';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const EVAL_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval');

// eval mode fixes maxTokens=4000 (cannot override, prevents truncated JSON causing signal loss)
export const EVAL_MAX_TOKENS = 4000;
// Truncation warning threshold (truncated when outputTokens == EVAL_MAX_TOKENS)
export const TRUNCATION_WARN_RATIO = 0.05;

// Built-in safe fallback function (bypasses known dispatcher sina/tencent bugs)
export async function fetchKlinesSafe(market, code, period, limit) {
  // Try eastmoney first
  try {
    const mod = await import('../data-sources/eastmoney.js');
    const r = await mod.fetchKlines({ market, code, period, limit });
    if (r && Array.isArray(r.klines) && r.klines.length >= 12) return r.klines || r;
  } catch (_) {}
  // Fallback to sina
  try {
    const mod = await import('../data-sources/sina.js');
    const r = await mod.fetchKlines({ market, code, period, limit });
    if (r && Array.isArray(r.klines) && r.klines.length >= 12) return r.klines || r;
  } catch (_) {}
  // Fallback to tencent
  const mod = await import('../data-sources/tencent.js');
  const r = await mod.fetchKlines({ market, code, period, limit });
  return r.klines || r;
}

// Scoring matrix
function mapSignal(s) {
  if (s === 'strong_bull') return 2;
  if (s === 'bull') return 1;
  if (s === 'neutral') return 0;
  if (s === 'bear') return -1;
  if (s === 'strong_bear') return -2;
  return null;
}

export function scorePrediction(predictedSignal, groundTruth) {
  if (!predictedSignal || !groundTruth) return 0;
  const p = mapSignal(predictedSignal);
  const g = mapSignal(groundTruth);
  if (p === null || g === null) return 0;
  if (p === g) return 1.0;
  if (p * g < 0) {
    return (Math.abs(p) === 2 && Math.abs(g) === 2) ? -1.0 : -0.5;
  }
  if (p === 0 || g === 0) return 0.3;
  return 0.5;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/**
 * @param {string} datasetPath
 * @param {object} options
 */
export async function runEvaluation(datasetPath, options = {}) {
  const {
    callDeepSeek,             // (prompt, apiKey) => { text, usage }
    fetchKlinesDispatcher = fetchKlinesSafe,  // Default to built-in safe fallback
    provider = 'deepseek',
    model = 'deepseek-chat',
    promptVersion = 'current',
    apiKey,
    templates = ['technical', 'trend', 'valuation', 'sentiment'],
    sectorAlphaFn = null,    // (code, cutoffDate) => sectorAlphaData | null
    // Concurrency control
    klineConcurrency = 2,
    klineDelayMs = 500,
    llmConcurrency = 5,
  } = options;

  if (!callDeepSeek) throw new Error('runEvaluation requires callDeepSeek');

  const dataset = JSON.parse(fs.readFileSync(datasetPath, 'utf-8'));
  const { testPoints, stocks } = dataset;
  const stockMap = new Map(stocks.map(s => [s.code, s]));
  const cache = new KlinesCache();

  // ---- Phase 1: Dedup + batch fetch klines ----
  console.log(`[eval] Phase 1: Fetching klines...`);

  // Collect unique (market, code, period, limit) combos
  const uniqueFetches = new Map(); // key → { market, code, period, limit }
  for (const tp of testPoints) {
    const stock = stockMap.get(tp.stockCode);
    if (!stock) continue;
    const key = cache.key(tp.stockCode, stock.market, 'monthly', 200);
    if (!uniqueFetches.has(key)) {
      uniqueFetches.set(key, { market: stock.market, code: tp.stockCode, period: 'monthly', limit: 200 });
    }
  }
  // CSI300
  uniqueFetches.set('1.000300:monthly:200', { market: '1', code: '000300', period: 'monthly', limit: 200 });

  console.log(`[eval] Dedup: ${testPoints.length} testPoints → ${uniqueFetches.size - 1} K-line fetches (+1 HS300)`);

  const fetchTasks = [...uniqueFetches.values()];
  let fetchCompleted = 0;
  const fetchStart = Date.now();

  // Concurrency 2, fetch batch by batch
  const batches = [];
  for (let i = 0; i < fetchTasks.length; i += klineConcurrency) {
    batches.push(fetchTasks.slice(i, i + klineConcurrency));
  }

  for (const batch of batches) {
    await Promise.allSettled(batch.map(async (task) => {
      try {
        const result = await fetchWithRetry(async () => {
          return fetchKlinesDispatcher(task.market, task.code, task.period, task.limit);
        }, 3);
        const klines = result.klines || result;
        if (Array.isArray(klines)) {
          cache.set(task.code, task.market, task.period, task.limit, klines);
        }
      } catch (err) {
        console.warn(`[eval] K-line fetch failed ${task.code}: ${err.message}`);
      }
      fetchCompleted++;
      if (fetchCompleted % 10 === 0) {
        console.log(`[eval] K-line progress: ${fetchCompleted}/${fetchTasks.length}`);
      }
    }));
    // Random delay 300-800ms between batches
    if (batches.indexOf(batch) < batches.length - 1) {
      await sleep(klineDelayMs + Math.random() * 300);
    }
  }

  const fetchElapsed = ((Date.now() - fetchStart) / 1000).toFixed(1);
  console.log(`[eval] K-lines done: ${cache.size()} cached, hit rate ${cache.hitRate()}, ${fetchElapsed}s`);

  // ---- Phase 2: Run LLM per testPoint×template ----
  console.log(`[eval] Phase 2: LLM analysis (concurrency ${llmConcurrency})...`);

  const total = testPoints.length * templates.length;
  let completed = 0;
  let succeeded = 0;
  let failed = 0;
  let totalCost = 0;
  const results = [];
  const startTime = Date.now();

  // Batch concurrent LLM execution
  const allJobs = [];
  for (const tp of testPoints) {
    for (const tpl of templates) {
      allJobs.push({ tp, tpl });
    }
  }

  const llmBatches = [];
  for (let i = 0; i < allJobs.length; i += llmConcurrency) {
    llmBatches.push(allJobs.slice(i, i + llmConcurrency));
  }

  for (const batch of llmBatches) {
    const batchPromises = batch.map(async ({ tp, tpl }) => {
      const stock = stockMap.get(tp.stockCode);
      if (!stock) { completed++; failed++; return; }

      const allKlines = cache.get(tp.stockCode, stock.market, 'monthly', 200);
      if (!allKlines || allKlines.length <= tp.cutoffIndex) { completed++; failed++; return; }
      const cutoffKlines = allKlines.slice(0, tp.cutoffIndex + 1);
      if (cutoffKlines.length < 12) { completed++; failed++; return; }

      try {
        // Compute MA/MACD indicators (needed by buildPromptByTemplate table)
        const closes = cutoffKlines.map(k => k.close);
        const ma5 = computeMA(closes, 5);
        const ma20 = computeMA(closes, 20);
        const ma60 = computeMA(closes, 60);
        const { dif, dea, hist } = computeMACD(closes);
        const klinesWithIndicators = cutoffKlines.map((k, i) => ({
          ...k, ma5: ma5[i], ma20: ma20[i], ma60: ma60[i], dif: dif[i], dea: dea[i], hist: hist[i],
        }));

        // sectorAlphaData (optional)
        let sectorAlphaData = null;
        if (sectorAlphaFn) {
          try { sectorAlphaData = sectorAlphaFn(tp.stockCode, tp.cutoffDate); } catch (_) {}
        }

        // Use same buildPromptByTemplate as production
        const prompt = await buildPromptByTemplate({
          templateKey: tpl,
          name: stock.name || tp.stockCode,
          code: tp.stockCode,
          klines: klinesWithIndicators,
          period: 'monthly',
          provider: 'deepseek',
          decisionMode: false,
          sectorAlphaData,
        });

        const resp = await callDeepSeek(prompt, apiKey);
        const cost = resp.usage
          ? (resp.usage.inputTokens || 8000) / 1e6 * 1 + (resp.usage.outputTokens || 2000) / 1e6 * 4
          : 0.02;
        totalCost += cost;

        // Parse structured JSON (HARD_CONSTRAINTS 7 requires LLM output)
        let scoreData = null;
        try {
          const m = resp.text.match(/```json\s*([\s\S]*?)```/);
          if (m) scoreData = JSON.parse(m[1].trim());
        } catch (_) { /* JSON parse failed, fallback to neutral */ }
        const predictedSignal = scoreData?.signal || 'neutral';
        const score = scorePrediction(predictedSignal, tp.groundTruth);
        succeeded++;

        // Truncation detection
        const outputTokens = resp.usage?.outputTokens || 0;
        const truncated = outputTokens >= EVAL_MAX_TOKENS;

        results.push({
          testPointId: tp.id, stockCode: tp.stockCode, cutoffDate: tp.cutoffDate,
          category: tp.category, template: tpl, model, promptVersion,
          predictedSignal, groundTruth: tp.groundTruth, score, cost, alpha: tp.alpha,
          rawResponse: resp.text,
          promptUsed: prompt,
          tokensUsed: resp.usage?.inputTokens
            ? { input: resp.usage.inputTokens, output: outputTokens }
            : null,
          truncated,
          parserVersion: PARSER_VERSION,
          evalRunnerVersion: EVAL_RUNNER_VERSION,
          timestamp: new Date().toISOString(),
        });
      } catch (err) {
        failed++;
        results.push({ testPointId: tp.id, template: tpl, error: err.message, score: null, parserVersion: PARSER_VERSION, evalRunnerVersion: EVAL_RUNNER_VERSION, timestamp: new Date().toISOString() });
      }
      completed++;
      if (completed % 20 === 0) {
        const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
        const eta = Math.max(0, (elapsed / completed * total - elapsed)).toFixed(0);
        console.log(`[eval] LLM: ${completed}/${total} ok=${succeeded} ¥${totalCost.toFixed(2)} ${elapsed}min ETA${eta}min`);
      }
    });
    await Promise.allSettled(batchPromises);
    // No sleep needed between LLM batches (deepseek has no rate limit)
  }

  const elapsedMin = ((Date.now() - startTime) / 60000).toFixed(1);
  const truncatedCount = results.filter(r => r.truncated).length;
  const truncPct = (truncatedCount / results.length * 100).toFixed(1);
  console.log(`\n[eval] Done: ${succeeded}/${total} ok, ${failed} fail, ${elapsedMin}min, ¥${totalCost.toFixed(4)}`);
  console.log(`[eval] Truncated: ${truncatedCount}/${results.length} (${truncPct}%)`);
  if (truncPct > 5) console.warn(`[eval] ⚠ Truncation rate ${truncPct}% exceeds threshold ${TRUNCATION_WARN_RATIO * 100}%, results may be incomplete`);
  console.log(`[eval] Cache hit rate: ${cache.hitRate()}`);

  // ---- Write results ----
  const runDir = path.join(EVAL_DIR, 'runs');
  fs.mkdirSync(runDir, { recursive: true });
  const runId = `${promptVersion}-${new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-')}`;
  const outPath = path.join(runDir, `${runId}.jsonl`);
  fs.writeFileSync(outPath, results.map(r => JSON.stringify(r)).join('\n') + '\n', 'utf-8');

  return {
    runId, path: outPath,
    totalCalls: total, succeeded, failed,
    totalCostCny: +totalCost.toFixed(4), elapsedMin,
    cacheHitRate: cache.hitRate(),
    results,
  };
}
