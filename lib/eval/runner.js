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

// eval 场景固定 maxTokens=4000（不可覆盖，防截断 JSON 块导致 signal 丢失）
export const EVAL_MAX_TOKENS = 4000;
// 截断警告阈值（outputTokens == EVAL_MAX_TOKENS 时视为截断）
export const TRUNCATION_WARN_RATIO = 0.05;

// 内建安全降级函数（绕过dispatcher的已知sina/tencent bug）
export async function fetchKlinesSafe(market, code, period, limit) {
  // 先试东财
  try {
    const mod = await import('../data-sources/eastmoney.js');
    const r = await mod.fetchKlines({ market, code, period, limit });
    if (r && Array.isArray(r.klines) && r.klines.length >= 12) return r.klines || r;
  } catch (_) {}
  // 降级新浪
  try {
    const mod = await import('../data-sources/sina.js');
    const r = await mod.fetchKlines({ market, code, period, limit });
    if (r && Array.isArray(r.klines) && r.klines.length >= 12) return r.klines || r;
  } catch (_) {}
  // 降级腾讯
  const mod = await import('../data-sources/tencent.js');
  const r = await mod.fetchKlines({ market, code, period, limit });
  return r.klines || r;
}

// 评分矩阵
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
    fetchKlinesDispatcher = fetchKlinesSafe,  // 默认用内建安全降级
    provider = 'deepseek',
    model = 'deepseek-chat',
    promptVersion = 'current',
    apiKey,
    templates = ['technical', 'trend', 'valuation', 'sentiment'],
    sectorAlphaFn = null,    // (code, cutoffDate) => sectorAlphaData | null
    // 并发控制
    klineConcurrency = 2,
    klineDelayMs = 500,
    llmConcurrency = 5,
  } = options;

  if (!callDeepSeek) throw new Error('runEvaluation 需要 callDeepSeek');

  const dataset = JSON.parse(fs.readFileSync(datasetPath, 'utf-8'));
  const { testPoints, stocks } = dataset;
  const stockMap = new Map(stocks.map(s => [s.code, s]));
  const cache = new KlinesCache();

  // ---- 阶段1：去重 + 批量拉K线 ----
  console.log(`[eval] 阶段1: 拉K线...`);

  // 收集唯一的 (market, code, period, limit) 组合
  const uniqueFetches = new Map(); // key → { market, code, period, limit }
  for (const tp of testPoints) {
    const stock = stockMap.get(tp.stockCode);
    if (!stock) continue;
    const key = cache.key(tp.stockCode, stock.market, 'monthly', 200);
    if (!uniqueFetches.has(key)) {
      uniqueFetches.set(key, { market: stock.market, code: tp.stockCode, period: 'monthly', limit: 200 });
    }
  }
  // 沪深300
  uniqueFetches.set('1.000300:monthly:200', { market: '1', code: '000300', period: 'monthly', limit: 200 });

  console.log(`[eval] 去重: ${testPoints.length} testPoints → ${uniqueFetches.size - 1} 次K线fetch (+1 HS300)`);

  const fetchTasks = [...uniqueFetches.values()];
  let fetchCompleted = 0;
  const fetchStart = Date.now();

  // 并发2，逐批拉取
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
        console.warn(`[eval] K线拉取失败 ${task.code}: ${err.message}`);
      }
      fetchCompleted++;
      if (fetchCompleted % 10 === 0) {
        console.log(`[eval] K线进度: ${fetchCompleted}/${fetchTasks.length}`);
      }
    }));
    // 批次间随机延迟 300-800ms
    if (batches.indexOf(batch) < batches.length - 1) {
      await sleep(klineDelayMs + Math.random() * 300);
    }
  }

  const fetchElapsed = ((Date.now() - fetchStart) / 1000).toFixed(1);
  console.log(`[eval] K线完成: ${cache.size()} 条缓存, 命中率 ${cache.hitRate()}, ${fetchElapsed}s`);

  // ---- 阶段2：逐testPoint×template跑LLM ----
  console.log(`[eval] 阶段2: LLM分析 (并发${llmConcurrency})...`);

  const total = testPoints.length * templates.length;
  let completed = 0;
  let succeeded = 0;
  let failed = 0;
  let totalCost = 0;
  const results = [];
  const startTime = Date.now();

  // 分批并发运行LLM
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
        // 计算 MA/MACD 指标（buildPromptByTemplate 表格需要）
        const closes = cutoffKlines.map(k => k.close);
        const ma5 = computeMA(closes, 5);
        const ma20 = computeMA(closes, 20);
        const ma60 = computeMA(closes, 60);
        const { dif, dea, hist } = computeMACD(closes);
        const klinesWithIndicators = cutoffKlines.map((k, i) => ({
          ...k, ma5: ma5[i], ma20: ma20[i], ma60: ma60[i], dif: dif[i], dea: dea[i], hist: hist[i],
        }));

        // sectorAlphaData（可选）
        let sectorAlphaData = null;
        if (sectorAlphaFn) {
          try { sectorAlphaData = sectorAlphaFn(tp.stockCode, tp.cutoffDate); } catch (_) {}
        }

        // 用与生产环境相同的 buildPromptByTemplate
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

        // 解析结构化 JSON（HARD_CONSTRAINTS 7 要求 LLM 输出）
        let scoreData = null;
        try {
          const m = resp.text.match(/```json\s*([\s\S]*?)```/);
          if (m) scoreData = JSON.parse(m[1].trim());
        } catch (_) { /* JSON 解析失败降级到 neutral */ }
        const predictedSignal = scoreData?.signal || 'neutral';
        const score = scorePrediction(predictedSignal, tp.groundTruth);
        succeeded++;

        // 截断检测
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
    // LLM批次间无需sleep（deepseek不限流）
  }

  const elapsedMin = ((Date.now() - startTime) / 60000).toFixed(1);
  const truncatedCount = results.filter(r => r.truncated).length;
  const truncPct = (truncatedCount / results.length * 100).toFixed(1);
  console.log(`\n[eval] 完成: ${succeeded}/${total} ok, ${failed} fail, ${elapsedMin}min, ¥${totalCost.toFixed(4)}`);
  console.log(`[eval] 截断: ${truncatedCount}/${results.length} (${truncPct}%)`);
  if (truncPct > 5) console.warn(`[eval] ⚠ 截断率 ${truncPct}% 超阈值 ${TRUNCATION_WARN_RATIO * 100}%，结果可能不完整`);
  console.log(`[eval] 缓存命中率: ${cache.hitRate()}`);

  // ---- 写结果 ----
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
