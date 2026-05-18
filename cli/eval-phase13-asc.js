// Phase 13: Adaptive Signal Calibration (ASC)
// 单次运行 with #13 confidence约束 → 后处理校准曲线
// 对比 frozen baseline score=0.1966
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadFrozenDataset } from '../lib/eval/load-frozen-dataset.js';
import { buildPromptByTemplate } from '../lib/build-prompt.js';
import { computeMA } from '../lib/compute-ma.js';
import { computeMACD } from '../lib/compute-macd.js';
import { scorePrediction } from '../lib/eval/compute-score.js';
import { calibrateByConfidence, formatCalibrationCurve, getHighConfScore } from '../lib/uncertainty/asc.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const RUNS_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs');

const EVAL_RUNNER_VERSION = '1.0.0';
const PARSER_VERSION = '1.0.0';
const PROMPT_VERSION = 'v13-asc';
const EVAL_MAX_TOKENS = 4000;
const LLM_CONCURRENCY = 2;
const LLM_DELAY_MS = 500;
const MAX_RETRIES = 3;
const RETRY_DELAYS_MS = [1000, 4000, 16000];

const FROZEN_BASELINE_SCORE = 0.1966;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function callDeepSeek(prompt, apiKey) {
  const resp = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model: 'deepseek-chat', messages: [{ role: 'user', content: prompt }], max_tokens: EVAL_MAX_TOKENS, temperature: 0.0 }),
  });
  if (!resp.ok) throw new Error(`DeepSeek HTTP ${resp.status}`);
  const data = await resp.json();
  return { text: data.choices?.[0]?.message?.content || '', usage: { inputTokens: data.usage?.prompt_tokens || 0, outputTokens: data.usage?.completion_tokens || 0 } };
}

async function callWithRetry(prompt, apiKey) {
  let lastErr;
  for (let a = 0; a <= MAX_RETRIES; a++) {
    try { const r = await callDeepSeek(prompt, apiKey); return { result: r, retries: a }; }
    catch (e) { lastErr = e; if (a < MAX_RETRIES) await sleep(RETRY_DELAYS_MS[a] || 16000); }
  }
  throw lastErr;
}

function loadCompleted(fp) {
  if (!fs.existsSync(fp)) return new Set();
  const c = fs.readFileSync(fp, 'utf-8').trim();
  if (!c) return new Set();
  const s = new Set();
  for (const l of c.split('\n').filter(Boolean)) {
    try { const r = JSON.parse(l); if (!r.error) s.add(`${r.stockCode}_${r.template}`); } catch (_) {}
  }
  return s;
}

function appendResult(fp, r) {
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  fs.appendFileSync(fp, JSON.stringify(r) + '\n', 'utf-8');
}

async function main() {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('DEEPSEEK_API_KEY');

  const dryRun = process.argv.includes('--dry-run');
  const dataset = loadFrozenDataset({ version: 'v1', subsetStocks: dryRun ? 3 : null });
  const testPoints = dataset.testPoints;
  const total = testPoints.length * 4;

  console.log('=== Phase 13 ASC ===');
  console.log(`模式: ${dryRun ? 'DRY-RUN' : 'FULL frozen-v1'}, stocks=${dataset.stocks.length}, testPoints=${testPoints.length}, calls=${total}`);
  console.log(`Frozen baseline score: ${FROZEN_BASELINE_SCORE}`);
  console.log(`EVAL_RUNNER_VERSION=${EVAL_RUNNER_VERSION} PROMPT_VERSION=${PROMPT_VERSION}`);
  console.log('新增: HARD_CONSTRAINTS #13 confidence诚实法则 + ASC后处理\n');

  const { getDb } = await import('../lib/db/connection.js');
  const db = getDb();

  const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  const outPath = path.join(RUNS_DIR, `phase13-asc-${timestamp}.jsonl`);
  console.log(`输出: ${path.basename(outPath)}`);

  const completed = loadCompleted(outPath);
  const stockMap = new Map(dataset.stocks.map(s => [s.code, s]));
  const pending = [];
  for (const tp of testPoints) {
    for (const tpl of ['technical', 'trend', 'valuation', 'sentiment']) {
      if (!completed.has(`${tp.stockCode}_${tpl}`)) pending.push({ tp, tpl });
    }
  }
  console.log(`已完成: ${total - pending.length}, 待跑: ${pending.length}\n`);

  if (pending.length === 0) {
    console.log('全部已完成!');
    const records = fs.readFileSync(outPath, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
    printASCReport(records);
    return;
  }

  // 预加载K线
  const klinesCache = new Map();
  for (const code of [...new Set(pending.map(j => j.tp.stockCode))]) {
    const rows = db.prepare('SELECT * FROM monthly_klines WHERE code=? ORDER BY date').all(code);
    if (rows.length >= 24) klinesCache.set(code, rows);
  }

  const startTime = Date.now();
  let completed_count = total - pending.length;
  let succeeded = completed_count;
  let failed = 0;
  let totalCost = 0;
  const retryStats = { firstTry: completed_count, retrySuccess: 0, finalFail: 0 };

  let jobIdx = 0;
  while (jobIdx < pending.length) {
    const batch = pending.slice(jobIdx, jobIdx + LLM_CONCURRENCY);
    await Promise.allSettled(batch.map(async ({ tp, tpl }) => {
      const stock = stockMap.get(tp.stockCode);
      const klines = klinesCache.get(tp.stockCode);
      if (!stock || !klines || tp.cutoffIndex >= klines.length || tp.cutoffIndex < 12) {
        appendResult(outPath, { testPointId: tp.id, stockCode: tp.stockCode, template: tpl, error: '数据不足', score: null });
        failed++; completed_count++; return;
      }

      const cutoffKlines = klines.slice(0, tp.cutoffIndex + 1);
      const closes = cutoffKlines.map(k => k.close);
      const ma5 = computeMA(closes, 5), ma20 = computeMA(closes, 20), ma60 = computeMA(closes, 60);
      const { dif, dea, hist } = computeMACD(closes);
      const kwi = cutoffKlines.map((k, i) => ({
        date: k.date, open: k.open, close: k.close, high: k.high, low: k.low,
        volume: k.volume, changePercent: k.change_percent, turnoverRate: k.turnover_rate,
        ma5: ma5[i], ma20: ma20[i], ma60: ma60[i], dif: dif[i], dea: dea[i], hist: hist[i],
      }));

      const prompt = buildPromptByTemplate({
        templateKey: tpl, name: stock.name || tp.stockCode, code: tp.stockCode,
        klines: kwi, period: 'monthly', provider: 'deepseek', decisionMode: false,
      });

      try {
        const { result, retries } = await callWithRetry(prompt, apiKey);
        if (retries > 0) retryStats.retrySuccess++; else retryStats.firstTry++;

        const cost = result.usage ? (result.usage.inputTokens||0)/1e6*1 + (result.usage.outputTokens||0)/1e6*4 : 0.02;
        totalCost += cost;

        let predictedSignal = 'parse_failed';
        try {
          const m = result.text.match(/```json\s*([\s\S]*?)```/);
          if (m) { const d = JSON.parse(m[1].trim()); predictedSignal = d.signal || 'parse_failed'; }
        } catch (_) {}

        const score = scorePrediction(predictedSignal, tp.groundTruth);

        appendResult(outPath, {
          testPointId: tp.id, stockCode: tp.stockCode, stockName: tp.stockName,
          cutoffDate: tp.cutoffDate, cutoffIndex: tp.cutoffIndex,
          template: tpl, model: 'deepseek-chat', promptVersion: PROMPT_VERSION,
          runMode: 'ASC',
          predictedSignal, groundTruth: tp.groundTruth, score, cost, alpha: tp.alpha,
          rawResponse: result.text, promptUsed: prompt,
          tokensUsed: result.usage?.inputTokens ? { input: result.usage.inputTokens, output: result.usage.outputTokens } : null,
          truncated: (result.usage?.outputTokens||0) >= EVAL_MAX_TOKENS, retries,
          parserVersion: PARSER_VERSION, evalRunnerVersion: EVAL_RUNNER_VERSION,
          timestamp: new Date().toISOString(),
        });
        succeeded++; completed_count++;
      } catch (err) {
        retryStats.finalFail++;
        appendResult(outPath, { testPointId: tp.id, stockCode: tp.stockCode, template: tpl, error: err.message, score: null });
        failed++; completed_count++;
      }
    }));

    if (completed_count % 20 === 0 || completed_count === total) {
      const elapsed = ((Date.now()-startTime)/60000).toFixed(1);
      const pct = (completed_count/total*100).toFixed(1);
      const eta = Math.max(0, (elapsed/completed_count*total-elapsed)).toFixed(0);
      console.log(`[ASC] ${completed_count}/${total} (${pct}%) ok=${succeeded} pf=${failed} ¥${totalCost.toFixed(2)} ${elapsed}min ETA${eta}min`);
    }
    jobIdx += LLM_CONCURRENCY;
    if (jobIdx < pending.length) await sleep(LLM_DELAY_MS);
  }

  const elapsedMin = ((Date.now()-startTime)/60000).toFixed(1);
  console.log(`\n[ASC] 完成: ${succeeded}/${total} ok, ${failed} fail, ${elapsedMin}min, ¥${totalCost.toFixed(4)}`);
  console.log(`[ASC] 重试: 一次成功=${retryStats.firstTry} 重试成功=${retryStats.retrySuccess} 最终失败=${retryStats.finalFail}\n`);

  // ASC 后处理
  const records = fs.readFileSync(outPath, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
  printASCReport(records);
}

function printASCReport(records) {
  const cal = calibrateByConfidence(records);
  const overallScore = cal.overall.weightedScore;
  const highScore = cal.high.weightedScore;

  console.log(formatCalibrationCurve(cal));
  console.log('');
  console.log(`Frozen baseline score: ${FROZEN_BASELINE_SCORE}`);
  console.log(`Phase 13 overall score: ${overallScore}`);
  console.log(`Phase 13 high-conf score: ${highScore}`);

  if (highScore != null) {
    const delta = highScore - FROZEN_BASELINE_SCORE;
    console.log(`Δ (high-conf - baseline): ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`);
    if (delta > 0.02) console.log('结论: high-conf 子集显著优于 baseline!');
    else if (delta > -0.01) console.log('结论: 边缘改善，confidence 过滤略有效');
    else console.log('结论: 无明显改善');
  }
}

main().catch(err => { console.error(err); process.exit(1); });
