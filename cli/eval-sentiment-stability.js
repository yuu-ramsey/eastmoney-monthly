// 反事实情绪稳定性评估
// 用法：
//   node cli/eval-sentiment-stability.js --dry-run     (3 stocks, 只打印 prompt)
//   node cli/eval-sentiment-stability.js --subset 5    (5 stocks, 完整 LLM 调用)

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildPromptByTemplate } from '../lib/build-prompt.js';
import { computeMA } from '../lib/compute-ma.js';
import { computeMACD } from '../lib/compute-macd.js';
import { scorePrediction } from '../lib/eval/compute-score.js';
import { loadFrozenDataset } from '../lib/eval/load-frozen-dataset.js';
import { buildCounterfactualEvents, computeStabilityScore, stabilityLevel } from '../lib/eval/counterfactual-sentiment.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const EVAL_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval');
const RUNS_DIR = path.join(EVAL_DIR, 'runs');
const DATA_DIR = path.join(PROJECT_DIR, 'data');

const EVAL_MAX_TOKENS = 4000;
const LLM_CONCURRENCY = 1;  // 串行调用（每个 stock 的 3 个情景必须顺序跑才能对比）
const LLM_DELAY_MS = 300;
const MAX_RETRIES = 3;
const RETRY_DELAYS_MS = [1000, 4000, 16000];

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// 加载 .env
function loadEnv() {
  const envPath = path.join(PROJECT_DIR, '.env');
  if (!fs.existsSync(envPath)) return {};
  const env = {};
  for (const line of fs.readFileSync(envPath, 'utf-8').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq > 0) env[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
  }
  return env;
}

async function callDeepSeek(prompt, apiKey, model = 'deepseek-chat') {
  const resp = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: EVAL_MAX_TOKENS, temperature: 0.0 }),
  });
  if (!resp.ok) { const e = await resp.text().catch(() => ''); throw new Error(`HTTP ${resp.status}: ${e.slice(0, 200)}`); }
  const d = await resp.json();
  return {
    text: d.choices?.[0]?.message?.content || '',
    usage: { inputTokens: d.usage?.prompt_tokens || 0, outputTokens: d.usage?.completion_tokens || 0 },
  };
}

async function callWithRetry(prompt, apiKey, model) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try { const r = await callDeepSeek(prompt, apiKey, model); return { result: r, retries: attempt }; }
    catch (err) { lastErr = err; if (attempt < MAX_RETRIES) await sleep(RETRY_DELAYS_MS[attempt] || 16000); }
  }
  throw lastErr;
}

function parseSignal(rawResponse) {
  try { const m = rawResponse.match(/```json\s*([\s\S]*?)```/); if (m) return JSON.parse(m[1].trim()).signal || 'parse_failed'; } catch (_) {}
  return 'parse_failed';
}

// 从 v1 数据集的真实 eval run 加载初始 events（模拟 content.js 抓取的"大事提醒"）
function loadMockEvents() {
  const mockPath = path.join(DATA_DIR, 'events-mock.json');
  if (fs.existsSync(mockPath)) return JSON.parse(fs.readFileSync(mockPath, 'utf-8'));
  // 使用默认空事件 + LLM 自行判断
  return [];
}

async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const subsetIdx = args.indexOf('--subset');
  const subsetN = subsetIdx >= 0 ? parseInt(args[subsetIdx + 1], 10) : (dryRun ? 3 : 40);

  const env = loadEnv();
  const apiKey = env.DEEPSEEK_API_KEY;
  if (!apiKey && !dryRun) throw new Error('DEEPSEEK_API_KEY 未设置');

  console.log('=== 反事实情绪稳定性评估 ===');
  console.log(`模式: ${dryRun ? 'DRY-RUN' : `FULL (${subsetN} stocks)`}`);

  const dataset = loadFrozenDataset({ version: 'v1', subsetStocks: subsetN });
  console.log(`数据集: ${dataset.stocks.length} stocks, ${dataset.testPoints.length} testPoints`);

  const { getDb } = await import('../lib/db/connection.js');
  const db = getDb();
  const klinesCache = new Map();
  for (const tp of dataset.testPoints) {
    if (klinesCache.has(tp.stockCode)) continue;
    const rows = db.prepare('SELECT * FROM monthly_klines WHERE code=? ORDER BY date').all(tp.stockCode);
    if (rows.length >= 24) klinesCache.set(tp.stockCode, rows);
  }

  const mockEvents = loadMockEvents();
  const scenarios = ['baseline', 'bullish', 'bearish'];
  const templates = ['sentiment'];  // sentiment 模板对事件最敏感

  if (dryRun) {
    // Dry-run：只打印第一个 test point 的 3 个 prompt
    const tp = dataset.testPoints[0];
    const stock = dataset.stocks.find(s => s.code === tp.stockCode);
    const klines = klinesCache.get(tp.stockCode);
    if (!klines) { console.log('K线不足'); return; }

    const cutoffKlines = klines.slice(0, tp.cutoffIndex + 1);
    const closes = cutoffKlines.map(k => k.close);
    const ma5 = computeMA(closes, 5), ma20 = computeMA(closes, 20), ma60 = computeMA(closes, 60);
    const { dif, dea, hist } = computeMACD(closes);
    const kwi = cutoffKlines.map((k, i) => ({
      date: k.date, open: k.open, close: k.close, high: k.high, low: k.low,
      volume: k.volume, changePercent: k.change_percent, turnoverRate: k.turnover_rate,
      ma5: ma5[i], ma20: ma20[i], ma60: ma60[i], dif: dif[i], dea: dea[i], hist: hist[i],
    }));

    for (const scenario of scenarios) {
      const events = buildCounterfactualEvents(mockEvents, scenario);
      const prompt = await buildPromptByTemplate({
        templateKey: 'sentiment', name: stock.name || tp.stockCode, code: tp.stockCode,
        klines: kwi, period: 'monthly', provider: 'deepseek', decisionMode: false,
        extraContext: { events },
      });
      console.log(`\n=== ${scenario.toUpperCase()} SCENARIO ===`);
      console.log(`Events: ${events.length} 条`);
      events.slice(0, 5).forEach(e => console.log(`  ${e.date} [${e.type}] ${e.title}`));
      console.log(`Prompt (前500字): ${prompt.slice(0, 500)}...\n`);
    }
    console.log('Dry-run 完成。检查 prompt 注入是否正确。');
    return;
  }

  // 完整模式
  const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  const outPath = path.join(RUNS_DIR, `sentiment-stability-${timestamp}.jsonl`);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });

  const startTime = Date.now();
  let completed = 0, ok = 0, fail = 0, totalCost = 0;
  const total = dataset.testPoints.length * scenarios.length * templates.length;
  console.log(`总任务: ${total} (${dataset.testPoints.length} testPoints × ${scenarios.length} 情景 × ${templates.length} 模板)\n`);

  for (const tp of dataset.testPoints) {
    const stock = dataset.stocks.find(s => s.code === tp.stockCode);
    const klines = klinesCache.get(tp.stockCode);
    if (!stock || !klines || tp.cutoffIndex >= klines.length || tp.cutoffIndex < 12) {
      completed += 3; fail += 3; continue;
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

    const scenarioSignals = {};

    for (const scenario of scenarios) {
      const template = 'sentiment';
      const events = buildCounterfactualEvents(mockEvents, scenario);

      try {
        const prompt = await buildPromptByTemplate({
          templateKey: template, name: stock.name || tp.stockCode, code: tp.stockCode,
          klines: kwi, period: 'monthly', provider: 'deepseek', decisionMode: false,
          extraContext: { events: events.length > 0 ? events : undefined },
        });

        const { result, retries } = await callWithRetry(prompt, apiKey, 'deepseek-chat');
        const signal = parseSignal(result.text);
        const score = scorePrediction(signal, tp.groundTruth);
        const cost = result.usage
          ? result.usage.inputTokens / 1e6 * 1 + result.usage.outputTokens / 1e6 * 4 : 0.02;
        totalCost += cost;
        ok++;
        scenarioSignals[scenario] = signal;

        fs.appendFileSync(outPath, JSON.stringify({
          stockCode: tp.stockCode, cutoffDate: tp.cutoffDate, scenario, template,
          predictedSignal: signal, groundTruth: tp.groundTruth, score, cost, retries,
          timestamp: new Date().toISOString(),
        }) + '\n');
      } catch (err) {
        fail++;
        fs.appendFileSync(outPath, JSON.stringify({
          stockCode: tp.stockCode, cutoffDate: tp.cutoffDate, scenario,
          error: String(err).slice(0, 200),
        }) + '\n');
      }
      completed++;

      if (completed % 6 === 0 || completed === total) {
        const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
        console.log(`进度: ${completed}/${total} | 成功:${ok} 失败:${fail} | 耗时:${elapsed}min | 费用:¥${totalCost.toFixed(2)}`);
      }

      await sleep(LLM_DELAY_MS);
    }

    // 计算该 test point 的稳定性得分
    if (Object.keys(scenarioSignals).length === 3) {
      const stability = computeStabilityScore([
        scenarioSignals.baseline,
        scenarioSignals.bullish,
        scenarioSignals.bearish,
      ]);
      fs.appendFileSync(outPath, JSON.stringify({
        stockCode: tp.stockCode, cutoffDate: tp.cutoffDate,
        type: 'stability_summary',
        baselineSignal: scenarioSignals.baseline,
        bullishSignal: scenarioSignals.bullish,
        bearishSignal: scenarioSignals.bearish,
        stabilityScore: stability,
        stabilityLevel: stabilityLevel(stability),
      }) + '\n');
    }
  }

  const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
  console.log(`\n=== 完成 ===`);
  console.log(`耗时: ${elapsed}min | 费用: ¥${totalCost.toFixed(2)}`);

  // 统计
  const allResults = [];
  for (const line of fs.readFileSync(outPath, 'utf-8').trim().split('\n').filter(Boolean)) {
    try { allResults.push(JSON.parse(line)); } catch (_) {}
  }
  const summaries = allResults.filter(r => r.type === 'stability_summary');
  const levels = { robust: 0, sensitive: 0, fragile: 0 };
  for (const s of summaries) levels[s.stabilityLevel] = (levels[s.stabilityLevel] || 0) + 1;
  const avgStability = summaries.reduce((s, r) => s + r.stabilityScore, 0) / Math.max(1, summaries.length);

  console.log(`\n=== 稳定性分布 ===`);
  console.log(`robust (>0.75):    ${levels.robust} (${(100*levels.robust/summaries.length).toFixed(0)}%)`);
  console.log(`sensitive (0.5-0.75): ${levels.sensitive} (${(100*levels.sensitive/summaries.length).toFixed(0)}%)`);
  console.log(`fragile (<0.5):    ${levels.fragile} (${(100*levels.fragile/summaries.length).toFixed(0)}%)`);
  console.log(`平均稳定性得分: ${avgStability.toFixed(4)}`);
  console.log(`\n输出: ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
