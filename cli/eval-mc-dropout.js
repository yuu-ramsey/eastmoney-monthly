// MC Dropout eval — compare With/Without MC Dropout data impact on LLM prediction quality
// Usage:
//   node cli/eval-mc-dropout.js --dry-run     (3 stocks)
//   node cli/eval-mc-dropout.js               (full 40 stocks)
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildPromptByTemplate } from '../lib/build-prompt.js';
import { computeMA } from '../lib/compute-ma.js';
import { computeMACD } from '../lib/compute-macd.js';
import { scorePrediction } from '../lib/eval/compute-score.js';
import { loadFrozenDataset } from '../lib/eval/load-frozen-dataset.js';
import { spawn } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const EVAL_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval');
const RUNS_DIR = path.join(EVAL_DIR, 'runs');
const PARQUET_PATH = path.join(PROJECT_DIR, '.eastmoney-ai', 'lstm', 'mc_dropout_signals.parquet');

const EVAL_MAX_TOKENS = 4000;
const LLM_CONCURRENCY = 2;
const LLM_DELAY_MS = 500;
const MAX_RETRIES = 3;
const RETRY_DELAYS_MS = [1000, 4000, 16000];

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ---- parse .env ----
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

// ---- LLM call ----
async function callDeepSeek(prompt, apiKey, model = 'deepseek-chat') {
  const resp = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model, messages: [{ role: 'user', content: prompt }], max_tokens: EVAL_MAX_TOKENS, temperature: 0.0 }),
  });
  if (!resp.ok) { const e = await resp.text().catch(() => ''); throw new Error(`HTTP ${resp.status}: ${e.slice(0, 200)}`); }
  const d = await resp.json();
  const text = d.choices?.[0]?.message?.content || '';
  const usage = d.usage || {};
  return { text, usage: { inputTokens: usage.prompt_tokens || 0, outputTokens: usage.completion_tokens || 0 } };
}

async function callWithRetry(prompt, apiKey, model) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try { const r = await callDeepSeek(prompt, apiKey, model); return { result: r, retries: attempt }; }
    catch (err) { lastErr = err; if (attempt < MAX_RETRIES) await sleep(RETRY_DELAYS_MS[attempt] || 16000); }
  }
  throw lastErr;
}

// ---- parse LLM JSON output ----
function parseSignal(rawResponse) {
  let scoreData = null;
  try { const m = rawResponse.match(/```json\s*([\s\S]*?)```/); if (m) scoreData = JSON.parse(m[1].trim()); } catch (_) {}
  return { predictedSignal: scoreData?.signal || 'parse_failed', scoreData };
}

// ---- checkpoint resume ----
function loadCompleted(fp) {
  if (!fs.existsSync(fp)) return new Set();
  const s = new Set();
  for (const line of fs.readFileSync(fp, 'utf-8').trim().split('\n').filter(Boolean)) {
    try { const r = JSON.parse(line); if (!r.error) s.add(`${r.stockCode}_${r.template}`); } catch (_) {}
  }
  return s;
}

function appendResult(fp, r) { fs.mkdirSync(path.dirname(fp), { recursive: true }); fs.appendFileSync(fp, JSON.stringify(r) + '\n'); }

// ---- Load MC Dropout data ----
function loadMcDropoutCache() {
  if (!fs.existsSync(PARQUET_PATH)) { console.warn('MC Dropout parquet not found, skipping'); return new Map(); }
  return new Promise((resolve) => {
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const py = spawn(pythonPath, ['-c', `
import pandas as pd, json
df = pd.read_parquet('${PARQUET_PATH}')
result = {}
for _, row in df.iterrows():
    code = str(row['code'])
    ulevel = row['uncertainty_level']
    result[code] = {
        'lstm_signal': float(row['signal']),
        'lstm_signal_raw': float(row['signal_raw']),
        'y3_mean': float(row['y3_mean']), 'y3_std': float(row['y3_std']),
        'y6_mean': float(row['y6_mean']), 'y6_std': float(row['y6_std']),
        'overall_confidence': float(row['overall_confidence']),
        'uncertainty_level': ulevel,
        'uncertainty_emoji': {'low': '\u{1F7E2}', 'medium': '\u{1F7E1}', 'high': '\u{1F534}'}.get(ulevel, '\u{1F7E1}'),
        'uncertainty_desc': {
            'low': 'Model predictions are highly consistent; signal has strong reliability',
            'medium': 'Model predictions show disagreement; signal needs technical verification',
            'high': 'Model predictions diverge significantly; signal unreliable, rely on technical analysis',
        }.get(ulevel, ''),
        'mc_samples': 50,
    }
print(json.dumps(result))
`], { cwd: PROJECT_DIR });
    let out = ''; py.stdout.on('data', d => out += d);
    py.on('close', () => { try { resolve(JSON.parse(out)); } catch (_) { resolve(new Map()); } });
  });
}

// ---- Main flow ----
async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const env = loadEnv();
  const apiKey = env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('DEEPSEEK_API_KEY not set in .env');

  console.log('=== MC Dropout A/B Evaluation ===');
  console.log(`Mode: ${dryRun ? 'DRY-RUN (3 stocks)' : 'FULL (40 stocks)'}`);

  // Load dataset
  const dataset = loadFrozenDataset({ version: 'v1', subsetStocks: dryRun ? 3 : null });
  const { testPoints, stocks } = dataset;
  const stockMap = new Map(stocks.map(s => [s.code, s]));
  console.log(`Dataset: ${stocks.length} stocks, ${testPoints.length} testPoints`);

  // Load MC Dropout cache
  const mcCache = await loadMcDropoutCache();
  console.log(`MC Dropout cache: ${Object.keys(mcCache).length} stocks`);

  // DB connection
  const { getDb } = await import('../lib/db/connection.js');
  const { calcSectorAlpha } = await import('../lib/sector/alpha.js');
  const db = getDb();

  // Preload K-lines
  const klinesCache = new Map();
  const uniqueCodes = [...new Set(testPoints.map(tp => tp.stockCode))];
  for (const code of uniqueCodes) {
    const rows = db.prepare('SELECT * FROM monthly_klines WHERE code=? ORDER BY date').all(code);
    if (rows.length >= 24) klinesCache.set(code, rows);
  }
  console.log(`K-line cache: ${klinesCache.size}/${uniqueCodes.length} stocks`);

  // Eval templates
  const templates = ['technical', 'trend', 'valuation', 'sentiment'];
  const total = testPoints.length * templates.length;
  console.log(`Total tasks: ${total} (${testPoints.length} stocks × ${templates.length} templates)`);

  const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  const outPath = path.join(RUNS_DIR, `mc-dropout-${timestamp}.jsonl`);
  const completed = loadCompleted(outPath);

  const pending = [];
  for (const tp of testPoints) {
    for (const tpl of templates) {
      if (!completed.has(`${tp.stockCode}_${tpl}`)) pending.push({ tp, tpl });
    }
  }
  console.log(`Pending: ${pending.length}/${total}\n`);

  if (pending.length === 0) { console.log('All done'); return; }

  const startTime = Date.now();
  let completed_count = total - pending.length, ok = completed_count, fail = 0;
  let totalCost = 0;
  const scoresWith = [];  // MC Dropout injected
  const scoresWithout = [];  // Control group (same stocks, different templates)

  let jobIdx = 0;
  while (jobIdx < pending.length) {
    const batch = pending.slice(jobIdx, jobIdx + LLM_CONCURRENCY);
    const batchProms = batch.map(async ({ tp, tpl }) => {
      const stock = stockMap.get(tp.stockCode);
      const klines = klinesCache.get(tp.stockCode);
      if (!stock || !klines || tp.cutoffIndex >= klines.length || tp.cutoffIndex < 12) {
        appendResult(outPath, { stockCode: tp.stockCode, template: tpl, error: 'Insufficient K-lines' });
        fail++; completed_count++; return;
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

      let sectorAlphaData = null;
      try { sectorAlphaData = calcSectorAlpha(db, tp.stockCode, 'monthly', 12, tp.cutoffDate); } catch (_) {}

      // MC Dropout data
      const lstmSignalData = mcCache[tp.stockCode] || null;

      const prompt = await buildPromptByTemplate({
        templateKey: tpl, name: stock.name || tp.stockCode, code: tp.stockCode,
        klines: kwi, period: 'monthly', provider: 'deepseek', decisionMode: false,
        sectorAlphaData, lstmSignalData,
      });

      try {
        const { result, retries } = await callWithRetry(prompt, apiKey, 'deepseek-chat');
        const { predictedSignal, scoreData } = parseSignal(result.text);
        const score = scorePrediction(predictedSignal, tp.groundTruth);
        const cost = result.usage
          ? result.usage.inputTokens / 1e6 * 1 + result.usage.outputTokens / 1e6 * 4 : 0.02;
        totalCost += cost;

        const record = {
          stockCode: tp.stockCode, stockName: tp.stockName,
          cutoffDate: tp.cutoffDate, template: tpl,
          predictedSignal, groundTruth: tp.groundTruth, score,
          actualReturn: tp.actualReturn, alpha: tp.alpha,
          hasMcDropout: !!lstmSignalData,
          mcUncertainty: lstmSignalData?.uncertainty_level || 'none',
          mcConfidence: lstmSignalData?.overall_confidence || null,
          mcSignal: lstmSignalData?.lstm_signal || null,
          cost, retries,
          outputTokens: result.usage?.outputTokens || 0,
          timestamp: new Date().toISOString(),
        };
        appendResult(outPath, record);
        ok++; completed_count++;
        return { record, ok: true, tp, tpl, lstmSignalData };
      } catch (err) {
        appendResult(outPath, { stockCode: tp.stockCode, template: tpl, error: String(err).slice(0, 200) });
        fail++; completed_count++;
        return null;
      }
    });

    const results = await Promise.all(batchProms);
    for (const r of results) {
      if (r && r.ok) {
        // Separate scores by with/without MC Dropout
        if (r.lstmSignalData) scoresWith.push(r.record);
        else scoresWithout.push(r.record);
      }
    }

    jobIdx += LLM_CONCURRENCY;
    if (completed_count % 4 === 0 || completed_count === total) {
      const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
      console.log(`Progress: ${completed_count}/${total} | ok:${ok} fail:${fail} | elapsed:${elapsed}min | cost:¥${totalCost.toFixed(2)}`);
    }
    if (jobIdx < pending.length) await sleep(LLM_DELAY_MS);
  }

  // ---- Report ----
  const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
  console.log(`\n=== Evaluation Complete ===`);
  console.log(`Elapsed: ${elapsed}min | Cost: ¥${totalCost.toFixed(2)}`);
  console.log(`Results: ${outPath}`);

  // Statistics
  const allResults = [];
  if (fs.existsSync(outPath)) {
    for (const line of fs.readFileSync(outPath, 'utf-8').trim().split('\n').filter(Boolean)) {
      try { allResults.push(JSON.parse(line)); } catch (_) {}
    }
  }

  const withMc = allResults.filter(r => r.hasMcDropout && !r.error);
  const withoutMc = allResults.filter(r => !r.hasMcDropout && !r.error);

  console.log(`\n=== A/B Comparison ===`);
  console.log(`With MC Dropout:    ${withMc.length} results, avg=${(withMc.reduce((s,r)=>s+r.score,0)/Math.max(1,withMc.length)).toFixed(3)}`);
  console.log(`Without MC Dropout: ${withoutMc.length} results, avg=${(withoutMc.reduce((s,r)=>s+r.score,0)/Math.max(1,withoutMc.length)).toFixed(3)}`);

  // Group by uncertainty level
  for (const level of ['low', 'medium', 'high']) {
    const subset = withMc.filter(r => r.mcUncertainty === level);
    if (subset.length > 0) {
      const avg = subset.reduce((s, r) => s + r.score, 0) / subset.length;
      console.log(`  MC ${level} uncertainty: ${subset.length} results, avg=${avg.toFixed(3)}`);
    }
  }

  console.log(`\nOutput file: ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
