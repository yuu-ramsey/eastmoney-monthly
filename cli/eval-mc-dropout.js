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

// ---- 断点续传 ----
function loadCompleted(fp) {
  if (!fs.existsSync(fp)) return new Set();
  const s = new Set();
  for (const line of fs.readFileSync(fp, 'utf-8').trim().split('\n').filter(Boolean)) {
    try { const r = JSON.parse(line); if (!r.error) s.add(`${r.stockCode}_${r.template}`); } catch (_) {}
  }
  return s;
}

function appendResult(fp, r) { fs.mkdirSync(path.dirname(fp), { recursive: true }); fs.appendFileSync(fp, JSON.stringify(r) + '\n'); }

// ---- 加载 MC Dropout 数据 ----
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
        'uncertainty_emoji': {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(ulevel, '🟡'),
        'uncertainty_desc': {
            'low': '模型预测一致性强，信号可信度较高',
            'medium': '模型预测存在分歧，信号需结合技术面验证',
            'high': '模型预测分歧大，信号不可靠，以技术分析为主',
        }.get(ulevel, ''),
        'mc_samples': 50,
    }
print(json.dumps(result))
`], { cwd: PROJECT_DIR });
    let out = ''; py.stdout.on('data', d => out += d);
    py.on('close', () => { try { resolve(JSON.parse(out)); } catch (_) { resolve(new Map()); } });
  });
}

// ---- 主流程 ----
async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const env = loadEnv();
  const apiKey = env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('DEEPSEEK_API_KEY 未在 .env 中设置');

  console.log('=== MC Dropout A/B 评估 ===');
  console.log(`模式: ${dryRun ? 'DRY-RUN (3 stocks)' : 'FULL (40 stocks)'}`);

  // Load dataset
  const dataset = loadFrozenDataset({ version: 'v1', subsetStocks: dryRun ? 3 : null });
  const { testPoints, stocks } = dataset;
  const stockMap = new Map(stocks.map(s => [s.code, s]));
  console.log(`数据集: ${stocks.length} stocks, ${testPoints.length} testPoints`);

  // 加载 MC Dropout 缓存
  const mcCache = await loadMcDropoutCache();
  console.log(`MC Dropout 缓存: ${Object.keys(mcCache).length} stocks`);

  // DB 连接
  const { getDb } = await import('../lib/db/connection.js');
  const { calcSectorAlpha } = await import('../lib/sector/alpha.js');
  const db = getDb();

  // 预加载 K 线
  const klinesCache = new Map();
  const uniqueCodes = [...new Set(testPoints.map(tp => tp.stockCode))];
  for (const code of uniqueCodes) {
    const rows = db.prepare('SELECT * FROM monthly_klines WHERE code=? ORDER BY date').all(code);
    if (rows.length >= 24) klinesCache.set(code, rows);
  }
  console.log(`K线缓存: ${klinesCache.size}/${uniqueCodes.length} stocks`);

  // 评估模板
  const templates = ['technical', 'trend', 'valuation', 'sentiment'];
  const total = testPoints.length * templates.length;
  console.log(`总任务: ${total} (${testPoints.length} stocks × ${templates.length} templates)`);

  const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  const outPath = path.join(RUNS_DIR, `mc-dropout-${timestamp}.jsonl`);
  const completed = loadCompleted(outPath);

  const pending = [];
  for (const tp of testPoints) {
    for (const tpl of templates) {
      if (!completed.has(`${tp.stockCode}_${tpl}`)) pending.push({ tp, tpl });
    }
  }
  console.log(`待跑: ${pending.length}/${total}\n`);

  if (pending.length === 0) { console.log('全部完成'); return; }

  const startTime = Date.now();
  let completed_count = total - pending.length, ok = completed_count, fail = 0;
  let totalCost = 0;
  const scoresWith = [];  // MC Dropout 注入
  const scoresWithout = [];  // 对照组（同样 stocks，不同 templates）

  let jobIdx = 0;
  while (jobIdx < pending.length) {
    const batch = pending.slice(jobIdx, jobIdx + LLM_CONCURRENCY);
    const batchProms = batch.map(async ({ tp, tpl }) => {
      const stock = stockMap.get(tp.stockCode);
      const klines = klinesCache.get(tp.stockCode);
      if (!stock || !klines || tp.cutoffIndex >= klines.length || tp.cutoffIndex < 12) {
        appendResult(outPath, { stockCode: tp.stockCode, template: tpl, error: 'K线不足' });
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

      // MC Dropout 数据
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
        // 分 with/without MC Dropout 收集分数
        if (r.lstmSignalData) scoresWith.push(r.record);
        else scoresWithout.push(r.record);
      }
    }

    jobIdx += LLM_CONCURRENCY;
    if (completed_count % 4 === 0 || completed_count === total) {
      const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
      console.log(`进度: ${completed_count}/${total} | 成功:${ok} 失败:${fail} | 耗时:${elapsed}min | 费用:¥${totalCost.toFixed(2)}`);
    }
    if (jobIdx < pending.length) await sleep(LLM_DELAY_MS);
  }

  // ---- 报告 ----
  const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
  console.log(`\n=== 评估完成 ===`);
  console.log(`耗时: ${elapsed}min | 费用: ¥${totalCost.toFixed(2)}`);
  console.log(`结果: ${outPath}`);

  // 统计
  const allResults = [];
  if (fs.existsSync(outPath)) {
    for (const line of fs.readFileSync(outPath, 'utf-8').trim().split('\n').filter(Boolean)) {
      try { allResults.push(JSON.parse(line)); } catch (_) {}
    }
  }

  const withMc = allResults.filter(r => r.hasMcDropout && !r.error);
  const withoutMc = allResults.filter(r => !r.hasMcDropout && !r.error);

  console.log(`\n=== A/B 对比 ===`);
  console.log(`With MC Dropout:    ${withMc.length} 结果, 均分=${(withMc.reduce((s,r)=>s+r.score,0)/Math.max(1,withMc.length)).toFixed(3)}`);
  console.log(`Without MC Dropout: ${withoutMc.length} 结果, 均分=${(withoutMc.reduce((s,r)=>s+r.score,0)/Math.max(1,withoutMc.length)).toFixed(3)}`);

  // 按不确定性分组
  for (const level of ['low', 'medium', 'high']) {
    const subset = withMc.filter(r => r.mcUncertainty === level);
    if (subset.length > 0) {
      const avg = subset.reduce((s, r) => s + r.score, 0) / subset.length;
      console.log(`  MC ${level}不确定性: ${subset.length} 个, 均分=${avg.toFixed(3)}`);
    }
  }

  console.log(`\n输出文件: ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
