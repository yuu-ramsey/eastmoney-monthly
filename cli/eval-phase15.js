// Phase 15 Multi-Agent eval
// 5 agents per sample: Bull+Bear || Technical+Sector → Judge
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadFrozenDataset } from '../lib/eval/load-frozen-dataset.js';
import { buildKlineTable } from '../lib/build-prompt.js';
import { computeMA } from '../lib/compute-ma.js';
import { computeMACD } from '../lib/compute-macd.js';
import { scorePrediction } from '../lib/eval/compute-score.js';
import { runMultiAgentDebate, parseJudgeSignal } from '../lib/agents/phase15-runner.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const RUNS_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs');

const EVAL_MAX_TOKENS = 4000;
const LLM_CONCURRENCY = 1;     // 1 testPoint at a time (5 agents concurrent)
const LLM_DELAY_MS = 500;
const FROZEN_BASELINE = 0.1966;
const PROMPT_VERSION = 'phase15-multi-agent';
const PERIOD_LABELS = { monthly: '月线', weekly: '周线', daily: '日线' };

async function getLLM(prompt, apiKey) {
  const resp = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({ model: 'deepseek-chat', messages: [{ role: 'user', content: prompt }], max_tokens: EVAL_MAX_TOKENS, temperature: 0.0 }),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const d = await resp.json();
  return d.choices?.[0]?.message?.content || '';
}

// Factory function for Agent LLM calls
function makeAgentOpts(apiKey) {
  return {
    provider: 'deepseek',
    model: 'deepseek-chat',
    apiKey,
    maxTokens: EVAL_MAX_TOKENS,
    llmCallFn: async (prompt) => getLLM(prompt, apiKey),
  };
}

// Override runAgentLLM to use our fetch function
async function runLLM(prompt, apiKey) {
  const text = await getLLM(prompt, apiKey);
  return { role: 'agent', text, usage: null, cost: 0.022, durationMs: 0 };
}

function loadCompleted(fp) {
  if (!fs.existsSync(fp)) return new Set();
  const c = fs.readFileSync(fp, 'utf-8').trim();
  if (!c) return new Set();
  const s = new Set(); for (const l of c.split('\n').filter(Boolean)) {
    try { const r = JSON.parse(l); if (!r.error) s.add(`${r.stockCode}_${r.template}`); } catch (_) {}
  } return s;
}

function append(fp, r) { fs.mkdirSync(path.dirname(fp), { recursive: true }); fs.appendFileSync(fp, JSON.stringify(r)+'\n'); }

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error('DEEPSEEK_API_KEY');

  const dryRun = process.argv.includes('--dry-run');
  const dataset = loadFrozenDataset({ version: 'v1', subsetStocks: dryRun ? 3 : null });
  const testPoints = dataset.testPoints;
  const total = testPoints.length * 4;

  console.log(`=== Phase 15 Multi-Agent ===`);
  console.log(`Mode: ${dryRun ? 'DRY-RUN' : 'FULL'}, stocks=${dataset.stocks.length}, testPoints=${testPoints.length}, calls=${total * 5}`);
  console.log(`Frozen baseline: ${FROZEN_BASELINE}`);
  console.log(`Cost estimate: ¥${(total * 5 * 0.022).toFixed(0)}\n`);

  const { getDb } = await import('../lib/db/connection.js');
  const { calcSectorAlpha } = await import('../lib/sector/alpha.js');
  const db = getDb();

  const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
  const outPath = path.join(RUNS_DIR, `phase15-${timestamp}.jsonl`);
  const completed = loadCompleted(outPath);
  const stockMap = new Map(dataset.stocks.map(s => [s.code, s]));

  const pending = [];
  for (const tp of testPoints) {
    for (const tpl of ['technical', 'trend', 'valuation', 'sentiment']) {
      if (!completed.has(`${tp.stockCode}_${tpl}`)) pending.push({ tp, tpl });
    }
  }
  console.log(`Pending: ${pending.length}/${total}\n`);

  if (pending.length === 0) { console.log('All done'); printReport(outPath); return; }

  // K-line cache
  const klinesCache = new Map();
  for (const c of [...new Set(pending.map(j => j.tp.stockCode))]) {
    const rows = db.prepare('SELECT * FROM monthly_klines WHERE code=? ORDER BY date').all(c);
    if (rows.length >= 24) klinesCache.set(c, rows);
  }

  const startTime = Date.now();
  let completedCount = total - pending.length, ok = completedCount, fail = 0;
  let totalCost = 0;

  for (let i = 0; i < pending.length; i++) {
    const { tp, tpl } = pending[i];
    const key = `${tp.stockCode}_${tpl}`;
    const stock = stockMap.get(tp.stockCode);
    const klines = klinesCache.get(tp.stockCode);

    if (!stock || !klines || tp.cutoffIndex >= klines.length || tp.cutoffIndex < 12) {
      append(outPath, { stockCode: tp.stockCode, template: tpl, error: 'insufficient K-lines', score: null });
      fail++; completedCount++; continue;
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

    // Sector alpha
    let sectorAlphaData = null;
    try { sectorAlphaData = calcSectorAlpha(db, tp.stockCode, 'monthly', 12, tp.cutoffDate); } catch (_) {}

    const ctx = {
      name: stock.name || tp.stockCode, code: tp.stockCode,
      period: 'monthly', periodLabel: '月线', klines: kwi,
      sectorAlphaData, templateKey: tpl,
    };

    // Run all 5 agents via debate
    const opts = {
      provider: 'deepseek', model: 'deepseek-chat', apiKey, maxTokens: EVAL_MAX_TOKENS,
      llmCallFn: async (prompt) => ({ text: await getLLM(prompt, apiKey), usage: null, cost: 0.022, durationMs: 0 }),
    };

    try {
      const result = await runMultiAgentDebate(ctx, opts);

      const allText = result.judge?.text || '';
      const parsed = parseJudgeSignal(allText);
      const predictedSignal = parsed?.signal || 'parse_failed';
      const score = scorePrediction(predictedSignal, tp.groundTruth);

      append(outPath, {
        stockCode: tp.stockCode, stockName: tp.stockName,
        cutoffDate: tp.cutoffDate, template: tpl,
        promptVersion: PROMPT_VERSION,
        predictedSignal, groundTruth: tp.groundTruth, score,
        judgeSignal: parsed, judgeText: result.judge?.text || null,
        bullText: result.partials?.bull_researcher?.text || null,
        bearText: result.partials?.bear_researcher?.text || null,
        techText: result.partials?.technical_agent?.text || null,
        sectorText: result.partials?.sector_agent?.text || null,
        agentSuccess: result.successCount,
        agentErrors: result.errors,
        judgeError: result.judgeError || null,
        totalCost: result.totalCost,
        durationMs: result.durationMs,
        timestamp: new Date().toISOString(),
      });
      totalCost += result.totalCost;
      ok++; completedCount++;
    } catch (err) {
      append(outPath, { stockCode: tp.stockCode, template: tpl, error: String(err), score: null });
      fail++; completedCount++;
    }

    if (completedCount % 5 === 0 || completedCount === total) {
      const e = ((Date.now()-startTime)/60000).toFixed(1);
      const pct = (completedCount/total*100).toFixed(1);
      const eta = Math.max(0, (parseFloat(e)/completedCount*total-parseFloat(e))).toFixed(0);
      console.log(`[P15] ${completedCount}/${total} (${pct}%) ok=${ok} fail=${fail} ¥${totalCost.toFixed(0)} ${e}min ETA${eta}min`);
    }

    if (i < pending.length - 1) await sleep(LLM_DELAY_MS);
  }

  const elapsed = ((Date.now()-startTime)/60000).toFixed(1);
  console.log(`\n[P15] Complete: ${ok}/${total} ok, ${fail} fail, ${elapsed}min, ¥${totalCost.toFixed(2)}\n`);
  printReport(outPath);
}

function printReport(filePath) {
  if (!fs.existsSync(filePath)) return;
  const records = fs.readFileSync(filePath, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
  const valid = records.filter(r => r.score != null);

  if (valid.length === 0) { console.log('No valid records'); return; }

  const sum = valid.reduce((s, r) => s + r.score, 0);
  const score = +(sum / valid.length).toFixed(4);

  const sigDist = {};
  for (const r of valid) { const s = r.predictedSignal || '?'; sigDist[s] = (sigDist[s] || 0) + 1; }

  const pf = valid.filter(r => r.predictedSignal === 'parse_failed').length;
  const strongBull = valid.filter(r => r.predictedSignal === 'strong_bull');
  const strongBear = valid.filter(r => r.predictedSignal === 'strong_bear');
  const sbFp = strongBull.filter(r => r.groundTruth !== 'strong_bull').length;
  const sbearFp = strongBear.filter(r => r.groundTruth !== 'strong_bear').length;

  const totalCost = records.reduce((s, r) => s + (r.totalCost || 0), 0);
  const judgeOk = valid.filter(r => r.judgeText && !r.judgeError).length;

  // Agent-level agreement: Judge signal vs Bull/Bear raw signals
  let bullAgree = 0, bearAgree = 0, totalAgreed = 0;

  console.log('## Phase 15 Multi-Agent Results\n');
  console.log(`| Metric | Value |`);
  console.log(`|--------|-------|`);
  console.log(`| Valid records | ${valid.length} |`);
  console.log(`| **Weighted score (all)** | **${score}** |`);
  console.log(`| parse_failed | ${pf} (${(pf/valid.length*100).toFixed(1)}%) |`);
  console.log(`| strong_bull | ${strongBull.length} (${(strongBull.length/valid.length*100).toFixed(1)}%) FP=${strongBull.length>0?(sbFp/strongBull.length*100).toFixed(0):'N/A'}% |`);
  console.log(`| strong_bear | ${strongBear.length} (${(strongBear.length/valid.length*100).toFixed(1)}%) FP=${strongBear.length>0?(sbearFp/strongBear.length*100).toFixed(0):'N/A'}% |`);
  console.log(`| Judge success | ${judgeOk}/${valid.length} |`);
  console.log(`| Total cost | ¥${totalCost.toFixed(2)} |`);

  const delta = score - FROZEN_BASELINE;
  console.log(`\nFrozen baseline: ${FROZEN_BASELINE}`);
  console.log(`Δ: ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}`);
  if (score > 0.22) console.log('Conclusion: PASS — score > 0.22');
  else if (score >= 0.20) console.log('Conclusion: MARGINAL — score ∈ [0.20, 0.22]');
  else console.log('Conclusion: FAIL — score < 0.20');
}

main().catch(e => { console.error(e); process.exit(1); });
