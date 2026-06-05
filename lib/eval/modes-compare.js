// Mode comparison: llm_only vs quant_only vs hybrid
// Reuses v1 eval data + kline cache, does not re-call LLM
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { computeQuantScore } from '../quant-factors.js';
import { fuseScores, scoreToSignal, detectStockRegime } from '../score-fusion.js';
import { fetchKlinesSafe } from './runner.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const EVAL_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval');

export async function runModesCompare(v1RunId) {
  const v1Path = path.join(EVAL_DIR, 'runs', `${v1RunId}.jsonl`);
  if (!fs.existsSync(v1Path)) throw new Error(`v1 run not found: ${v1RunId}`);

  const v1Results = fs.readFileSync(v1Path, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
  const datasetPath = path.join(EVAL_DIR, 'dataset.json');
  const dataset = fs.existsSync(datasetPath)
    ? JSON.parse(fs.readFileSync(datasetPath, 'utf-8'))
    : { stocks: [], testPoints: [] };
  // 从seed-stocks补全category+market
  const seedsPath = path.join(PROJECT_DIR, 'lib', 'eval', 'seed-stocks.json');
  const seeds = JSON.parse(fs.readFileSync(seedsPath, 'utf-8'));
  const catMap = new Map();
  const marketMap = new Map();
  for (const [cat, list] of Object.entries(seeds.stocks)) {
    for (const s of list) { catMap.set(s.code, cat); marketMap.set(s.code, s.market); }
  }

  const tpMap = new Map(dataset.testPoints.map(tp => [tp.id, tp]));

  // 批量拉K线
  const cache = new Map();
  const uniqueStocks = new Set(v1Results.map(r => r.stockCode));
  console.log(`Loading klines for ${uniqueStocks.size} stocks...`);

  let ok = 0, fail = 0;
  for (const code of uniqueStocks) {
    const market = marketMap.get(code);
    if (!market) { fail++; continue; }
    try {
      const klines = await fetchKlinesSafe(market, code, 'monthly', 200);
      if (Array.isArray(klines)) { cache.set(code, klines); ok++; }
      else fail++;
    } catch (e) { fail++; }
  }
  console.log(`Loaded ${ok} stocks' klines (${fail} failed)`);

  // 对每条v1结果添加quant评分
  const enriched = [];
  for (const r of v1Results) {
    if (r.error) { enriched.push(r); continue; }

    const allKlines = cache.get(r.stockCode);
    if (!allKlines) { enriched.push(r); continue; }

    // 用最近的24根K线（模拟cutoff时能看到的数据）
    // 因为testPoint的cutoffIndex信息在dataset中，这里用-24作为近似
    const cutoffKlines = allKlines.slice(-Math.min(60, allKlines.length));
    if (cutoffKlines.length < 24) { enriched.push(r); continue; }

    // 补全category
    r.category = r.category || catMap.get(r.stockCode) || 'unknown';

    const quantResult = computeQuantScore(cutoffKlines);
    if (!quantResult) { enriched.push(r); continue; }

    // LLM结果（从v1 eval重建）
    const llmResult = {
      score: r.score != null ? Math.max(0, Math.min(100, 50 + r.score * 50)) : 50,
      signal: r.predictedSignal || 'neutral',
      confidence: 'medium',
    };

    // 三种模式
    // mode 1: llm_only = 原v1结果
    const llmOnly = { ...r, mode: 'llm_only' };

    // mode 2: quant_only
    const quantOnlyScore = quantResult.score;
    const quantOnly = {
      ...r,
      mode: 'quant_only',
      predictedSignal: scoreToSignal(quantOnlyScore),
      score: computeEvalScore(scoreToSignal(quantOnlyScore), r.groundTruth),
    };

    // mode 3: hybrid (旧)
    const fusedOld = fuseScores(llmResult, quantResult, { useAdaptive: false });
    const hybrid = {
      ...r,
      mode: 'hybrid',
      predictedSignal: fusedOld.final_signal,
      score: computeEvalScore(fusedOld.final_signal, r.groundTruth),
      quantScore: quantResult.score,
      fusionAgreement: fusedOld.agreement,
    };

    // mode 4: adaptive (新)
    const fusedAdaptive = fuseScores(llmResult, quantResult, { useAdaptive: true });
    const adaptive = {
      ...r,
      mode: 'adaptive',
      predictedSignal: fusedAdaptive.final_signal,
      score: computeEvalScore(fusedAdaptive.final_signal, r.groundTruth),
      quantScore: quantResult.score,
      fusionAgreement: fusedAdaptive.agreement,
      regime: fusedAdaptive.regime,
    };

    enriched.push(llmOnly, quantOnly, hybrid, adaptive);
  }

  // 按mode分组统计
  const modes = ['llm_only', 'quant_only', 'hybrid', 'adaptive'];
  const cats = [...new Set(enriched.filter(r => !r.error && r.category).map(r => r.category))];

  console.log(`\n=== Modes Comparison ===`);
  console.log(`| mode | overall | ${cats.join(' | ')} |`);
  console.log(`|------|---------|${cats.map(() => '--------|').join('')}`);

  for (const mode of modes) {
    const modeResults = enriched.filter(r => r.mode === mode && !r.error && r.score != null);
    const overall = avgScore(modeResults);
    const catScores = cats.map(cat => {
      const cr = modeResults.filter(r => r.category === cat);
      return cr.length > 0 ? avgScore(cr).toFixed(2) : '-';
    });
    console.log(`| ${mode.padEnd(10)} | ${overall.toFixed(2)} | ${catScores.join(' | ')} |`);
  }

  // Regime分布（仅adaptive模式）
  console.log(`\n=== Adaptive Regime Distribution ===`);
  const adaptiveResults = enriched.filter(r => r.mode === 'adaptive' && !r.error);
  console.log(`| 类别 | total | strong_trend% | sideways% | high_vol% | mixed% |`);
  console.log(`|------|-------|--------------|-----------|-----------|--------|`);
  for (const cat of cats) {
    const cr = adaptiveResults.filter(r => r.category === cat);
    if (cr.length === 0) continue;
    const total = cr.length;
    const regimeCounts = { strong_trend: 0, sideways: 0, high_vol: 0, mixed: 0 };
    for (const r of cr) { if (regimeCounts[r.regime] !== undefined) regimeCounts[r.regime]++; }
    console.log(`| ${cat.padEnd(14)} | ${String(total).padEnd(5)} | ${pctStr(regimeCounts.strong_trend, total)} | ${pctStr(regimeCounts.sideways, total)} | ${pctStr(regimeCounts.high_vol, total)} | ${pctStr(regimeCounts.mixed, total)} |`);
  }

  // 保存 enriched results
  const outPath = path.join(EVAL_DIR, 'runs', `modes-compare-${v1RunId}.jsonl`);
  fs.writeFileSync(outPath, enriched.map(r => JSON.stringify(r)).join('\n') + '\n', 'utf-8');

  return { path: outPath, modes, cats, enriched };
}

function pctStr(n, total) { return total > 0 ? (n / total * 100).toFixed(0) + '%'.padEnd(5) : '0%'.padEnd(5); }

function avgScore(results) {
  if (results.length === 0) return 0;
  return results.reduce((s, r) => s + (r.score || 0), 0) / results.length;
}

function computeEvalScore(predicted, groundTruth) {
  const map = { strong_bull: 2, bull: 1, neutral: 0, bear: -1, strong_bear: -2 };
  const p = map[predicted] ?? 0;
  const g = map[groundTruth] ?? 0;
  if (p === g) return 1.0;
  if (p * g < 0) return (Math.abs(p) === 2 && Math.abs(g) === 2) ? -1.0 : -0.5;
  if (p === 0 || g === 0) return 0.3;
  return 0.5;
}
