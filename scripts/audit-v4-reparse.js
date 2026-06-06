// Audit script: re-parse v4-signals jsonl with current parser, verify score reproducibility
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const RUNS_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs');

// Score logic identical to lib/eval/runner.js
function mapSignal(s) {
  if (s === 'strong_bull') return 2;
  if (s === 'bull') return 1;
  if (s === 'neutral') return 0;
  if (s === 'bear') return -1;
  if (s === 'strong_bear') return -2;
  return null;
}

function scorePrediction(predictedSignal, groundTruth) {
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

function reparseRecord(record) {
  const raw = record.rawResponse;
  if (!raw) return { error: 'rawResponse missing' };

  // Use exact same JSON extraction logic as runner.js
  let scoreData = null;
  try {
    const m = raw.match(/```json\s*([\s\S]*?)```/);
    if (m) scoreData = JSON.parse(m[1].trim());
  } catch (_) { /* JSON parse failed */ }

  const reparsedSignal = scoreData?.signal || 'neutral';
  const groundTruth = record.groundTruth;
  const originalSignal = record.signal || record.predictedSignal || 'neutral';
  const originalScore = record.score;

  const newScore = scorePrediction(reparsedSignal, groundTruth);

  return {
    code: record.code || record.stockCode,
    cutoffDate: record.cutoffDate,
    template: record.template,
    groundTruth,
    originalSignal,
    reparsedSignal,
    signalMatch: originalSignal === reparsedSignal,
    originalScore,
    newScore,
    scoreMatch: originalScore === newScore,
    hasRawResponse: !!raw,
    hasJsonBlock: !!(raw && raw.match(/```json\s*([\s\S]*?)```/)),
  };
}

function computeStats(results) {
  const parsed = results.filter(r => !r.error);
  const total = parsed.length;
  let scoreSum = 0, perfect = 0, dirCorrect = 0, dirWrong = 0;
  for (const r of parsed) {
    const s = r.originalScore ?? r.newScore;
    if (s != null) scoreSum += s;
    if (s === 1.0) perfect++;
    if (s != null && s >= 0.5) dirCorrect++;
    if (s != null && s < 0) dirWrong++;
  }
  return {
    total,
    weightedScore: total > 0 ? +(scoreSum / total).toFixed(4) : null,
    perfectPct: total > 0 ? (perfect / total * 100).toFixed(1) + '%' : '0%',
    dirCorrectPct: total > 0 ? (dirCorrect / total * 100).toFixed(1) + '%' : '0%',
    dirWrongPct: total > 0 ? (dirWrong / total * 100).toFixed(1) + '%' : '0%',
  };
}

function signalDistribution(results) {
  const parsed = results.filter(r => !r.error);
  const dist = { strong_bull: 0, bull: 0, neutral: 0, bear: 0, strong_bear: 0 };
  for (const r of parsed) {
    const s = r.reparsedSignal || r.originalSignal;
    if (dist[s] !== undefined) dist[s]++;
  }
  for (const k of Object.keys(dist)) {
    dist[k] = { count: dist[k], pct: (dist[k] / parsed.length * 100).toFixed(1) + '%' };
  }
  return dist;
}

// ---- v4-signals ----
console.log('='.repeat(60));
console.log('Task 1c: Re-parse v4-signals (latest file: 00-41)');
console.log('='.repeat(60));

const v4Path = path.join(RUNS_DIR, 'v4-signals-2026-05-17-00-41.jsonl');
if (!fs.existsSync(v4Path)) {
  console.log('FATAL: v4 jsonl does not exist');
  process.exit(1);
}

const v4Lines = fs.readFileSync(v4Path, 'utf-8').trim().split('\n').filter(Boolean);
console.log(`File lines: ${v4Lines.length}`);

const v4Records = v4Lines.map(l => JSON.parse(l));
const v4Results = v4Records.map(reparseRecord);

// Statistics
const v4Stats = computeStats(v4Results);
console.log('\n--- v4 Re-parse Statistics ---');
console.log(`Total records: ${v4Stats.total}`);
console.log(`Weighted score (re-parsed): ${v4Stats.weightedScore}`);
console.log(`Perfect accuracy: ${v4Stats.perfectPct}`);
console.log(`Direction correct rate: ${v4Stats.dirCorrectPct}`);
console.log(`Direction wrong rate: ${v4Stats.dirWrongPct}`);

const v4SigDist = signalDistribution(v4Results);
console.log('\n--- Signal Distribution After Re-parse ---');
for (const [sig, info] of Object.entries(v4SigDist)) {
  console.log(`  ${sig}: ${info.count} (${info.pct})`);
}

// Compatibility check
const v4NoRaw = v4Results.filter(r => !r.hasRawResponse).length;
const v4NoJson = v4Results.filter(r => r.hasRawResponse && !r.hasJsonBlock).length;
const v4SignalMismatch = v4Results.filter(r => !r.error && !r.signalMatch).length;
const v4ScoreMismatch = v4Results.filter(r => !r.error && !r.scoreMatch).length;
const v4Error = v4Results.filter(r => r.error).length;

console.log('\n--- Compatibility Check ---');
console.log(`rawResponse missing: ${v4NoRaw}/${v4Results.length}`);
console.log(`JSON block missing: ${v4NoJson}/${v4Results.length}`);
console.log(`signal mismatch: ${v4SignalMismatch}/${v4Results.length}`);
console.log(`score mismatch: ${v4ScoreMismatch}/${v4Results.length}`);
console.log(`Parse errors: ${v4Error}/${v4Results.length}`);

if (v4SignalMismatch > 0) {
  console.log('\n--- signal mismatch samples (first 5) ---');
  v4Results.filter(r => !r.error && !r.signalMatch).slice(0, 5).forEach(r => {
    console.log(`  ${r.code} ${r.cutoffDate} ${r.template}: orig=${r.originalSignal} reparse=${r.reparsedSignal} gt=${r.groundTruth}`);
  });
}

if (v4ScoreMismatch > 0) {
  console.log('\n--- score mismatch samples (first 5) ---');
  v4Results.filter(r => !r.error && !r.scoreMatch).slice(0, 5).forEach(r => {
    console.log(`  ${r.code} ${r.cutoffDate} ${r.template}: orig=${r.originalScore} new=${r.newScore} signal=[${r.originalSignal}→${r.reparsedSignal}]`);
  });
}

// ---- v5-resonance ----
console.log('\n' + '='.repeat(60));
console.log('Task 1d: Re-parse v5-resonance');
console.log('='.repeat(60));

const v5Path = path.join(RUNS_DIR, 'v5-resonance-2026-05-17-03-16.jsonl');
if (!fs.existsSync(v5Path)) {
  console.log('FATAL: v5 jsonl does not exist');
  process.exit(1);
}

const v5Lines = fs.readFileSync(v5Path, 'utf-8').trim().split('\n').filter(Boolean);
console.log(`File lines: ${v5Lines.length}`);

const v5Records = v5Lines.map(l => JSON.parse(l));
const v5Results = v5Records.map(reparseRecord);

const v5Stats = computeStats(v5Results);
console.log('\n--- v5 Re-parse Statistics ---');
console.log(`Total records: ${v5Stats.total}`);
console.log(`Weighted score (re-parsed): ${v5Stats.weightedScore}`);
console.log(`Perfect accuracy: ${v5Stats.perfectPct}`);
console.log(`Direction correct rate: ${v5Stats.dirCorrectPct}`);

const v5SigDist = signalDistribution(v5Results);
console.log('\n--- Signal Distribution After Re-parse ---');
for (const [sig, info] of Object.entries(v5SigDist)) {
  console.log(`  ${sig}: ${info.count} (${info.pct})`);
}

const v5NoRaw = v5Results.filter(r => !r.hasRawResponse).length;
const v5SignalMismatch = v5Results.filter(r => !r.error && !r.signalMatch).length;
const v5ScoreMismatch = v5Results.filter(r => !r.error && !r.scoreMatch).length;

console.log('\n--- Compatibility Check ---');
console.log(`rawResponse missing: ${v5NoRaw}/${v5Results.length}`);
console.log(`signal mismatch: ${v5SignalMismatch}/${v5Results.length}`);
console.log(`score mismatch: ${v5ScoreMismatch}/${v5Results.length}`);

if (v5ScoreMismatch > 0) {
  console.log('\n--- score mismatch samples (first 5) ---');
  v5Results.filter(r => !r.error && !r.scoreMatch).slice(0, 5).forEach(r => {
    console.log(`  ${r.code} ${r.cutoffDate} ${r.template}: orig=${r.originalScore} new=${r.newScore} signal=[${r.originalSignal}→${r.reparsedSignal}]`);
  });
}

// Final conclusion
console.log('\n' + '='.repeat(60));
console.log('Conclusion');
console.log('='.repeat(60));
const v4AllMatch = v4SignalMismatch === 0 && v4ScoreMismatch === 0 && v4NoRaw === 0;
const v5AllMatch = v5SignalMismatch === 0 && v5ScoreMismatch === 0 && v5NoRaw === 0;

console.log(`v4: ${v4AllMatch ? '✓ Fully reproducible — data is authentic' : '✗ Inconsistencies exist'}`);
console.log(`v5: ${v5AllMatch ? '✓ Fully reproducible — data is authentic' : '✗ Inconsistencies exist'}`);
