// Deep audit: analyze root cause of v4 re-parse inconsistencies
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const RUNS_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval', 'runs');

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

const v4Path = path.join(RUNS_DIR, 'v4-signals-2026-05-17-00-41.jsonl');
const v4Lines = fs.readFileSync(v4Path, 'utf-8').trim().split('\n').filter(Boolean);
const v4Records = v4Lines.map(l => JSON.parse(l));

// Consistency categorization
let mismatchStats = {
  parseFailed_to_neutral: { count: 0, oldScoreSum: 0, newScoreSum: 0, cases: [] },
  parseFailed_to_actual: { count: 0, oldScoreSum: 0, newScoreSum: 0, cases: [] },
  signalChanged: { count: 0, oldScoreSum: 0, newScoreSum: 0, cases: [] },
  jsonMissing: { count: 0, oldScoreSum: 0, newScoreSum: 0, cases: [] },
};

for (const rec of v4Records) {
  const raw = rec.rawResponse;
  if (!raw) continue;

  // Extract JSON
  let scoreData = null;
  try {
    const m = raw.match(/```json\s*([\s\S]*?)```/);
    if (m) scoreData = JSON.parse(m[1].trim());
  } catch (_) {}

  const reparsedSignal = scoreData?.signal || 'neutral';
  const originalSignal = rec.signal || rec.predictedSignal || 'neutral';
  const groundTruth = rec.groundTruth;
  const newScore = scorePrediction(reparsedSignal, groundTruth);
  const oldScore = rec.score;

  if (originalSignal === reparsedSignal) continue;

  // Mismatch, categorize
  if (originalSignal === 'parse_failed' && reparsedSignal === 'neutral') {
    mismatchStats.parseFailed_to_neutral.count++;
    mismatchStats.parseFailed_to_neutral.oldScoreSum += oldScore;
    mismatchStats.parseFailed_to_neutral.newScoreSum += newScore;
    if (mismatchStats.parseFailed_to_neutral.cases.length < 3) {
      mismatchStats.parseFailed_to_neutral.cases.push({ code: rec.code, gt: groundTruth, old: oldScore, new: newScore });
    }
  } else if (originalSignal === 'parse_failed') {
    mismatchStats.parseFailed_to_actual.count++;
    mismatchStats.parseFailed_to_actual.oldScoreSum += oldScore;
    mismatchStats.parseFailed_to_actual.newScoreSum += newScore;
    if (mismatchStats.parseFailed_to_actual.cases.length < 3) {
      mismatchStats.parseFailed_to_actual.cases.push({
        code: rec.code, template: rec.template, gt: groundTruth,
        reparse: reparsedSignal, old: oldScore, new: newScore,
      });
    }
  } else {
    mismatchStats.signalChanged.count++;
    mismatchStats.signalChanged.oldScoreSum += oldScore;
    mismatchStats.signalChanged.newScoreSum += newScore;
    if (mismatchStats.signalChanged.cases.length < 5) {
      mismatchStats.signalChanged.cases.push({
        code: rec.code, template: rec.template, gt: groundTruth,
        orig: originalSignal, reparse: reparsedSignal, old: oldScore, new: newScore,
        json: raw.match(/```json\s*([\s\S]*?)```/)?.[1]?.substring(0, 200),
      });
    }
  }
}

console.log('=== v4 Re-parse Inconsistency Root Cause Analysis ===\n');

// Category 1: parse_failed → neutral
const pf1 = mismatchStats.parseFailed_to_neutral;
console.log(`1. parse_failed → neutral (JSON block exists but signal field missing/invalid): ${pf1.count} entries`);
console.log(`   Old score sum: ${pf1.oldScoreSum.toFixed(2)}, New score sum: ${pf1.newScoreSum.toFixed(2)}`);
console.log(`   Δ: ${(pf1.newScoreSum - pf1.oldScoreSum).toFixed(2)}`);

// Category 2: parse_failed → actual signal
const pf2 = mismatchStats.parseFailed_to_actual;
console.log(`\n2. parse_failed → actual signal (old parser JSON parse failed, new parser succeeded): ${pf2.count} entries`);
console.log(`   Old score sum: ${pf2.oldScoreSum.toFixed(2)}, New score sum: ${pf2.newScoreSum.toFixed(2)}`);
console.log(`   Δ: ${(pf2.newScoreSum - pf2.oldScoreSum).toFixed(2)}`);
pf2.cases.forEach(c => {
  console.log(`   Example: ${c.code} ${c.template} gt=${c.gt} parse_failed→${c.reparse} score=${c.old}→${c.new}`);
});

// Category 3: signal field changed
const sc = mismatchStats.signalChanged;
console.log(`\n3. signal field directly changed (JSON block parsed successfully but signal value differs): ${sc.count} entries`);
console.log(`   Old score sum: ${sc.oldScoreSum.toFixed(2)}, New score sum: ${sc.newScoreSum.toFixed(2)}`);
console.log(`   Δ: ${(sc.newScoreSum - sc.oldScoreSum).toFixed(2)}`);
sc.cases.forEach(c => {
  console.log(`   Example: ${c.code} ${c.template} gt=${c.gt} ${c.orig}→${c.reparse} score=${c.old}→${c.new}`);
  if (c.json) console.log(`      JSON: ${c.json}`);
});

// Total delta
const totalOldScore = v4Records.reduce((s, r) => s + (r.score || 0), 0);
let totalNewScore = 0;
for (const rec of v4Records) {
  const raw = rec.rawResponse;
  if (!raw) { totalNewScore += (rec.score || 0); continue; }
  let scoreData = null;
  try {
    const m = raw.match(/```json\s*([\s\S]*?)```/);
    if (m) scoreData = JSON.parse(m[1].trim());
  } catch (_) {}
  const reparsedSignal = scoreData?.signal || 'neutral';
  totalNewScore += scorePrediction(reparsedSignal, rec.groundTruth);
}

console.log(`\n=== Summary ===`);
console.log(`Total records: ${v4Records.length}`);
console.log(`Old weighted score: ${(totalOldScore / v4Records.length).toFixed(4)}`);
console.log(`New weighted score: ${(totalNewScore / v4Records.length).toFixed(4)}`);
console.log(`Δ: ${((totalNewScore - totalOldScore) / v4Records.length).toFixed(4)}`);
console.log(`Total mismatches: ${pf1.count + pf2.count + sc.count} entries (${((pf1.count + pf2.count + sc.count) / v4Records.length * 100).toFixed(1)}%)`);
console.log(`Old score sum: ${totalOldScore.toFixed(1)} → New score sum: ${totalNewScore.toFixed(1)} (Δ=${(totalNewScore - totalOldScore).toFixed(1)})`);

// Extra verification: what logic was used to compute the score claimed by the original v4 jsonl?
console.log(`\n=== Verification: original score calculation logic ===`);
// Manually verify a few samples
for (let i = 0; i < 5; i++) {
  const r = v4Records[i];
  const s = mapSignal(r.signal);
  const g = mapSignal(r.groundTruth);
  const expected = scorePrediction(r.signal, r.groundTruth);
  console.log(`  [${i}] signal=${r.signal}(${s}) gt=${r.groundTruth}(${g}) stored=${r.score} expected=${expected} ${r.score === expected ? '✓' : '✗'}`);
}

// Manually verify a few parse_failed samples
const pfRecords = v4Records.filter(r => r.signal === 'parse_failed');
console.log(`\nparse_failed record count: ${pfRecords.length}`);
for (let i = 0; i < 3; i++) {
  const r = pfRecords[i];
  console.log(`  [${i}] ${r.code} ${r.cutoffDate} ${r.template} signal=${r.signal} gt=${r.groundTruth} score=${r.score}`);
  console.log(`       rawResponse JSON: ${(r.rawResponse || '').match(/```json\s*([\s\S]*?)```/)?.[1]?.substring(0, 150) || '(no JSON block)'}`);
}
