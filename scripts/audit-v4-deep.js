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

// 分类一致性
let mismatchStats = {
  parseFailed_to_neutral: { count: 0, oldScoreSum: 0, newScoreSum: 0, cases: [] },
  parseFailed_to_actual: { count: 0, oldScoreSum: 0, newScoreSum: 0, cases: [] },
  signalChanged: { count: 0, oldScoreSum: 0, newScoreSum: 0, cases: [] },
  jsonMissing: { count: 0, oldScoreSum: 0, newScoreSum: 0, cases: [] },
};

for (const rec of v4Records) {
  const raw = rec.rawResponse;
  if (!raw) continue;

  // 提取 JSON
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

  // 不一致，分类
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

console.log('=== v4 重解析不一致根因分析 ===\n');

// 类别1: parse_failed → neutral
const pf1 = mismatchStats.parseFailed_to_neutral;
console.log(`1. parse_failed → neutral (JSON块存在但signal字段缺失/无效): ${pf1.count}条`);
console.log(`   旧 score 总和: ${pf1.oldScoreSum.toFixed(2)}, 新 score 总和: ${pf1.newScoreSum.toFixed(2)}`);
console.log(`   Δ: ${(pf1.newScoreSum - pf1.oldScoreSum).toFixed(2)}`);

// 类别2: parse_failed → 实际信号
const pf2 = mismatchStats.parseFailed_to_actual;
console.log(`\n2. parse_failed → 实际信号 (旧parser JSON解析失败,新parser成功): ${pf2.count}条`);
console.log(`   旧 score 总和: ${pf2.oldScoreSum.toFixed(2)}, 新 score 总和: ${pf2.newScoreSum.toFixed(2)}`);
console.log(`   Δ: ${(pf2.newScoreSum - pf2.oldScoreSum).toFixed(2)}`);
pf2.cases.forEach(c => {
  console.log(`   例: ${c.code} ${c.template} gt=${c.gt} parse_failed→${c.reparse} score=${c.old}→${c.new}`);
});

// 类别3: signal 字段变化
const sc = mismatchStats.signalChanged;
console.log(`\n3. signal 字段直接变化 (JSON块解析成功但signal值不同): ${sc.count}条`);
console.log(`   旧 score 总和: ${sc.oldScoreSum.toFixed(2)}, 新 score 总和: ${sc.newScoreSum.toFixed(2)}`);
console.log(`   Δ: ${(sc.newScoreSum - sc.oldScoreSum).toFixed(2)}`);
sc.cases.forEach(c => {
  console.log(`   例: ${c.code} ${c.template} gt=${c.gt} ${c.orig}→${c.reparse} score=${c.old}→${c.new}`);
  if (c.json) console.log(`      JSON: ${c.json}`);
});

// 总 Δ
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

console.log(`\n=== 总结 ===`);
console.log(`总记录: ${v4Records.length}`);
console.log(`旧加权得分: ${(totalOldScore / v4Records.length).toFixed(4)}`);
console.log(`新加权得分: ${(totalNewScore / v4Records.length).toFixed(4)}`);
console.log(`Δ: ${((totalNewScore - totalOldScore) / v4Records.length).toFixed(4)}`);
console.log(`不一致总计: ${pf1.count + pf2.count + sc.count}条 (${((pf1.count + pf2.count + sc.count) / v4Records.length * 100).toFixed(1)}%)`);
console.log(`旧 score 总和: ${totalOldScore.toFixed(1)} → 新 score 总和: ${totalNewScore.toFixed(1)} (Δ=${(totalNewScore - totalOldScore).toFixed(1)})`);

// 额外验证：原始 v4 jsonl 自身声称的 score 用什么逻辑算的？
console.log(`\n=== 验证：原始 score 计算逻辑 ===`);
// 抽几条做手工验证
for (let i = 0; i < 5; i++) {
  const r = v4Records[i];
  const s = mapSignal(r.signal);
  const g = mapSignal(r.groundTruth);
  const expected = scorePrediction(r.signal, r.groundTruth);
  console.log(`  [${i}] signal=${r.signal}(${s}) gt=${r.groundTruth}(${g}) stored=${r.score} expected=${expected} ${r.score === expected ? '✓' : '✗'}`);
}

// 抽几条 parse_failed 的验证
const pfRecords = v4Records.filter(r => r.signal === 'parse_failed');
console.log(`\nparse_failed 记录数: ${pfRecords.length}`);
for (let i = 0; i < 3; i++) {
  const r = pfRecords[i];
  console.log(`  [${i}] ${r.code} ${r.cutoffDate} ${r.template} signal=${r.signal} gt=${r.groundTruth} score=${r.score}`);
  console.log(`       rawResponse JSON: ${(r.rawResponse || '').match(/```json\s*([\s\S]*?)```/)?.[1]?.substring(0, 150) || '(无JSON块)'}`);
}
