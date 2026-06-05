// 折扣多月回报标签对比评估
// 用法：
//   node cli/eval-discounted-labels.js <eval-jsonl-path>
//   node cli/eval-discounted-labels.js .eastmoney-ai/eval/runs/mc-dropout-2026-05-20-18-50-15.jsonl
// 将已有 eval 结果用 v2 折扣回报标签重新打分，对比两种标签下的得分差异

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadFrozenDataset } from '../lib/eval/load-frozen-dataset.js';
import { scorePrediction } from '../lib/eval/compute-score.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');

const args = process.argv.slice(2);
const jsonlPath = args[0];
if (!jsonlPath) {
  console.error('用法: node cli/eval-discounted-labels.js <eval-jsonl-path>');
  process.exit(1);
}

const fullPath = path.resolve(PROJECT_DIR, jsonlPath);
if (!fs.existsSync(fullPath)) {
  console.error(`文件不存在: ${fullPath}`);
  process.exit(1);
}

// 加载 v2 dataset
console.log('加载 v2 dataset...');
const v2 = loadFrozenDataset({ version: 'v2' });
console.log(`  ${v2.stocks.length} stocks, ${v2.testPoints.length} testPoints`);

// 构建 v2 lookup: stockCode + cutoffDate → testPoint
const v2Lookup = new Map();
for (const tp of v2.testPoints) {
  v2Lookup.set(tp.stockCode + '_' + tp.cutoffDate, tp);
}

// 读取 eval 结果
console.log(`读取 eval 结果: ${path.basename(fullPath)}`);
const lines = fs.readFileSync(fullPath, 'utf-8').trim().split('\n').filter(Boolean);
const evalResults = lines.map(JSON.parse).filter(r => !r.error && r.predictedSignal !== 'parse_failed');
console.log(`  ${evalResults.length} valid results`);

// 匹配并重新打分
let matched = 0;
let unmatched = 0;
const pairs = [];  // { stockCode, cutoffDate, predictedSignal, scoreOld, scoreDiscounted, oldLabel, newLabel }

for (const r of evalResults) {
  const key = r.stockCode + '_' + r.cutoffDate;
  const tp = v2Lookup.get(key);
  if (!tp) {
    unmatched++;
    continue;
  }
  matched++;

  const scoreOld = scorePrediction(r.predictedSignal, tp.groundTruth);
  const scoreDiscounted = scorePrediction(r.predictedSignal, tp.groundTruthDiscounted);

  pairs.push({
    stockCode: r.stockCode,
    cutoffDate: r.cutoffDate,
    template: r.template,
    predictedSignal: r.predictedSignal,
    mcUncertainty: r.mcUncertainty || 'none',
    scoreOld,
    scoreDiscounted,
    oldLabel: tp.groundTruth,
    newLabel: tp.groundTruthDiscounted,
    labelAgreement: tp.labelAgreement,
  });
}

console.log(`\n匹配: ${matched}, 未匹配: ${unmatched}`);

// 统计对比
const oldScores = pairs.map(p => p.scoreOld);
const newScores = pairs.map(p => p.scoreDiscounted);

const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
const std = arr => {
  const m = avg(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
};

console.log('\n=== 旧标签 vs 新标签 得分对比 ===');
console.log(`旧标签(单月alpha):   n=${oldScores.length} mean=${avg(oldScores).toFixed(4)} std=${std(oldScores).toFixed(4)}`);
console.log(`新标签(折扣回报):    n=${newScores.length} mean=${avg(newScores).toFixed(4)} std=${std(newScores).toFixed(4)}`);
console.log(`差异:               Δmean=${(avg(newScores) - avg(oldScores)).toFixed(4)} Δstd=${(std(newScores) - std(oldScores)).toFixed(4)}`);

// 按标签一致性分组
const agreeSet = pairs.filter(p => p.labelAgreement);
const disagreeSet = pairs.filter(p => !p.labelAgreement);
console.log('\n=== 按标签一致性分组 ===');
console.log(`标签一致 (n=${agreeSet.length}):      oldMean=${avg(agreeSet.map(p=>p.scoreOld)).toFixed(4)} newMean=${avg(agreeSet.map(p=>p.scoreDiscounted)).toFixed(4)}`);
console.log(`标签不一致 (n=${disagreeSet.length}): oldMean=${avg(disagreeSet.map(p=>p.scoreOld)).toFixed(4)} newMean=${avg(disagreeSet.map(p=>p.scoreDiscounted)).toFixed(4)}`);

// 按 MC uncertainty 分组
console.log('\n=== 按 MC 不确定性 × 标签方案 ===');
for (const level of ['low', 'medium', 'high']) {
  const subset = pairs.filter(p => p.mcUncertainty === level);
  if (subset.length === 0) continue;
  console.log(`${level}: n=${subset.length} oldMean=${avg(subset.map(p=>p.scoreOld)).toFixed(4)} newMean=${avg(subset.map(p=>p.scoreDiscounted)).toFixed(4)}  Δ=${(avg(subset.map(p=>p.scoreDiscounted))-avg(subset.map(p=>p.scoreOld))).toFixed(4)}`);
}

// 按模板分组
console.log('\n=== 按模板 × 标签方案 ===');
const tmpls = [...new Set(pairs.map(p => p.template))];
for (const tpl of tmpls) {
  const subset = pairs.filter(p => p.template === tpl);
  console.log(`${tpl.padEnd(12)} n=${String(subset.length).padStart(4)} oldMean=${avg(subset.map(p=>p.scoreOld)).toFixed(4)} newMean=${avg(subset.map(p=>p.scoreDiscounted)).toFixed(4)}`);
}

// 得分相关性
const n = oldScores.length;
const oldMean = avg(oldScores), newMean = avg(newScores);
let cov = 0, varO = 0, varN = 0;
for (let i = 0; i < n; i++) {
  cov += (oldScores[i] - oldMean) * (newScores[i] - newMean);
  varO += (oldScores[i] - oldMean) ** 2;
  varN += (newScores[i] - newMean) ** 2;
}
const corr = cov / Math.sqrt(varO * varN);
console.log(`\n旧得分 vs 新得分 Pearson r: ${corr.toFixed(4)}`);
console.log(`标签一致率: ${pairs.filter(p => p.labelAgreement).length}/${n} (${(100* pairs.filter(p=>p.labelAgreement).length / n).toFixed(1)}%)`);
