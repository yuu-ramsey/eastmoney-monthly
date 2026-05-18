// Phase 15 诊断：抽 5 样本读完整 agent 输出
import * as fs from 'node:fs';

const files = fs.readdirSync('.eastmoney-ai/eval/runs').filter(f => f.startsWith('phase15-'));
const file = '.eastmoney-ai/eval/runs/' + files[files.length - 1];
const records = fs.readFileSync(file, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
console.log(`Total: ${records.length} records`);

// 选 5 个样本：strong GT 2-3 个 + bull/bear GT 2-3 个
const strongSamples = records.filter(r => r.groundTruth === 'strong_bull' || r.groundTruth === 'strong_bear');
const normalSamples = records.filter(r => r.groundTruth === 'bull' || r.groundTruth === 'bear');

// 每个 sample 选不同 stockCode + template 的第一条
function pickDiverse(arr, n) {
  const seen = new Set();
  const picked = [];
  for (const r of arr) {
    const key = r.stockCode;
    if (!seen.has(key) && r.judgeText) {
      seen.add(key);
      picked.push(r);
      if (picked.length >= n) break;
    }
  }
  return picked;
}

const samples = [...pickDiverse(strongSamples, 2), ...pickDiverse(normalSamples, 3)];

for (let i = 0; i < samples.length; i++) {
  const r = samples[i];
  console.log('\n' + '='.repeat(80));
  console.log(`SAMPLE ${i + 1}: ${r.stockCode} ${r.template} | gt=${r.groundTruth} | Judge signal=${r.predictedSignal} | score=${r.score}`);
  console.log('='.repeat(80));

  // Each agent output
  const agents = [
    ['Bull Researcher', r.bullText],
    ['Bear Researcher', r.bearText],
    ['Technical Agent', r.techText],
    ['Sector Agent', r.sectorText],
  ];

  for (const [name, text] of agents) {
    console.log(`\n--- ${name} ---`);
    if (!text) { console.log('  [MISSING]'); continue; }
    // 截取前 400 字
    console.log(text.substring(0, 400).replace(/\n/g, '\n  '));
    if (text.length > 400) console.log('  ... (truncated, total ' + text.length + ' chars)');
  }

  // Judge output
  console.log(`\n--- Judge ---`);
  if (!r.judgeText) { console.log('  [MISSING]'); continue; }
  console.log(r.judgeText.substring(0, 600).replace(/\n/g, '\n  '));

  // Quick analysis: which agent is Judge most aligned with?
  const judgeSignal = r.predictedSignal;
  const bullMentions = (r.judgeText?.match(/Bull|bull|看多/g) || []).length;
  const bearMentions = (r.judgeText?.match(/Bear|bear|看空/g) || []).length;
  const techMentions = (r.judgeText?.match(/Technical|技术/g) || []).length;
  const sectorMentions = (r.judgeText?.match(/Sector|行业|sector|alpha/g) || []).length;

  console.log(`\n--- Agent 引用统计 ---`);
  console.log(`Judge 引用 Bull: ${bullMentions}x | Bear: ${bearMentions}x | Technical: ${techMentions}x | Sector: ${sectorMentions}x`);

  // Agent independence check: compare Bull first sentence vs Bear first sentence
  const bullFirst = (r.bullText || '').split('\n').find(l => l.trim().startsWith('###')) || '';
  const bearFirst = (r.bearText || '').split('\n').find(l => l.trim().startsWith('###')) || '';
  console.log(`Bull 首标题: ${bullFirst.trim()}`);
  console.log(`Bear 首标题: ${bearFirst.trim()}`);
}

// Summary
console.log('\n' + '='.repeat(80));
console.log('SUMMARY');
console.log('='.repeat(80));
const allBull = records.filter(r => r.bullText).length;
const allBear = records.filter(r => r.bearText).length;
const allTech = records.filter(r => r.techText).length;
const allSector = records.filter(r => r.sectorText).length;
console.log(`Agent 成功率: Bull=${allBull}/48 Bear=${allBear}/48 Tech=${allTech}/48 Sector=${allSector}/48`);

// Agent输出平均长度
const avgLen = (arr) => arr.reduce((s, r) => s + (r?.length || 0), 0) / Math.max(1, arr.length);
console.log(`平均长度: Bull=${Math.round(avgLen(records.map(r=>r.bullText)))} Bear=${Math.round(avgLen(records.map(r=>r.bearText)))} Tech=${Math.round(avgLen(records.map(r=>r.techText)))} Sector=${Math.round(avgLen(records.map(r=>r.sectorText)))} Judge=${Math.round(avgLen(records.map(r=>r.judgeText)))}`);
