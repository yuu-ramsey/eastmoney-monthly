import * as fs from 'node:fs';

const files = fs.readdirSync('.eastmoney-ai/eval/runs').filter(f => f.startsWith('phase15-'));
const file = '.eastmoney-ai/eval/runs/' + files[files.length - 1];
const records = fs.readFileSync(file, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));

// Signal distribution
const sigDist = {};
for (const r of records) { const s = r.predictedSignal || '?'; sigDist[s] = (sigDist[s] || 0) + 1; }
console.log('Signal distribution:', JSON.stringify(sigDist));

// 抽样 3 条
for (let i = 0; i < Math.min(3, records.length); i++) {
  const r = records[i];
  console.log(`\n=== ${r.stockCode} ${r.template} signal=${r.predictedSignal} gt=${r.groundTruth} score=${r.score} ===`);
  const jt = r.judgeText || '';
  const m = jt.match(/```json\s*([\s\S]*?)```/);
  if (m) console.log('Judge JSON:', m[1].trim().substring(0, 150));
  const dirIdx = jt.indexOf('综合方向判断');
  if (dirIdx >= 0) console.log('方向判断:', jt.substring(dirIdx, dirIdx + 200).replace(/\n/g, ' '));
}
