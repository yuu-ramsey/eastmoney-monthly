// Extract frozen eval dataset from Phase 12 Run A
import * as fs from 'node:fs';

const records = fs.readFileSync('.eastmoney-ai/eval/runs/runA-no-sector-2026-05-18-05-12-20.jsonl', 'utf-8')
  .trim().split('\n').filter(Boolean).map(l => JSON.parse(l));

// Extract stocks
const stocks = new Map();
for (const r of records) {
  if (!stocks.has(r.stockCode)) {
    stocks.set(r.stockCode, {
      code: r.stockCode,
      name: r.stockName || r.stockCode,
      market: r.stockCode.startsWith('6') ? '1' : '0',
      category: r.category || 'hs300',
    });
  }
}

// Extract testPoints (unique code+cutoff)
const tpMap = new Map();
for (const r of records) {
  const key = r.stockCode + '_' + r.cutoffDate;
  if (!tpMap.has(key)) {
    tpMap.set(key, {
      id: r.testPointId || ('tp_' + r.stockCode + '_' + r.cutoffDate.replace(/-/g, '')),
      stockCode: r.stockCode,
      cutoffDate: r.cutoffDate,
      cutoffIndex: r.cutoffIndex || 0,
      groundTruth: r.groundTruth,
      alpha: r.alpha || 0,
    });
  }
}

const stockList = [...stocks.values()].sort((a, b) => a.code.localeCompare(b.code));
const testPoints = [...tpMap.values()];

const gtDist = {};
for (const tp of testPoints) gtDist[tp.groundTruth] = (gtDist[tp.groundTruth] || 0) + 1;

const dataset = {
  version: 'frozen-v1',
  createdAt: new Date().toISOString(),
  baseline: {
    runId: 'runA-no-sector-2026-05-18-05-12-20',
    score: 0.1966, n: records.length,
    config: 'no sector, no resonance, old #12 constraint, deepseek-chat, maxTokens=4000',
  },
  stocks: stockList,
  testPoints,
  templates: ['technical', 'trend', 'valuation', 'sentiment'],
};

fs.mkdirSync('data', { recursive: true });
fs.writeFileSync('data/frozen-eval-dataset-v1.json', JSON.stringify(dataset, null, 2), 'utf-8');

console.log(`Frozen dataset: ${stockList.length} stocks, ${testPoints.length} testPoints, ${testPoints.length * 4} calls`);
console.log(`GT: ${JSON.stringify(gtDist)}`);
console.log(`Baseline score: 0.1966`);
