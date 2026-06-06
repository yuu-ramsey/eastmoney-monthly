// Step 2: rebuild from v4 jsonl dataset-v6.json
// Directly reuse v4 groundTruth/stockReturn/alpha,
// fill cutoffIndex from local SQLite
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const EVAL_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval');
const RUNS_DIR = path.join(EVAL_DIR, 'runs');

const v4Path = path.join(RUNS_DIR, 'v4-signals-2026-05-17-00-41.jsonl');
const v4Lines = fs.readFileSync(v4Path, 'utf-8').trim().split('\n').filter(Boolean);
const v4Records = v4Lines.map(l => JSON.parse(l));

// Extract unique stocks (code → name)
const stockMap = new Map();
for (const r of v4Records) {
  const code = r.code || r.stockCode;
  if (!stockMap.has(code)) {
    const nameMatch = (r.promptUsed || '').match(/以下是\s*(.+?)\(/);
    stockMap.set(code, nameMatch ? nameMatch[1].trim() : code);
  }
}
console.log(`Unique stocks: ${stockMap.size}`);

// Extract unique testPoints (dedup: code + cutoffDate unique)
const tpMap = new Map();
for (const r of v4Records) {
  const code = r.code || r.stockCode;
  const key = `${code}_${r.cutoffDate}`;
  if (!tpMap.has(key)) {
    tpMap.set(key, {
      stockCode: code,
      stockName: stockMap.get(code),
      cutoffDate: r.cutoffDate,
      groundTruth: r.groundTruth,
    });
  }
}
console.log(`Unique testPoints: ${tpMap.size}`);

// Connect SQLite to fill cutoffIndex
const { getDb } = await import('../lib/db/connection.js');
const db = getDb();

function getMonthlyKlines(code) {
  return db.prepare('SELECT date, close FROM monthly_klines WHERE code=? ORDER BY date').all(code);
}

function findCutoffIndex(klines, cutoffDate) {
  for (let i = 0; i < klines.length; i++) {
    if (String(klines[i].date).trim().startsWith(cutoffDate)) return i;
  }
  return -1;
}

const testPoints = [];
let skipped = 0;

for (const [key, tp] of tpMap) {
  const klines = getMonthlyKlines(tp.stockCode);
  if (klines.length < 24) { skipped++; continue; }

  const idx = findCutoffIndex(klines, tp.cutoffDate);
  if (idx < 0) { skipped++; continue; }
  if (idx < 12) { skipped++; continue; }

  testPoints.push({
    id: `tp_${tp.stockCode}_${tp.cutoffDate.replace(/-/g, '')}`,
    stockCode: tp.stockCode,
    stockName: tp.stockName,
    category: 'hs300',
    industry: 'unknown',
    cutoffDate: tp.cutoffDate,
    cutoffIndex: idx,
    evaluationHorizonMonths: 6,
    groundTruth: tp.groundTruth,
    // actualReturn/alpha computed live by runner in v4 eval, extractable from jsonl
    actualReturn: 0,  // placeholder, runner will recalculate
    indexReturn: 0,
    alpha: 0,
  });
}

console.log(`TestPoints built: ${testPoints.length} (skipped ${skipped})`);

// Fill actualReturn/alpha from v4 jsonl (take first record value for this code+cutoffDate)
for (const tp of testPoints) {
  const rec = v4Records.find(r =>
    (r.code || r.stockCode) === tp.stockCode && r.cutoffDate === tp.cutoffDate
  );
  if (rec) {
    tp.actualReturn = rec.stockReturn ?? rec.actualReturn ?? 0;
    tp.indexReturn = rec.indexReturn ?? 0;
    tp.alpha = rec.alpha ?? (rec.stockReturn ?? 0) - (rec.indexReturn ?? 0);
  }
}

// GT distribution
const gtDist = {};
testPoints.forEach(tp => { gtDist[tp.groundTruth] = (gtDist[tp.groundTruth] || 0) + 1; });
console.log(`GT distribution: ${JSON.stringify(gtDist)}`);
console.log(`LLM calls: ${testPoints.length * 4}`);

// Write output
const stocks = [...stockMap.entries()].map(([code, name]) => ({
  code,
  market: code.startsWith('6') ? '1' : '0',
  name,
  category: 'hs300',
  industry: 'unknown',
}));

const datasetOut = {
  version: 'v6',
  createdAt: new Date().toISOString(),
  rebuildNote: 'Extracted 40 stocks+groundTruth from v4-signals-2026-05-17-00-41.jsonl, filled cutoffIndex from local SQLite',
  stocks,
  testPoints,
};

const outPath = path.join(EVAL_DIR, 'dataset-v6.json');
fs.writeFileSync(outPath, JSON.stringify(datasetOut, null, 2), 'utf-8');
console.log(`\nWritten to: ${outPath}`);
console.log(`stocks=${stocks.length}, testPoints=${testPoints.length}, calls=${testPoints.length * 4}`);
