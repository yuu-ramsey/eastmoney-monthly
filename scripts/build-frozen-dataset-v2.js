// 构建 v2 frozen eval dataset —— 从 v1 精确匹配 cutoff dates，添加折扣多月回报标签
// 用法：
//   node scripts/build-frozen-dataset-v2.js                  (全量 298 股，自动生成 cutoff)
//   node scripts/build-frozen-dataset-v2.js --match-v1       (精确使用 v1 的 stockCode+cutoffDate)
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { getDb } from '../lib/db/connection.js';
import { computeDiscountedReturn, discretizeDiscountedReturn } from '../lib/eval/discounted-return.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const DATA_DIR = path.join(PROJECT_DIR, 'data');

const GAMMA = 0.9;
const MONTHS = 3;
const MIN_KLINES = 36;
const MIN_FUTURE = MONTHS;

function discretizeAlpha(alpha) {
  if (alpha >= 20) return 'strong_bull';
  if (alpha >= 10) return 'bull';
  if (alpha > -10) return 'neutral';
  if (alpha > -20) return 'bear';
  return 'strong_bear';
}

const matchV1 = process.argv.includes('--match-v1');
const db = getDb();

if (matchV1) {
  // ====== 精确匹配 v1 模式：复用 v1 的 stockCode + cutoffDate ======
  const v1Path = path.join(DATA_DIR, 'frozen-eval-dataset-v1.json');
  if (!fs.existsSync(v1Path)) {
    console.error('v1 dataset 不存在');
    process.exit(1);
  }
  const v1 = JSON.parse(fs.readFileSync(v1Path, 'utf-8'));

  const v1Codes = [...new Set(v1.testPoints.map(tp => tp.stockCode))];
  const allKlines = db.prepare(`
    SELECT code, date, close FROM monthly_klines
    WHERE code IN (${v1Codes.map(() => '?').join(',')})
    ORDER BY code, date
  `).all(...v1Codes);

  const klinesByCode = new Map();
  for (const row of allKlines) {
    if (!klinesByCode.has(row.code)) klinesByCode.set(row.code, []);
    klinesByCode.get(row.code).push(row);
  }

  // 构建 date → index 查找
  const dateIndexCache = new Map();
  for (const [code, klines] of klinesByCode) {
    const dateMap = new Map();
    klines.forEach((k, i) => dateMap.set(k.date, i));
    dateIndexCache.set(code, dateMap);
  }

  const testPoints = [];
  let skipped = 0;

  for (const tp of v1.testPoints) {
    const klines = klinesByCode.get(tp.stockCode);
    const dateMap = dateIndexCache.get(tp.stockCode);
    if (!klines || !dateMap) { skipped++; continue; }

    const idx = dateMap.get(tp.cutoffDate);
    if (idx === undefined || idx < 12) { skipped++; continue; }
    if (idx + MIN_FUTURE >= klines.length) { skipped++; continue; }

    const currentClose = klines[idx].close;
    const futureClose = klines[idx + 1]?.close;
    const alpha = futureClose ? ((futureClose - currentClose) / currentClose) * 100 : 0;
    const discounted = computeDiscountedReturn(klines, idx, { gamma: GAMMA, months: MONTHS });

    testPoints.push({
      ...tp,
      groundTruthDiscounted: discretizeDiscountedReturn(discounted.discountedReturn),
      discountedReturn: discounted.discountedReturn,
      discountMonthsUsed: discounted.monthsAvailable,
      individualReturns: discounted.individualReturns,
      alpha,
      labelAgreement: tp.groundTruth === discretizeDiscountedReturn(discounted.discountedReturn),
    });
  }

  console.log(`v1 testPoints: ${v1.testPoints.length}, 匹配成功: ${testPoints.length}, 跳过: ${skipped}`);

  // 标签分布对比
  function labelDist(points, key) {
    const dist = {};
    for (const tp of points) dist[tp[key]] = (dist[tp[key]] || 0) + 1;
    return dist;
  }

  console.log('\n=== 标签分布对比 ===');
  console.log('旧(单月alpha):', JSON.stringify(labelDist(testPoints, 'groundTruth')));
  console.log('新(折扣回报): ', JSON.stringify(labelDist(testPoints, 'groundTruthDiscounted')));
  const agreeCount = testPoints.filter(tp => tp.labelAgreement).length;
  console.log(`标签一致率: ${agreeCount}/${testPoints.length} (${(agreeCount / testPoints.length * 100).toFixed(1)}%)`);

  const alphas = testPoints.map(tp => tp.alpha ?? 0);
  const discounted = testPoints.map(tp => tp.discountedReturn);
  const alphaRanks = [...Array(alphas.length).keys()].sort((a, b) => alphas[a] - alphas[b]);
  const discRanks = [...Array(discounted.length).keys()].sort((a, b) => discounted[a] - discounted[b]);
  const rankDiffs = alphaRanks.map(r => (r - discRanks.indexOf(r)) ** 2);
  const n = testPoints.length;
  const spearman = 1 - (6 * rankDiffs.reduce((a, b) => a + b, 0)) / (n * (n * n - 1));
  console.log(`单月alpha vs 折扣回报 Spearman rank: ${spearman.toFixed(4)}`);

  const dataset = {
    version: 'frozen-v2',
    createdAt: new Date().toISOString(),
    config: { gamma: GAMMA, discountMonths: MONTHS, matchedFromV1: true },
    stocks: v1.stocks,
    testPoints,
    templates: v1.templates,
  };

  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.writeFileSync(path.join(DATA_DIR, 'frozen-eval-dataset-v2.json'), JSON.stringify(dataset, null, 2), 'utf-8');
  console.log(`\n保存: data/frozen-eval-dataset-v2.json (${dataset.stocks.length} stocks, ${testPoints.length} testPoints)`);

} else {
  // ====== 全量模式：自动生成 cutoff dates ======
  const stocks = db.prepare(`
    SELECT DISTINCT m.code, COALESCE(s.stock_name, m.code) AS name
    FROM monthly_klines m
    LEFT JOIN stock_industry_mapping s ON m.code = s.stock_code
    WHERE m.code IN (SELECT stock_code FROM stock_industry_mapping)
    ORDER BY m.code
  `).all();
  console.log(`股票数: ${stocks.length}`);

  const allKlines = db.prepare(`
    SELECT code, date, close FROM monthly_klines
    WHERE code IN (${stocks.map(() => '?').join(',')})
    ORDER BY code, date
  `).all(...stocks.map(s => s.code));

  const klinesByCode = new Map();
  for (const row of allKlines) {
    if (!klinesByCode.has(row.code)) klinesByCode.set(row.code, []);
    klinesByCode.get(row.code).push(row);
  }

  const testPoints = [];
  let skippedShort = 0;

  for (const stock of stocks) {
    const klines = klinesByCode.get(stock.code) || [];
    if (klines.length < MIN_KLINES) { skippedShort++; continue; }

    for (let i = 24; i < klines.length - MIN_FUTURE; i += 4) {
      const currentClose = klines[i].close;
      const nextClose = klines[i + 1].close;
      const alpha = ((nextClose - currentClose) / currentClose) * 100;
      const discounted = computeDiscountedReturn(klines, i, { gamma: GAMMA, months: MONTHS });

      testPoints.push({
        id: `tp_${stock.code}_${klines[i].date}`,
        stockCode: stock.code,
        stockName: stock.name || stock.code,
        cutoffDate: klines[i].date,
        cutoffIndex: i,
        groundTruth: discretizeAlpha(alpha),
        alpha,
        groundTruthDiscounted: discretizeDiscountedReturn(discounted.discountedReturn),
        discountedReturn: discounted.discountedReturn,
        discountMonthsUsed: discounted.monthsAvailable,
        individualReturns: discounted.individualReturns,
        labelAgreement: discretizeAlpha(alpha) === discretizeDiscountedReturn(discounted.discountedReturn),
      });
    }
  }

  console.log(`跳过低数据股票: ${skippedShort}`);
  console.log(`Test points: ${testPoints.length}`);

  const activeCodes = new Set(testPoints.map(tp => tp.stockCode));
  const stockMeta = stocks
    .filter(s => activeCodes.has(s.code))
    .map(s => ({
      code: s.code,
      name: s.name || s.code,
      market: s.code.startsWith('6') ? '1' : '0',
      category: 'hs300',
    }));

  const dataset = {
    version: 'frozen-v2',
    createdAt: new Date().toISOString(),
    config: { gamma: GAMMA, discountMonths: MONTHS, minKlines: MIN_KLINES, minFuture: MIN_FUTURE, stepMonths: 4 },
    stocks: stockMeta.sort((a, b) => a.code.localeCompare(b.code)),
    testPoints,
    templates: ['technical', 'trend', 'valuation', 'sentiment'],
  };

  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.writeFileSync(path.join(DATA_DIR, 'frozen-eval-dataset-v2.json'), JSON.stringify(dataset, null, 2), 'utf-8');
  console.log(`保存: data/frozen-eval-dataset-v2.json (${stockMeta.length} stocks, ${testPoints.length} testPoints)`);
}
