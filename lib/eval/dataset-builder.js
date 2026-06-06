// Evaluation dataset auto-generation
// buildDataset() — fetch klines → pick cutoff → compute actual return → label groundTruth
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const EVAL_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'eval');

function ensureDir() { if (!fs.existsSync(EVAL_DIR)) fs.mkdirSync(EVAL_DIR, { recursive: true }); }

// groundTruth classification
const GT_RULES = [
  { minAlpha: 10, label: 'strong_bull' },
  { minAlpha: 3, label: 'bull' },
  { minAlpha: -3, label: 'neutral' },
  { minAlpha: -10, label: 'bear' },
  { minAlpha: -Infinity, label: 'strong_bear' },
];

function getGroundTruth(alpha) {
  for (const rule of GT_RULES) {
    if (alpha >= rule.minAlpha) return rule.label;
  }
  return 'strong_bear';
}

/**
 * @param {Array} stockList — [{code, market, name, category, industry}]
 * @param {object} options
 * @param {Function} options.fetchKlines — (market, code, period, limit) => {klines}
 * @param {Function} options.fetchIndexKlines — (period, limit) => {klines}
 * @param {number} [options.testPointsPerStock=4]
 * @param {number} [options.minDateGap=90]
 * @param {number} [options.earliestMonthsAgo=24]
 * @param {number} [options.evaluationHorizonMonths=6]
 */
export async function buildDataset(stockList, options = {}) {
  const {
    fetchKlines,
    fetchIndexKlines,
    testPointsPerStock = 4,
    minDateGap = 90,
    earliestMonthsAgo = 24,
    evaluationHorizonMonths = 6,
  } = options;

  const testPoints = [];
  const allKlines = {};
  const indexCache = {};

  // Pre-fetch CSI300 monthly klines (once)
  try {
    const idxRaw = await fetchIndexKlines('monthly', 200);
    indexCache.klines = idxRaw.klines || idxRaw;
  } catch (_) {
    indexCache.klines = [];
  }

  for (const stock of stockList) {
    // Fetch full monthly klines
    let raw;
    try {
      raw = await fetchKlines(stock.market, stock.code, 'monthly', 200);
    } catch (_) {
      continue;
    }
    const klines = raw.klines || raw;
    if (!Array.isArray(klines) || klines.length < 24) continue;

    allKlines[stock.code] = klines;

    // Available cutoff range: from earliestMonthsAgo to before horizon
    const now = new Date();
    const earliestDate = new Date(now);
    earliestDate.setMonth(earliestDate.getMonth() - earliestMonthsAgo);
    const latestDate = new Date(now);
    latestDate.setMonth(latestDate.getMonth() - evaluationHorizonMonths);

    // Find indices in klines between earliestDate ~ latestDate
    const candidates = [];
    for (let i = 0; i < klines.length; i++) {
      const d = parseDate(klines[i].date);
      if (!d) continue;
      if (d >= earliestDate && d <= latestDate) {
        candidates.push(i);
      }
    }

    if (candidates.length < 2) continue;

    // Uniformly sample testPointsPerStock points, ensure gap >= minDateGap days
    const selected = [];
    let step = Math.max(1, Math.floor(candidates.length / testPointsPerStock));
    for (let j = 0; j < candidates.length && selected.length < testPointsPerStock; j += step) {
      // Check gap from previous
      const idx = candidates[j];
      const kd = parseDate(klines[idx].date);
      if (selected.length > 0) {
        const prevDate = parseDate(klines[selected[selected.length - 1]].date);
        if (kd && prevDate && (kd - prevDate) / 86400000 < minDateGap) continue;
      }
      selected.push(idx);
    }

    for (const idx of selected) {
      const kd = parseDate(klines[idx].date);
      if (!kd) continue;
      const cutoffDate = kd.toISOString().slice(0, 10);

      // Compute actual return after horizon
      const horizonEnd = new Date(kd);
      horizonEnd.setMonth(horizonEnd.getMonth() + evaluationHorizonMonths);

      let endIdx = -1;
      for (let j = idx + 1; j < klines.length; j++) {
        const d = parseDate(klines[j].date);
        if (d && d >= horizonEnd) { endIdx = j; break; }
      }
      if (endIdx < 0) endIdx = klines.length - 1;

      const fromClose = klines[idx].close;
      const toClose = klines[endIdx].close;
      if (!fromClose || !toClose || fromClose <= 0) continue;
      const stockReturn = +((toClose - fromClose) / fromClose * 100).toFixed(2);

      // CSI300 alpha
      let indexReturn = 0;
      if (indexCache.klines.length > 0) {
        let idxFrom = -1, idxTo = -1;
        for (let j = 0; j < indexCache.klines.length; j++) {
          const d = parseDate(indexCache.klines[j].date);
          if (d && d <= kd && (idxFrom < 0 || d >= parseDate(indexCache.klines[idxFrom].date))) idxFrom = j;
        }
        for (let j = 0; j < indexCache.klines.length; j++) {
          const d = parseDate(indexCache.klines[j].date);
          if (d && d >= horizonEnd) { idxTo = j; break; }
        }
        if (idxTo < 0) idxTo = indexCache.klines.length - 1;
        if (idxFrom >= 0 && idxTo > idxFrom) {
          const iFrom = indexCache.klines[idxFrom].close;
          const iTo = indexCache.klines[idxTo].close;
          if (iFrom && iTo && iFrom > 0) indexReturn = +((iTo - iFrom) / iFrom * 100).toFixed(2);
        }
      }

      const alpha = +(stockReturn - indexReturn).toFixed(2);
      const groundTruth = getGroundTruth(alpha);

      testPoints.push({
        id: `tp_${stock.code}_${idx}`,
        stockCode: stock.code,
        stockName: stock.name,
        category: stock.category,
        industry: stock.industry,
        cutoffDate,
        cutoffIndex: idx,
        evaluationHorizonMonths,
        actualReturn: stockReturn,
        indexReturn,
        alpha,
        groundTruth,
      });
    }
  }

  const dataset = {
    version: '1.0',
    createdAt: new Date().toISOString().slice(0, 10),
    stocks: stockList.map((s) => ({ code: s.code, market: s.market, name: s.name, category: s.category, industry: s.industry })),
    testPoints,
  };

  ensureDir();
  const outPath = path.join(EVAL_DIR, 'dataset.json');
  fs.writeFileSync(outPath, JSON.stringify(dataset, null, 2), 'utf-8');
  return { path: outPath, stockCount: stockList.length, testPointCount: testPoints.length };
}

function parseDate(dateStr) {
  if (!dateStr) return null;
  const s = String(dateStr).slice(0, 10);
  const d = new Date(s + 'T00:00:00');
  return isNaN(d.getTime()) ? null : d;
}
