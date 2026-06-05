// Self-backtest decision log
// Let AI see its own judgments on this stock at historical time points + actual returns + alpha,
// and calibrate current confidence accordingly
//
// 3 exported functions:
//   runHistoricalAnalysis()  — invoke LLM at historical cutoff points for judgments
//   calculateActualReturn() — pure numeric computation of actual return + alpha
//   buildSelfCalibrationBlock() — generate markdown calibration block

import { buildPromptByTemplate } from './build-prompt.js';
import { computeMA } from './compute-ma.js';
import { computeMACD } from './compute-macd.js';
import { getProvider } from './llm/index.js';

// Fixed lightweight model for backtest (cost saving)
const BACKTEST_MODEL = {
  anthropic: 'claude-sonnet-4-6',
  deepseek: 'deepseek-chat',
};

const BACKTEST_CACHE_PREFIX = 'backtest';
const BACKTEST_CACHE_TTL_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

// ---- Judgment extraction ----

// Extract directional judgment and key price levels from LLM output
function extractJudgment(text) {
  const judgmentPatterns = [
    { pattern: /bullish/i, judgment: 'bullish' },
    { pattern: /偏空/, judgment: '偏空' },
    { pattern: /中性震荡/, judgment: '中性震荡' },
    { pattern: /中性/, judgment: '中性' },
  ];

  let judgment = '无法判断';
  for (const { pattern, judgment: j } of judgmentPatterns) {
    if (pattern.test(text)) {
      judgment = j;
      break;
    }
  }

  // Extract key price levels (XX.XX format, take first 3)
  const pricePattern = /(\d{2,4}\.\d{2})/g;
  const prices = [...text.matchAll(pricePattern)].map((m) => parseFloat(m[1]));
  const keyLevels = [...new Set(prices)].slice(0, 3);

  return { judgment, keyLevels };
}

// ---- Backtest judgment cache ----

function backtestCacheKey(code, template, cutoffIndex, providerId) {
  return `${BACKTEST_CACHE_PREFIX}:${code}:${template}:${cutoffIndex}:${providerId}`;
}

/**
 * Backtest cache read (storage injected by caller)
 * @param {object} storage — implements get([keys]) interface
 * @param {string} key
 */
async function getBacktestCache(storage, key) {
  try {
    const items = await storage.get([key]);
    const record = items[key];
    if (record && (Date.now() - record.timestamp) < BACKTEST_CACHE_TTL_MS) {
      return record.data;
    }
  } catch (_) { /* ignore */ }
  return null;
}

/**
 * Backtest cache write (storage injected by caller)
 * @param {object} storage — implements set(items) interface
 * @param {string} key
 * @param {*} data
 */
async function setBacktestCache(storage, key, data) {
  try {
    await storage.set({ [key]: { data, timestamp: Date.now() } });
  } catch (_) { /* ignore */ }
}

// ---- Main functions ----

/**
 * Run analysis at historical cutoff point, return LLM judgment
 * @param {Array} klines - full kline array (with MA/MACD)
 * @param {number} cutoffIndex - cutoff point (exclusive), uses klines.slice(0, cutoffIndex)
 * @param {string} template - analysis dimension key
 * @param {string} providerId - 'anthropic' | 'deepseek'
 * @param {object} settings - { apiKey, template, ... }
 * @returns {{ date: string, judgment: string, keyLevels: number[], rawText: string }}
 */
export async function runHistoricalAnalysis(klines, cutoffIndex, template, providerId, settings) {
  if (cutoffIndex < 12) {
    throw new Error(`Backtest data insufficient: cutoffIndex=${cutoffIndex}, need at least 12 bars`);
  }

  const subKlines = klines.slice(0, cutoffIndex);
  // Recompute MA/MACD (indicators must be recalculated after truncation)
  const closes = subKlines.map((k) => k.close);
  const ma5 = computeMA(closes, 5);
  const ma20 = computeMA(closes, 20);
  const ma60 = computeMA(closes, 60);
  const { dif, dea, hist } = computeMACD(closes);
  const klinesWithIndicators = subKlines.map((k, i) => ({
    ...k, ma5: ma5[i], ma20: ma20[i], ma60: ma60[i], dif: dif[i], dea: dea[i], hist: hist[i],
  }));

  const date = subKlines[subKlines.length - 1].date;

  // Build prompt (no tool_use, no market benchmark, save tokens)
  const prompt = await buildPromptByTemplate({
    templateKey: template,
    name: settings.name || '?',
    code: settings.code || '?',
    klines: klinesWithIndicators,
    period: 'monthly',
    provider: providerId,
    decisionMode: false,
    // no market → no tool_use directive, backtest doesn't need it
  });

  // Call LLM (force lightweight model)
  const model = BACKTEST_MODEL[providerId] || 'claude-sonnet-4-6';
  const provider = getProvider(providerId);

  const result = await provider.call(prompt, {
    model,
    apiKey: settings.apiKey,
    maxTokens: 2000, // Backtest uses fewer tokens, 2000 is enough for judgment
  });

  const { judgment, keyLevels } = extractJudgment(result.text);

  return { date, judgment, keyLevels, rawText: result.text };
}

/**
 * Calculate actual return from fromIndex to toIndex + CSI300 alpha
 * @param {Array} klines - full kline array (only close + date needed)
 * @param {number} fromIndex
 * @param {number} toIndex
 * @param {Array|null} indexKlines - CSI300 kline array
 * @returns {{ fromDate: string, toDate: string, stockReturn: number, indexReturn: number|null, alpha: number|null }}
 */
export function calculateActualReturn(klines, fromIndex, toIndex, indexKlines) {
  if (fromIndex < 0 || toIndex >= klines.length || fromIndex >= toIndex) {
    throw new Error(`Invalid index: fromIndex=${fromIndex} toIndex=${toIndex} klines.length=${klines.length}`);
  }

  const fromClose = klines[fromIndex].close;
  const toClose = klines[toIndex].close;
  const stockReturn = ((toClose - fromClose) / fromClose) * 100;

  const fromDate = klines[fromIndex].date;
  const toDate = klines[toIndex].date;

  let indexReturn = null;
  let alpha = null;

  if (indexKlines && indexKlines.length > 0) {
    // Find closest CSI300 bar in the same period
    const idxFrom = findClosestIndex(indexKlines, fromDate);
    const idxTo = findClosestIndex(indexKlines, toDate);
    if (idxFrom >= 0 && idxTo > idxFrom) {
      const idxFromClose = indexKlines[idxFrom].close;
      const idxToClose = indexKlines[idxTo].close;
      indexReturn = ((idxToClose - idxFromClose) / idxFromClose) * 100;
      alpha = stockReturn - indexReturn;
    }
  }

  return { fromDate, toDate, stockReturn, indexReturn, alpha };
}

// Find closest bar index in indexKlines for targetDate
function findClosestIndex(indexKlines, targetDate) {
  const target = String(targetDate).slice(0, 7); // YYYY-MM
  for (let i = indexKlines.length - 1; i >= 0; i--) {
    if (String(indexKlines[i].date).slice(0, 7) <= target) return i;
  }
  return 0;
}

/**
 * Generate "historical self-calibration" markdown block
 * @param {Array} results - [{ date, judgment, keyLevels, actualReturn }, ...]
 * @returns {string}
 */
export function buildSelfCalibrationBlock(results) {
  if (!results || results.length === 0) return '';

  const lines = [
    '## 历史自我校准',
    '',
    '以下是模型在该股票历史时点的判断与实际后续走势对比：',
    '',
  ];

  for (const r of results) {
    const stockDir = r.actualReturn.stockReturn >= 0 ? '涨' : '跌';
    const stockAbs = Math.abs(r.actualReturn.stockReturn).toFixed(2);
    const alphaStr = r.actualReturn.alpha != null
      ? `alpha = ${r.actualReturn.alpha >= 0 ? '+' : ''}${r.actualReturn.alpha.toFixed(2)}%`
      : '（无大盘对照数据）';
    const idxStr = r.actualReturn.indexReturn != null
      ? `沪深300 同期 ${r.actualReturn.indexReturn >= 0 ? '+' : ''}${r.actualReturn.indexReturn.toFixed(2)}%`
      : '';

    lines.push(`- **${r.date}**：判断【${r.judgment}】`);
    if (r.keyLevels && r.keyLevels.length > 0) {
      lines.push(`  关键价位：${r.keyLevels.map((p) => p.toFixed(2)).join(' / ')}`);
    }
    lines.push(`  → 后续至 ${r.actualReturn.toDate} 实际：${stockDir} ${stockAbs}%，${idxStr}，${alphaStr}`);
    lines.push('');
  }

  lines.push('请综合上述自我校准结果，调整本次判断的置信度。');
  lines.push('注意：模型在历史时点的判断也可能存在偏差，以上信息仅供校准参考，不作为绝对依据。');

  return lines.join('\n');
}

// Export cache functions for background.js use
export { getBacktestCache, setBacktestCache, backtestCacheKey };
