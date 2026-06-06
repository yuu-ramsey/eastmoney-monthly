// Multi-period directional vectorization — returns 'bull' | 'bear' | 'neutral' per period
import { calculateAll } from '../indicators/calculate.js';
import config from './config.json' with { type: 'json' };

/** Linear regression slope (last n periods) */
function slope(series, n) {
  const valid = series.filter(v => v != null);
  if (valid.length < n) return 0;
  const data = valid.slice(-n);
  const m = data.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < m; i++) {
    sumX += i; sumY += data[i]; sumXY += i * data[i]; sumX2 += i * i;
  }
  const denom = (m * sumX2 - sumX * sumX);
  return denom === 0 ? 0 : (m * sumXY - sumX * sumY) / denom;
}

/**
 * @param {Array} klines - Monthly K-line
 * @param {Object} indicators — result of calculateAll
 * @returns {'bull'|'bear'|'neutral'}
 */
export function calcMonthlyDirection(klines, indicators) {
  const n = klines.length;
  if (n < 60) return 'neutral';
  const last = n - 1;
  const close = klines[last].close;
  const ma60 = indicators.ma60[last];
  if (close == null || ma60 == null) return 'neutral';
  const ma20Slope = slope(indicators.ma20, config.monthly.slopeLookback);
  const macdVal = indicators.macd.dif[last];
  if (close > ma60 && ma20Slope > config.monthly.slopeMinDegree && macdVal != null && macdVal > 0) return 'bull';
  if (close < ma60 && ma20Slope < config.monthly.slopeMinDegree && macdVal != null && macdVal < 0) return 'bear';
  return 'neutral';
}

/**
 * @returns {'bull'|'bear'|'neutral'}
 */
export function calcWeeklyDirection(klines, indicators) {
  const n = klines.length;
  if (n < 20) return 'neutral';
  const last = n - 1;
  const close = klines[last].close;
  const ma20 = indicators.ma20[last];
  if (close == null || ma20 == null) return 'neutral';
  const ma20Slope = slope(indicators.ma20, config.weekly.slopeLookback);
  if (close > ma20 && ma20Slope > config.weekly.slopeMinDegree) return 'bull';
  if (close < ma20 && ma20Slope < config.weekly.slopeMinDegree) return 'bear';
  return 'neutral';
}

/**
 * @returns {'bull'|'bear'|'neutral'}
 */
export function calcDailyDirection(klines, indicators) {
  const n = klines.length;
  if (n < 20) return 'neutral';
  const last = n - 1;
  const close = klines[last].close;
  const ma5 = indicators.ma5[last];
  const ma20 = indicators.ma20[last];
  const rsi = indicators.rsi14[last];
  if (close == null || ma20 == null || ma5 == null || rsi == null) return 'neutral';
  if (close > ma20 && ma5 > ma20 && rsi > config.daily.rsiMin && rsi < config.daily.rsiMax) return 'bull';
  if (close < ma20 && ma5 < ma20 && rsi > config.daily.rsiMin && rsi < config.daily.rsiMax) return 'bear';
  return 'neutral';
}
