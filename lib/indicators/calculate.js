// Unified indicator calculation entry — input kline array, return all indicators
import { sma, ema, macd } from './trend.js';
import { rsi, kdj, wr, cci, stochastic } from './momentum.js';
import { boll, atr } from './volatility.js';
import { obv, mfi } from './volume.js';

/**
 * @param {Array} klines — [{ open, close, high, low, volume, ... }]
 * @returns All indicator arrays (same length as klines, leading nulls)
 */
export function calculateAll(klines) {
  const closes = klines.map(k => k.close);
  const highs = klines.map(k => k.high);
  const lows = klines.map(k => k.low);
  const volumes = klines.map(k => k.volume || 0);

  const macdResult = macd(closes);

  return {
    // Trend
    ma5: sma(closes, 5),
    ma10: sma(closes, 10),
    ma20: sma(closes, 20),
    ma60: sma(closes, 60),
    ma120: sma(closes, 120),
    ema12: ema(closes, 12),
    ema26: ema(closes, 26),
    macd: macdResult, // { dif, dea, hist }

    // Momentum
    rsi6: rsi(closes, 6),
    rsi14: rsi(closes, 14),
    rsi24: rsi(closes, 24),
    kdj: kdj(highs, lows, closes),
    wr: wr(highs, lows, closes),
    cci: cci(highs, lows, closes),
    stoch: stochastic(highs, lows, closes),

    // Volatility
    boll: boll(closes), // { upper, mid, lower }
    atr14: atr(highs, lows, closes),

    // Volume-price
    obv: obv(closes, volumes),
    mfi14: mfi(highs, lows, closes, volumes),
  };
}

/** Take last n non-null indicator values (for prompt table) */
export function tailIndicators(indicators, n = 5) {
  const result = {};
  for (const [key, value] of Object.entries(indicators)) {
    if (key === 'macd') {
      result.macd_dif = tailArray(value.dif, n);
      result.macd_dea = tailArray(value.dea, n);
      result.macd_hist = tailArray(value.hist, n);
    } else if (key === 'kdj') {
      result.kdj_k = tailArray(value.k, n);
      result.kdj_d = tailArray(value.d, n);
      result.kdj_j = tailArray(value.j, n);
    } else if (key === 'boll') {
      result.boll_upper = tailArray(value.upper, n);
      result.boll_mid = tailArray(value.mid, n);
      result.boll_lower = tailArray(value.lower, n);
    } else if (key === 'stoch') {
      result.stoch_k = tailArray(value.k, n);
      result.stoch_d = tailArray(value.d, n);
    } else {
      result[key] = tailArray(value, n);
    }
  }
  return result;
}

function tailArray(arr, n) {
  const valid = arr.filter(v => v != null);
  return valid.slice(-n);
}
