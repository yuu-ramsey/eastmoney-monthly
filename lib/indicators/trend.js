// Trend indicators — SMA / EMA / MACD
import { avg, emaInit } from './core.js';

/** Simple Moving Average */
export function sma(closes, period) {
  return avg(closes, period);
}

/** Exponential Moving Average (EMA1=SMA, then weighted by 2/(N+1)) */
export function ema(closes, period) {
  const out = new Array(closes.length).fill(null);
  if (closes.length < period) return out;

  const init = emaInit(closes, period);
  out[period - 1] = init;
  const k = 2 / (period + 1);

  for (let i = period; i < closes.length; i++) {
    if (closes[i] != null && out[i - 1] != null) {
      out[i] = closes[i] * k + out[i - 1] * (1 - k);
    }
  }
  return out;
}

/** MACD: 返回 { dif, dea, hist } 三个数组（与 K 线等长） */
export function macd(closes, fast = 12, slow = 26, signal = 9) {
  const n = closes.length;
  const result = {
    dif: new Array(n).fill(null),
    dea: new Array(n).fill(null),
    hist: new Array(n).fill(null),
  };

  if (n < slow) return result;

  const emaFast = ema(closes, fast);
  const emaSlow = ema(closes, slow);

  // DIF = EMA(fast) - EMA(slow)
  for (let i = 0; i < n; i++) {
    if (emaFast[i] != null && emaSlow[i] != null) {
      result.dif[i] = emaFast[i] - emaSlow[i];
    }
  }

  // DEA = EMA(DIF, signal)
  const difValues = result.dif;
  const deaInit = emaInit(difValues.filter(v => v != null), signal);
  if (deaInit == null) return result;

  let deaPrev = deaInit;
  const k = 2 / (signal + 1);
  const slowMinus1 = slow - 1 + signal - 1; // 需要 slow + signal 根才出 DEA
  for (let i = slowMinus1; i < n; i++) {
    if (i === slowMinus1) {
      result.dea[i] = deaInit;
    } else if (result.dif[i] != null && result.dea[i - 1] != null) {
      result.dea[i] = result.dif[i] * k + result.dea[i - 1] * (1 - k);
    }
  }

  // HIST = 2 × (DIF - DEA)
  for (let i = 0; i < n; i++) {
    if (result.dif[i] != null && result.dea[i] != null) {
      result.hist[i] = 2 * (result.dif[i] - result.dea[i]);
    }
  }

  return result;
}
