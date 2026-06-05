// Momentum indicators — RSI / KDJ / WR / CCI / Stochastic
// Formulas strictly match Tongdaxin / Tonghuashun
import { max, min, smaSmoothing } from './core.js';

/** TDX RSI(N)
 *  LC = REF(CLOSE,1), MAX(CLOSE-LC,0) / ABS(CLOSE-LC) 的 SMA(N,1) * 100
 */
export function rsi(closes, period = 14) {
  const n = closes.length;
  const out = new Array(n).fill(null);
  if (n < period + 1) return out;

  // Compute daily change
  const up = new Array(n).fill(0);
  const down = new Array(n).fill(0);
  for (let i = 1; i < n; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff > 0) up[i] = diff;
    else down[i] = -diff;
  }

  // SMA(up, period, 1) / SMA(up+down, period, 1) * 100
  let upSma = 0, downSma = 0;
  // First valid bar: simple average at day 'period'
  let upAcc = 0, downAcc = 0;
  for (let i = 1; i <= period; i++) { upAcc += up[i]; downAcc += down[i]; }
  upSma = upAcc / period;
  downSma = downAcc / period;

  if (upSma + downSma === 0) out[period] = 50;
  else out[period] = (upSma / (upSma + downSma)) * 100;

  for (let i = period + 1; i < n; i++) {
    upSma = smaSmoothing(upSma, up[i], period, 1);
    downSma = smaSmoothing(downSma, down[i], period, 1);
    if (upSma + downSma === 0) out[i] = out[i - 1] || 50;
    else out[i] = (upSma / (upSma + downSma)) * 100;
  }

  return out;
}

/** 通达信 KDJ(N, M1, M2)
 *  RSV = (C - LLV(L,N)) / (HHV(H,N) - LLV(L,N)) * 100
 *  K = SMA(RSV, M1, 1)  // 即 1/M1*RSV + (M1-1)/M1*前K, 初始K=50
 *  D = SMA(K, M2, 1)
 *  J = 3K - 2D
 */
export function kdj(highs, lows, closes, n = 9, m1 = 3, m2 = 3) {
  const len = closes.length;
  const k = new Array(len).fill(null);
  const d = new Array(len).fill(null);
  const j = new Array(len).fill(null);

  if (len < n) return { k, d, j };

  const hhv = max(highs, n);
  const llv = min(lows, n);

  // Initialize K=50, D=50
  let prevK = 50, prevD = 50;
  let firstValid = -1;

  for (let i = n - 1; i < len; i++) {
    if (hhv[i] == null || llv[i] == null) continue;
    const range = hhv[i] - llv[i];
    const rsv = range === 0 ? 50 : ((closes[i] - llv[i]) / range) * 100;

    if (firstValid < 0) {
      k[i] = smaSmoothing(50, rsv, m1, 1);
      d[i] = smaSmoothing(50, k[i], m2, 1);
      firstValid = i;
    } else {
      k[i] = smaSmoothing(prevK, rsv, m1, 1);
      d[i] = smaSmoothing(prevD, k[i], m2, 1);
    }

    j[i] = 3 * k[i] - 2 * d[i];
    prevK = k[i];
    prevD = d[i];
  }

  return { k, d, j };
}

/** 威廉指标 W%R(N) = (HHV(N) - C) / (HHV(N) - LLV(N)) * 100 */
export function wr(highs, lows, closes, period = 14) {
  const n = closes.length;
  const out = new Array(n).fill(null);
  const hhv = max(highs, period);
  const llv = min(lows, period);
  for (let i = period - 1; i < n; i++) {
    const range = hhv[i] - llv[i];
    if (range === 0) out[i] = 50;
    else out[i] = (hhv[i] - closes[i]) / range * 100;
  }
  return out;
}

/** CCI(N) = (TP - MA(TP,N)) / (0.015 * MeanDeviation)
 *  TP = (H + L + C) / 3
 */
export function cci(highs, lows, closes, period = 14) {
  const n = closes.length;
  const out = new Array(n).fill(null);
  if (n < period) return out;

  const tp = new Array(n);
  for (let i = 0; i < n; i++) tp[i] = (highs[i] + lows[i] + closes[i]) / 3;

  for (let i = period - 1; i < n; i++) {
    let sumTp = 0;
    for (let j = i - period + 1; j <= i; j++) sumTp += tp[j];
    const ma = sumTp / period;

    let sumDev = 0;
    for (let j = i - period + 1; j <= i; j++) sumDev += Math.abs(tp[j] - ma);
    const meanDev = sumDev / period;

    if (meanDev === 0) out[i] = 0;
    else out[i] = (tp[i] - ma) / (0.015 * meanDev);
  }
  return out;
}

/** 美式 Stochastic (Fast) — %K: (C - LLV) / (HHV - LLV) * 100, %D: SMA(%K, d) */
export function stochastic(highs, lows, closes, kPeriod = 14, dPeriod = 3) {
  const n = closes.length;
  const k = new Array(n).fill(null);
  const d = new Array(n).fill(null);

  const hhv = max(highs, kPeriod);
  const llv = min(lows, kPeriod);

  for (let i = kPeriod - 1; i < n; i++) {
    const range = hhv[i] - llv[i];
    if (range === 0) k[i] = 50;
    else k[i] = (closes[i] - llv[i]) / range * 100;
  }

  // SMA of %K for %D
  for (let i = kPeriod - 1 + dPeriod - 1; i < n; i++) {
    let sum = 0;
    for (let j = i - dPeriod + 1; j <= i; j++) {
      if (k[j] != null) sum += k[j];
    }
    d[i] = sum / dPeriod;
  }

  return { k, d };
}
