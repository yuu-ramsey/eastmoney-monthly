// Volatility indicators — Bollinger Bands / ATR
import { avg, std } from './core.js';

/** 布林带 (N, K) — 返回 { upper, mid, lower } */
export function boll(closes, n = 20, k = 2) {
  const len = closes.length;
  const mid = avg(closes, n);
  const s = std(closes, n);
  const upper = new Array(len).fill(null);
  const lower = new Array(len).fill(null);

  for (let i = 0; i < len; i++) {
    if (mid[i] != null && s[i] != null) {
      upper[i] = mid[i] + k * s[i];
      lower[i] = mid[i] - k * s[i];
    }
  }
  return { upper, mid, lower };
}

/** ATR(N) — 真实波幅的 N 期均线 */
export function atr(highs, lows, closes, period = 14) {
  const n = closes.length;
  const out = new Array(n).fill(null);
  if (n < 2) return out;

  // True Range
  const tr = new Array(n).fill(null);
  for (let i = 1; i < n; i++) {
    const h = highs[i], l = lows[i], prevC = closes[i - 1];
    if (h == null || l == null || prevC == null) continue;
    tr[i] = Math.max(
      h - l,
      Math.abs(h - prevC),
      Math.abs(l - prevC),
    );
  }

  // EMA of TR (use standard EMA weight 2/(N+1))
  let acc = 0;
  let count = 0;
  for (let i = 1; i <= period && i < n; i++) {
    if (tr[i] != null) { acc += tr[i]; count++; }
  }
  if (count === 0) return out;

  out[period] = acc / count;
  const k = 2 / (period + 1);
  for (let i = period + 1; i < n; i++) {
    if (tr[i] != null && out[i - 1] != null) {
      out[i] = tr[i] * k + out[i - 1] * (1 - k);
    }
  }
  return out;
}
