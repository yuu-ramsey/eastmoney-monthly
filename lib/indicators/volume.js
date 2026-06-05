// Volume-price indicators — OBV / MFI
import { max, min } from './core.js';

/** OBV — On Balance Volume: add volume on up days, subtract on down days */
export function obv(closes, volumes) {
  const n = closes.length;
  const out = new Array(n).fill(null);
  if (n === 0) return out;

  out[0] = volumes[0] || 0;
  for (let i = 1; i < n; i++) {
    if (closes[i] == null) continue;
    const prev = out[i - 1] || 0;
    const vol = volumes[i] || 0;
    if (closes[i] > closes[i - 1]) out[i] = prev + vol;
    else if (closes[i] < closes[i - 1]) out[i] = prev - vol;
    else out[i] = prev;
  }
  return out;
}

/** MFI(N) — 资金流向指标
 *  TP = (H+L+C)/3, MF = TP * Volume
 *  PMF = sum of MF where TP > prev TP, NMF = sum where TP < prev TP
 *  MFI = 100 - 100/(1 + PMF/NMF)
 */
export function mfi(highs, lows, closes, volumes, period = 14) {
  const n = closes.length;
  const out = new Array(n).fill(null);
  if (n < period + 1) return out;

  const tp = new Array(n);
  for (let i = 0; i < n; i++) {
    if (highs[i] == null || lows[i] == null || closes[i] == null) { tp[i] = null; continue; }
    tp[i] = (highs[i] + lows[i] + closes[i]) / 3;
  }

  for (let i = period; i < n; i++) {
    let pmf = 0, nmf = 0;
    for (let j = i - period + 1; j <= i; j++) {
      if (tp[j] == null || tp[j - 1] == null) continue;
      const mf = tp[j] * (volumes[j] || 0);
      if (tp[j] > tp[j - 1]) pmf += mf;
      else if (tp[j] < tp[j - 1]) nmf += mf;
    }
    if (nmf === 0) out[i] = 100;
    else out[i] = 100 - 100 / (1 + pmf / nmf);
  }
  return out;
}
