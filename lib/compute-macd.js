// MACD 指标计算
// DIF = EMA(fast) - EMA(slow)
// DEA = EMA(DIF, signal)
// HIST = 2 * (DIF - DEA)
// EMA 初始值用前 period 项的简单平均

function ema(values, period) {
  const out = new Array(values.length).fill(null);
  if (values.length < period) return out;

  let sum = 0;
  for (let i = 0; i < period; i++) sum += values[i];
  out[period - 1] = sum / period;

  const k = 2 / (period + 1);
  for (let i = period; i < values.length; i++) {
    out[i] = values[i] * k + out[i - 1] * (1 - k);
  }
  return out;
}

export function computeMACD(closes, fast = 12, slow = 26, signal = 9) {
  if (!Array.isArray(closes)) return { dif: [], dea: [], hist: [] };
  if (!Number.isInteger(fast) || fast <= 0) return { dif: [], dea: [], hist: [] };
  if (!Number.isInteger(slow) || slow <= 0) return { dif: [], dea: [], hist: [] };
  if (!Number.isInteger(signal) || signal <= 0) return { dif: [], dea: [], hist: [] };
  if (fast >= slow) return { dif: [], dea: [], hist: [] };

  const n = closes.length;
  const emaFast = ema(closes, fast);
  const emaSlow = ema(closes, slow);

  // DIF = EMA_fast - EMA_slow
  const dif = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    if (emaFast[i] !== null && emaSlow[i] !== null) {
      dif[i] = emaFast[i] - emaSlow[i];
    }
  }

  // 收集非 null 的 DIF,计算其 EMA 作为 DEA
  const difDense = [];
  const difIndices = [];
  for (let i = 0; i < n; i++) {
    if (dif[i] !== null) {
      difDense.push(dif[i]);
      difIndices.push(i);
    }
  }

  const deaDense = ema(difDense, signal);
  const dea = new Array(n).fill(null);
  for (let i = 0; i < difIndices.length; i++) {
    dea[difIndices[i]] = deaDense[i];
  }

  // HIST = 2 * (DIF - DEA)
  const hist = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    if (dif[i] !== null && dea[i] !== null) {
      hist[i] = 2 * (dif[i] - dea[i]);
    }
  }

  return { dif, dea, hist };
}
