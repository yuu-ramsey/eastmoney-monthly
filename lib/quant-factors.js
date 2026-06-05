// Quantitative factor computation — pure functions, no side effects
// Returns null when klines < 24 bars, does not throw

/**
 * f1: Trend strength (-1~+1)
 * Comprehensive judgment using short/medium/long-term MA slopes
 */
export function computeTrendStrength(klines) {
  if (!Array.isArray(klines) || klines.length < 24) return null;

  const closes = klines.map(k => k.close).filter(v => v != null);
  if (closes.length < 24) return null;

  // Compute 3 MAs: MA5, MA20, MA60
  const ma5 = ma(closes, 5);
  const ma20 = ma(closes, 20);
  const ma60 = ma(closes, Math.min(60, closes.length));

  const last = closes.length - 1;
  if (ma5[last] == null || ma20[last] == null) return null;

  // Short-term slope: MA5 change rate over last 5 bars
  const shortSlope = ma5[last - 4] ? (ma5[last] - ma5[last - 4]) / Math.abs(ma5[last - 4]) : 0;
  // Medium-term slope: MA20 change rate over last 10 bars
  const midSlope = ma20[last - 9] ? (ma20[last] - ma20[last - 9]) / Math.abs(ma20[last - 9]) : 0;

  // MA alignment: MA5 > MA20 > MA60
  const alignment = (ma5[last] > ma20[last] ? 0.25 : -0.25)
    + (ma20[last] > (ma60[last] || ma20[last]) ? 0.25 : -0.25);

  // Composite: slope + alignment
  const raw = shortSlope * 30 + midSlope * 20 + alignment;
  const value = Math.max(-1, Math.min(1, raw));

  return {
    value: +value.toFixed(3),
    raw,
    components: { shortSlope: +shortSlope.toFixed(4), midSlope: +midSlope.toFixed(4), alignment: +alignment.toFixed(2) },
  };
}

/**
 * f2: Price position (0~1)
 * Current price percentile within last N bars
 */
export function computePricePosition(klines, window = 36) {
  if (!Array.isArray(klines) || klines.length < Math.min(12, window)) return null;

  const closes = klines.map(k => k.close).filter(v => v != null);
  if (closes.length < Math.min(12, window)) return null;

  const n = Math.min(window, closes.length);
  const slice = closes.slice(-n);
  const current = slice[slice.length - 1];
  const min = Math.min(...slice);
  const max = Math.max(...slice);

  if (max <= min) return null;

  const raw = (current - min) / (max - min);
  return {
    value: +raw.toFixed(3),
    raw,
    components: { current, min, max, window: n },
  };
}

/**
 * f3: Volatility percentile (0~1)
 * Current volatility's position in historical distribution
 */
export function computeVolatilityPercentile(klines, window = 36) {
  if (!Array.isArray(klines) || klines.length < Math.min(12, window)) return null;

  const n = Math.min(window, klines.length);
  const slice = klines.slice(-n);

  // Compute amplitude for each bar
  const amplitudes = slice.map(k => {
    if (k.high == null || k.low == null || k.open == null) return null;
    return Math.abs(k.high - k.low) / Math.abs(k.open) || 0;
  }).filter(v => v != null);

  if (amplitudes.length < 6) return null;

  const current = amplitudes[amplitudes.length - 1];
  const sorted = [...amplitudes].sort((a, b) => a - b);
  const pos = sorted.findIndex(v => v >= current);
  const raw = pos / sorted.length;

  // Low score near median, high score at extremes (suited for trend stock definition)
  return {
    value: +raw.toFixed(3),
    raw,
    components: { currentAmp: +current.toFixed(4), medianAmp: +sorted[Math.floor(sorted.length / 2)].toFixed(4), window: n },
  };
}

/**
 * f4: Volume-price confirmation (-1 / 0 / +1)
 * Volume-price relationship of the last 3 bars
 */
export function computeVolumePriceConfirm(klines) {
  if (!Array.isArray(klines) || klines.length < 4) return null;

  const last = klines.slice(-3);
  const prevs = klines.slice(-6, -3); // previous 3 bars for volume comparison

  let confirmScore = 0;
  let validPairs = 0;

  for (let i = 0; i < last.length; i++) {
    const k = last[i];
    if (k.close == null || k.open == null || k.volume == null) continue;
    const priceUp = k.close > k.open;
    // Compare with avg volume of previous 3 bars
    const prevVol = prevs.length > i && prevs[i]?.volume != null ? prevs[i].volume : k.volume;
    const avgVol = (k.volume + prevVol) / 2;
    const volUp = k.volume > avgVol * 1.1;

    if (priceUp && volUp) confirmScore += 1;   // Price up + volume up
    else if (!priceUp && volUp) confirmScore -= 1; // Price down + volume up
    else if (priceUp && !volUp) confirmScore -= 0.5; // Price up + volume down
    else confirmScore += 0.5; // Price down + volume down (possible exhaustion)
    validPairs++;
  }

  if (validPairs === 0) return null;
  const raw = confirmScore / validPairs;
  return {
    value: raw > 0.3 ? 1 : raw < -0.3 ? -1 : 0,
    raw: +raw.toFixed(3),
    components: { pairs: validPairs, confirmScore: +confirmScore.toFixed(1) },
  };
}

/**
 * Composite quantitative score (-100 ~ +100)
 */
export function computeQuantScore(klines) {
  if (!Array.isArray(klines) || klines.length < 24) return null;

  const trend = computeTrendStrength(klines);
  const position = computePricePosition(klines);
  const volatility = computeVolatilityPercentile(klines);
  const volume = computeVolumePriceConfirm(klines);

  if (!trend || !position || !volatility || !volume) return null;

  // Weights
  // trend: 35%, position: 25%, volatility: 15%, volume: 25%
  const rawScore = trend.value * 35 + (position.value - 0.5) * 2 * 25 + (volatility.value - 0.5) * 2 * 15 + volume.value * 25;
  const score = Math.round(Math.max(-100, Math.min(100, rawScore * 100)));

  // confidence: based on factor agreement
  const signs = [
    Math.sign(trend.value),
    Math.sign(position.value - 0.5),
    Math.sign(volatility.value - 0.5),
    Math.sign(volume.value),
  ];
  const posCount = signs.filter(s => s > 0).length;
  const negCount = signs.filter(s => s < 0).length;
  const agreement = Math.max(posCount, negCount) / 4;
  const confidence = +(agreement * 0.8 + 0.2).toFixed(2);

  return {
    score,
    factors: { f1: trend, f2: position, f3: volatility, f4: volume },
    confidence,
  };
}

function ma(arr, period) {
  const result = new Array(arr.length).fill(null);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i];
    if (i >= period) sum -= arr[i - period];
    if (i >= period - 1) result[i] = +((sum / period).toFixed(4));
  }
  return result;
}
