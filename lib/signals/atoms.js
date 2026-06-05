// Atomic operations — MyTT style, for signal factory composition

/** Detect if A crosses B (within recent lookback period)
 *  @returns {{ crossed: boolean, direction: 'up'|'down'|null, atIndex: number }} */
export function cross(seriesA, seriesB, lookback = 1) {
  const result = { crossed: false, direction: null, atIndex: -1 };
  const n = Math.min(seriesA.length, seriesB.length);
  if (n < 2) return result;

  const start = Math.max(1, n - lookback);
  for (let i = start; i < n; i++) {
    if (seriesA[i] == null || seriesB[i] == null) continue;
    const prevA = seriesA[i - 1], prevB = seriesB[i - 1];
    if (prevA == null || prevB == null) continue;
    if (prevA <= prevB && seriesA[i] > seriesB[i]) {
      result.crossed = true; result.direction = 'up'; result.atIndex = i; return result;
    }
    if (prevA >= prevB && seriesA[i] < seriesB[i]) {
      result.crossed = true; result.direction = 'down'; result.atIndex = i; return result;
    }
  }
  return result;
}

/** 过去 n 期内 condition 是否曾为 true */
export function exist(condition, n) {
  const start = Math.max(0, condition.length - n);
  for (let i = start; i < condition.length; i++) {
    if (condition[i]) return true;
  }
  return false;
}

/** 过去 n 期内 condition 为 true 的次数 */
export function count(condition, n) {
  const start = Math.max(0, condition.length - n);
  let c = 0;
  for (let i = start; i < condition.length; i++) {
    if (condition[i]) c++;
  }
  return c;
}

/** N 期最高值 */
export function hhv(series, n) {
  const start = Math.max(0, series.length - n);
  let max = -Infinity;
  for (let i = start; i < series.length; i++) {
    if (series[i] != null && series[i] > max) max = series[i];
  }
  return max === -Infinity ? null : max;
}

/** N 期最低值 */
export function llv(series, n) {
  const start = Math.max(0, series.length - n);
  let min = Infinity;
  for (let i = start; i < series.length; i++) {
    if (series[i] != null && series[i] < min) min = series[i];
  }
  return min === Infinity ? null : min;
}

/** 过去 n 期内 condition 是否始终满足 */
export function every(condition, n) {
  const start = Math.max(0, condition.length - n);
  for (let i = start; i < condition.length; i++) {
    if (!condition[i]) return false;
  }
  return true;
}
