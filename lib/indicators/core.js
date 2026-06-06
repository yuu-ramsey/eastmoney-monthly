// Core math utilities — rolling window statistics + EMA/SMA atomic operations
// All functions return arrays of same length as input, first N-1 entries are null

/** Rolling N-period sum */
export function sum(arr, n) {
  const out = new Array(arr.length).fill(null);
  if (arr.length < n || n <= 0) return out;
  let acc = 0;
  for (let i = 0; i < n; i++) acc += arr[i] || 0;
  out[n - 1] = acc;
  for (let i = n; i < arr.length; i++) {
    acc += (arr[i] || 0) - (arr[i - n] || 0);
    out[i] = acc;
  }
  return out;
}

/** Rolling N-period mean */
export function avg(arr, n) {
  const s = sum(arr, n);
  return s.map((v, i) => (v != null && i >= n - 1) ? v / n : null);
}

/** Rolling N-period max */
export function max(arr, n) {
  const out = new Array(arr.length).fill(null);
  for (let i = n - 1; i < arr.length; i++) {
    let m = -Infinity;
    for (let j = i - n + 1; j <= i; j++) {
      if (arr[j] != null && arr[j] > m) m = arr[j];
    }
    out[i] = m === -Infinity ? null : m;
  }
  return out;
}

/** Rolling N-period min */
export function min(arr, n) {
  const out = new Array(arr.length).fill(null);
  for (let i = n - 1; i < arr.length; i++) {
    let m = Infinity;
    for (let j = i - n + 1; j <= i; j++) {
      if (arr[j] != null && arr[j] < m) m = arr[j];
    }
    out[i] = m === Infinity ? null : m;
  }
  return out;
}

/** Rolling N-period standard deviation */
export function std(arr, n) {
  const a = avg(arr, n);
  const out = new Array(arr.length).fill(null);
  for (let i = n - 1; i < arr.length; i++) {
    let sumSq = 0;
    for (let j = i - n + 1; j <= i; j++) {
      sumSq += Math.pow(arr[j] - a[i], 2);
    }
    out[i] = Math.sqrt(sumSq / n);
  }
  return out;
}

/** EMA initial value (SMA of first period) */
export function emaInit(arr, period) {
  const slice = arr.slice(0, period).filter(v => v != null);
  if (slice.length === 0) return null;
  return slice.reduce((a, b) => a + b, 0) / slice.length;
}

/** Tongdaxin SMA(X, N, M): returns M/N * X + (N-M)/N * prev
 *  M=1 gives standard SMA smoothing
 */
export function smaSmoothing(prev, current, n, m = 1) {
  return (m / n) * current + ((n - m) / n) * prev;
}
