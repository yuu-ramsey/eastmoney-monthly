// Simple Moving Average (SMA)
// Input: numeric array + window size
// Output: array of same length as input, first period-1 entries are null
// Rolling sum implementation, O(n)

export function computeMA(values, period) {
  if (!Array.isArray(values)) return [];
  if (!Number.isInteger(period) || period <= 0) return [];

  const out = new Array(values.length).fill(null);
  if (values.length === 0) return out;

  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    if (i >= period - 1) out[i] = sum / period;
  }
  return out;
}
