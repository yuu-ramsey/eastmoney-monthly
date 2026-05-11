// 简单移动平均 (Simple Moving Average)
// 输入:数值数组 + 窗口大小
// 输出:与输入等长的数组,前 period-1 项为 null
// 滚动求和实现,O(n)

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
