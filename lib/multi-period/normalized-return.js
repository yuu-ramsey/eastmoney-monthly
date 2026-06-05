// Multi-time-window normalized return — compute cross-window reward/risk ratios from kline arrays
// No LSTM dependency, pure numeric computation, injected into prompt as soft constraint
//
// Normalization formula: reward/risk ratio = window return / window annualized volatility
// Also computes historical percentile, answering "where does current ratio rank historically"

const PERIOD_CONFIG = {
  monthly: {
    windows: [3, 6, 12],
    labels: ['季度(3月)', '半年(6月)', '年度(12月)'],
    periodsPerYear: 12,
  },
  weekly: {
    windows: [4, 12, 26],
    labels: ['月度(4周)', '季度(12周)', '半年(26周)'],
    periodsPerYear: 52,
  },
  daily: {
    windows: [20, 60, 120],
    labels: ['月度(20日)', '季度(60日)', '半年(120日)'],
    periodsPerYear: 252,
  },
};

/**
 * 计算单窗口收益 + 波动率
 * @param {number[]} closes
 * @param {number} windowLen
 * @param {number} periodsPerYear
 * @returns {{ ret: number, vol: number, ratio: number }}
 */
function calcWindowRatio(closes, windowLen, periodsPerYear) {
  const n = closes.length;
  if (n <= windowLen) return { ret: 0, vol: 0, ratio: 0 };

  const currentRet = (closes[n - 1] - closes[n - 1 - windowLen]) / closes[n - 1 - windowLen];

  // 滚动窗口收益率，用于计算波动率（取最近 min(n-windowLen, periodsPerYear) 个窗口）
  const rollingRets = [];
  const maxSamples = Math.min(n - windowLen, periodsPerYear * 2);
  const start = Math.max(0, n - windowLen - maxSamples);
  for (let i = start; i < n - windowLen; i++) {
    rollingRets.push((closes[i + windowLen] - closes[i]) / closes[i]);
  }

  const mean = rollingRets.reduce((a, b) => a + b, 0) / rollingRets.length;
  const variance = rollingRets.reduce((s, r) => s + (r - mean) ** 2, 0) / rollingRets.length;
  const annualVol = Math.sqrt(variance) * Math.sqrt(periodsPerYear);

  const ratio = annualVol > 1e-8 ? currentRet / annualVol : 0;
  return { ret: currentRet, vol: annualVol, ratio };
}

/**
 * 计算当前性价比在所有历史窗口中的分位
 * @param {number[]} closes
 * @param {number} windowLen
 * @returns {number} 0-100 百分位
 */
function calcPercentileRank(closes, windowLen) {
  const ratios = [];
  for (let i = 0; i < closes.length - windowLen; i++) {
    const ret = (closes[i + windowLen] - closes[i]) / closes[i];
    ratios.push(ret);
  }
  if (ratios.length === 0) return 50;

  const currentRet = ratios[ratios.length - 1];
  const sorted = [...ratios].sort((a, b) => a - b);
  const rank = sorted.filter((r) => r < currentRet).length;
  return (rank / sorted.length) * 100;
}

/**
 * 主入口：计算多窗口归一化回报
 * @param {Array} klines — K 线数组 [{ close, ... }]
 * @param {string} period — 'monthly' | 'weekly' | 'daily'
 * @returns {{ windows: Array, summary: string, bestWindow: string, trend: string }}
 */
export function calcNormalizedReturns(klines, period) {
  const cfg = PERIOD_CONFIG[period];
  if (!cfg || klines.length < Math.min(...cfg.windows) + 1) return null;

  const closes = klines.map((k) => k.close).filter((c) => c != null && c > 0);
  if (closes.length < Math.min(...cfg.windows) + 1) return null;

  const windows = cfg.windows.map((w, idx) => {
    const { ret, vol, ratio } = calcWindowRatio(closes, w, cfg.periodsPerYear);
    const percentile = calcPercentileRank(closes, w);
    return {
      window: w,
      label: cfg.labels[idx],
      ret,
      retPct: (ret * 100),
      vol,
      volPct: (vol * 100),
      ratio,
      percentile,
    };
  });

  // 找性价比最高的窗口
  const best = windows.reduce((a, b) => (b.ratio > a.ratio ? b : a), windows[0]);

  // 趋势：最近 3 个最短窗口的 ratio 序列
  const shortWin = cfg.windows[0];
  const recentRatios = [];
  const trendN = Math.min(closes.length - shortWin - 1, cfg.periodsPerYear);
  for (let i = closes.length - shortWin - trendN; i <= closes.length - shortWin - 1; i++) {
    if (i >= shortWin) {
      const r = (closes[i] - closes[i - shortWin]) / closes[i - shortWin];
      recentRatios.push(r);
    }
  }
  const trendStart = recentRatios.slice(0, Math.floor(recentRatios.length / 2)).reduce((a, b) => a + b, 0) / (Math.floor(recentRatios.length / 2) || 1);
  const trendEnd = recentRatios.slice(-Math.floor(recentRatios.length / 2)).reduce((a, b) => a + b, 0) / (Math.floor(recentRatios.length / 2) || 1);
  const trend = trendEnd > trendStart * 1.1 ? 'improving' : trendEnd < trendStart * 0.9 ? 'deteriorating' : 'stable';

  // 生成一句话摘要
  const bestLabel = best.label;
  const trendLabel = trend === 'improving' ? '上升' : trend === 'deteriorating' ? '下降' : '稳定';
  const summary = `性价比最高的窗口是${bestLabel}(${(best.ratio >= 0 ? '+' : '') + best.ratio.toFixed(3)})，` +
    `处于历史 P${best.percentile.toFixed(0)}。` +
    `近${trendN}期趋势：${trendLabel}`;

  return { windows, summary, bestWindow: bestLabel, trend };
}
