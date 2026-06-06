// Counterfactual sentiment simulation — test LLM prediction sensitivity to sentiment events
// Three scenarios: baseline (real events) / bullish (inject positive news) / bearish (inject negative news)

// Bullish event library
const BULLISH_EVENTS = [
  { date: '06-15', type: '业绩预增', title: '预计2025H1归母净利润同比增长50%-80%' },
  { date: '06-10', type: '重大合同', title: '中标国家电网12.5亿元特高压工程总承包项目' },
  { date: '06-08', type: '机构调研', title: '近30日获86家机构密集调研，AI+业务布局受关注' },
  { date: '06-05', type: '研报', title: '中信证券：上调至"买入"评级，目标价提升30%' },
  { date: '05-28', type: '股东增持', title: '控股股东累计增持1.2亿元，均价高于当前价15%' },
];

// Bearish event library
const BEARISH_EVENTS = [
  { date: '06-15', type: '业绩预亏', title: '预计2025H1归母净利润亏损2-3亿元' },
  { date: '06-12', type: '股东减持', title: '第二大股东拟减持不超过3%股份' },
  { date: '06-10', type: '监管问询', title: '收到深交所年报问询函，涉及收入确认合规性' },
  { date: '06-05', type: '研报', title: '华泰证券：下调至"中性"评级，行业竞争加剧压缩毛利' },
  { date: '05-30', type: '诉讼公告', title: '子公司涉及3.8亿元知识产权侵权诉讼' },
];

// Positive keywords
const POSITIVE_KEYWORDS = ['增持', '预增', '扭亏', '中标', '突破', '获批', '回购', '买入', '上调', '增长', '预盈'];
// Negative keywords
const NEGATIVE_KEYWORDS = ['减持', '亏损', '预亏', '问询', '诉讼', '立案', '退市', '冻结', '下调', '预降', '违规'];

/**
 * Scan sentiment keywords in event title
 * @returns {{ positive: number, negative: number }}
 */
function scanSentiment(event) {
  const title = event.title || '';
  let pos = 0, neg = 0;
  for (const kw of POSITIVE_KEYWORDS) if (title.includes(kw)) pos++;
  for (const kw of NEGATIVE_KEYWORDS) if (title.includes(kw)) neg++;
  return { positive: pos, negative: neg };
}

/**
 * Build counterfactual event array
 * @param {Array<{date: string, type: string, title: string}>} baseEvents original events
 * @param {'baseline'|'bullish'|'bearish'} scenario
 * @returns {Array<{date: string, type: string, title: string}>}
 */
export function buildCounterfactualEvents(baseEvents, scenario) {
  if (scenario === 'baseline') return [...baseEvents];

  // Filter out opposite events, inject same-direction events
  const filtered = baseEvents.filter(event => {
    const { positive, negative } = scanSentiment(event);
    if (scenario === 'bullish') return negative === 0;  // Remove bearish
    if (scenario === 'bearish') return positive === 0;  // Remove bullish
    return true;
  });

  const inject = scenario === 'bullish' ? BULLISH_EVENTS : BEARISH_EVENTS;

  // Inject at front (latest events), keep same-direction from original
  return [...inject, ...filtered].slice(0, 12); // Max 12 entries
}

/**
 * Signal mapping
 */
function mapSignal(signal) {
  const m = { strong_bull: 2, bull: 1, neutral: 0, bear: -1, strong_bear: -2 };
  return m[signal] ?? 0;
}

/**
 * Compute cross-scenario stability score
 * @param {string[]} signals [baseline, bullish, bearish]
 * @returns {number} 0=complete reversal, 1=perfectly consistent
 */
export function computeStabilityScore(signals) {
  if (signals.length < 3) return 1.0;
  const numeric = signals.map(mapSignal);
  const maxDelta = Math.max(...numeric) - Math.min(...numeric);
  // maxDelta=4 (strong_bull vs strong_bear) → 0.0, maxDelta=0 → 1.0
  return Math.max(0, 1.0 - maxDelta / 4.0);
}

/**
 * Stability classification
 */
export function stabilityLevel(score) {
  if (score >= 0.75) return 'robust';       // Sentiment does not affect judgment
  if (score >= 0.5) return 'sensitive';     // Has impact but not reversed
  return 'fragile';                          // Sentiment may cause reversal
}
