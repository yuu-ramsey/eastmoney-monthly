// Signal summary generator — aggregate all factory signals, produce prompt-injectable text
import { SIGNAL_CONFIG as CFG } from './config.js';
import {
  isMACDGoldenCross, isKDJGoldenCross, isRSIOversold, isOversoldBottom,
  isBreakResistance, isMA60Uptrend, isMA20Uptrend, isVolumeBreak, isBullishAlignment,
  isMACDDeathCross, isMACDDivergence, isLongUpperShadow, isVolStagnation, isShrinkingVolDecline,
  isBearishAlignment, isMACrossDown, isInRange, isNearHigh,
} from './factory.js';

/** Summarize all signals
 * @returns {{ bull: [], bear: [], neutral: [], rawCount: {bull:number, bear:number, neutral:number} }} */
export function generateSignalSummary(klines, indicators) {
  const bull = [];
  const bear = [];
  const neutral = [];

  const addIf = (category, signal) => {
    if (signal.triggered) {
      category.push({ name: signal.name || '', description: signal.description });
    }
  };

  // Bullish
  const macdGC = isMACDGoldenCross(indicators); macdGC.name = 'MACD金叉'; addIf(bull, macdGC);
  const kdjGC = isKDJGoldenCross(indicators); kdjGC.name = 'KDJ金叉'; addIf(bull, kdjGC);
  const rsiOS = isRSIOversold(indicators); rsiOS.name = 'RSI超卖'; addIf(bull, rsiOS);
  const overBot = isOversoldBottom(klines, indicators); overBot.name = '超跌底部'; addIf(bull, overBot);
  const brkRes = isBreakResistance(klines, indicators); brkRes.name = '突破阻力'; addIf(bull, brkRes);
  const ma20Up = isMA20Uptrend(indicators); ma20Up.name = 'MA20上行'; addIf(bull, ma20Up);
  const ma60Up = isMA60Uptrend(klines, indicators); ma60Up.name = 'MA60上行'; addIf(bull, ma60Up);
  const volBrk = isVolumeBreak(klines); volBrk.name = '量能突破'; addIf(bull, volBrk);
  const bullAlign = isBullishAlignment(indicators); bullAlign.name = '多头排列'; addIf(bull, bullAlign);

  // Bearish
  const macdDC = isMACDDeathCross(indicators); macdDC.name = 'MACD死叉'; addIf(bear, macdDC);
  const macdDiv = isMACDDivergence(klines, indicators); macdDiv.name = 'MACD顶背离'; addIf(bear, macdDiv);
  const upShadow = isLongUpperShadow(klines); upShadow.name = '长上影线'; addIf(bear, upShadow);
  const volStag = isVolStagnation(klines); volStag.name = '放量滞涨'; addIf(bear, volStag);
  const shrinkDecl = isShrinkingVolDecline(klines); shrinkDecl.name = '缩量下跌'; addIf(bear, shrinkDecl);
  const maCrossDn = isMACrossDown(indicators); maCrossDn.name = 'MA5死叉'; addIf(bear, maCrossDn);
  const bearAlign = isBearishAlignment(indicators); bearAlign.name = '空头排列'; addIf(bear, bearAlign);

  // Neutral
  const inRange = isInRange(klines); inRange.name = '横盘震荡'; addIf(neutral, inRange);
  const nearHigh = isNearHigh(klines); nearHigh.name = '接近高点'; addIf(neutral, nearHigh);

  return {
    bull, bear, neutral,
    rawCount: { bull: bull.length, bear: bear.length, neutral: neutral.length },
  };
}

/** Generate text for prompt injection */
export function formatSignalSummary(summary) {
  if (!summary || (summary.bull.length === 0 && summary.bear.length === 0 && summary.neutral.length === 0)) {
    return '';
  }
  const lines = ['\n## 结构化信号识别（程序完成）'];
  if (summary.bull.length > 0) {
    lines.push('\n✓ 看多信号（' + summary.bull.length + ' 个）：');
    for (const s of summary.bull) lines.push('  - ' + s.description);
  }
  if (summary.bear.length > 0) {
    lines.push('\n✗ 看空信号（' + summary.bear.length + ' 个）：');
    for (const s of summary.bear) lines.push('  - ' + s.description);
  }
  if (summary.neutral.length > 0) {
    lines.push('\n⚠ 中性/参考信号（' + summary.neutral.length + ' 个）：');
    for (const s of summary.neutral) lines.push('  - ' + s.description);
  }
  lines.push(`\n信号统计：bull=${summary.rawCount.bull} bear=${summary.rawCount.bear} neutral=${summary.rawCount.neutral}`);
  return lines.join('\n') + '\n';
}

/** Generate HARD_CONSTRAINTS #9 guidance text (based on signal count) */
export function buildSignalGuidance(summary) {
  if (!summary) return '';
  const b = summary.rawCount.bull, be = summary.rawCount.bear;

  if (b >= CFG.strongBullThreshold.bullMin && be <= CFG.strongBullThreshold.bearMax) {
    return `⚠ 信号指引：bull 信号 ${b} 个 ≥ ${CFG.strongBullThreshold.bullMin}，bear 信号 ${be} 个 ≤ ${CFG.strongBullThreshold.bearMax}。证据充分，signal 字段应判 strong_bull，不要回避。`;
  }
  if (be >= CFG.strongBearThreshold.bearMin && b <= CFG.strongBearThreshold.bullMax) {
    return `⚠ 信号指引：bear 信号 ${be} 个 ≥ ${CFG.strongBearThreshold.bearMin}，bull 信号 ${b} 个 ≤ ${CFG.strongBearThreshold.bullMax}。证据充分，signal 字段应判 strong_bear，不要回避。`;
  }
  if (b >= CFG.considerBullThreshold.bullMin) {
    return `⚠ 信号指引：bull 信号 ${b} 个 ≥ ${CFG.considerBullThreshold.bullMin}，signal 字段应至少判 bull，不应退到 neutral。`;
  }
  if (be >= CFG.considerBullThreshold.bullMin) {
    return `⚠ 信号指引：bear 信号 ${be} 个 ≥ ${CFG.considerBullThreshold.bullMin}，signal 字段应至少判 bear，不应退到 neutral。`;
  }
  return '';
}
