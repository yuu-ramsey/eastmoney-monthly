// Signal factory — 15 structured signals, based on indicators + atomic operations
// Signals return { triggered, strength(0-1), description }
import { cross, exist, count, hhv, llv, every } from './atoms.js';
import { SIGNAL_CONFIG as CFG } from './config.js';

function sig(triggered, strength, description) {
  return { triggered, strength: Math.min(1, Math.max(0, strength)), description };
}

// ═══ Bullish signals ═══

/** MACD golden cross (DIF crosses above DEA, latest cross is upward) */
export function isMACDGoldenCross(indicators) {
  const rUp = cross(indicators.macd.dif, indicators.macd.dea, CFG.macd.crossLookback);
  const rDn = cross(indicators.macd.dea, indicators.macd.dif, CFG.macd.crossLookback);
  // 只有最近穿越是向上时才报金叉
  if (rUp.crossed && (!rDn.crossed || rUp.atIndex > rDn.atIndex)) {
    return sig(true, 0.8, `MACD 金叉（DIF 上穿 DEA，距今 ${indicators.macd.dif.length - 1 - rUp.atIndex} 期）`);
  }
  return sig(false, 0, '');
}

/** KDJ 金叉（K 上穿 D，lookback 期内） */
export function isKDJGoldenCross(indicators) {
  const r = cross(indicators.kdj.k, indicators.kdj.d, CFG.kdj.crossLookback);
  return sig(r.crossed, 0.7, r.crossed ? `KDJ 金叉（K 上穿 D）` : '');
}

/** RSI 超卖：RSI < 20 持续 2 期 */
export function isRSIOversold(indicators) {
  const n = indicators.rsi14.length;
  if (n < 2) return sig(false, 0, '');
  const cond = indicators.rsi14.map(v => v != null && v < CFG.rsi.oversold);
  const sustained = count(cond, CFG.oversoldBottom.sustainPeriods) >= CFG.oversoldBottom.sustainPeriods;
  const lastRsi = indicators.rsi14[n - 1];
  return sig(sustained, 0.6, sustained ? `RSI 超卖（RSI=${lastRsi?.toFixed(1)}，持续 ≥2 期 < ${CFG.rsi.oversold}）` : '');
}

/** 超跌底部：RSI < 30 且距 60 月低点 < 10% */
export function isOversoldBottom(klines, indicators) {
  const n = klines.length;
  if (n < 12) return sig(false, 0, '');
  const lastClose = klines[n - 1]?.close, lastRsi = indicators.rsi14[n - 1];
  if (lastClose == null || lastRsi == null) return sig(false, 0, '');
  const low60 = llv(klines.map(k => k.close), 60);
  if (low60 == null) return sig(false, 0, '');
  const distFromLow = (lastClose - low60) / low60 * 100;
  const triggered = lastRsi < CFG.oversoldBottom.rsiMax && distFromLow < CFG.oversoldBottom.positionMax;
  return sig(triggered, 0.9, triggered ? `超跌底部（RSI=${lastRsi.toFixed(1)}，距 60 月低点 ${low60.toFixed(2)} 仅 ${distFromLow.toFixed(1)}%）` : '');
}

/** 突破阻力：当前价突破 60 月最高（近 12 月内首次） */
export function isBreakResistance(klines, indicators) {
  const n = klines.length;
  if (n < 12) return sig(false, 0, '');
  const closes = klines.map(k => k.close);
  const high60 = hhv(closes, CFG.breakResistance.priceLookback);
  const high12 = hhv(closes, 12);
  if (high60 == null || high12 == null) return sig(false, 0, '');
  const lastClose = closes[n - 1];
  const triggered = lastClose >= high12 && high12 >= high60 * 0.99;
  return sig(triggered, 0.7, triggered ? `突破阻力（当前价 ${lastClose.toFixed(2)} ≥ 近 12 月最高 ${high12.toFixed(2)}）` : '');
}

/** MA20 上行（最近 3 期持续向上） */
export function isMA20Uptrend(indicators) {
  const n = indicators.ma20.length;
  if (n < 4) return sig(false, 0, '');
  const last = n - 1;
  const up = indicators.ma20[last] > indicators.ma20[last - 3] && indicators.ma20[last] > indicators.ma20[last - 1];
  return sig(up, 0.4, up ? 'MA20 上行趋势' : '');
}

/** MA60 上行且价格站稳 MA60 之上连续 3 期 */
export function isMA60Uptrend(klines, indicators) {
  const n = indicators.ma60.length;
  if (n < 4) return sig(false, 0, '');
  const last = n - 1;
  const ma60Before = indicators.ma60[last - CFG.ma.uptrendLookback];
  if (ma60Before == null || indicators.ma60[last] == null) return sig(false, 0, '');
  const ma60Up = indicators.ma60[last] > ma60Before;
  const closes = klines.map(k => k.close);
  const aboveMa60 = closes[last] > indicators.ma60[last];
  return sig(ma60Up && aboveMa60, 0.5, (ma60Up && aboveMa60) ? 'MA60 上行且价格在上方' : '');
}

/** 均线多头排列：MA5 > MA20 > MA60 */
export function isBullishAlignment(indicators) {
  const n = indicators.ma5.length;
  if (n < 3) return sig(false, 0, '');
  const last = n - 1;
  const aligned = indicators.ma5[last] > indicators.ma20[last] && indicators.ma20[last] > indicators.ma60[last];
  return sig(aligned, 0.7, aligned ? '均线多头排列（MA5 > MA20 > MA60）' : '');
}

/** 均线空头排列：MA5 < MA20 < MA60 */
export function isBearishAlignment(indicators) {
  const n = indicators.ma5.length;
  if (n < 3) return sig(false, 0, '');
  const last = n - 1;
  const aligned = indicators.ma5[last] < indicators.ma20[last] && indicators.ma20[last] < indicators.ma60[last];
  return sig(aligned, 0.7, aligned ? '均线空头排列（MA5 < MA20 < MA60）' : '');
}

/** 量能突破：本期成交量 > 20 期均量的 2 倍 */
export function isVolumeBreak(klines) {
  const n = klines.length;
  if (n < 20) return sig(false, 0, '');
  const volumes = klines.map(k => k.volume || 0);
  const avg20 = volumes.slice(-21, -1).reduce((a, b) => a + b, 0) / 20;
  const lastVol = volumes[n - 1];
  const triggered = lastVol > avg20 * CFG.volume.surgeMultiple;
  return sig(triggered, 0.6, triggered ? `量能突破（本期量 ${(lastVol/1e6).toFixed(0)}M > 20 期均量 ${(avg20/1e6).toFixed(0)}M 的 ${(lastVol/avg20).toFixed(1)}x）` : '');
}

// ═══ Bearish signals ═══

/** MACD 死叉 */
export function isMACDDeathCross(indicators) {
  const r = cross(indicators.macd.dea, indicators.macd.dif, CFG.macd.crossLookback);
  return sig(r.crossed, 0.8, r.crossed ? 'MACD 死叉（DEA 上穿 DIF）' : '');
}

/** MACD 顶背离：价创新高 MACD 不创新高 */
export function isMACDDivergence(klines, indicators) {
  const n = klines.length;
  if (n < CFG.divergence.macdLookback) return sig(false, 0, '');
  const closes = klines.map(k => k.close);
  const priceHigh = hhv(closes.slice(-CFG.divergence.priceLookback), CFG.divergence.priceLookback);
  const macdHigh = hhv(indicators.macd.dif.slice(-CFG.divergence.macdLookback), CFG.divergence.macdLookback);
  if (priceHigh == null || macdHigh == null) return sig(false, 0, '');
  const prevPriceHigh = hhv(closes.slice(-CFG.divergence.priceLookback * 2, -CFG.divergence.priceLookback), CFG.divergence.priceLookback);
  const prevMacdHigh = hhv(indicators.macd.dif.slice(-CFG.divergence.macdLookback * 2, -CFG.divergence.macdLookback), CFG.divergence.macdLookback);
  const triggered = prevPriceHigh != null && prevMacdHigh != null && priceHigh > prevPriceHigh && macdHigh < prevMacdHigh;
  return sig(triggered, 0.9, triggered ? 'MACD 顶背离（价创新高，MACD 未创新高）' : '');
}

/** 长上影线：上影 > 实体 2x 且位置 > 70% */
export function isLongUpperShadow(kline) {
  if (!kline || kline.length === 0) return sig(false, 0, '');
  const k = Array.isArray(kline) ? kline[kline.length - 1] : kline;
  const { open, close, high, low } = k;
  if (!open || !close || !high || !low || open === close) return sig(false, 0, '');
  const body = Math.abs(close - open);
  const upperShadow = high - Math.max(open, close);
  const ratio = upperShadow / (body + 0.001);
  return sig(ratio > CFG.shadow.ratio, 0.5, ratio > CFG.shadow.ratio ? `长上影线（上影/实体=${ratio.toFixed(1)}x）` : '');
}

/** 高位放量滞涨：量前 10% 但涨幅 < 3% */
export function isVolStagnation(klines) {
  const n = klines.length;
  if (n < 20) return sig(false, 0, '');
  const volumes = klines.map(k => k.volume || 0);
  const threshold = volumes.slice(-20).sort((a, b) => b - a)[Math.floor(20 * (1 - CFG.stagnation.volPct / 100))];
  const lastVol = volumes[n - 1];
  const lastChange = Math.abs((klines[n - 1].close - klines[n - 1].open) / (klines[n - 1].open + 0.001) * 100);
  const triggered = lastVol >= threshold && lastChange < CFG.stagnation.priceChangeMax;
  return sig(triggered, 0.7, triggered ? `高位放量滞涨（量处前 10%，涨幅仅 ${lastChange.toFixed(1)}%）` : '');
}

/** 缩量下跌：量 < 20 期均量 0.5x 且 close < open */
export function isShrinkingVolDecline(klines) {
  const n = klines.length;
  if (n < 20) return sig(false, 0, '');
  const volumes = klines.map(k => k.volume || 0);
  const avg20 = volumes.slice(-21, -1).reduce((a, b) => a + b, 0) / 20;
  const lastVol = volumes[n - 1];
  const lastClose = klines[n - 1].close, lastOpen = klines[n - 1].open;
  const triggered = lastVol < avg20 * CFG.volume.shrinkRatio && lastClose < lastOpen;
  return sig(triggered, 0.5, triggered ? `缩量下跌（量=${(lastVol/1e6).toFixed(0)}M < 均量 ${(avg20/1e6).toFixed(0)}M 的 0.5x）` : '');
}

// ═══ Neutral / Reference signals ═══

/** 均线死叉：MA5 跌破 MA20 */
export function isMACrossDown(indicators) {
  const r = cross(indicators.ma20, indicators.ma5, CFG.ma.crossLookback); // ma20 上穿 ma5 = ma5 下穿 ma20
  return sig(r.crossed, 0.6, r.crossed ? 'MA5 跌破 MA20' : '');
}

/** 横盘震荡：20 期价格波动 < 10% */
export function isInRange(klines) {
  const n = klines.length;
  if (n < CFG.range.lookback) return sig(false, 0, '');
  const closes = klines.map(k => k.close).slice(-CFG.range.lookback);
  const h = Math.max(...closes), l = Math.min(...closes);
  const range = (h - l) / l * 100;
  return sig(range < CFG.range.tightPct, 0.4, range < CFG.range.tightPct ? `横盘震荡（${CFG.range.lookback} 期波动仅 ${range.toFixed(1)}%）` : '');
}

/** 接近高点：距 60 月高点 < 5% */
export function isNearHigh(klines) {
  const n = klines.length;
  if (n < 60) return sig(false, 0, '');
  const lastClose = klines[n - 1].close;
  const high60 = hhv(klines.map(k => k.close), 60);
  if (high60 == null) return sig(false, 0, '');
  const dist = (lastClose - high60) / high60 * 100;
  return sig(dist > -CFG.position.nearHigh, 0.3, dist > -CFG.position.nearHigh ? `接近 60 月高点（距 ${high60.toFixed(2)} 仅 ${dist.toFixed(1)}%）` : '');
}
