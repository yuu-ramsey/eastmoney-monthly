// Signal threshold configuration — centralized parameter management
export const SIGNAL_CONFIG = {
  rsi: { oversold: 20, overbought: 80, oversoldWarn: 30 },
  macd: { crossLookback: 2 },
  kdj: { oversold: 20, overbought: 80, crossLookback: 3 },
  position: { nearBottom: 10, nearHigh: 5 },
  volume: { surgeMultiple: 2, shrinkRatio: 0.5, surgePct: 90 },
  ma: { crossLookback: 3, uptrendLookback: 12 },
  range: { tightPct: 10, lookback: 20 },
  shadow: { ratio: 2, position: 70 },
  divergence: { priceLookback: 12, macdLookback: 20 },
  stagnation: { volPct: 90, priceChangeMax: 3 },
  breakResistance: { priceLookback: 60, maLookback: 12 },
  oversoldBottom: { rsiMax: 30, positionMax: 10, sustainPeriods: 2 },
  strongBullThreshold: { bullMin: 3, bearMax: 1 },
  strongBearThreshold: { bearMin: 3, bullMax: 1 },
  considerBullThreshold: { bullMin: 2 },
};
