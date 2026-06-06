// Discounted multi-month return: r_{t+1} + γ*r_{t+2} + γ²*r_{t+3}
// A-share asymmetric thresholds (accounting for upward bias)

/**
 * @param {Array<{close: number}>} klines Monthly kline array
 * @param {number} cutoffIndex prediction point (decision made at this index)
 * @param {Object} options { gamma = 0.9, months = 3 }
 * @returns {{ discountedReturn: number, monthsAvailable: number, individualReturns: number[], gamma: number }}
 */
export function computeDiscountedReturn(klines, cutoffIndex, { gamma = 0.9, months = 3 } = {}) {
  const currentClose = klines[cutoffIndex].close;
  const individualReturns = [];
  let discountedReturn = 0;

  for (let i = 0; i < months; i++) {
    const futureIdx = cutoffIndex + i + 1;
    if (futureIdx >= klines.length) break;
    const ri = (klines[futureIdx].close - currentClose) / currentClose;
    individualReturns.push(ri);
    discountedReturn += ri * Math.pow(gamma, i);
  }

  return {
    discountedReturn,
    monthsAvailable: individualReturns.length,
    individualReturns,
    gamma,
  };
}

/**
 * A-share asymmetric thresholds. Upward thresholds are looser than downward because A-shares have a long-term upward drift.
 */
export const DEFAULT_THRESHOLDS = {
  strongBull: 0.15,   // >= +15% discounted return
  bull: 0.05,         // >= +5%
  neutralUpper: 0.05,  // < +5%
  neutralLower: -0.05, // >= -5%
  bear: -0.10,        // <= -10%
  strongBear: -0.20,  // <= -20%
};

/**
 * @param {number} discountedReturn
 * @param {Object} thresholds
 * @returns {'strong_bull'|'bull'|'neutral'|'bear'|'strong_bear'}
 */
export function discretizeDiscountedReturn(discountedReturn, thresholds = DEFAULT_THRESHOLDS) {
  const t = thresholds;
  if (discountedReturn >= t.strongBull) return 'strong_bull';
  if (discountedReturn >= t.bull) return 'bull';
  if (discountedReturn > t.neutralLower) return 'neutral';  // between -5% and +5%, not reaching bull
  if (discountedReturn > t.bear) return 'neutral';          // between -10% and -5%, still neutral
  if (discountedReturn > t.strongBear) return 'bear';       // between -20% and -10%
  return 'strong_bear';
}

/**
 * Signal-to-number mapping (for rank correlation etc.)
 */
export function mapSignalToNumber(signal) {
  const map = { strong_bull: 2, bull: 1, neutral: 0, bear: -1, strong_bear: -2 };
  return map[signal] ?? 0;
}
