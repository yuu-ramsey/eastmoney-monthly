// LLM + quantitative hybrid score fusion v2 — adaptive regime detection
// Dynamically select fusion weights based on quantitative factor strength, avoiding catastrophic deterioration in sideways/high_vol regimes

/**
 * Detect what regime the stock is currently in
 * Uses F2_pricePosition + F3_volatilityPct combo instead of F1_trend absolute threshold
 * @param {object} quantResult — return value of computeQuantScore
 * @returns {'strong_trend' | 'sideways' | 'high_vol' | 'mixed'}
 */
export function detectStockRegime(quantResult) {
  if (!quantResult || !quantResult.factors) return 'mixed';
  const { f2, f3 } = quantResult.factors;
  const pos = f2?.value ?? 0.5;
  const vol = f3?.value ?? 0.5;

  // Extreme position + low volatility = true trend (price at extreme with low fluctuation)
  if ((pos > 0.80 || pos < 0.20) && vol < 0.6) return 'strong_trend';

  // Mid-range + low volatility = true sideways (price stuck in middle of range)
  if (pos > 0.30 && pos < 0.70 && vol < 0.5) return 'sideways';

  // High volatility
  if (vol > 0.7) return 'high_vol';

  return 'mixed';
}

/** Adaptive weight table */
const ADAPTIVE_WEIGHTS = {
  strong_trend: { llm: 0.30, quant: 0.70 },
  sideways:     { llm: 0.92, quant: 0.08 },
  high_vol:     { llm: 0.85, quant: 0.15 },
  mixed:        { llm: 0.50, quant: 0.50 },
};

export const SIGNAL_THRESHOLDS = {
  strong_bull: 60,
  bull: 20,
  neutral: -20,
  bear: -60,
  strong_bear: -100,
};

/**
 * score → signal mapping
 */
export function scoreToSignal(score) {
  if (score >= 60) return 'strong_bull';
  if (score >= 20) return 'bull';
  if (score >= -20) return 'neutral';
  if (score >= -60) return 'bear';
  return 'strong_bear';
}

/**
 * Fuse LLM and quant scores
 * @param {object} llmResult — { score: 0-100, signal: string, confidence: string }
 * @param {object} quantResult — return value of computeQuantScore { score: -100~100, factors, confidence }
 * @returns {object}
 */
/**
 * @param {object} llmResult — { score, signal, confidence }
 * @param {object} quantResult — return value of computeQuantScore
 * @param {object} [options]
 * @param {boolean} [options.useAdaptive=true]
 */
export function fuseScores(llmResult, quantResult, options = {}) {
  const { useAdaptive = true } = options;

  // Handle missing data
  if (!llmResult && !quantResult) return null;
  if (!quantResult) {
    const s = llmResult.score ?? 50;
    return {
      final_score: s, final_signal: scoreToSignal(s),
      final_confidence: confidenceToNum(llmResult.confidence),
      llm_score: s, quant_score: null,
      agreement: 0, regime: 'unknown', components: { mode: 'llm_only' },
    };
  }
  if (!llmResult) {
    const s = quantResult.score + 50;
    const clamped = Math.max(0, Math.min(100, s));
    return {
      final_score: clamped, final_signal: scoreToSignal(quantResult.score),
      final_confidence: quantResult.confidence,
      llm_score: null, quant_score: quantResult.score,
      agreement: 0, regime: 'unknown', components: { mode: 'quant_only' },
    };
  }

  // LLM score: 0-100 → -100~100
  const llmScore = llmResult.score - 50;
  const quantScore = quantResult.score;

  // Directional agreement
  const llmDir = Math.sign(llmScore);
  const quantDir = Math.sign(quantScore);
  const agreement = llmDir === quantDir ? (llmDir === 0 ? 0 : 0.8 + Math.abs(llmDir) * 0.2) : -0.5;

  // Adaptive regime detection
  const regime = detectStockRegime(quantResult);
  let llmWeight, quantWeight;
  if (useAdaptive) {
    const w = ADAPTIVE_WEIGHTS[regime] || ADAPTIVE_WEIGHTS.mixed;
    llmWeight = w.llm;
    quantWeight = w.quant;
  } else {
    // Old logic: weight by agreement
    if (agreement >= 0.8) { llmWeight = 0.6; quantWeight = 0.4; }
    else if (agreement >= 0) { llmWeight = 0.7; quantWeight = 0.3; }
    else { llmWeight = 0.4; quantWeight = 0.6; }
  }

  const finalScore = llmScore * llmWeight + quantScore * quantWeight;
  const clampedFinal = Math.max(0, Math.min(100, finalScore + 50));

  return {
    final_score: Math.round(clampedFinal),
    final_signal: scoreToSignal(finalScore),
    final_confidence: +((llmResult.confidence === 'high' ? 0.9 : llmResult.confidence === 'medium' ? 0.65 : 0.4) * (1 + agreement) / 2).toFixed(2),
    llm_score: llmResult.score,
    quant_score: quantScore,
    agreement: +agreement.toFixed(2),
    regime,
    components: {
      llmWeight: +llmWeight.toFixed(2),
      quantWeight: +quantWeight.toFixed(2),
      llmContribution: +(llmScore * llmWeight).toFixed(1),
      quantContribution: +(quantScore * quantWeight).toFixed(1),
    },
  };
}

function confidenceToNum(c) {
  if (c === 'high') return 0.9;
  if (c === 'medium') return 0.65;
  return 0.4;
}
