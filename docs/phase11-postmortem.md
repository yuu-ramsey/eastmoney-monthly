# Phase 11: Multi-Period Resonance Constraint — Postmortem

## Objective

Inject monthly/weekly/daily three-period resonance signals into the LLM prompt (HARD_CONSTRAINTS #11), constraining the model to output judgments aligned with resonance direction when resonance is clear.

## Original Basis

v4-signals eval: score 0.168 (stored) vs v5-resonance: score 0.130 — resonance constraint caused score decline. The decision to close was correct at the time, but the predictive power of the resonance signal itself was not verified.

## Offline Quantitative Matrix (Zero LLM Cost)

298 HS300 stocks x 2018-2025 monthly data; at each evaluation point, compute three-period direction + resonance level (using lib/multi-period/ production code logic), strict walk-forward.

### Sample Sizes

| signal | n (per holding) | Description |
|--------|----------------|------|
| strong_bull | 1,114-1,118 | All three periods bullish |
| strong_bear | 1,114 | All three periods bearish |
| mild_bull | 2,720-2,721 | Two periods bullish |
| mild_bear | 2,939 | Two periods bearish |

### Avg Forward Return (deducting HS300 equal-weight benchmark alpha)

| signal \ holding | 1m | 3m | 6m | 12m |
|---|---|---|---|---|
| strong_bull | 0.59% | 2.61% | 4.31% | 9.25% |
| **strong_bear** | **1.53%** | **3.48%** | **7.67%** | **11.15%** |

### Long-Short (strong_bull - strong_bear, alpha)

| 1m | 3m | 6m | 12m |
|---|---|---|---|
| -0.94% | -0.88% | -3.35% | -1.90% |
| Sharpe=-0.195 | Sharpe=-0.051 | Sharpe=-0.097 | Sharpe=-0.028 |

## Key Finding

**Three-period resonance is a contrarian indicator.** Forward returns after all-three-period bearish (strong_bear) are systematically higher than after all-three-period bullish (strong_bull); the effect persists after deducting benchmark.

Economic explanation: Three periods aligned -> trend already fully priced -> mean reversion. The A-share folk wisdom of "three white soldiers mark the top, three black crows mark the bottom" is quantitatively validated here.

However, predictive power is weak (|Sharpe| approx 0.10), making it a marginal signal.

## Why LLM v5 Score Declined

HARD_CONSTRAINTS #11 required LLM to follow resonance direction when strong resonance was present, outputting strong_bull/strong_bear accordingly. But resonance is a contrarian signal — actual returns after strong_bull resonance are lower than after strong_bear. LLM was forced by the constraint to use a contrarian signal, naturally worsening the score.

## Final Conclusion

|Sharpe| approx 0.10, marginal contrarian signal. Direct use by LLM causes misjudgment (v5 score 0.130), but may be effective as an LSTM feature.

## Follow-up

- Code retained in `lib/multi-period/`, default `ENABLE_RESONANCE=false`
- Phase 17 (LSTM) will use resonance as a contrarian feature
- Optional path: change #11 to contrarian logic and re-eval
