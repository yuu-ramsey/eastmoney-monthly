# Phase 12: HS300 Sector Alpha — Postmortem

## Objective

Improve the model's monthly trend judgment accuracy by injecting HS300 internal sector alpha (individual stock excess return relative to its sector benchmark) into the LLM prompt.

## Experiment Design

### Experiment 1: v1 Injection (Weak Constraint)

- Prompt difference: Run A (no sector) vs Run B (with sector alpha block + #12 constraint)
- 640 samples (40 stocks x 4 templates x 4 cutoffDates)
- Model: deepseek-chat, maxTokens=4000
- **Result**: Delta score = -0.0032 (statistically insignificant)
  - Run A weighted score: 0.1966
  - Run B weighted score: 0.1934
  - strong_bull FP: 57.9% -> 57.8%
  - parse_failed: 7 -> 14

### Experiment 2: v2 Strong Constraint (Strengthened #12)

- Strengthened #12 constraint: must reason about alpha tier (strong/neutral/weak), strong_bull requires strong tier, strong_bear requires weak tier
- SectorAlphaBlock presented with tiered formatting (table format)
- 160 samples (10 stocks x 4 templates x 4 cutoffDate)
- **Result**: Delta score = -0.0306 (worsened)
  - Run A weighted score: 0.0806
  - Run B weighted score: 0.0500
  - strong_bull FP: 56.0% -> 71.4% (deteriorated)
  - bear bias: 31.3% -> 40.0% (bearish bias increased)

### Offline Quantitative Matrix (Zero LLM Cost)

- 298 HS300 stocks x 2018-2025 monthly data = 88,511 observations
- 16 combinations (4 lookback x 4 holding)
- Metrics: Spearman IC / Hit Rate / Long-Short Sharpe

**IC Matrix**:

| lookback \ holding | 1m | 3m | 6m | 12m |
|---|---|---|---|---|
| 3m | -0.0075 | -0.0013 | -0.0121 | -0.0015 |
| 6m | -0.0008 | -0.0069 | -0.0103 | -0.0086 |
| 12m | +0.0075 | +0.0056 | +0.0004 | -0.0242 |
| 24m | -0.0017 | -0.0174 | -0.0409 | **-0.0738** |

**Hit Rate Matrix**:

All 47.8%-51.4%, indistinguishable from random.

**Long-Short Sharpe Matrix**:

**All negative** (-0.119 to -0.017). Best combination sharpe=-0.017.

## Root Cause Analysis

1. **12-month lookback sector Alpha has no predictive power for 6-month forward returns**: IC in [-0.01, +0.01] range, indistinguishable from zero
2. **24-month lookback has weak contrarian prediction** (IC=-0.074): individual stocks that outperform peers long-term tend to mean-revert, but predictive strength is too low (IC < 0.10) to serve as LLM hard constraint
3. **LLM was forced to make wrong judgments using weak signals**: In the v2 strong constraint experiment, LLM did comply with #12 instructions (rawResponse audit 10/10 explicitly cited alpha tiers), but in cases where alpha direction was inconsistent with ground truth, the constraint introduced systematic noise instead
4. **Temporal autocorrelation of individual stock alpha within the same sector is extremely low**: Sector alpha is fundamentally a highly mean-reverting signal, suitable for contrarian trading strategies rather than trend following

## Final Conclusion

**Phase 12: Closed. All 16 cells of IC matrix have |IC| < 0.10, cannot serve as LLM prompt constraint.**

## Byproducts

- **IC matrix methodology**: Reusable sector alpha predictive power evaluation framework (cli/analyze-sector-predictive-power.js)
- **24m lookback, 12m holding strongest signal** (IC=-0.07): reserved as feature candidate for Phase 17 (LSTM training)
- **lib/sector/ code retained**: calSectorAlpha function still usable for post-filtering or other offline analysis scenarios; prompt injection path closed

## Lessons Learned

1. **Run offline quantitative validation before prompt injection**: Phase 12 should have run the IC matrix first before deciding whether to inject into LLM. Doing it in reverse wasted 29 CNY eval cost (640 experiment) + 3.6 CNY (160 experiment)
2. **The strength of constraint wording does not change the information content of the signal itself**: v1->v2 constraint strengthening did not improve results because information did not increase
3. **Signals with IC < 0.10 are unsuitable to feed directly to LLM**: LLM cannot distinguish "informative constraints" from "noise constraints"; forcing low signal-to-noise ratio data leads to systematic judgment bias
