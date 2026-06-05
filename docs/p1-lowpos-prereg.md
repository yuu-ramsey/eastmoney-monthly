# P1 Low-Position Pool LLM Pre-Registration — Phase C0

> Branch: `p1-eval-lowpos-run` | Date: 2026-05-29 | **Commit before running LLM; no changes after run**

## Primary Metric

**(LLM Winsorize spread - Momentum Winsorize spread) block bootstrap difference CI**

Compute both sides' spread on the same set of resampled blocks, then take difference; CI comes from difference distribution.

## Judgment Rules (three-way)

| Difference CI | Conclusion |
|---------|------|
| Lower bound > 0 | LLM has incremental edge above momentum (conservatively credible) |
| Contains 0 | **inconclusive** (default expectation — Test MDE=31.7%, insufficient power; not a negative conclusion) |
| Upper bound < 0 | LLM inferior to momentum |

## Reference System

| Baseline | Expected |
|------|------|
| Momentum factor | +29.5%, CI [19.6%, 40.7%] |
| Reversal factor | -32% |
| Always-bullish | N/A |
| LLM | To be measured |

## Experimental Parameters

- Dataset: `frozen-eval-lowpos-v1.json` (732 pairs)
- Prompt: technical template, DeepSeek-chat, temperature=0.0
- Train: 2018-06, 2018-12, 2019-06, 2020-09, 2021-06, 2022-06
- Test: 2020-03, 2021-12, 2022-10, 2023-06, 2024-02, 2024-10
- Cost cap: 38 CNY

## Hard Constraints

- Pass: "LLM has/has no detectable increment relative to momentum factor (power-limited)"
- Fail: "Low-position stock returns X%" / "LLM verified effective"
- Absolute return metrics stamped "survivorship bias, internal reference only"
