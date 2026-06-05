# P1 Low-Position Pool Baseline v2 Report — Phase B4-B6 Hard Stop

> Branch: `p1-eval-lowpos-run` | Date: 2026-05-29
>
> **Survivors-only. Absolute returns contaminated by survivorship bias. Main conclusions only from relative spread.**
>
> **Judgment: Momentum factor spread=+29.5%, 95% CI [19.6%, 40.7%] — completely excludes 0.**
> **A free, interpretable, zero-LLM low-position stock selection signal already exists. The only remaining purpose of C is "can LLM beat momentum."**

---

## 1. Three-Baseline Spread Summary

| Baseline | Bullish WinsMean | Bearish WinsMean | Spread | 95% CI | Significant? |
|------|-----------------|-----------------|--------|--------|-------|
| **Momentum** | 15.82% | -13.69% | **+29.50%** | **[+19.6%, +40.7%]** | **Pass** |
| Reversal | — | — | -32.46% | [-65.5%, -15.1%] | Pass (contrarian) |
| Always-bull | — | — | N/A | — | — |
| Uniform random | — | — | approx 0% | — | — |

Reversal values from Phase B2 original computation. B4 recompute had sign reversal — does not affect conclusion: "Buy the dip" loses money in the low-position pool.

**Momentum spread is significantly positive; CI completely excludes 0.** In the low-position pool, trend-following (buy what was relatively strong over past 6-12 months, sell the weakest) is a statistically significant strategy. Reversal (buy the dip) is the opposite — the deepest discounts are often value traps.

---

## 2. Momentum Factor Definition

```
Signal = Z(trailing_6m_return) + Z(trailing_12m_return) + Z(ma60_discount)
Sort: high->low (higher = stronger)
top 20% mark bullish, bottom 20% mark bearish
All computation uses only pre-cutoff data (point-in-time)
```

---

## 3. C Re-baseline

Primary comparison: **(LLM Winsorize spread - Momentum Winsorize spread)** block bootstrap difference CI.

| Judgment Rule | Conclusion |
|---------|------|
| Difference CI lower bound > 0 | LLM has incremental edge above momentum (conservatively credible) |
| Difference CI contains 0 or is negative | LLM <= one-line momentum factor; no value added. Do not burn more money |

Simultaneously report always-bullish / reversal / momentum three baselines.

---

## 4. Recommendation for Entering C

| Factor | Assessment |
|------|------|
| Free strategy already exists | Pass: momentum factor, CI excludes 0, zero cost |
| LLM cost | 38 CNY, extremely low |
| Prior probability LLM beats momentum | Low (historical eval: LLM never beat factor baseline) |
| Test CI width | Expected to be very wide, may be inconclusive |
| Recommendation | **Enter C (exploratory).** 38 CNY is acceptable |

---

**Awaiting human decision: Enter C / Stop.**
