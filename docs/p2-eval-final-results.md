# P2 Eval Final Results — C3

> Branch: `p2-rebuild-unbiased-pool` | Date: 2026-05-30
>
> ⚠ Contains 0.7% delisted stocks ⚠ Survivorship bias substantially corrected ⚠ MDE 8.1%

---

## Primary Metric

| | Spread | 95% CI |
|---|--------|--------|
| **LLM** | **+8.94%** | **[+3.7, +14.1]** |
| Reversal | +6.63% | [+1.3, +11.2] |
| **Difference** | **+2.31%** | **[-4.7, +10.0]** |

**INCONCLUSIVE (CI contains 0). But LLM itself is significant — the biggest change v1->v2.**

---

## LLM Signal Distribution

strong_bull 6% | bull 23% | neutral 66% | bear 4% | strong_bear 0%

Bull bias improved from v1's 67%/1% to 29%/4%.

---

## Four Baselines

| Baseline | Spread | CI |
|------|--------|-----|
| LLM | +8.94% | [+3.7, +14.1] |
| Reversal | +6.63% | [+1.3, +11.2] |
| Momentum | +0.44% | [-4.2, +5.1] |

---

## Train/Test

Train: +10.10% | Test: +6.35% | Cost: **0.45 CNY**

---

## v1->v2 Summary

| Metric | v1 | v2 |
|------|-----|-----|
| MDE | 24% | 8.1% |
| Reversal | -32.5% artifact | +6.6% CI positive |
| LLM | -7.6% CI contains 0 | **+8.9% CI positive** |
| Difference CI | [-39,+56] | [-4.7,+10.0] |

LLM has independent predictive power on the unbiased pool. Complementary to reversal, not substitutive.
