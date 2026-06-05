# P1 Eval Rebuild Results — Phase C3

> Branch: `p1-eval-lowpos-run` | Date: 2026-05-29
>
> ⚠ Survivors-only ⚠ Exploratory ⚠ Test-MDE limited (31.7%)

---

## Primary Metric: (LLM - Momentum) Spread Difference CI

| Metric | Value |
|------|-----|
| LLM Winsorize spread | -7.64% |
| Momentum Winsorize spread | -13.09% |
| **Difference (LLM - Momentum)** | **+5.45%** |
| **Difference 95% CI** | **[-39.0%, +56.2%]** |
| CI width | **95.2pp** |

### Judgment: **INCONCLUSIVE — CI contains 0, power-limited**

Difference +5.45% but CI spans 95pp. Consistent with pre-registered Test MDE=31.7% — any reasonable effect's CI would cross zero at this power. **Not a negative conclusion (not equivalent to "LLM adds no value"); it is insufficient power.**

---

## Key Finding: LLM Structural Bullish Bias

```
Signal distribution: bull=490 (67%) | neutral=234 (32%) | bear=8 (1%)
```

LLM almost never predicts bearish in the low-position pool. Falls into the same trap as the reversal factor — seeing "cheap" means "buy," not distinguishing value traps from real opportunities. The momentum factor correctly follows the trend.

---

## Four Baseline Reference System

| Baseline | Spread | 95% CI | Description |
|------|--------|--------|------|
| Momentum factor | +29.5% | [+19.6, +40.7] | **Significantly positive (Pass)** |
| Reversal factor | -32.5% | [-65.5, -15.1] | Significantly contrarian |
| **LLM** | **-7.6%** | **[-51.8, +30.1]** | **inconclusive** |
| Always-bullish | N/A | — | 100% bull |

---

## Conclusion (pre-registered language)

> LLM's incremental edge relative to the momentum factor **cannot be detected** (difference CI contains 0, power-limited).
> A one-line momentum factor (+29.5% spread, CI entirely positive) is already the optimal free signal for this universe.
> Before expanding sample, continuing to burn LLM is meaningless.

## Cost

- LLM calls: 732 | Parse failed: 0
