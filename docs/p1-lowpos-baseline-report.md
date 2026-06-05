# P1 Low-Position Pool Baseline Report — Phase B Hard Stop

> Branch: `p1-eval-lowpos-run` | Date: 2026-05-29
>
> **Survivors-only. Absolute returns contaminated by survivorship bias; not for conclusions. Main conclusions only from relative spread.**
>
> **Judgment: Pool resolution barely sufficient (full-sample MDE=24.0%); reversal factor itself has spread=-32% with CI spanning 50pp.**
> **LLM-vs-reversal comparison depends on spread difference CI; entering C is feasible but remains exploratory.**

---

## 1. Pool Overview

| Metric | Value |
|------|-----|
| Total testPoints | 732 |
| Valid reversal factor pairs | 573 (unique) |
| Unique stocks | 268 |
| Total (stock, timepoint) | 573 |

### Per-Timepoint Inclusion Counts

```
2018-06:  76 (Train)    2020-03:  84 (Test)
2018-12: 115 (Train)    2021-12:  33 (Test)
2019-06:  22 ⚠ THIN    2022-10: 138 (Test)
2020-09:  26 ⚠ THIN    2023-06:  64 (Test)
2021-06:  29 ⚠ THIN    2024-02:  89 (Test)
2022-06:  44 (Train)    2024-10:  12 ⚠ THIN
```

4 thin timepoints < 30. Low-position stocks naturally scarce during bull/recovery periods.

---

## 2. Reversal Factor Baseline

```
bullish (deepest 20% decline):  WinsMean =   6.14%
bearish (shallowest 20% decline):  WinsMean =  38.60%
spread:                        -32.46%
95% CI:              [-65.54%, -15.08%]
nBlocks:             573
```

**The reversal factor is contrarian.** In the low-position pool, "buy the deepest dips" stocks perform far worse than "did not decline deeply" stocks. This is consistent with the value trap hypothesis — the deepest discounts often correspond to fundamental problems, not buying opportunities.

---

## 3. Alpha Distribution

| Pool | n | Mean | WinsMean | WinsStd | alpha>0% |
|------|---|------|---------|---------|------|
| Full | 573 | 17.13% | 15.35% | 47.1% | — |
| Train | 242 | 20.93% | 20.21% | 53.9% | 64% |
| Test | 331 | 14.35% | 10.48% | 36.0% | 56% |

Train is bull-skewed (bear bottom + early recovery), Test includes turning points (bull top -> bear, crash recovery).

---

## 4. Power Recheck

| Pool | n_unique | n_sb approx 20% | MDE |
|------|---------|----------|-----|
| Full | 573 | 115 | **24.0%** |
| Train | 242 | 48 | 37.1% |
| Test | 331 | 66 | 31.7% |

Full-sample MDE=24.0%: can only detect spread differences >24pp. Test alone is unusable (31.7%).

**Reversal factor spread=-32% in absolute value is large** — the effect size is big enough to potentially be captured by MDE. LLM only needs to beat reversal's -32% spread.

---

## 5. Hard Stop Judgment

| Condition | Status |
|------|------|
| Reversal factor has measurable spread | Pass (-32%, CI spans 50pp but excludes 0) |
| Test timepoints cross regimes | Pass (crash/reversal/bear/sideways) |
| Risk of entering C | Difference CI will be very wide, may be inconclusive |

**Recommendation: Enter C (exploratory).** Reversal spread has large absolute value; 38 CNY LLM cost is extremely low. But must follow: if test CI is wide -> do not draw strong conclusions. Alternative: first expand timepoints >12 to reduce MDE, then run LLM.

---

**Awaiting human decision: Enter C / Expand timepoints first / Stop.**
