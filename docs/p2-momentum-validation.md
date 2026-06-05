# P2 Momentum Factor Validation: Hold-Out + Regime + Tradability

> Branch: `p2-momentum-validate` | Date: 2026-05-29
>
> ⚠ Survivors-only; delisted stocks missing (`docs/p1-survivorship-stress.md`); absolute returns contaminated by survivorship bias
>
> **⚠ B4 Correction: Phase B4's +29.5% confirmed as bug upon recalculation (sort array shared object reference). Correct full-sample momentum spread = -13.1%, CI contains 0.**

---

## V1: Hold-Out

| Pool | n | spread | 95% CI | Thin |
|------|---|--------|--------|-----|
| Full | 573 | -13.1% | [-44.7, +4.8] | — |
| Train | 242 | -24.1% | [-57.4, +5.5] | 4/6 |
| Test | 331 | -8.1% | [-47.3, +18.1] | 2/6 |

**Train/Test both CI contain 0 -> Not passed.** Test magnitude shrinks to 1/3 of full sample.

---

## V2: Regime

| Regime | Timepoints | n | spread | 95% CI |
|--------|------|---|--------|--------|
| Up | 2018-06~2020-09 (5, CSI300 +7~+90%) | 260 | **-42.2%** | **[-94.5, -2.5]** |
| Down | 2021-06~2024-02 (5, CSI300 -6~-15%) | 193 | **+20.6%** | **[+10.8, +32.4]** |
| Sway | 2022-10, 2024-10 (2) | 120 | -9.1% | [-30.9, +12.5] |

**Momentum is beta exposure, not independent alpha.** Loses big in up markets, gains small in down markets — profit/loss direction determined by market regime.

---

## V3: Tradability

| Metric | Value |
|------|-----|
| Long-only excess | **-2.79%** |
| Winsorize spread | -13.1% |
| Extreme contribution | -4.3pp |
| Turnover | 91%/period |
| Net long after 0.2% cost | -3.15% |
| Net long after 0.3% cost | -3.33% |

**Buy momentum top 20% underperforms equal-weight hold; extremely high turnover; net costs = total loss.**

---

## Overall Judgment

| Dimension | Conclusion |
|------|------|
| V1 hold-out | Not passed — CI contains 0 |
| V2 regime | Beta exposure — loses in up markets/wins in down markets |
| V3 tradability | Not tradable — net excess negative |

**+29.5% was a B4 bug. Momentum is not a free signal. Prerequisites for productization: expand sample + add delisted stocks + pre-register regime.**
