# P2 Unbiased Low-Position Pool — Baostock Single-Source Rebuild

> Branch: `p2-rebuild-unbiased-pool` | Date: 2026-05-30
>
> Single-source stamp: Baostock (adjustflag=2 pre-adjusted), 3294 codes

---

## 1. Pool Comparison

| Metric | v1 Survivors-only | v2 Baostock | Delta |
|------|-------------|------------|----|
| Pairs | 732 | **1568** | +2.1x |
| Thin (<30) | 4/12 | **0/12** | All eliminated |
| Alpha mean | 15.54% | **7.11%** | **-8.43pp** |
| Winsorize mean | 13.95% | 6.49% | -7.46pp |
| Std | 61.3% | 31.0% | -49% |
| Winsorize std | — | 26.2% | — |

### Per-Timepoint (all >=70)

```
2018-06:140  2018-12:140  2019-06:140  2020-03:140
2020-09:140  2021-06:140  2021-12: 75  2022-06:140
2022-10:140  2023-06:140  2024-02:140  2024-10:100
```

---

## 2. Survivorship Bias Quantified

**Survivor pool overstates alpha by 8.4pp.** Missing delisted stocks systematically exclude the worst subsequent performance.

---

## 3. MDE

| Pool | n | wStd | MDE |
|------|---|------|-----|
| v1 | 573 | ~47% | 24.0% |
| v2 Full | 1568 | 26.2% | **8.1%** |
| v2 Train | 835 | — | 11.1% |
| v2 Test | 733 | — | 11.8% |

**MDE improved 3x (24% -> 8.1%).** Test MDE=11.8%; single-digit edges detectable on full sample.

---

## 4. To Be Continued

Momentum/reversal baseline recalculation needs Baostock klines cache persisted before running.
