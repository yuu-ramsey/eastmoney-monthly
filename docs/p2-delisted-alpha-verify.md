# P2 Delisted Stock Alpha Verification

> Branch: `p2-rebuild-unbiased-pool` | Date: 2026-05-30

## Conclusion: Alpha algorithm is clean. 11 delisted pairs account for only 0.7%, all >=6m horizon, drag down rather than inflate reversal spread.

---

## 1. Horizon Rules

```python
fwd = [data[j][1] for j in range(ci+1, min(len(data), ci+HORIZON+1)) if data[j][1]>0.001]
alpha = (fwd[HORIZON-1] - close) / close * 100 if len(fwd) >= HORIZON else None
```

Insufficient 6 months -> None, excluded from pool. All included pairs have strict 6m horizon.

---

## 2. Spot-Check Sample

| Code | Cutoff | Fwd Months | Alpha | revZ | Bucket |
|------|--------|-------|-------|------|--------|
| 000018 | 2019-06 | 6 | -80.3% | -2.21 | bull |
| 600240 | 2019-06 | 7 | -64.7% | -3.09 | bull |
| 000018 | 2018-12 | 12 | -35.4% | -1.23 | bull |
| 600242 | 2022-06 | 11 | -32.4% | -1.88 | bull |
| 600209 | 2018-06 | 47 | -26.6% | -1.39 | bull |
| 600077 | 2018-06 | 60 | -25.2% | +0.07 | neutral |
| 600074 | 2018-06 | 23 | -21.1% | -0.22 | neutral |
| 600069 | 2019-06 | 13 | +9.9% | -1.26 | bull |

All >=6m. Large negative alpha is reasonable (value destruction before delisting).

---

## 3. Reversal Bucket Distribution

| Bucket | Delisted Stocks |
|----|-------|
| Bullish | 9 |
| Bearish | **0** |

All delisted stocks fall in bullish (deepest decline) bucket; negative alpha drags down bullish mean.

---

## 4. Contribution Decomposition

```
With delisted: spread = +6.63%
Without delisted: spread = +7.47%
Delta: -0.84pp
```

**Delisted stocks are depressing, not inflating, the reversal spread.** +6.6% is a conservative estimate. The expansion root cause is Baostock covering 990 new codes (non-delisted), not delisted stocks flooding in.
