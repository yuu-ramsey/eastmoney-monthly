# P1 Power Analysis: How Many Samples Needed to Detect strong_bull Effect

> Branch: p0a-verify-3-4 | Date: 2026-05-29
>
> **Conclusion: With current alpha std=33.7%, 34 strong_bull independent pairs can only detect effects >31.5%.
> The current point estimate 12.5% is far below the detection threshold — all prior "not significant" results are not due to model problems but insufficient N.
> Winsorize + effect=6% -> need ~330 strong_bull approx 1660 total pairs.**

---

## Data Foundation

### Stock Selection Criteria

**`scripts/build-frozen-dataset.js` + `lib/eval/seed-stocks.json`**:
- 40 stocks, all marked "hs300", from user hand-picked seed pool (6 morphology categories, 5-8 each)
- Selection criteria: listed >=5 years, data >=24 monthly bars -> **survivorship bias**
- 40/2247 A-shares = 1.8%, no sector/market-cap representativeness

### Key Parameters

```
n_unique (stock, timepoint): 160
strong_bull unique:  34 (21.3%)
alpha cross-sectional std:    33.7%
alpha mean:          6.1%
design effect deff:       1.90 (ICC=0.3, 4 records/pair)
```

**Real output**:
```
alpha std: 33.7 %
avg records/pair: 4.0
design effect: 1.90
```

---

## 1. MDE Curve

Fixed power=0.8, alpha=0.05, two independent group mean difference, design effect=1.90:

| n_sb | total pairs | MDE | Description |
|------|------------|-----|------|
| **34 (current)** | ~170 | **31.5%** | Can only detect effects >37.6% |
| 60 | ~300 | 23.7% | Can detect >29.8% |
| 100 | ~500 | 18.4% | Can detect >24.5% |
| 160 | ~800 | 14.5% | Can detect >20.6% |
| 240 | ~1200 | 11.9% | Can detect >18.0% |
| 360 | ~1800 | 9.7% | Can detect >15.8% |
| 600 | ~3000 | 7.5% | Can detect >13.6% |

**Current MDE=31.5%, point estimate=12.5%**. That the point estimate is below MDE means: even if the true effect really is 12.5%, with current sample size it cannot be reliably detected (power < 0.8).

**Real output**:
```
  n_sb=  34  total~170  MDE=31.5%
  n_sb=  60  total~300  MDE=23.7%
  n_sb= 100  total~500  MDE=18.4%
  n_sb= 160  total~800  MDE=14.5%
```

---

## 2. Reverse: Given Effect -> Required Sample

| Effect Size | n_sb Needed | total pairs | Scale (6 timepoints/stock) |
|---------|----------|------------|-----------------|
| 12.5% (point estimate) | 216 | ~1080 | 180 stocks x 6t |
| 6.0% (half) | 938 | ~4690 | 782 stocks x 6t |
| 4.2% (1/3) | 1913 | ~9565 | 1595 stocks x 6t |
| 3.0% (conservative) | 3750 | ~18750 | 3125 stocks x 6t |

**Real output**:
```
  12.5%  n_sb= 216  total~ 1080  180stocks x 6t
  6.0%   n_sb= 938  total~ 4690  782stocks x 6t
  4.2%   n_sb=1913  total~ 9565  1595stocks x 6t
  3.0%   n_sb=3750  total~18750  3125stocks x 6t
```

At std=33.7%, even for "effect = half of point estimate", 782 stocks x 6 timepoints is needed — far beyond reasonable scope. **Reducing std first is the only path.**

---

## 3. After Winsorize (trim p1/p99, std approx 20%)

6 extreme alphas (<=-30% or >=150%) contribute ~60% of variance. Winsorize can reduce std to ~20%.

| Effect Size | n_sb Needed | total pairs | Scale (6 timepoints/stock) |
|---------|----------|------------|-----------------|
| 12.5% (point estimate) | 77 | ~385 | 65 stocks x 6t |
| 6.0% (half) | 332 | ~1660 | 277 stocks x 6t |
| 4.2% (1/3) | 676 | ~3380 | 564 stocks x 6t |
| 3.0% (conservative) | 1325 | ~6625 | 1105 stocks x 6t |

**Real output**:
```
  12.5%  n_sb=  77  total~  385  65stocks x 6t
  6.0%   n_sb= 332  total~ 1660  277stocks x 6t
```

---

## 4. Scale Recommendation

**One-liner: Reduce alpha std from 33.7% to 20% (Winsorize), effect >= 6% -> need ~330 strong_bull approx 1660 total unique pairs approx 280 stocks x 6 timepoints.**

Two decision paths:

| Path | Plan | Scale | Pros | Cons |
|------|------|------|------|------|
| A (Recommended) | Winsorize + expand to 100 stocks x 12 timepoints | 1200 pairs, ~240 sb | Covers bull/bear, 4-year span | Needs more historical data |
| B | Winsorize + expand to 200 stocks x 4 timepoints | 800 pairs, ~160 sb | Keeps it simple | Too few timepoints, seasonal effects |

**If the target is effect=6% (half point estimate) significant at power=0.8, Path A is the minimum viable plan.**

Current 40 stocks x 4 timepoints = 160 pairs is only an exploratory pilot, not a sample capable of statistical inference.

---

## Methodology Notes

- **MDE formula**: Two independent group mean difference t-test, two-sided alpha=0.05, power=0.8
- **Design effect**: deff = 1 + (m - 1) x ICC, m=4 records/pair, ICC=0.3 (conservative estimate of template correlation)
- **strong_bull proportion**: Fixed at 20% (based on v4's 34/160 = 21.3%)

## Script

`cli/eval-power-analysis.js`:
```
node cli/eval-power-analysis.js
```

## Untouched

| Module | Status |
|------|------|
| Any eval logic | Untouched |
| Any LLM call | Not triggered |
