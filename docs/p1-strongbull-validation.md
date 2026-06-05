# P1 strong_bull Validation: LLM's Only Edge vs Naive Momentum Baseline

> Branch: p0a-verify-3-4 | Date: 2026-05-29
>
> **Judgment: C — strong_bull itself is not significant. unique-pair level CI lower bound (5.02%) < unconditional mean (6.07%).
> Error bars from 34 independent samples cover everything. Must expand sample before drawing any conclusions.**

---

## Prerequisite: How 40 Stocks Were Selected

**`scripts/build-frozen-dataset.js:1-10`** — frozen dataset extracted from runA eval JSONL:

```js
stocks.set(r.stockCode, { category: r.category || 'hs300' });
```

**`lib/eval/seed-stocks.json`** — seed stocks in 6 morphology categories, 5-8 each, listed >=5 years. User hand-picked.

40 stocks come from hand-picked seed pool + data availability filter (>=24 monthly K-lines). **Survivorship bias** is obvious: stocks that survived to 2024 with 15+ years of continuous monthly data are inherently stronger (unconditional alpha mean 6.07%/6 months).

---

## Method

**`cli/eval-strongbull-vs-momentum.js`** — Analysis at unique (stock, cutoffDate) granularity:

1. For each pair (stock, cutoffDate), compute naive momentum from monthly data before cutoff (trailing 6m/12m return + MA60 deviation, Z-score equal-weight composite)
2. Take same number of top momentum pairs as strong_bull count
3. All CIs via block bootstrap (block = (stock, cutoffDate))
4. No LLM rerun

---

## 1. strong_bull Self-Significance

| Metric | Value |
|------|-----|
| Full sample unique pairs | 160 |
| strong_bull unique pairs | 34 (21.3%) |
| strong_bull alpha mean | 20.13% |
| Unconditional alpha mean | 6.07% |
| **95% CI (block bootstrap)** | **[5.02%, 41.06%]** |

**Real command output**:
```
alpha mean: 20.13% (unconditional: 6.07%)
95% CI: [5.02%, 41.06%]
-> CI lower bound below unconditional -> not significant (Fail)
```

CI lower bound 5.02% < unconditional 6.07%. Even though strong_bull mean is a high 20%, with only 34 independent samples, null cannot be rejected.

---

## 2. Momentum Baseline

| Metric | Value |
|------|-----|
| Momentum top-34 alpha mean | 12.72% |
| **95% CI** | **[4.32%, 22.58%]** |

**Real command output**:
```
alpha mean: 12.72%
95% CI: [4.32%, 22.58%]
```

Momentum itself is also not significant (CI contains unconditional mean). For top-34 out of 160 pairs, any ranking metric's CI will be wide.

---

## 3. Overlap

| Metric | Value |
|------|-----|
| LLM strong_bull | 34 |
| Momentum top-34 | 34 |
| Intersection | 16 |
| **Jaccard** | **0.308** |

**Real command output**: `LLM: 34  Momentum: 34  Intersection: 16  Jaccard: 0.308`

Low overlap. LLM and momentum only share ~1/3 of picks. Even though both are insignificant, they select different stocks.

---

## 4. Difference Significance

| Metric | Value |
|------|-----|
| strong_bull alpha | 20.13% |
| Momentum alpha | 12.72% |
| Difference (sb - mom) | +7.40% |
| **95% CI** | **[-7.01%, +27.52%]** |

**Real command output**: `Difference: 7.40%, 95% CI: [-7.01%, 27.52%]`

Difference CI crosses zero, span 34pp. Cannot say strong_bull is significantly better than momentum.

---

## Judgment

```
-> C) strong_bull itself not significant -> expand sample before drawing conclusions
```

| Judgment Condition | Result |
|----------|------|
| strong_bull CI lower bound > unconditional 6.07%? | **No** (5.02% < 6.07%) |
| Difference CI > 0? | **No** (contains 0) |

**Not A and not B.** The strong_bull bucket has no statistically significant alpha advantage at the unique-pair level. The problem is not in the model — it is in the sample.

---

## Script

`cli/eval-strongbull-vs-momentum.js`:
```
node cli/eval-strongbull-vs-momentum.js
```

## Untouched

| Module | Status |
|------|------|
| Any eval logic | Untouched |
| prompt/LLM | Untouched |
| frozen dataset | Read-only |
