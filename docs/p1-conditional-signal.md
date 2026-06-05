# P1 Conditional Signal Analysis: Real Alpha Assessment Without Scoring Matrix

> Branch: p0a-verify-3-4 | Date: 2026-05-29
> 
> **Conclusion first: The model can identify extreme momentum stocks (strong_bull bucket alpha=18-20% vs unconditional 6%),
> but has no significant ability to differentiate general bullish vs bearish (spread 95% CI contains 0).
> Directional accuracy 59% is only +8pp above always-bullish, statistically unreliable.
> The original score improvement 0.187->0.197 is completely invisible in alpha space.**

---

## Analysis Method

**`cli/eval-reanalyze-conditional.js`** — Completely discards the `scorePrediction` scoring matrix,
using only real 6-month alpha (from `frozen-eval-dataset-v1.json`) to evaluate prediction informativeness.

Comparing two evals:
- v4-signals (0.187 exclPF, 640 records)
- frozen-baseline (0.1966, 640 records)

---

## 1. Conditional Realized Returns

### v4-signals

```
Full sample (n=640): alpha mean=6.07%  median=0.88%  alpha>0%=51.2%

Prediction bucket   n    alpha mean  alpha median  alpha>0%   vs unconditional Delta
--------------------------------------------------
strong_bull      104    18.54%     6.19%    61.5% +   12.47%
bull             141     4.50%     2.41%    58.9%    -1.57%
neutral           64     1.17%    -4.22%    45.3%    -4.89%
bear             203     0.79%    -3.56%    39.9%    -5.27%
strong_bear       63     9.94%    -2.27%    49.2% +    3.87%
```

**Real command output** (`node cli/eval-reanalyze-conditional.js`):
```
strong_bull      104    18.54%     6.19%    61.5% +   12.47%
bull             141     4.50%     2.41%    58.9%    -1.57%
neutral           64     1.17%    -4.22%    45.3%    -4.89%
bear             203     0.79%    -3.56%    39.9%    -5.27%
strong_bear       63     9.94%    -2.27%    49.2% +    3.87%
```

**Judgment**:
- strong_bull bucket: alpha=18.54%, far above unconditional 6.07%. (Pass) The model can identify extreme momentum stocks
- bull bucket: alpha=4.50%, **below** unconditional mean. (Fail) "Generally bullish" does not bring excess returns
- bear bucket: alpha=0.79%, **positive**. (Fail) Stocks predicted "bearish" are actually still rising
- **strong_bear bucket: alpha=9.94%, positive and above unconditional**. (Fail, Fail) Stocks the model considers "worst" actually rose better — this is not a contrarian signal; it is non-discrimination

### frozen-baseline

```
Prediction bucket   n    alpha mean  alpha median  alpha>0%   vs unconditional Delta
--------------------------------------------------
strong_bull      107    20.22%     7.24%    65.4% +   14.16%
bull             150     3.44%     1.36%    58.0%    -2.62%
neutral           72     4.49%    -3.74%    48.6%    -1.57%
bear             250     1.55%    -3.56%    40.8%    -4.52%
strong_bear       54     7.24%     1.77%    53.7% +    1.18%
```

Same pattern: strong_bull stands out alone; remaining buckets are disordered.

---

## 2. Directional Discrimination (spread)

| Metric | v4 (0.187) | baseline (0.197) |
|------|-----------|-----------------|
| bullish alpha mean | 10.46% | 10.43% |
| bearish alpha mean | 2.96% | 2.56% |
| **spread** | **7.50%** | **7.87%** |
| **95% CI (block bootstrap)** | **[-2.16%, +19.77%]** | **[-1.67%, +19.45%]** |
| n_eff (blocks) | 160 | 160 |

**Real command output**:
```
v4: spread 95% CI: [-2.16%, 19.77%], n_eff approx 160
bl: spread 95% CI: [-1.67%, 19.45%], n_eff approx 160
-> (Fail) CI contains 0 -> cannot reject spread=0; prediction has no significant directional discrimination
```

**spread 7.5% looks large but CI width is +/-11%.** This is a direct consequence of 40 stocks x 4 timepoints = 160 blocks — 4 templates within a block are highly correlated; effective sample size is insufficient to separate signal from noise.

v4 (0.187) and baseline (0.197) have nearly identical spread (7.50% vs 7.87%); the 0.01 score difference is **completely invisible** in alpha space.

---

## 3. Directional Accuracy

| Metric | v4 (0.187) | baseline (0.197) |
|------|-----------|-----------------|
| Directional predictions n | 511 | 561 |
| Hits n | 301 | 330 |
| **Hit rate** | **58.9%** | **58.8%** |
| Always-bullish comparison | 50.7% | 51.3% |
| **Excess** | **+8.2pp** | **+7.5pp** |

By bucket (v4):
```
strong_bull  n=104  hits=64 (61.5%)  <- indeed elevated
bull         n=141  hits=83 (58.9%)  <- near prior
bear         n=203  hits=122 (60.1%) <- but bear needs alpha<0 to "hit"
strong_bear  n= 63  hits=32 (50.8%)  <- near coin flip
```

Note: "hit" for bear/strong_bear is defined as alpha < 0. The bear bucket alpha mean is +0.79% (actually rising), but 60.1% of bear predictions correspond to alpha < 0 -> this is "a stopped clock is right twice a day" — because the full-sample prior of 49% alpha < 0 is already high.

---

## 4. Comparison with Original Scoring Matrix

| Dimension | Original Scoring Matrix | Real Alpha |
|------|----------|---------|
| Always neutral | 0.401 (strongest strategy) | 1.17% (below unconditional mean) |
| v4 vs baseline difference | 0.187 vs 0.197 (+5.3%) | spread 7.50 vs 7.87 (nearly identical) |
| Statistical significance | Not reported | (Fail) Not significant |
| Can distinguish bull vs bear | Matrix guarantees score (0.5) | Actually cannot |

The original scoring matrix has structural flaws (0.3 floor), and makes neutral the optimal strategy. Real alpha analysis shows the actual return difference between "predict bull" and "predict bear" is wide and not significant.

---

## 5. Summary

1. **Model information is concentrated in the strong_bull bucket** — approximately 100/640 predictions can select stocks with alpha ~19%, but account for less than 1/6
2. **No significant differentiation among bull / bear / neutral / strong_bear buckets** — disordered ranking, overlapping CIs
3. **spread 7.5% is statistically insignificant at n_eff=160** — needs ~4x sample (>=40 stocks x >=20 timepoints) to push CI to exclude 0
4. **0.187 vs 0.197 is a no-op in alpha space** — the two prompt versions have completely identical prediction quality. Minute scoring matrix differences come from parse_failed proportion and matrix noise, not prediction capability improvement

**Recommendation for next steps**: Before expanding sample (>=100 stocks, >=12 timepoints), any score difference <0.02 should not be interpreted as "improvement." Prioritize sample expansion over continued prompt tuning.

---

## Script

`cli/eval-reanalyze-conditional.js` — standalone CLI, usage:
```
node cli/eval-reanalyze-conditional.js
```
Does not call LLM, does not write files; only reads existing jsonl + frozen dataset for pure numerical statistics.

## Untouched

| Module | Status |
|------|------|
| lib/eval/compute-score.js | Untouched |
| Any LLM call | Not triggered |
| frozen dataset | Read-only |
