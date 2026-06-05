# P1 Eval Characterization Audit: What 0.187 Actually Measures

> Branch: p0a-verify-3-4 (read-only audit) | Date: 2026-05-29
> 
> **Conclusion first**: v4=0.187 is a score computed on 40 stocks, 4 timepoints, 640 samples, using 6-month alpha threshold labels + weighted hit matrix. The score is slightly above uniform random (0.144), but **far below "always predict neutral" (0.401)** — the scoring function structurally rewards neutral. Sample size is too small (40 stocks/4 timepoints), error bars cover most "improvements." No code-level look-ahead leakage detected, but all prompt iterations share the same frozen dataset, creating overfitting risk.

---

## 1. What is Ground Truth?

### Definition: 6-month alpha (excess return) discretized label

**`lib/eval/dataset-builder.js:14-27`**:

```js
const GT_RULES = [
  { minAlpha: 10, label: 'strong_bull' },   // alpha >= 10%
  { minAlpha: 3, label: 'bull' },            // alpha >= 3%
  { minAlpha: -3, label: 'neutral' },        // alpha >= -3%
  { minAlpha: -10, label: 'bear' },          // alpha >= -10%
  { minAlpha: -Infinity, label: 'strong_bear' }, // alpha < -10%
];
```

### Alpha Calculation: stockReturn - CSI300 indexReturn

**`lib/eval/dataset-builder.js:122-148`**:

```js
const fromClose = klines[idx].close;
const toClose = klines[endIdx].close;
const stockReturn = +((toClose - fromClose) / fromClose * 100).toFixed(2);

// CSI300 same period
const iFrom = indexCache.klines[idxFrom].close;
const iTo = indexCache.klines[idxTo].close;
const indexReturn = +((iTo - iFrom) / iFrom * 100).toFixed(2);

const alpha = +(stockReturn - indexReturn).toFixed(2);
```

- **Horizon**: 6 months (`evaluationHorizonMonths = 6`, `dataset-builder.js:46`)
- **Price**: close-to-close (monthly closing price)
- **Adjustment**: pre-adjusted (Eastmoney API default)
- **Suspension/limit-up-down**: Not handled. Suspension month close unchanged -> return approx 0; limit-up/down locked months: monthly close reflects extreme price
- **endIdx selection**: From cutoff, find first bar with `date >= horizonEnd`; if not found, take last kline (`dataset-builder.js:116-121`)

### GT Distribution (v4 eval 640 samples)

```
strong_bull: 180 (28.1%)  <- alpha >= 10%
strong_bear: 176 (27.5%)  <- alpha < -10%
bear:        104 (16.3%)  <- -10% <= alpha < -3%
neutral:      92 (14.4%)  <- -3% <= alpha < 3%
bull:         88 (13.8%)  <- 3% <= alpha < 10%
```

**Real command output** (node computing v4 jsonl):
```
GT dist: {"bear":104,"bull":88,"strong_bull":180,"strong_bear":176,"neutral":92}
```

Strong labels account for 55.6% (strong_bull + strong_bear), neutral only 14.4%. GT distribution is highly polarized.

---

## 2. What is the Score Formula?

### Core weighted score = sum(scores) / n

**`lib/eval/compute-score.js:13-24`**:

```js
export function scorePrediction(predictedSignal, groundTruth) {
  const p = mapSignal(predictedSignal);  // strong_bull=2, bull=1, neutral=0, bear=-1, strong_bear=-2
  const g = mapSignal(groundTruth);
  if (p === g) return 1.0;              // exact match
  if (p * g < 0) {                       // opposite direction
    return (Math.abs(p) === 2 && Math.abs(g) === 2) ? -1.0 : -0.5;  // strong opposite=-1.0, normal opposite=-0.5
  }
  if (p === 0 || g === 0) return 0.3;   // one side neutral -> 0.3
  return 0.5;                            // same direction different strength
}
```

Scoring matrix:

| pred \ GT -> | strong_bull | bull | neutral | bear | strong_bear |
|---------------|-------------|------|---------|------|-------------|
| strong_bull   | 1.0 | 0.5 | 0.3 | -0.5 | **-1.0** |
| bull          | 0.5 | 1.0 | 0.3 | -0.5 | -0.5 |
| neutral       | 0.3 | 0.3 | **1.0** | 0.3 | 0.3 |
| bear          | -0.5 | -0.5 | 0.3 | 1.0 | 0.5 |
| strong_bear   | **-1.0** | -0.5 | 0.3 | 0.5 | 1.0 |

### Weighted Score Calculation

**`lib/eval/compute-score.js:31-47` (computeScoreTransparent)**:

- `full.weightedScore` = sum of all record scores / total records (including parse_failed, score=0)
- `exclPf.weightedScore` = mean excluding `signal === 'parse_failed'`

v4 actual:
```
full.weightedScore = 107.4 / 640 = 0.1678
exclPf.weightedScore = 107.4 / 575 = 0.1868  (excluding 65 parse_failed)
```

**PROGRESS.md's 0.187 is the exclPf value** (107.4/575), from `audit-v4-deep.js:140`.

### How is "bear bias 43%" reflected in the formula?

Not in the formula. bear bias is a **diagnostic metric** of signal distribution (`report.js:202-216`), used to flag model directional bias, not participate in score calculation:

```js
if ((s === 'bear' || s === 'strong_bear') && diff > 0) bearBias += diff;
```

v4 actual: bear 203 + strong_bear 63 = 266/640 = **41.6%** predicted bearish.

### Key Finding: "Always Neutral" Strategy Crushes Models

The neutral row in the scoring matrix **has a floor of 0.3 against every GT** (except neutral->neutral perfect 1.0). Actual:

```
Always neutral:    0.4006  <- optimal null strategy
Always bear:       0.1337
Always bull:       0.1025
Always strong_bull: 0.0369
Uniform random:    0.1444
GT-weighted random: 0.1176
v4 model:          0.1678  <- only +0.023 above random
Frozen baseline:   0.1966  <- known best prompt
```

**Real command output**:
```
=== Null baselines (n=640) ===
Always strong_bull: 23.6  weighted=0.0369
Always bull       : 65.6  weighted=0.1025
Always neutral    : 256.4  weighted=0.4006
Always bear       : 85.6  weighted=0.1337
Always strong_bear: 31.6  weighted=0.0494
Uniform random:   weighted=0.1444
```

0.187 is neither "too weak to be meaningful" nor "too high to be anomalous/leaked" — it is slightly above random baseline, but suppressed by the neutral floor-defined upper bound (0.401). Models making directional predictions bear -0.5/-1.0 penalties, while "always neutral" takes zero risk for 0.3.

---

## 3. Sample Size and Time Range

### Frozen Dataset v1

**`lib/eval/load-frozen-dataset.js` -> `data/frozen-eval-dataset-v1.json`**

**Real command output**:
```
stocks: 40
testPoints: 160
version: frozen-v1
createdAt: 2026-05-18T13:35:49.837Z
uniqueMonths: 4
months: 2024-05,2024-11,2025-05,2025-11
```

### v4 eval Scale

**Real command output**:
```
total lines: 640          (160 testPoints x 4 templates)
valid: 640 | errors: 0
unique codes: 40
per stock: 16 (4 dates x 4 templates)
date range: 2024-05 ~ 2025-11
date list: 2024-05, 2024-11, 2025-05, 2025-11
```

### Statistical Power Assessment

- **n = 640** LLM calls, but only **40 stocks x 4 timepoints = 160 independent (stock, timepoint) pairs**
- 4 templates repeat-evaluate the same samples -> effective independent sample count is closer to 160, not 640
- 40 stocks represent only ~1% of A-shares, no sector/market-cap representativeness
- 4 timepoints are all May or November (semi-annual intervals), covering 1.5 years
- **Error bar for 0.187**: Assuming n_eff approx 160, score std approx 0.5, then SE approx 0.5/sqrt(160) approx 0.04, 95% CI approx +/-0.08
- 0.187 CI covers [0.11, 0.27], cannot distinguish "better than random" from "indistinguishable from random"

---

## 4. Train/Test Methodology Leakage Check

### 4a. Feature Computation: No Look-Ahead Leakage (Pass)

**`lib/eval/runner.js:187-194`** (eval-mc-dropout.js same logic):

```js
const cutoffKlines = allKlines.slice(0, tp.cutoffIndex + 1);
const closes = cutoffKlines.map(k => k.close);
const ma5 = computeMA(closes, 5);
const ma20 = computeMA(closes, 20);
const ma60 = computeMA(closes, 60);
const { dif, dea, hist } = computeMACD(closes);
```

All indicators computed only on `[0, cutoffIndex]` interval; no future data involved. MA/MACD functions (compute-ma.js, compute-macd.js) are standard historical window computations.

### 4b. Data Split: Not walk-forward; same dataset iterated for tuning

- **Not** walk-forward / temporal extrapolation / rolling window
- All 4 timepoints from 2024-05 ~ 2025-11 are **fixed once** as frozen dataset
- All prompt versions (baseline -> v4 -> v13 -> phase15) evaluated on **the same dataset**
- Frozen baseline 0.1966 is the first high score tuned on this dataset (`runA-no-sector-2026-05-18`)
- **Risk**: prompt engineering is essentially "hyperparameter tuning on the same test set"; 0.187->0.197 gains may come from overfitting to 40 stocks/4 timepoints

### 4c. Does v4 eval inject LSTM signal: No (Pass)

**Real command output**: spot-check v4 prompt start:
```
LSTM in prompt: false
Prompt start: You are an A-share technical analyst. Below are the last 194 months of monthly data for 000001...
```

v4 prompt contains only K-line table + MA/MACD, no LSTM/ML signal injection.

### 4d. LSTM MC Dropout eval train/eval overlap: Needs Verification

`cli/eval-mc-dropout.js` uses `mc_dropout_signals.parquet` (pre-computed LSTM prediction JSON), but the training time window for that file is unknown. If LSTM training used data >=2024-05, it is look-ahead leakage. Need to confirm LSTM training cutoff date.

---

## 5. Null Baseline Comparison

See Section 2. Full null baselines:

| Strategy | weightedScore |
|------|--------------|
| Always neutral | **0.4006** |
| Frozen baseline prompt | 0.1966 |
| v4 model (excl PF) | 0.1868 |
| v4 model (full) | 0.1678 |
| Uniform random | 0.1444 |
| Always bear | 0.1337 |
| GT-weighted random | 0.1176 |
| Always bull | 0.1025 |

**Key conclusion**: 0.187 is only +0.023 above random, -0.214 below "always neutral." This score is not statistically distinguishable from random (given n_eff approx 160). It is neither "weak signal" nor "leakage" — it is a number measured on a tiny sample with overfitting risk, under a scoring function that structurally rewards neutral.

### Context Required to Interpret 0.187

Any report of 0.187 (or subsequent scores) must simultaneously report:
1. Same-period null baselines (especially always neutral = 0.401)
2. Sample size n + effective independent sample count
3. Whether sharing same dataset with frozen baseline (0.1966)
4. Number of excluded PF samples

---

## Untouched (Audit Constraint)

| Module | Status |
|------|------|
| lib/eval/*.js | Read-only |
| lib/evaluation/*.js | Read-only |
| cli/eval*.js | Read-only |
| Any LLM call | Not triggered |
| Prompt templates | Untouched |

## Post-Audit Recommendations (no code changes, for human decision)

1. **Prioritize changing scoring function**: Remove neutral's structural advantage (0.3 floor), or switch to rank-based IC and hit rate dual metrics
2. **Expand sample**: 40 stocks 4 timepoints is insufficient. Target >=100 stocks, >=12 timepoints (covering bull/bear cycles)
3. **Time split**: Reserve half the timepoints as hold-out; do not tune prompts on them
4. **Confirm LSTM training cutoff date**: If mc_dropout_signals.parquet training data covers eval timepoints, all mc-dropout eval results are invalid
