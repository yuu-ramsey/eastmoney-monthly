# P1 Eval Rebuild Design: Low-Position Subset + Dual Rulers + Hold-Out

> Branch: `p1-eval-rebuild` | Date: 2026-05-29
>
> UNIVERSE = **low_position**
>
> Background audits: `docs/p1-eval-characterization.md` (pathological scoring matrix), `docs/p1-power-analysis.md` (MDE=31.5%)
>
> **Design goal**: Retire the manual scoring matrix; evaluate LLM prediction quality on point-in-time low-position subset using two independent rulers; hold-out timepoints are never used for prompt tuning. Win condition: whether LLM can beat the "buy the dip" reversal factor.

---

## 0. Retire Scoring Matrix

`lib/eval/compute-score.js`'s `scorePrediction()`:
- Keep code but do not delete; add comment at top marking known flaws
- No longer serve as primary metric. Subsequent cli/eval-*.js switch to new dual rulers

```js
// ⚠ DEPRECATED as primary metric — see audit docs:
//   docs/p1-eval-characterization.md (neutral 0.3 floor, always-neutral=0.401)
//   docs/p1-conditional-signal.md   (spread 7.5% not significant, score matrix noise)
//   docs/p1-power-analysis.md       (MDE=31.5%, 34 sb pairs cannot detect effect)
```

---

## 1. Low-Position Criteria Definition

### Primary Definition (fixed once, never change)

```
N = 12  (lookback months)
X = 20% (range bottom threshold)

Condition 1: close_position_12m = (close - min(close_t-12 .. close_t-1)) / (max_t-12..t-1 - min_t-12..t-1) <= 0.20
Condition 2: close < MA60 (60-month moving average, includes current close)

Included = Condition 1 AND Condition 2
```

- **Condition 1**: close is in the bottom 20% of the past 12-month range -> recent sustained decline / testing support
- **Condition 2**: price below long-term moving average -> medium-to-long-term also at low position
- Lookback window and MA computation use **only pre-cutoff data** (point-in-time)

### Sensitivity Analysis (run separately)

Fix Condition 2 (`< MA60`), run for X in {10%, 20%, 30%} each, check whether conclusions are robust.

### Per-Timepoint Inclusion Count Volatility (actual)

**Real DB query output** (including MA60 "both" column):
```
2018-12 (bear bottom):  110 stocks
2019-06 (recovery):      22 stocks
2020-03 (COVID):         84 stocks
2020-12 (bull):          29 stocks
2022-10 (bear):         139 stocks
2024-10 (strong bull):   12 stocks
```

**Inclusion count fluctuates 12~170; processing rules:**
1. Per-timepoint **cap 140 stocks** (if exceeds, randomly sample 140, seed fixed for reproducibility)
2. If insufficient, do not supplement; truthfully record actual inclusion count
3. Report per-timepoint n; mark "thin month" (< 30)

---

## 2. Timepoint Selection & Hold-Out

### Timepoints (12, across 4 regimes, 6m horizon room)

| Timepoint | Regime | Next 6m Direction | Train/Test |
|------|--------|-----------|------------|
| 2018-06 | bear mid | Down (2018 H2 bear) | Train |
| 2018-12 | bear bottom | Up (2019 H1 recovery) | Train |
| 2019-06 | recovery mid | Up (2019 H2) | Train |
| 2020-03 | COVID crash | Up (V-recovery) | **Test** |
| 2020-09 | recovery | Up (2021 bull) | Train |
| 2021-06 | bull mid | Up->Down (top) | Train |
| 2021-12 | bull top | Down (2022 bear) | **Test** |
| 2022-06 | bear mid | Down-> | Train |
| 2022-10 | bear bottom | Up (2023 H1) | **Test** |
| 2023-06 | sideways | Down-> | **Test** |
| 2024-02 | recovery dip | Up (2024 H1) | **Test** |
| 2024-10 | bull | ? (2025 H1) | **Test** |

**Hold-out**: 6 Test timepoints. Covers crash/reversal/bear-bottom/sideways/bull — ensures test spans regimes.

**Train timepoints (6) are for prompt iteration only. Test timepoints (6) are only run once at final acceptance.**

### Estimated Sample Sizes

| Metric | Train | Test | Total |
|------|-------|------|-------|
| Unique pairs (~80/tp) | ~480 | ~480 | ~960 |
| Records (x4 templates) | ~1920 | ~1920 | ~3840 |
| Winsorize MDE (std approx 20%, n_sb~96/side) | ~8% | ~8% | — |
| LLM cost (DeepSeek ~0.01 CNY/call) | 19 CNY | 19 CNY | 38 CNY |

vs current 160 pairs, MDE=31.5%: independent samples expanded **6x**, MDE reduced from 31.5% to **~8%**.

---

## 3. Suspension/Limit-Up-Down/Delisted Alpha Handling

### Delisted

**Must include**. Stocks existing at point-in-time construction, even if later delisted, are included.
After delisting, close = delisting price (0 if unavailable); alpha computed to last month.

### Suspension

Suspension-month close unchanged (Eastmoney fills with last traded close).
Continuous suspension > 3 months -> mark `suspended: true`, report separately in analysis.

### Limit-Up/Down Locked

Monthly close reflects actual limit price; no correction.
Intra-month trading days < 5 -> mark `thin_trading: true`.

---

## 4. Dual Rulers

All metrics report 95% CI via **(stock, timepoint) block bootstrap**.

### Ruler 1: Robust Center (Winsorize)

| Metric | Definition |
|------|------|
| alpha median | Median of bucket alpha (naturally robust to extremes) |
| Winsorize mean | Trim alpha < p1 or > p99 to boundary, then compute mean |
| Winsorize spread | Winsorize bullish mean - Winsorize bearish mean |

**Judgment**: spread > 0 and 95% CI excludes 0 -> directional discrimination.

### Ruler 2: Tail Capture

| Metric | Definition |
|------|------|
| Top-20% capture rate | Proportion of real alpha top-20% predicted as bullish |
| >20% mega-bull hit rate | Proportion of alpha > 20% predicted as strong_bull/bull |
| Bullish portfolio expected return | Buy all bullish predictions, equal-weight hold 6m, real mean return |

**Judgment**: capture rate > 20% -> model identifies alpha outliers.

---

## 5. Null + Reversal Factor Baseline

⚠ **Do NOT use** momentum factor — low-position stock momentum is universally negative; momentum is not a valid null.

### 5a. Simple Null

| Baseline | Strategy |
|------|------|
| Always-bullish | All pairs predict bull |
| Always-neutral | All pairs predict neutral |
| Uniform random | 5-way uniform random x 1000 iterations |

### 5b. Reversal/Oversold Factor (the target LLM must beat)

Per pair, computed from pre-cutoff data (no look-ahead):

| Factor | Formula | Direction |
|------|------|------|
| Short-term reversal 1m | trailing 1-month return | Lower = more bullish |
| Short-term reversal 3m | trailing 3-month return | Lower = more bullish |
| MA60 discount | (close - MA60) / MA60 | More negative = more bullish |
| RSI oversold | RSI(14) < 30 | binary bullish |

**Composite reversal score**: Equal-weight Z-score average -> sort -> top 20% mark bullish, bottom 20% mark bearish.

### 5c. Key Judgment

```
Model spread > Reversal factor spread AND CI excludes 0 -> LLM has independent edge
Model spread <= Reversal factor spread               -> LLM adds no value; one-line factor replaces it
```

---

## 6. Implementation Plan

### Stage A: Build Sample Pool (this design; implementation on separate branch)

File: `lib/eval/dataset-builder-lowpos.js`

1. Load full monthly data + sector mapping from SQLite
2. For 12 cutoff months, point-in-time compute low-position filter + MA60
3. For each (stock, cutoff), compute 6m forward alpha (including delisted/suspension handling)
4. Record per-timepoint actual inclusion count
5. Output: `data/frozen-eval-lowpos-v1.json`
6. Run X in {10%, 20%, 30%} sensitivity

### Stage B: New Scoring + Baselines (implementation on separate branch)

File: `lib/eval/rulers.js`

1. Retire `compute-score.js` (add deprecation comment)
2. Implement `dualRulers(results)` + `blockBootstrap()`
3. Implement `reversalFactor(klines, cutoffIdx)` — pure pre-cutoff data
4. Implement `evalLowPosition(dataset, results) -> report`

### Stage C: Run LLM + Generate Report (implementation on separate branch)

1. Build baseline prompt on Train timepoints
2. Test timepoint acceptance (once only)
3. Compare against reversal factor baseline
4. Output `docs/p1-eval-rebuild-results.md`

---

## 7. Do-Not-Touch List

| Module | Status |
|------|------|
| Agent prompt (bull/bear/predictor/judge) | Do not touch |
| LLM provider (anthropic/deepseek) | Do not touch |
| score-fusion | Do not touch |
| Structured output parsing | Do not touch |
| `lib/eval/compute-score.js` | Retain + deprecation comment |
| Low-position criteria formula | Fixed after this design, do not change |
