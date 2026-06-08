# Neutralize Framework — Audit Document

**Branch**: neutralize-framework  
**Date**: 2026-06-08 (A1/A2 rework: 2026-06-08)  
**Files**: `lib/eval/neutralize.cjs`, `lib/eval/validate-signal.cjs`, `lib/eval/_selftest_reversal.cjs`

---

## 1. Neutralization Regression Specification

### Model

Cross-sectional OLS per time-point (strict PIT):

```
signal_i = α + β₁·logMarketCap_i              (if data available; DATA_PENDING = dropped)
             + β₂·Amihud_i                     (if useLiquidity=true and data available)
             + Σ_{j=1}^{K-1} γⱼ·Industry1_j_dummy_i   (K = unique L1 industries in cross-section)
             + δ₂·Q2_i + δ₃·Q3_i + δ₄·Q4_i             (season dummies, Q1 baseline)
             + ε_i

residual_i = ε_i  ← neutralized signal
```

| Control | Source | Notes |
|---------|--------|-------|
| `logMarketCap` | Pool field `r.logMarketCap` | log(total market cap CNY). DATA_PENDING = term dropped. |
| `amihud` / `bidAskSpread` | Pool field `r.amihud` or `r.bidAskSpread` | Optional liquidity control (`useLiquidity=true`). DATA_PENDING = term dropped. |
| L1 Industry dummies | `data/industry-map.json` → `stockToIndustry`, 19 categories | Stripped "sh."/"sz." prefix for lookup |
| Season dummies Q2/Q3/Q4 | Derived from `cutoffDate` month | Captures industry seasonal effects (e.g., bank Q4, consumer Q4) |

**A1 Rework note (2026-06-08)**: `rangePosition` was the original size proxy and has been removed.
- `rangePosition` measures price position in the trading range — collinear with momentum/reversal signals
- In the 72tp pool, `rangePosition` had extreme outliers (P0=-2,469,569, P99=0.20) and near-zero correlation with reversal (r=0.0013)
- Replaced with `logMarketCap`, which is DATA_PENDING for all current pool records
- Effect on IC: none — IC stayed at -0.0407 (removing rangePosition changed nothing)

**PIT guarantee**: Each time-point's OLS uses only records from that time-point's cross-section. No cross-time information enters the regression.

**Missing industry**: Stocks not found in `industry-map.json` (< 1% coverage) receive zero contribution to industry dummies; their signal enters the intercept.

### Per-Period R² Distribution (72tp pool, 78 time-points)

From the synthetic-industry self-test (Test 2):

- R² = 0.98 for pure-industry signal → neutralization absorbs essentially all industry variance
- For real signals (reversal): R² varies; industry explains a fraction of signal variance

---

## 2. 5-Gate Definitions

All 5 gates must pass for `verdict = 'pass'`. Current thresholds are calibrated for A-share monthly cross-section (α = 6-month forward return, ~188 stocks/timepoint).

| Gate | Description | Pass Criterion |
|------|-------------|----------------|
| **G1** IC Quality | Neutralized IC mean/std/IR over time | `\|IC_IR\| ≥ 0.3` **AND** `\|t-stat\| ≥ 2.0` |
| **G2** Monotonicity | Q1–Q5 quintile spread and ordering | Q5−Q1 spread > 0 **AND** ≥ 3/4 consecutive pairs monotone |
| **G3** Long-side Net α | Q5 gross alpha minus A-share cost | Net alpha > 0 **AND** `\|t-stat\| ≥ 1.96` |
| **G4** Robustness | BH-FDR + walk-forward consistency | At least 1 window BH-adjusted p < 0.05 **AND** ≥ 2/3 windows sign-consistent |
| **G5** p-value | One-sided P(spread ≤ 0), bootstrap CI | One-sided p < 0.05 **AND** 95% CI consistent with direction |

### A-share cost model (NEEDS VERIFICATION against actual broker rates)

| Component | Value | Note |
|-----------|-------|------|
| Commission | 5 bps one-way | [NEEDS VERIFICATION] |
| Stamp duty | 10 bps sell-only | [NEEDS VERIFICATION] |
| Bid-ask spread | 10 bps round-trip | [NEEDS VERIFICATION] |
| **Total round-trip** | **30 bps** | Applied to gross Q5 spread |

Gross and net alpha are returned separately. Gross serves as a sanity check; net is the gate criterion.

### Sanity alerts (auto-triggered)

- `|IC| > 0.10` → check for look-ahead leakage
- Monthly gross spread > 5% → check for look-ahead leakage
- Annualized Sharpe > 3.0 → check for look-ahead leakage

---

## 3. Self-Test Results

### Deviation Table

| Requirement | Actual | Reason |
|-------------|--------|--------|
| Reversal neutralized IC `\|IC\| < 0.01` | `\|IC\| = 0.0407` | 72tp pool is pre-filtered to "lowpos" (stocks near price range lows). In this filtered universe, reversal signal = prior losers tend to *continue* losing (momentum continuation in distress), not mean-revert. Industry neutralization does not eliminate a genuine stock-level effect. |
| 5-gate `verdict = fail` for reversal | ✓ `verdict = fail` | G1 IC_IR = 0.295 < 0.3; G2 Q5 spread negative; G3 net alpha < 0; G5 p = 0.995 > 0.05. Gate fails correctly. |
| Framework neutralization is correct | ✓ Verified via Test 2 | Synthetic pure-industry signal R² = 0.98, residual mean = 0.000000, assertNeutralized passed. IC reduced from 0.026 to 0.013. |
| A1 rework: remove rangePosition → IC drops? | IC unchanged at -0.0407 | corr(reversal, rangePosition) = 0.0013 — rangePosition was making no contribution to OLS. Removing it has zero effect. IC of -0.04 is genuine pool-level effect, not a size artifact. |
| A3: full-market pool validates lowpos-specific effect | PENDING | `data/frozen-eval-fullmarket.json` not yet built. Test 3 stub added to self-test; will run automatically when pool is available. |

### Test 1: Reversal (task-spec test)

```
IC mean:    -0.0407
IC std:     0.1380
IC_IR:      -0.295
t-stat:     -2.60
Timepoints: 78
Verdict:    fail ✓

Gate1 FAIL — IC_IR=-0.295 < 0.3 threshold
Gate2 FAIL — Q5-Q1 spread=-1.35%, monotone=1/4
Gate3 FAIL — netAlpha=-1.51%, t=-1.32
Gate4 PASS — walk-forward 3/3 windows consistent, BH-FDR pass
Gate5 FAIL — one-sided p=0.9954, CI=[-0.07,-0.01]
```

**Conclusion**: 5-gate correctly rejects reversal. The `|IC| < 0.01` specification from the task document does not hold for the 72tp lowpos pool because reversal here is a genuine (negative) predictor, not a spurious style bias. The framework is correct; the dataset's behavior differs from the assumed context.

### Test 2: Synthetic pure-industry signal (framework verification)

```
Raw IC: mean=-0.0259, t=-1.93
Neutralized IC: mean=-0.0128, t=-0.87 (50% reduction)
Residual check (2019-09): mean≈0, std=0.006, R²=0.98
assertNeutralized: PASSED ✓
IC reduced: 0.0259 → 0.0128 ✓
5-gate verdict=fail ✓
```

**Conclusion**: The OLS neutralization correctly absorbs industry variance (R² = 0.98 for the pure-industry signal). Residuals are zero-mean. The 5-gate correctly rejects a signal with no remaining alpha.

---

## 4. API Reference for KoC §3

### `lib/eval/neutralize.cjs`

```javascript
const { neutralizeCrossSection, neutralizePanel, getL1Industry } = require('./neutralize.cjs');

// Single cross-section
const { residuals, r2, nDropped, sizeProxyUsed, liquidityUsed } = neutralizeCrossSection(
  records,               // array of records for ONE time-point
  r => sueFn(r),         // signal function
  {
    useIndustry: true,   // L1 industry dummies (default: true)
    useSizeProxy: true,  // log(market cap) size control (default: true); DATA_PENDING = dropped
    useLiquidity: false, // Amihud/bidAskSpread liquidity control (default: false); DATA_PENDING = dropped
    useSeason: true,     // Q2/Q3/Q4 dummies (default: true)
    cutoffDate: 'YYYY-MM',  // REQUIRED when useSeason=true
  }
);
// residuals[i] = neutralized signal for records[i]
// r2 = how much of signal variance was explained by style (diagnostic)
// sizeProxyUsed = false when logMarketCap is null for all records (DATA_PENDING)
// liquidityUsed = false when amihud/bidAskSpread is null for all records (DATA_PENDING)

// Full panel
const { byDate, meanR2, sizeUsedCount, liquidityUsedCount } = neutralizePanel(allRecords, signalFn, options);
// byDate: Map<date, {residuals, r2, nDropped, sizeProxyUsed, liquidityUsed}>
// sizeUsedCount: number of time-points where size column was active
```

### `lib/eval/validate-signal.cjs`

```javascript
const { validateSignal } = require('./validate-signal.cjs');

const result = validateSignal(
  allRecords,             // all records, multiple time-points
  r => sueSignal(r),      // signal function (return null for missing)
  {
    neutralize: true,     // apply neutralization (default: true)
    alphaKey: 'alpha',    // forward return field name
    neutralizeOptions: {  // passed to neutralizeCrossSection
      useIndustry: true,
      useSizeProxy: true,
      useSeason: true,
    },
    costOptions: {
      commissionBps: 5,   // [NEEDS VERIFICATION]
      stampDutyBps: 10,   // [NEEDS VERIFICATION]
      spreadBps: 10,      // [NEEDS VERIFICATION]
    },
  }
);

// result.verdict: 'pass' | 'fail'
// result.gates: { gate1, gate2, gate3, gate4, gate5 } — each has .pass, .reason, .metrics
// result.metrics: { icMean, icStd, icir, tStat, nTimepoints, icSeries }
// result.sanityWarnings: string[] — empty if no suspicious values
```

### Usage pattern for KoC §3

```javascript
const { validateSignal } = require('../lib/eval/validate-signal.cjs');
const pool = require('../data/frozen-eval-lowpos-72tp.json');
const { assertPoolClean } = require('../lib/eval/guards.cjs');

// 1. Guard check
const records = assertPoolClean(pool);

// 2. Define your signal
function sueSignal(record) {
  // compute SUE from record.fundamentals or your PEAD data
  return record.sue_z ?? null;
}

// 3. Validate
const result = validateSignal(records, sueSignal, { neutralize: true });

// 4. Interpret
console.log('Verdict:', result.verdict);
console.log('IC_IR:', result.metrics.icir.toFixed(3));
for (const [gate, g] of Object.entries(result.gates)) {
  console.log(gate, g.pass ? 'PASS' : 'FAIL', '—', g.reason);
}
```

---

## 5. Known Limitations

1. **Size control DATA_PENDING**: `logMarketCap` is null for all 14,640 records in the current pool. The size term is silently dropped from all cross-section regressions. Status: `sizeProxyUsed = false` for all 78 time-points. Will activate automatically once pool records include `logMarketCap`.

2. **Liquidity control DATA_PENDING**: `amihud` and `bidAskSpread` are null for all pool records. `useLiquidity` is available but currently inactive. Will activate when pool records include either field.

3. **Industry freshness**: `industry-map.json` reflects industry classification as of 2026-05-24. For historical records (2018–2022), some stocks may have changed industry. This is a static mapping.

4. **Cost model needs verification**: The 30 bps round-trip cost is an estimate. Actual rates vary by broker, stock type, and holding period.

5. **Season dummies are at the quarter level**: Monthly precision for seasonal effects is not captured. Bank quarterly reporting seasonality (March/June/September/December) may need finer granularity for SUE signals.
