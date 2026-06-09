# Neutralize Framework — Audit Document

**Branch**: neutralize-framework  
**Date**: 2026-06-08 (interface rework: 2026-06-08)  
**Files**: `lib/eval/neutralize.cjs`, `lib/eval/validate-signal.cjs`, `lib/eval/_selftest_reversal.cjs`, `lib/eval/_selftest_synthetic.cjs`

---

## 1. Neutralization Regression Specification

### Model

Cross-sectional OLS per time-point (strict PIT):

```
signal_i = α + β₁·log(market_cap_yi_i)        (from record.market_cap_yi, §16 PIT join)
             + β₂·log(amihud_i)                (from record.amihud, §15 calculation)
             + Σ_{j=1}^{K-1} γⱼ·Industry1_j_dummy_i   (K = unique L1 industries in cross-section)
             + ε_i

residual_i = ε_i  ← neutralized signal
```

| Control | Source | Coverage | Notes |
|---------|--------|----------|-------|
| `log(market_cap)` | `record.market_cap_yi` (亿元, from §16 as-of join) | DATA_PENDING until §16 | Requires ≥30% non-null to activate; missing → median impute |
| `log(Amihud)` | `record.amihud` (from §15 calculation) | DATA_PENDING until §15 | Requires ≥30% non-null to activate |
| L1 Industry dummies | `data/industry-map.json` → `stockToIndustry`, ~16-19 categories | ~99% on known pools | Stripped "sh."/"sz." prefix for lookup |

**DATA_PENDING**: when `market_cap_yi` or `amihud` are null on records (§15/§16 not yet run),
those controls are dropped and a one-time warning is printed. `neutralizationStatus` is set to `'partial'`.

**PIT guarantee**: Each time-point's OLS uses only records from that time-point's cross-section.
`record.market_cap_yi` must be the as-of value at `cutoffDate` (set by §16 as-of join).

**Season dummies**: Omitted from per-cross-section OLS. Within a cross-section all records share
the same cutoffDate/quarter, making season dummies constant (collinear with the intercept). They
are only meaningful in pooled panel regression.

### neutralizationStatus

| Status | Condition | Meaning |
|--------|-----------|---------|
| `'full'` | Both `log_mktcap` AND `log_amihud` included (≥30% coverage) | Complete neutralization; results are final |
| `'partial'` | Either size or liquidity missing (null/low coverage) | DATA_PENDING; results not final |

**downstream rule**: Any result with `neutralizationStatus='partial'` is preliminary. Do not
record as final IC/verdict until `'full'` is achieved after §15/§16.

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

### Path A: Synthetic size-signal test — PASSED (2026-06-08)

**Script**: `lib/eval/_selftest_synthetic.cjs` — runs NOW, no real data needed.

```
Setup: 300 stocks × 24 months
Signal  = 2.0 × z_size + 0.3 × noise  (pure size exposure)
Alpha   = 0.8 × z_size + 1.0 × noise  (forward return also has size beta)

Raw IC (no neutralization):     0.5831   t=68.3
Neutralized IC (log_MC+amihud): -0.0120  t=-1.4   (97.9% reduction)

R² on sample cross-section:     0.977    (size explains 97.7% of signal)
Residual mean:                  ≈ 0      (assertNeutralized PASSED)
neutralizationStatus:           full     ✓

ASSERTIONS:
  ✓ neutralizationStatus = full
  ✓ assertNeutralized: residual mean ≈ 0
  ✓ Raw IC = 0.58 > 0.20 (size exposure confirmed)
  ✓ |Neutralized IC| = 0.012 < 0.10 (size bias removed)
  ✓ IC reduction = 97.9% ≥ 60%
```

**Conclusion**: Neutralization mechanism is correct. A purely size-driven signal (IC=0.58)
is reduced to noise (IC=-0.012) after neutralizing by `log(market_cap_yi)`.

---

### Path B/C: Real data tests — DATA_PENDING (auto-activate when §15/§16 data arrives)

**Path B** (Test 4 in `_selftest_reversal.cjs`): reversal with real `market_cap_yi` from §16.
- Currently SKIP (0% market_cap_yi coverage on pool)
- Will auto-run when pool records have §16 fields
- Question: does reversal IC=-0.04 drop to ≈0 with full neutralization?

**Path C** (Test 3 in `_selftest_reversal.cjs`): full-market pool reversal.
- Currently SKIP (`data/frozen-eval-fullmarket.json` not yet built)
- Will auto-run when full-market pool is available

---

### Deviation Table (current partial-neutralization state)

| Requirement | Actual | Reason |
|-------------|--------|--------|
| Reversal neutralized IC `\|IC\| < 0.01` | `\|IC\| = 0.0336` (partial) | neutralizationStatus='partial'; size/liquidity DATA_PENDING. Result is preliminary. |
| 5-gate `verdict = fail` for reversal | ✓ `verdict = fail` | G1 IC_IR=-0.266 < 0.3; G2/G3/G5 fail. Gate fails correctly. |
| Framework neutralization is correct | ✓ Verified via Path A | Synthetic size signal IC=0.58 → 0.012 after log(MC) neutralization (97.9% reduction). |
| Full neutralization (§15/§16) result | DATA_PENDING | Path B auto-activates when records have market_cap_yi. |

### Test 2: Synthetic pure-industry signal (framework verification)

```
Raw IC: mean=-0.0259, t=-1.93
Neutralized IC: mean=-0.0128, t=-0.87 (50% reduction)
Residual check (2019-09): mean≈0, std=0.006, R²=0.980
assertNeutralized: PASSED ✓
5-gate verdict=fail ✓
```

**Conclusion**: When the signal IS industry-driven, neutralization absorbs it (R²=0.98, IC drops 50%).
Framework is mechanically correct.

---

## 4. API Reference for KoC §3

### `lib/eval/neutralize.cjs`

```javascript
const { neutralizeCrossSection, neutralizePanel,
        getLogMarketCap, getLogAmihud, getL1Industry } = require('./neutralize.cjs');

// Single cross-section
const {
  residuals,              // neutralized signal per record (same order)
  r2,                     // R² of style regression (0 = signal is pure alpha)
  nDropped,               // records with null signal
  controlsUsed,           // e.g. ['log_mktcap', 'log_amihud', 'industry_L1(16)']
  missingControls,        // e.g. ['size', 'liquidity'] when DATA_PENDING
  neutralizationStatus,   // 'full' | 'partial'
  beta,
} = neutralizeCrossSection(
  records,               // records for ONE time-point
  r => sueFn(r),         // signal function
  {
    useIndustry: true,   // L1 industry dummies (default: true)
    useMarketCap: true,  // log(record.market_cap_yi) from §16 (default: true)
    useAmihud: true,     // log(record.amihud) from §15 (default: true)
    cutoffDate: 'YYYY-MM',
  }
);
// DATA_PENDING: if record.market_cap_yi / record.amihud are null → silently dropped,
//   one-time warning printed, neutralizationStatus='partial'

// Full panel
const { byDate, meanR2, neutralizationStatus, controlsUsed } =
  neutralizePanel(allRecords, signalFn, options);
// neutralizationStatus: 'full' only if ALL cross-sections were full
```

### `lib/eval/validate-signal.cjs`

```javascript
const { validateSignal } = require('./validate-signal.cjs');

const result = validateSignal(
  allRecords,
  r => sueSignal(r),
  {
    neutralize: true,
    alphaKey: 'alpha',
    neutralizeOptions: { useIndustry: true, useMarketCap: true, useAmihud: true },
    costOptions: { commissionBps: 5, stampDutyBps: 10, spreadBps: 10 }, // [NEEDS VERIFICATION]
  }
);

// result.verdict: 'pass' | 'fail'
// result.gates: { gate1, gate2, gate3, gate4, gate5 }
// result.metrics.neutralizationStatus: 'full'|'partial'|'none'
// result.metrics.controlsUsed: string[]
// WARN printed if neutralizationStatus='partial' — result is not final
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

1. **market_cap_yi is DATA_PENDING** until §16 (PIT as-of join) completes. Until then, `neutralizationStatus='partial'` and IC results are preliminary.

2. **amihud is DATA_PENDING** until §15 (daily kline fetch + Amihud calc) completes. Until then, liquidity neutralization is skipped.

3. **Coverage threshold**: ≥30% non-null required to activate a control. Below threshold, cross-section median is NOT imputed — the control is dropped entirely. This avoids imputation bias when data is genuinely missing.

4. **Industry map is static** (2026-05-24 snapshot). Historical reclassifications not tracked.

5. **Season dummies omitted**: Season is constant within a cross-section (all stocks share the same cutoffDate/quarter), making season dummies collinear with the intercept. Only meaningful in pooled panel regression.

6. **Cost model needs verification**: 30 bps round-trip is an estimate. Actual rates vary by broker and stock type.
