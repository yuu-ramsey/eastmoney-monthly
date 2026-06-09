'use strict';

/**
 * Path A: Synthetic size-signal self-test.
 *
 * PURPOSE
 *   Prove the neutralization MECHANISM is correct — independent of real data.
 *   Run NOW (no §15/§16 needed). If this fails, the framework has a bug.
 *   If this passes, the framework is correct; real-data tests (Path B/C) are
 *   gated only on data availability.
 *
 * DESIGN
 *   Construct 300 synthetic stocks × 24 monthly time-points where:
 *     signal_i = 2.0 · z_size_i + 0.3 · noise_signal_i   (pure size exposure + small noise)
 *     alpha_i  = 0.8 · z_size_i + 1.0 · noise_alpha_i    (forward return also has size beta)
 *     z_size_i = (log(MC_i) - mean) / std                 (standardized size)
 *
 *   Raw IC(signal, alpha) is driven by the shared z_size component — typically 0.3–0.5.
 *   After neutralizing signal w.r.t. log(market_cap_yi), residuals ≈ noise_signal.
 *   IC(residuals, alpha) ≈ 0 because noise_signal ⊥ z_size ⊥ noise_alpha.
 *
 * ASSERTIONS
 *   1. neutralizationStatus = 'full' (market_cap_yi and amihud both provided)
 *   2. assertNeutralized: residual mean ≈ 0 per cross-section (OLS property)
 *   3. Raw IC > 0.20 (shared size exposure creates meaningful raw correlation)
 *   4. |Neutralized IC| < 0.10 (size exposure removed; only noise left)
 *   5. IC reduction ≥ 60% (neutralization substantially removes size bias)
 *
 * NOTE: synthetic stocks use codes 'synth_XXXX' — no industry mapping.
 *   The neutralization uses log(market_cap_yi) only (industry='partial-industry'
 *   since coverage is 0). neutralizationStatus = 'full' is verified via
 *   the size+liquidity fields being present.
 */

const { validateSignal, computeICSeries } = require('./validate-signal.cjs');
const { neutralizeCrossSection, assertNeutralized } = require('./neutralize.cjs');

// ── Seeded PRNG (LCG) for reproducibility ────────────────────────────────────

function makePRNG(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function makeRandn(rng) {
  // Box-Muller
  return function () {
    const u = 1 - rng();
    const v = 1 - rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

// ── Synthetic data construction ───────────────────────────────────────────────

const NUM_STOCKS = 300;
const NUM_MONTHS = 24;
const SEED = 42;

function buildSyntheticDates(numMonths) {
  const dates = [];
  let year = 2022, month = 1;
  for (let i = 0; i < numMonths; i++) {
    dates.push(`${year}-${String(month).padStart(2, '0')}`);
    month++;
    if (month > 12) { month = 1; year++; }
  }
  return dates;
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function stdDev(arr, m) {
  if (m === undefined) m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

/**
 * Generate synthetic records.
 * Each stock has a stable log(MC) drawn once; signal and alpha are generated per time-point.
 * Records include market_cap_yi and amihud so neutralizationStatus = 'full'.
 */
function buildSyntheticRecords() {
  const rng = makePRNG(SEED);
  const randn = makeRandn(rng);
  const dates = buildSyntheticDates(NUM_MONTHS);

  // Per-stock stable log(MC) in 元; MC_yi in 亿元
  const stockLogMC = Array.from({ length: NUM_STOCKS }, () => 24 + 2 * randn()); // log(元)
  const stockMCyi = stockLogMC.map(lmc => Math.exp(lmc) / 1e8); // 亿元

  // Per-stock stable Amihud baseline
  const stockAmihudBase = Array.from({ length: NUM_STOCKS }, () => Math.exp(-2 + 0.5 * randn()));

  const allRecords = [];
  for (const date of dates) {
    const mLogMC = mean(stockLogMC);
    const sLogMC = stdDev(stockLogMC, mLogMC);

    for (let s = 0; s < NUM_STOCKS; s++) {
      const zSize = (stockLogMC[s] - mLogMC) / sLogMC; // standardized size
      const signal = 2.0 * zSize + 0.3 * randn();
      const alpha = 0.8 * zSize + 1.0 * randn();
      const amihud = stockAmihudBase[s] * Math.exp(0.2 * randn()); // small time variation

      allRecords.push({
        stockCode: `synth_${String(s).padStart(4, '0')}`,
        cutoffDate: date,
        market_cap_yi: stockMCyi[s],
        amihud,
        alpha,
        _signal: signal,
        // No klines — not needed for this test
      });
    }
  }
  return allRecords;
}

// ── Test execution ────────────────────────────────────────────────────────────

function runSyntheticSizeTest() {
  console.log('\n=== PATH A: Synthetic size-signal self-test ===');
  console.log(`Setup: ${NUM_STOCKS} stocks × ${NUM_MONTHS} months, signal = 2·z_size + noise`);

  const records = buildSyntheticRecords();
  const signalFn = r => r._signal;

  // --- Raw IC (no neutralization) ---
  const rawResult = validateSignal(records, signalFn, { neutralize: false, alphaKey: 'alpha' });
  const rawIC = rawResult.metrics.icMean;
  console.log(`\nRaw (no neutralization):  IC = ${rawIC.toFixed(4)}, t = ${rawResult.metrics.tStat.toFixed(2)}`);

  // --- Neutralized IC ---
  const neutrResult = validateSignal(records, signalFn, {
    neutralize: true,
    alphaKey: 'alpha',
    neutralizeOptions: { useIndustry: false, useMarketCap: true, useAmihud: true },
  });
  const neutrIC = neutrResult.metrics.icMean;
  const { neutralizationStatus, controlsUsed } = neutrResult.metrics;

  console.log(`Neutralized (log_MC+amihud): IC = ${neutrIC.toFixed(4)}, t = ${neutrResult.metrics.tStat.toFixed(2)}`);
  console.log(`neutralizationStatus: ${neutralizationStatus}`);
  console.log(`controlsUsed: ${controlsUsed.join(', ')}`);

  // --- Direct residual check on one cross-section ---
  const sampleDate = '2022-06';
  const sampleRecs = records.filter(r => r.cutoffDate === sampleDate);
  const { residuals, r2, neutralizationStatus: csStatus } = neutralizeCrossSection(
    sampleRecs,
    signalFn,
    { useIndustry: false, useMarketCap: true, useAmihud: true, cutoffDate: sampleDate }
  );
  const finiteResids = residuals.filter(isFinite);
  const residMean = mean(finiteResids);

  console.log(`\nCross-section check [${sampleDate}]:`);
  console.log(`  Residual mean = ${residMean.toFixed(6)} (expect ≈ 0)`);
  console.log(`  R² = ${r2.toFixed(4)} (fraction of signal variance explained by size)`);
  console.log(`  neutralizationStatus = ${csStatus}`);

  // ── Assertions ──
  let pass = true;

  // 1. neutralizationStatus = 'full'
  if (neutralizationStatus !== 'full') {
    console.error(`✗ FAIL: neutralizationStatus='${neutralizationStatus}', expected 'full'. Check market_cap_yi/amihud fields.`);
    pass = false;
  } else {
    console.log('✓ neutralizationStatus = full');
  }

  // 2. assertNeutralized: residual mean ≈ 0
  try {
    assertNeutralized(finiteResids, 0.01);
    console.log(`✓ assertNeutralized: residual mean = ${residMean.toFixed(6)}`);
  } catch (e) {
    console.error(`✗ FAIL: ${e.message}`);
    pass = false;
  }

  // 3. Raw IC > 0.20 (size exposure creates real correlation with alpha)
  if (Math.abs(rawIC) < 0.20) {
    console.error(`✗ FAIL: Raw IC = ${rawIC.toFixed(4)} < 0.20 — size signal not generating expected correlation.`);
    pass = false;
  } else {
    console.log(`✓ Raw IC = ${rawIC.toFixed(4)} > 0.20 (size exposure confirmed)`);
  }

  // 4. |Neutralized IC| < 0.10 (size bias removed)
  if (Math.abs(neutrIC) >= 0.10) {
    console.error(`✗ FAIL: |Neutralized IC| = ${Math.abs(neutrIC).toFixed(4)} ≥ 0.10 — size not fully removed.`);
    pass = false;
  } else {
    console.log(`✓ |Neutralized IC| = ${Math.abs(neutrIC).toFixed(4)} < 0.10 (size bias removed)`);
  }

  // 5. IC reduction ≥ 60%
  const reduction = 1 - Math.abs(neutrIC) / Math.abs(rawIC);
  if (reduction < 0.60) {
    console.error(`✗ FAIL: IC reduction = ${(reduction * 100).toFixed(1)}% < 60% — neutralization insufficient.`);
    pass = false;
  } else {
    console.log(`✓ IC reduction = ${(reduction * 100).toFixed(1)}% ≥ 60% (substantial size removal)`);
  }

  // 6. R² sanity: size should explain a meaningful fraction of the signal
  if (r2 < 0.20) {
    console.warn(`⚠ R² = ${r2.toFixed(4)} < 0.20 — size explains less than expected fraction of signal.`);
  } else {
    console.log(`✓ R² = ${r2.toFixed(4)} (size explains ${(r2 * 100).toFixed(1)}% of signal variance)`);
  }

  return { pass, rawIC, neutrIC, reduction, r2, neutralizationStatus };
}

// ── Main ──────────────────────────────────────────────────────────────────────

if (require.main === module) {
  const result = runSyntheticSizeTest();

  console.log('\n══════════════════════════════════════════');
  console.log(`PATH A (synthetic size neutralization): ${result.pass ? '✓ PASS' : '✗ FAIL'}`);
  console.log(`  Raw IC: ${result.rawIC.toFixed(4)}`);
  console.log(`  Neutralized IC: ${result.neutrIC.toFixed(4)}`);
  console.log(`  IC reduction: ${(result.reduction * 100).toFixed(1)}%`);
  console.log(`  neutralizationStatus: ${result.neutralizationStatus}`);
  console.log('══════════════════════════════════════════');

  if (!result.pass) {
    console.error('\n✗ PATH A FAILED — neutralization mechanism has a bug. Do NOT proceed to Path B/C.');
    process.exit(1);
  }

  console.log('\n✅ PATH A PASSED — neutralization mechanism is correct.');
  console.log('   Framework is ready for Path B (real §16 data) and Path C (full-market pool).');
  console.log('   Next: wait for §15/§16 completion, then run _selftest_reversal.cjs.');
}

module.exports = { runSyntheticSizeTest };
