'use strict';

/**
 * Generic 5-gate signal validation module.
 * Evaluates any signal against future returns using standard criteria.
 * Returns pass/fail per gate plus full metrics.
 *
 * Gate thresholds are calibrated for A-share monthly cross-section research.
 */

const { neutralizeCrossSection } = require('./neutralize.cjs');

// ── Thresholds ────────────────────────────────────────────────────────────────

const THRESHOLDS = {
  gate1_icir_min: 0.3,       // minimum |IC_IR| to pass
  gate1_tstat_min: 2.0,      // minimum |IC t-stat|
  gate2_spread_min_pct: 0,   // Q5−Q1 spread must be > 0 in test set
  gate2_monotone_min: 3,     // at least 3 of 4 consecutive quintile pairs must be monotone
  gate3_net_alpha_min: 0,    // net-of-cost alpha > 0
  gate3_tstat_min: 1.96,
  gate4_bh_fdr_alpha: 0.05,  // BH-FDR threshold for walk-forward window p-values
  gate4_wf_consistent_frac: 2 / 3,  // ≥2/3 walk-forward windows must show same sign
  gate5_pvalue_max: 0.05,
  ic_sanity_warn: 0.10,      // |IC| > this triggers a leakage warning
  sharpe_sanity_warn: 3.0,
  spread_sanity_warn: 5.0,   // monthly spread in pct > this → check leakage
};

// ── Math utilities ────────────────────────────────────────────────────────────

function mean(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function stdDev(arr, m) {
  if (m === undefined) m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
}

function rank(arr) {
  const indexed = arr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const r = new Array(arr.length);
  let i = 0;
  while (i < arr.length) {
    let j = i;
    while (j < arr.length - 1 && indexed[j + 1][0] === indexed[j][0]) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[indexed[k][1]] = avgRank;
    i = j + 1;
  }
  return r;
}

function pearsonCorr(a, b) {
  const n = a.length;
  const ma = mean(a), mb = mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  const denom = Math.sqrt(da * db);
  return denom < 1e-12 ? 0 : num / denom;
}

function spearmanIC(signals, alphas) {
  return pearsonCorr(rank(signals), rank(alphas));
}

// Normal approximation: p-value for two-sided t-test
function pValueFromT(t, df) {
  // Abramowitz & Stegun approximation for the normal CDF tail
  const z = Math.abs(t);
  const a = [0.2316419, 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429];
  const k = 1 / (1 + a[0] * z);
  const poly = k * (a[1] + k * (a[2] + k * (a[3] + k * (a[4] + k * a[5]))));
  const norm = Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI) * poly;
  return 2 * norm;
}

// Benjamini-Hochberg FDR correction
function bhFDR(pValues, alpha = 0.05) {
  const n = pValues.length;
  const indexed = pValues.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
  const adjusted = new Array(n);
  let minAdj = 1;
  for (let i = n - 1; i >= 0; i--) {
    const adj = Math.min(1, indexed[i][0] * n / (i + 1));
    minAdj = Math.min(minAdj, adj);
    adjusted[indexed[i][1]] = minAdj;
  }
  return adjusted;
}

// ── Signal processing ─────────────────────────────────────────────────────────

/**
 * Group records by cutoffDate, compute signal per record.
 */
function groupByDate(records, signalFn) {
  const groups = new Map();
  for (const r of records) {
    if (!groups.has(r.cutoffDate)) groups.set(r.cutoffDate, []);
    groups.get(r.cutoffDate).push(r);
  }
  return groups;
}

/**
 * Compute IC series across time-points.
 * @param {object[]} allRecords
 * @param {function} signalFn - record → number|null
 * @param {object} options
 * @returns {{date: string, ic: number, n: number, grossSpread: number}[]}
 */
function computeICSeries(allRecords, signalFn, options = {}) {
  const {
    neutralize: doNeutralize = true,
    alphaKey = 'alpha',
    neutralizeOptions = {},
    holdingPeriod = null,  // not used here, reserved for overlap check
  } = options;

  const groups = groupByDate(allRecords, signalFn);
  const icSeries = [];

  for (const [date, recs] of [...groups.entries()].sort()) {
    const valid = recs.filter(r => r[alphaKey] != null && isFinite(r[alphaKey]));
    if (valid.length < 10) continue;

    const rawSignals = valid.map(r => signalFn(r));
    const validMask = rawSignals.map(s => s != null && isFinite(s));
    const validIdxs = rawSignals.map((s, i) => (s != null && isFinite(s)) ? i : -1).filter(i => i >= 0);
    if (validIdxs.length < 10) continue;

    const filteredRecords = validIdxs.map(i => valid[i]);
    const filteredSignals = validIdxs.map(i => rawSignals[i]);
    const filteredAlphas = validIdxs.map(i => valid[i][alphaKey]);

    if (doNeutralize) {
      const { residuals } = neutralizeCrossSection(
        filteredRecords,
        r => signalFn(r),
        { ...neutralizeOptions, cutoffDate: date }
      );
      // Align residuals with filteredAlphas
      const pairSignals = [];
      const pairAlphas = [];
      for (let i = 0; i < filteredRecords.length; i++) {
        const s = residuals[i];
        if (isFinite(s)) { pairSignals.push(s); pairAlphas.push(filteredAlphas[i]); }
      }
      if (pairSignals.length < 10) continue;
      const ic = spearmanIC(pairSignals, pairAlphas);

      // Quintile spread for gross alpha
      const sorted = pairSignals.map((s, i) => [s, pairAlphas[i]]).sort((a, b) => a[0] - b[0]);
      const quintileSize = Math.floor(sorted.length / 5);
      const q1Alpha = mean(sorted.slice(0, quintileSize).map(x => x[1]));
      const q5Alpha = mean(sorted.slice(-quintileSize).map(x => x[1]));
      icSeries.push({ date, ic, n: pairSignals.length, grossSpread: q5Alpha - q1Alpha });
    } else {
      const ic = spearmanIC(filteredSignals, filteredAlphas);
      const sorted = filteredSignals.map((s, i) => [s, filteredAlphas[i]]).sort((a, b) => a[0] - b[0]);
      const quintileSize = Math.floor(sorted.length / 5);
      const q1Alpha = mean(sorted.slice(0, quintileSize).map(x => x[1]));
      const q5Alpha = mean(sorted.slice(-quintileSize).map(x => x[1]));
      icSeries.push({ date, ic, n: filteredSignals.length, grossSpread: q5Alpha - q1Alpha });
    }
  }
  return icSeries;
}

// ── Cost model ────────────────────────────────────────────────────────────────

/**
 * A-share long-only cost model (needs verification against actual broker rates).
 * @param {object} options
 * @param {number} [options.commissionBps=5] - one-way commission (bps) [NEEDS VERIFICATION]
 * @param {number} [options.stampDutyBps=10] - sell stamp duty (bps) [NEEDS VERIFICATION]
 * @param {number} [options.spreadBps=10] - round-trip bid-ask spread (bps) [NEEDS VERIFICATION]
 * @returns {number} total round-trip cost in bps
 */
function computeCost(options = {}) {
  const {
    commissionBps = 5,   // [NEEDS VERIFICATION]
    stampDutyBps = 10,   // [NEEDS VERIFICATION]
    spreadBps = 10,      // [NEEDS VERIFICATION]
  } = options;
  // Round trip: buy + sell
  return (commissionBps * 2) + stampDutyBps + spreadBps;
}

// ── 5-Gate implementation ─────────────────────────────────────────────────────

/**
 * Gate 1: Cross-sectional IC quality.
 * Pass if |IC_IR| ≥ 0.3 AND |t-stat| ≥ 2.
 */
function gate1IC(icSeries) {
  const icVals = icSeries.map(x => x.ic);
  if (icVals.length < 5) return { pass: false, reason: 'too few time-points', metrics: {} };
  const m = mean(icVals);
  const s = stdDev(icVals, m);
  const t = m / (s / Math.sqrt(icVals.length));
  const icir = s > 0 ? m / s : 0;
  const pass = Math.abs(t) >= THRESHOLDS.gate1_tstat_min &&
               Math.abs(icir) >= THRESHOLDS.gate1_icir_min;
  return {
    pass,
    reason: pass ? 'IC_IR and t-stat pass' : `IC_IR=${icir.toFixed(3)}, t=${t.toFixed(2)} — below threshold`,
    metrics: { icMean: m, icStd: s, icir, tStat: t, nTimepoints: icVals.length },
  };
}

/**
 * Gate 2: Quintile monotonicity.
 * Long-side Q5 return > Q1 (spread > 0) and at least 3/4 consecutive pairs are monotone.
 */
function gate2Monotonicity(allRecords, signalFn, options = {}) {
  const { neutralize: doNeutralize = true, alphaKey = 'alpha', neutralizeOptions = {} } = options;
  const groups = groupByDate(allRecords, signalFn);
  const quintileMeans = [0, 0, 0, 0, 0];
  const quintileCounts = [0, 0, 0, 0, 0];

  for (const [date, recs] of groups) {
    const valid = recs.filter(r => {
      const s = signalFn(r);
      return s != null && isFinite(s) && r[alphaKey] != null;
    });
    if (valid.length < 10) continue;

    let signals;
    if (doNeutralize) {
      const { residuals } = neutralizeCrossSection(valid, signalFn, { ...neutralizeOptions, cutoffDate: date });
      signals = residuals;
    } else {
      signals = valid.map(r => signalFn(r));
    }

    const pairs = signals.map((s, i) => [s, valid[i][alphaKey]]).sort((a, b) => a[0] - b[0]);
    const qSize = Math.floor(pairs.length / 5);
    for (let q = 0; q < 5; q++) {
      const qPairs = pairs.slice(q * qSize, (q + 1) * qSize);
      const qAlpha = mean(qPairs.map(x => x[1]));
      quintileMeans[q] += qAlpha;
      quintileCounts[q]++;
    }
  }

  const avgMeans = quintileMeans.map((s, i) => quintileCounts[i] > 0 ? s / quintileCounts[i] : 0);
  let monotoneCount = 0;
  for (let i = 0; i < 4; i++) {
    if (avgMeans[i + 1] > avgMeans[i]) monotoneCount++;
  }
  const spread = avgMeans[4] - avgMeans[0];
  const pass = spread > THRESHOLDS.gate2_spread_min_pct &&
               monotoneCount >= THRESHOLDS.gate2_monotone_min;

  return {
    pass,
    reason: pass ? 'Quintile monotone and spread > 0' :
      `spread=${spread.toFixed(2)}, monotone=${monotoneCount}/4`,
    metrics: { quintileMeans: avgMeans, spread, monotoneCount },
  };
}

/**
 * Gate 3: Long-side net alpha.
 * Top quintile (Q5) gross and net alpha, both positive and t-stat > 1.96.
 */
function gate3LongAlpha(icSeries, costOptions = {}) {
  const spreads = icSeries.map(x => x.grossSpread);
  if (spreads.length < 5) return { pass: false, reason: 'too few time-points', metrics: {} };

  const totalCostBps = computeCost(costOptions);
  // grossSpread is in pct points; cost is in bps (÷100 to convert)
  const netSpreads = spreads.map(s => s - totalCostBps / 100);

  const grossMean = mean(spreads);
  const netMean = mean(netSpreads);
  const netStd = stdDev(netSpreads, netMean);
  const netT = netMean / (netStd / Math.sqrt(netSpreads.length));

  const pass = netMean > THRESHOLDS.gate3_net_alpha_min &&
               Math.abs(netT) >= THRESHOLDS.gate3_tstat_min;

  return {
    pass,
    reason: pass ? 'Net alpha > 0 and t-stat passes' :
      `netAlpha=${netMean.toFixed(2)}%, t=${netT.toFixed(2)}`,
    metrics: {
      grossAlphaMean: grossMean,
      netAlphaMean: netMean,
      netAlphaStd: netStd,
      netTStat: netT,
      totalCostBps,
    },
  };
}

/**
 * Gate 4: Robustness — BH-FDR + walk-forward consistency.
 */
function gate4Robustness(icSeries) {
  if (icSeries.length < 6) {
    return { pass: false, reason: 'too few time-points for walk-forward', metrics: {} };
  }

  // Walk-forward: split into 3 equal windows
  const wSize = Math.floor(icSeries.length / 3);
  const windows = [
    icSeries.slice(0, wSize),
    icSeries.slice(wSize, 2 * wSize),
    icSeries.slice(2 * wSize),
  ].filter(w => w.length >= 3);

  const windowStats = windows.map(w => {
    const vals = w.map(x => x.ic);
    const m = mean(vals);
    const s = stdDev(vals, m);
    const t = m / (s / Math.sqrt(vals.length));
    const p = pValueFromT(t, vals.length - 1);
    return { mean: m, t, p, n: vals.length };
  });

  // BH-FDR on window p-values
  const pValues = windowStats.map(w => w.p);
  const adjP = bhFDR(pValues, THRESHOLDS.gate4_bh_fdr_alpha);

  // Overall IC sign
  const allIC = icSeries.map(x => x.ic);
  const overallMean = mean(allIC);
  const overallSign = Math.sign(overallMean);

  // Count windows consistent with overall sign
  const consistent = windowStats.filter(w => Math.sign(w.mean) === overallSign).length;
  const consistentFrac = consistent / windowStats.length;

  // Overall BH-FDR pass: at least 1 window has adjusted p < alpha
  const bhPass = adjP.some(p => p < THRESHOLDS.gate4_bh_fdr_alpha);
  const wfPass = consistentFrac >= THRESHOLDS.gate4_wf_consistent_frac;
  const pass = bhPass && wfPass;

  return {
    pass,
    reason: pass ? 'BH-FDR and walk-forward pass' :
      `BH-FDR=${bhPass}, walk-forward=${consistent}/${windowStats.length} consistent`,
    metrics: {
      windowStats,
      adjustedPValues: adjP,
      consistentFrac,
      bhPass,
      wfPass,
    },
  };
}

/**
 * Gate 5: p-value consistency — one-sided test P(spread ≤ 0), CI must agree with p.
 */
function gate5PValueConsistency(icSeries) {
  const icVals = icSeries.map(x => x.ic);
  const m = mean(icVals);
  const s = stdDev(icVals, m);
  const t = m / (s / Math.sqrt(icVals.length));
  const twoSidedP = pValueFromT(t, icVals.length - 1);
  const oneSidedP = m > 0 ? twoSidedP / 2 : 1 - twoSidedP / 2;

  // Bootstrap CI for IC mean (1000 samples)
  const nBoot = 1000;
  const bootMeans = [];
  for (let b = 0; b < nBoot; b++) {
    let bSum = 0;
    for (let i = 0; i < icVals.length; i++) {
      bSum += icVals[Math.floor(Math.random() * icVals.length)];
    }
    bootMeans.push(bSum / icVals.length);
  }
  bootMeans.sort((a, b) => a - b);
  const ciLo = bootMeans[Math.floor(nBoot * 0.025)];
  const ciHi = bootMeans[Math.floor(nBoot * 0.975)];

  // Consistency: p < 0.05 AND CI agrees with direction
  const pPass = oneSidedP < THRESHOLDS.gate5_pvalue_max;
  const ciPositive = ciLo > 0;
  const ciNegative = ciHi < 0;
  const directionAgrees = m > 0 ? ciPositive : ciNegative;
  const pass = pPass && directionAgrees;

  return {
    pass,
    reason: pass ? `One-sided p=${oneSidedP.toFixed(4)} < 0.05, CI consistent` :
      `p=${oneSidedP.toFixed(4)}, CI=[${ciLo.toFixed(3)},${ciHi.toFixed(3)}], direction=${directionAgrees}`,
    metrics: { oneSidedP, twoSidedP, ciLo, ciHi, icMean: m, tStat: t },
  };
}

// ── Sanity checks ─────────────────────────────────────────────────────────────

function runSanityChecks(icSeries, gate3Metrics) {
  const warnings = [];
  const icVals = icSeries.map(x => x.ic);
  const m = mean(icVals);
  if (Math.abs(m) > THRESHOLDS.ic_sanity_warn) {
    warnings.push(`!!! SANITY: |IC| = ${m.toFixed(3)} > ${THRESHOLDS.ic_sanity_warn} — CHECK FOR LEAKAGE`);
  }
  if (gate3Metrics && Math.abs(gate3Metrics.grossAlphaMean || 0) > THRESHOLDS.spread_sanity_warn) {
    warnings.push(`!!! SANITY: grossSpread = ${gate3Metrics.grossAlphaMean.toFixed(2)}% > ${THRESHOLDS.spread_sanity_warn}% — CHECK FOR LEAKAGE`);
  }
  if (gate3Metrics && gate3Metrics.netTStat) {
    const sharpe = gate3Metrics.netAlphaMean / (gate3Metrics.netAlphaStd || 1) * Math.sqrt(12);
    if (Math.abs(sharpe) > THRESHOLDS.sharpe_sanity_warn) {
      warnings.push(`!!! SANITY: Sharpe ≈ ${sharpe.toFixed(1)} > ${THRESHOLDS.sharpe_sanity_warn} — CHECK FOR LEAKAGE`);
    }
  }
  return warnings;
}

// ── Main API ──────────────────────────────────────────────────────────────────

/**
 * Run the full 5-gate validation on a signal.
 *
 * @param {object[]} allRecords - all records (multiple time-points)
 * @param {function} signalFn - record → number|null (the signal to test)
 * @param {object} [options]
 * @param {boolean} [options.neutralize=true] - apply neutralization
 * @param {string} [options.alphaKey='alpha'] - field name for forward return
 * @param {object} [options.neutralizeOptions={}] - passed to neutralizeCrossSection
 * @param {object} [options.costOptions={}] - cost model params
 * @returns {{
 *   gates: {gate1: object, gate2: object, gate3: object, gate4: object, gate5: object},
 *   metrics: {icSeries: object[], icMean: number, icStd: number, icir: number, tStat: number},
 *   verdict: 'pass' | 'fail',
 *   sanityWarnings: string[],
 * }}
 */
function validateSignal(allRecords, signalFn, options = {}) {
  const {
    neutralize: doNeutralize = true,
    alphaKey = 'alpha',
    neutralizeOptions = {},
    costOptions = {},
  } = options;

  // Assertions before computation
  if (!Array.isArray(allRecords) || allRecords.length === 0) {
    throw new Error('validateSignal: allRecords must be a non-empty array');
  }
  if (typeof signalFn !== 'function') {
    throw new Error('validateSignal: signalFn must be a function');
  }

  const validateOptions = { neutralize: doNeutralize, alphaKey, neutralizeOptions };
  const icSeries = computeICSeries(allRecords, signalFn, validateOptions);

  if (icSeries.length === 0) {
    return {
      gates: { gate1: { pass: false, reason: 'no valid IC series' }, gate2: { pass: false }, gate3: { pass: false }, gate4: { pass: false }, gate5: { pass: false } },
      metrics: { icSeries: [], icMean: 0, icStd: 0, icir: 0, tStat: 0 },
      verdict: 'fail',
      sanityWarnings: [],
    };
  }

  const g1 = gate1IC(icSeries);
  const g2 = gate2Monotonicity(allRecords, signalFn, validateOptions);
  const g3 = gate3LongAlpha(icSeries, costOptions);
  const g4 = gate4Robustness(icSeries);
  const g5 = gate5PValueConsistency(icSeries);

  const sanityWarnings = runSanityChecks(icSeries, g3.metrics);

  const verdict = g1.pass && g2.pass && g3.pass && g4.pass && g5.pass ? 'pass' : 'fail';

  const icVals = icSeries.map(x => x.ic);
  const icMean = mean(icVals);
  const icStd = stdDev(icVals, icMean);
  const icir = icStd > 0 ? icMean / icStd : 0;
  const tStat = icMean / (icStd / Math.sqrt(icVals.length));

  return {
    gates: { gate1: g1, gate2: g2, gate3: g3, gate4: g4, gate5: g5 },
    metrics: { icSeries, icMean, icStd, icir, tStat, nTimepoints: icVals.length },
    verdict,
    sanityWarnings,
  };
}

module.exports = {
  validateSignal,
  computeICSeries,
  computeCost,
  spearmanIC,
  THRESHOLDS,
  // Exposed for testing
  gate1IC,
  gate2Monotonicity,
  gate3LongAlpha,
  gate4Robustness,
  gate5PValueConsistency,
};
