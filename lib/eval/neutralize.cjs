'use strict';

/**
 * Generic cross-sectional neutralization module.
 * Regresses a signal on style controls via OLS per time-point.
 * Returns residuals = signal with style exposures removed.
 *
 * Control variables (read directly from record fields — NO external file):
 *   1. log(market_cap) = log(record.market_cap_yi * 1e8)  — from §16 PIT join
 *   2. log(Amihud)     = log(record.amihud)               — from §15 Amihud calc
 *   3. L1 industry dummies                                 — industry style
 *
 * DATA_PENDING: when market_cap_yi / amihud are null on records (§15/§16 not yet run),
 *   those controls are silently dropped and neutralizationStatus = 'partial'.
 *   A warning is printed once per missing variable.
 *
 * PIT constraint: all control variables use data <= cutoffDate only.
 *   Caller must ensure records for a given cutoffDate were constructed PIT-correctly.
 *
 * NOTE on season dummies: season is constant within a cross-section (all records share
 *   the same cutoffDate/quarter), making season dummies collinear with the intercept.
 *   They are intentionally omitted from per-cross-section OLS. For panel-level seasonal
 *   effects, use a Fama-MacBeth setup across time-points.
 */

const industryMap = require('../../data/industry-map.json');

// ── Industry lookup ──────────────────────────────────────────────────────────

const _stockToL2 = industryMap.stockToIndustry || {};
const _l2ToL1 = {};
(industryMap.industries || []).forEach(ind => { _l2ToL1[ind.name] = ind.l1Name; });

function _stripPrefix(stockCode) {
  return stockCode.replace(/^(sh\.|sz\.)/, '');
}

function getL1Industry(stockCode) {
  const l2 = _stockToL2[_stripPrefix(stockCode)];
  if (!l2) return null;
  return _l2ToL1[l2] || l2;
}

// ── Record-level control accessors ───────────────────────────────────────────

/**
 * Read log(market_cap) from record.market_cap_yi (亿元, from §16 PIT join).
 * Returns null if field is absent or non-positive.
 */
function getLogMarketCap(record) {
  const myi = record.market_cap_yi;
  if (myi == null || myi <= 0) return null;
  return Math.log(myi * 1e8); // 亿元 → 元 → log
}

/**
 * Read log(Amihud) from record.amihud (from §15 calculation).
 * Returns null if field is absent or non-positive.
 */
function getLogAmihud(record) {
  const a = record.amihud;
  if (a == null || a <= 0) return null;
  return Math.log(a);
}

// ── DATA_PENDING warning tracker ─────────────────────────────────────────────

const _dataPendingWarned = new Set();

function _warnDataPending(variable) {
  if (!_dataPendingWarned.has(variable)) {
    console.warn(
      `  WARN neutralize.cjs: ${variable} DATA_PENDING — ` +
      `neutralization will be 'partial' until §15/§16 data is available.`
    );
    _dataPendingWarned.add(variable);
  }
}

// ── Regression helpers ───────────────────────────────────────────────────────

/**
 * Solve Ax = b via Gauss-Jordan elimination with partial pivoting.
 */
function gaussJordan(A, b) {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);
  for (let p = 0; p < n; p++) {
    let maxRow = p;
    for (let i = p + 1; i < n; i++) {
      if (Math.abs(aug[i][p]) > Math.abs(aug[maxRow][p])) maxRow = i;
    }
    [aug[p], aug[maxRow]] = [aug[maxRow], aug[p]];
    const pivot = aug[p][p];
    if (Math.abs(pivot) < 1e-12) return null;
    for (let j = 0; j <= n; j++) aug[p][j] /= pivot;
    for (let i = 0; i < n; i++) {
      if (i !== p) {
        const factor = aug[i][p];
        for (let j = 0; j <= n; j++) aug[i][j] -= factor * aug[p][j];
      }
    }
  }
  return aug.map(row => row[n]);
}

/**
 * OLS: regress y on X, return residuals and R².
 */
function ols(X, y) {
  const n = X.length;
  const k = X[0].length;
  const XtX = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) =>
      X.reduce((s, row) => s + row[i] * row[j], 0)
    )
  );
  const Xty = Array.from({ length: k }, (_, i) =>
    X.reduce((s, row, ri) => s + row[i] * y[ri], 0)
  );
  const beta = gaussJordan(XtX, Xty);
  if (!beta) {
    const yMean = y.reduce((s, v) => s + v, 0) / n;
    return { residuals: y.map(v => v - yMean), r2: 0, beta: null };
  }
  const yHat = X.map(row => row.reduce((s, v, j) => s + v * beta[j], 0));
  const residuals = y.map((v, i) => v - yHat[i]);
  const yMean = y.reduce((s, v) => s + v, 0) / n;
  const ssTot = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
  const ssRes = residuals.reduce((s, v) => s + v * v, 0);
  const r2 = ssTot > 1e-12 ? 1 - ssRes / ssTot : 0;
  return { residuals, r2, beta };
}

// ── Main API ─────────────────────────────────────────────────────────────────

/**
 * Neutralize a signal over one cross-section (single time-point).
 *
 * Regression:
 *   signal_i = α + β·log(market_cap_yi_i) + γ·log(amihud_i)
 *              + Σ δ_j·industry_L1_j_dummy_i + ε_i
 *
 * neutralizationStatus:
 *   'full'    — log_mktcap AND log_amihud both included (≥30% coverage each)
 *   'partial' — one or both of size/liquidity missing (DATA_PENDING)
 *
 * @param {object[]} records - records for ONE time-point
 * @param {function} signalFn - record → number|null
 * @param {object} [options]
 * @param {boolean} [options.useIndustry=true]    — include L1 industry dummies
 * @param {boolean} [options.useMarketCap=true]   — include log(market_cap_yi)
 * @param {boolean} [options.useAmihud=true]      — include log(amihud)
 * @param {boolean} [options.useSizeProxy]        — legacy alias for useMarketCap
 * @param {string}  [options.cutoffDate]          — 'YYYY-MM', passed for context only
 * @returns {{
 *   residuals: number[],
 *   r2: number,
 *   nDropped: number,
 *   controlsUsed: string[],
 *   missingControls: string[],
 *   neutralizationStatus: 'full'|'partial',
 *   industries: (string|null)[],
 *   beta: number[]|null
 * }}
 */
function neutralizeCrossSection(records, signalFn, options = {}) {
  const {
    useIndustry = true,
    useMarketCap = options.useSizeProxy !== undefined ? options.useSizeProxy : true,
    useAmihud = true,
    cutoffDate = null,
  } = options;

  const rawSignals = records.map(r => signalFn(r));
  const validIdxs = rawSignals
    .map((s, i) => (s != null && isFinite(s)) ? i : -1)
    .filter(i => i >= 0);

  if (validIdxs.length < 5) {
    const vs = validIdxs.map(i => rawSignals[i]);
    const mean = vs.length > 0 ? vs.reduce((a, b) => a + b, 0) / vs.length : 0;
    return {
      residuals: rawSignals.map(s => (s != null && isFinite(s)) ? s - mean : 0),
      r2: 0,
      nDropped: records.length - validIdxs.length,
      controlsUsed: [],
      missingControls: ['size', 'liquidity'],
      neutralizationStatus: 'partial',
      industries: records.map(() => null),
      beta: null,
    };
  }

  const validRecords = validIdxs.map(i => records[i]);
  const y = validIdxs.map(i => rawSignals[i]);

  // Industry labels
  const industries = validRecords.map(r =>
    useIndustry ? getL1Industry(r.stockCode) : null
  );
  const uniqueInds = [...new Set(industries.filter(x => x != null))].sort();

  // Continuous controls from record fields (record.market_cap_yi, record.amihud)
  const controlsUsed = [];
  const missingControls = [];
  const contCols = [];

  if (useMarketCap) {
    const logMC = validRecords.map(r => getLogMarketCap(r));
    const coverage = logMC.filter(v => v != null).length / logMC.length;
    if (coverage >= 0.3) {
      const valid = logMC.filter(v => v != null).sort((a, b) => a - b);
      const median = valid[Math.floor(valid.length / 2)] ?? 0;
      contCols.push(logMC.map(v => v ?? median)); // fill missing with median
      controlsUsed.push('log_mktcap');
    } else {
      missingControls.push('size');
      _warnDataPending('market_cap_yi (size)');
    }
  }

  if (useAmihud) {
    const logAmihud = validRecords.map(r => getLogAmihud(r));
    const coverage = logAmihud.filter(v => v != null).length / logAmihud.length;
    if (coverage >= 0.3) {
      const valid = logAmihud.filter(v => v != null).sort((a, b) => a - b);
      const median = valid[Math.floor(valid.length / 2)] ?? 0;
      contCols.push(logAmihud.map(v => v ?? median));
      controlsUsed.push('log_amihud');
    } else {
      missingControls.push('liquidity');
      _warnDataPending('amihud (liquidity)');
    }
  }

  // Build design matrix: [intercept, ...contCols, industry_dummies...]
  const X = validRecords.map((_, i) => {
    const row = [1];
    for (const col of contCols) row.push(col[i]);
    // Industry dummies: drop first industry as baseline
    for (let j = 1; j < uniqueInds.length; j++) {
      row.push(industries[i] === uniqueInds[j] ? 1 : 0);
    }
    return row;
  });

  if (useIndustry && uniqueInds.length > 0) {
    controlsUsed.push(`industry_L1(${uniqueInds.length})`);
  }

  // Determine neutralizationStatus: 'full' requires BOTH size and liquidity
  const hasSizeControl = useMarketCap && controlsUsed.includes('log_mktcap');
  const hasLiquidityControl = useAmihud && controlsUsed.includes('log_amihud');
  const sizeRequested = useMarketCap;
  const liquidityRequested = useAmihud;
  const neutralizationStatus =
    (sizeRequested && !hasSizeControl) || (liquidityRequested && !hasLiquidityControl)
      ? 'partial'
      : 'full';

  // Warn once if partial
  if (neutralizationStatus === 'partial' && missingControls.length > 0) {
    // Individual variable warnings already emitted above via _warnDataPending
  }

  const { residuals: validResiduals, r2, beta } = ols(X, y);

  const residuals = new Array(records.length).fill(0);
  validIdxs.forEach((origIdx, vi) => { residuals[origIdx] = validResiduals[vi]; });

  return {
    residuals,
    r2,
    nDropped: records.length - validIdxs.length,
    controlsUsed,
    missingControls,
    neutralizationStatus,
    industries: records.map(r => useIndustry ? getL1Industry(r.stockCode) : null),
    beta,
  };
}

/**
 * Neutralize across all time-points (panel).
 * Returns aggregated neutralizationStatus: 'full' only if ALL cross-sections are full.
 */
function neutralizePanel(allRecords, signalFn, options = {}) {
  const groups = new Map();
  for (const r of allRecords) {
    if (!groups.has(r.cutoffDate)) groups.set(r.cutoffDate, []);
    groups.get(r.cutoffDate).push(r);
  }

  const byDate = new Map();
  const r2Series = [];
  let anyPartial = false;
  const allControlsUsed = new Set();

  for (const [date, recs] of groups) {
    const result = neutralizeCrossSection(recs, signalFn, { ...options, cutoffDate: date });
    byDate.set(date, result);
    r2Series.push(result.r2);
    if (result.neutralizationStatus === 'partial') anyPartial = true;
    result.controlsUsed.forEach(c => allControlsUsed.add(c));
  }

  const meanR2 = r2Series.length > 0 ? r2Series.reduce((a, b) => a + b, 0) / r2Series.length : 0;
  return {
    byDate,
    r2Series,
    meanR2,
    neutralizationStatus: anyPartial ? 'partial' : 'full',
    controlsUsed: [...allControlsUsed],
  };
}

// ── Assertions ───────────────────────────────────────────────────────────────

function assertNeutralized(residuals, tolerance = 0.001) {
  const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
  if (Math.abs(mean) > tolerance) {
    throw new Error(
      `assertNeutralized FAILED: residual mean = ${mean.toFixed(6)}, expected < ${tolerance}. ` +
      'OLS residuals should be zero-mean.'
    );
  }
}

function assertNoLookaheadInControls(records, cutoffDate) {
  if (!cutoffDate) throw new Error('assertNoLookaheadInControls: cutoffDate is required');
  let violations = 0;
  for (const r of records) {
    if (r.klines && r.klines.length > 0) {
      const lastDate = r.klines[r.klines.length - 1].date;
      if (lastDate > cutoffDate) {
        violations++;
        if (violations <= 2) console.warn(`  WARN: ${r.stockCode} kline ${lastDate} > cutoff ${cutoffDate}`);
      }
    }
  }
  if (violations > 0) throw new Error(`assertNoLookaheadInControls: ${violations} kline dates after cutoff`);
}

module.exports = {
  neutralizeCrossSection,
  neutralizePanel,
  getL1Industry,
  getLogMarketCap,
  getLogAmihud,
  assertNeutralized,
  assertNoLookaheadInControls,
  ols,
  gaussJordan,
};
