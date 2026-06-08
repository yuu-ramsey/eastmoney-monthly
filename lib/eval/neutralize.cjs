'use strict';

/**
 * Generic cross-sectional neutralization module.
 * Regresses a signal on style controls via OLS per time-point.
 * Returns residuals = signal with style exposures removed.
 *
 * Control variables (in order of priority):
 *   1. log(market_cap) = log(closeAtCutoff × totalShare)  — true size (PIT approx)
 *   2. log(Amihud)     = log(|R_monthly| / amount_B)      — illiquidity proxy
 *   3. L1 industry dummies                                 — industry style
 *   4. Season dummies Q2/Q3/Q4                             — seasonal effects
 *
 * Data dependency:
 *   data/neutralize-controls.json — fetched by scripts/fetch_neutralize_controls.py
 *   If the file does not exist, a warning is logged once and size/Amihud are dropped.
 *
 * PIT constraint: all control variables use data <= cutoffDate only.
 */

const path = require('path');
const fs = require('fs');
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

// ── Controls data (lazy load) ────────────────────────────────────────────────

const CONTROLS_PATH = path.resolve(__dirname, '../../data/neutralize-controls.json');
let _controls = null;
let _controlsWarned = false;
let _controlsLoaded = false;

function _loadControls() {
  if (_controlsLoaded) return _controls;
  _controlsLoaded = true;

  if (!fs.existsSync(CONTROLS_PATH)) {
    if (!_controlsWarned) {
      console.warn(
        '  WARN neutralize.cjs: data/neutralize-controls.json not found. ' +
        'Run scripts/fetch_neutralize_controls.py to enable true market cap + Amihud. ' +
        'Size and liquidity controls will be DATA_PENDING (silently dropped).'
      );
      _controlsWarned = true;
    }
    _controls = null;
    return null;
  }

  try {
    _controls = JSON.parse(fs.readFileSync(CONTROLS_PATH, 'utf8'));
    console.log(
      `  neutralize.cjs: controls loaded — ` +
      `mc=${_controls.metadata?.mc_coverage_pct}% ` +
      `monthly=${_controls.metadata?.monthly_coverage_pct}%`
    );
  } catch (e) {
    console.warn(`  WARN neutralize.cjs: failed to parse controls file: ${e.message}`);
    _controls = null;
  }
  return _controls;
}

/**
 * Get log(market_cap) for a record at a given cutoffDate.
 * market_cap = closeAtCutoff × totalShare_snapshot (PIT approximation).
 * Returns null if controls not available or data missing.
 */
function getLogMarketCap(record, cutoffDate) {
  const controls = _loadControls();
  if (!controls) return null;
  const ts = controls.total_share?.[record.stockCode];
  if (!ts) return null;
  const close = record.closeAtCutoff;
  if (!close || close <= 0) return null;
  const mc = ts * close;
  return mc > 0 ? Math.log(mc) : null;
}

/**
 * Get log(Amihud illiquidity) for a record at a given cutoffDate.
 * Amihud_monthly = |R_monthly| / (amount_CNY / 1e8).
 * Returns null if controls not available or data missing.
 */
function getLogAmihud(record, cutoffDate) {
  const controls = _loadControls();
  if (!controls) return null;
  const monthlyData = controls.monthly?.[record.stockCode];
  if (!monthlyData) return null;
  const m = monthlyData[cutoffDate];
  if (!m) return null;
  const ret = m.return;
  const amount = m.amount;
  if (ret == null || amount == null || amount <= 0) return null;
  const amihud = Math.abs(ret) / (amount / 1e8);
  return amihud > 0 ? Math.log(amihud) : null;
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
 *   signal_i = α + β·log(mktcap_i) + γ·log(amihud_i)
 *              + Σ δ_j·industry_j_dummy_i + Σ θ_q·season_q_dummy_i + ε_i
 *
 * @param {object[]} records - records for ONE time-point
 * @param {function} signalFn - record → number|null
 * @param {object} [options]
 * @param {boolean} [options.useIndustry=true]
 * @param {boolean} [options.useSeason=true]
 * @param {boolean} [options.useMarketCap=true]  — log(mktcap) from controls file
 * @param {boolean} [options.useAmihud=true]     — log(Amihud) from controls file
 * @param {boolean} [options.useSizeProxy]       — alias for useMarketCap (legacy option name)
 * @param {string}  [options.cutoffDate]         — 'YYYY-MM', required for season + Amihud
 * @returns {{
 *   residuals: number[],
 *   r2: number,
 *   nDropped: number,
 *   controlsUsed: string[],
 *   industries: (string|null)[],
 *   beta: number[]|null
 * }}
 */
function neutralizeCrossSection(records, signalFn, options = {}) {
  const {
    useIndustry = true,
    useSeason = true,
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
      r2: 0, nDropped: records.length - validIdxs.length,
      controlsUsed: [], industries: records.map(() => null), beta: null,
    };
  }

  const validRecords = validIdxs.map(i => records[i]);
  const y = validIdxs.map(i => rawSignals[i]);

  // Season
  let quarter = null;
  if (useSeason && cutoffDate) {
    quarter = Math.floor((parseInt(cutoffDate.slice(5, 7), 10) - 1) / 3) + 1;
  }

  // Industry labels
  const industries = validRecords.map(r =>
    useIndustry ? getL1Industry(r.stockCode) : null
  );
  const uniqueInds = [...new Set(industries.filter(x => x != null))].sort();

  // Continuous controls: log_mktcap and log_amihud from controls file
  const controlsUsed = [];
  const contCols = [];

  if (useMarketCap) {
    const logMC = validRecords.map(r => getLogMarketCap(r, cutoffDate));
    const coverage = logMC.filter(v => v != null).length / logMC.length;
    if (coverage >= 0.3) {
      const sorted = logMC.filter(v => v != null).sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
      contCols.push(logMC.map(v => v ?? median));
      controlsUsed.push('log_mktcap');
    }
    // No fallback to rangePosition — if coverage < 30%, silently drop (DATA_PENDING)
  }

  if (useAmihud && cutoffDate) {
    const logAmihud = validRecords.map(r => getLogAmihud(r, cutoffDate));
    const coverage = logAmihud.filter(v => v != null).length / logAmihud.length;
    if (coverage >= 0.3) {
      const sorted = logAmihud.filter(v => v != null).sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
      contCols.push(logAmihud.map(v => v ?? median));
      controlsUsed.push('log_amihud');
    }
  }

  // Build design matrix
  const X = validRecords.map((_, i) => {
    const row = [1];
    for (const col of contCols) row.push(col[i]);
    for (let j = 1; j < uniqueInds.length; j++) {
      row.push(industries[i] === uniqueInds[j] ? 1 : 0);
    }
    if (useSeason && quarter != null) {
      row.push(quarter === 2 ? 1 : 0);
      row.push(quarter === 3 ? 1 : 0);
      row.push(quarter === 4 ? 1 : 0);
    }
    return row;
  });

  if (useIndustry && uniqueInds.length > 0) controlsUsed.push(`industry_L1(${uniqueInds.length})`);
  if (useSeason && quarter != null) controlsUsed.push('season');

  const { residuals: validResiduals, r2, beta } = ols(X, y);

  const residuals = new Array(records.length).fill(0);
  validIdxs.forEach((origIdx, vi) => { residuals[origIdx] = validResiduals[vi]; });

  return {
    residuals,
    r2,
    nDropped: records.length - validIdxs.length,
    controlsUsed,
    industries: records.map(r => useIndustry ? getL1Industry(r.stockCode) : null),
    beta,
  };
}

/**
 * Neutralize across all time-points (panel).
 */
function neutralizePanel(allRecords, signalFn, options = {}) {
  const groups = new Map();
  for (const r of allRecords) {
    if (!groups.has(r.cutoffDate)) groups.set(r.cutoffDate, []);
    groups.get(r.cutoffDate).push(r);
  }
  const byDate = new Map();
  const r2Series = [];
  for (const [date, recs] of groups) {
    const result = neutralizeCrossSection(recs, signalFn, { ...options, cutoffDate: date });
    byDate.set(date, result);
    r2Series.push(result.r2);
  }
  const meanR2 = r2Series.length > 0 ? r2Series.reduce((a, b) => a + b, 0) / r2Series.length : 0;
  return { byDate, r2Series, meanR2 };
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
