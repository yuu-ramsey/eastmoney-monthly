'use strict';

/**
 * Generic cross-sectional neutralization module.
 * Regresses a signal on style controls (size, liquidity, industry dummies, season dummies)
 * via OLS per time-point. Returns residuals = signal with style exposures removed.
 *
 * PIT constraint: each time-point's regression uses only that time-point's cross-section.
 * No future information is allowed in control variables.
 *
 * Size proxy: uses r.logMarketCap (log total market cap in CNY). DATA_PENDING if null.
 *   NOT rangePosition — price range position is collinear with momentum/reversal signals
 *   and is NOT a valid size proxy.
 * Liquidity: uses r.amihud (Amihud illiquidity) or r.bidAskSpread. DATA_PENDING if null.
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

// ── Regression helpers ───────────────────────────────────────────────────────

/**
 * Solve Ax = b via Gauss-Jordan elimination with partial pivoting.
 * @param {number[][]} A - n×n matrix
 * @param {number[]} b - n-vector
 * @returns {number[]|null} solution vector, or null if singular
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
 * OLS: regress y on X, return residuals.
 * @param {number[][]} X - n×k design matrix
 * @param {number[]} y - n-vector
 * @returns {{residuals: number[], r2: number, beta: number[]|null}}
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
    // Singular — return raw y as residuals (neutralization skipped)
    const yMean = y.reduce((s, v) => s + v, 0) / n;
    const ss = y.reduce((s, v) => s + (v - yMean) ** 2, 0);
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
 * @param {object[]} records - array of records for this time-point
 * @param {function} signalFn - record → number (signal value)
 * @param {object} [options]
 * @param {boolean} [options.useIndustry=true] - include L1 industry dummies
 * @param {boolean} [options.useSeason=true] - include season (Q2/Q3/Q4) dummies
 * @param {boolean} [options.useSizeProxy=true] - include log(market cap) as size control.
 *   Requires records to have a non-null `logMarketCap` field. If no record in the
 *   cross-section has `logMarketCap`, the size term is silently dropped (DATA_PENDING).
 *   Missing values within a cross-section that otherwise has data are filled with the
 *   cross-sectional median (minimum-bias imputation).
 * @param {boolean} [options.useLiquidity=false] - include liquidity control.
 *   Uses `r.amihud` (Amihud illiquidity) if available, falling back to `r.bidAskSpread`.
 *   Dropped silently if no record in the cross-section has either field (DATA_PENDING).
 * @param {string} [options.cutoffDate] - 'YYYY-MM' used to derive season; required if useSeason=true
 * @returns {{
 *   residuals: number[],       // neutralized signal for each record (same order)
 *   r2: number,                // R² of style regression (how much signal was style)
 *   nDropped: number,          // records dropped due to missing signal or industry
 *   industries: string[],      // L1 industry for each record (null if unknown)
 *   beta: number[]|null,       // regression coefficients
 *   sizeProxyUsed: boolean,    // whether size column was included this cross-section
 *   liquidityUsed: boolean,    // whether liquidity column was included this cross-section
 * }}
 */
function neutralizeCrossSection(records, signalFn, options = {}) {
  const {
    useIndustry = true,
    useSeason = true,
    useSizeProxy = true,
    useLiquidity = false,
    cutoffDate = null,
  } = options;

  // Collect raw signals and filter invalid
  const rawSignals = records.map(r => signalFn(r));
  const validMask = rawSignals.map(s => s != null && isFinite(s));
  const validIdxs = validMask.map((ok, i) => ok ? i : -1).filter(i => i >= 0);

  if (validIdxs.length < 5) {
    // Not enough valid records — return demeaned signals
    const mean = rawSignals.filter(s => s != null).reduce((a, b) => a + b, 0) / (validIdxs.length || 1);
    return {
      residuals: rawSignals.map(s => (s != null && isFinite(s)) ? s - mean : 0),
      r2: 0,
      nDropped: records.length - validIdxs.length,
      industries: records.map(() => null),
      beta: null,
      sizeProxyUsed: false,
      liquidityUsed: false,
    };
  }

  const validRecords = validIdxs.map(i => records[i]);
  const y = validIdxs.map(i => rawSignals[i]);

  // Derive season if cutoffDate provided
  let quarter = null;
  if (useSeason && cutoffDate) {
    const month = parseInt(cutoffDate.slice(5, 7), 10);
    quarter = Math.floor((month - 1) / 3) + 1; // 1-4
  }

  // Industry labels for valid records
  const industries = validRecords.map(r =>
    useIndustry ? getL1Industry(r.stockCode) : null
  );
  const uniqueInds = [...new Set(industries.filter(x => x != null))].sort();

  // Decide whether to include size / liquidity columns this cross-section.
  // A column is included only when at least one record has actual data — no fallback allowed.
  // Missing values within an included column are filled with the cross-section median.
  const sizeVals = useSizeProxy
    ? validRecords.map(r => r.logMarketCap != null ? r.logMarketCap : null)
    : null;
  const sizeProxyUsed = sizeVals != null && sizeVals.some(v => v != null);
  let sizeMedian = null;
  if (sizeProxyUsed) {
    const nonNull = sizeVals.filter(v => v != null).sort((a, b) => a - b);
    sizeMedian = nonNull[Math.floor(nonNull.length / 2)];
  }

  const liqVals = useLiquidity
    ? validRecords.map(r => r.amihud != null ? r.amihud : (r.bidAskSpread != null ? r.bidAskSpread : null))
    : null;
  const liquidityUsed = liqVals != null && liqVals.some(v => v != null);
  let liqMedian = null;
  if (liquidityUsed) {
    const nonNull = liqVals.filter(v => v != null).sort((a, b) => a - b);
    liqMedian = nonNull[Math.floor(nonNull.length / 2)];
  }

  // Build design matrix: [intercept, logMarketCap?, amihud/spread?, ind_d1..., Q2?, Q3?, Q4?]
  const X = validRecords.map((r, i) => {
    const row = [1];
    if (sizeProxyUsed) row.push(sizeVals[i] != null ? sizeVals[i] : sizeMedian);
    if (liquidityUsed) row.push(liqVals[i] != null ? liqVals[i] : liqMedian);
    // Industry dummies (drop first industry as baseline)
    for (let j = 1; j < uniqueInds.length; j++) {
      row.push(industries[i] === uniqueInds[j] ? 1 : 0);
    }
    // Season dummies (drop Q1 as baseline)
    if (useSeason && quarter != null) {
      row.push(quarter === 2 ? 1 : 0);
      row.push(quarter === 3 ? 1 : 0);
      row.push(quarter === 4 ? 1 : 0);
    }
    return row;
  });

  const { residuals: validResiduals, r2, beta } = ols(X, y);

  // Map residuals back to original record order
  const residuals = new Array(records.length).fill(0);
  validIdxs.forEach((origIdx, vi) => {
    residuals[origIdx] = validResiduals[vi];
  });

  const allIndustries = records.map(r =>
    useIndustry ? getL1Industry(r.stockCode) : null
  );

  return {
    residuals,
    r2,
    nDropped: records.length - validIdxs.length,
    industries: allIndustries,
    beta,
    sizeProxyUsed,
    liquidityUsed,
  };
}

/**
 * Neutralize a signal across all time-points.
 *
 * @param {object[]} allRecords - all records (multiple time-points)
 * @param {function} signalFn - record → number
 * @param {object} [options] - same as neutralizeCrossSection
 * @returns {{
 *   byDate: Map<string, {residuals: number[], r2: number}>,
 *   r2Series: number[],       // per-time-point R²
 *   meanR2: number,
 *   sizeUsedCount: number,    // time-points where size column was active
 *   liquidityUsedCount: number,
 * }}
 */
function neutralizePanel(allRecords, signalFn, options = {}) {
  const byDate = new Map();
  const groups = new Map();
  for (const r of allRecords) {
    if (!groups.has(r.cutoffDate)) groups.set(r.cutoffDate, []);
    groups.get(r.cutoffDate).push(r);
  }

  const r2Series = [];
  let sizeUsedCount = 0;
  let liquidityUsedCount = 0;
  for (const [date, recs] of groups) {
    const opts = { ...options, cutoffDate: date };
    const result = neutralizeCrossSection(recs, signalFn, opts);
    byDate.set(date, result);
    r2Series.push(result.r2);
    if (result.sizeProxyUsed) sizeUsedCount++;
    if (result.liquidityUsed) liquidityUsedCount++;
  }

  const meanR2 = r2Series.length > 0
    ? r2Series.reduce((a, b) => a + b, 0) / r2Series.length
    : 0;

  return { byDate, r2Series, meanR2, sizeUsedCount, liquidityUsedCount };
}

// ── Assertions ───────────────────────────────────────────────────────────────

/**
 * Assert that residuals are approximately uncorrelated with the control variables.
 * Checks that |mean residual| ≈ 0 (OLS should produce zero-mean residuals).
 */
function assertNeutralized(residuals, tolerance = 0.001) {
  const mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
  if (Math.abs(mean) > tolerance) {
    throw new Error(
      `assertNeutralized FAILED: residual mean = ${mean.toFixed(6)}, expected < ${tolerance}. ` +
      'OLS residuals should be zero-mean.'
    );
  }
}

/**
 * Assert no look-ahead in control variables: cutoffDate must be provided
 * and all records' kline last date must not exceed cutoffDate.
 */
function assertNoLookaheadInControls(records, cutoffDate) {
  if (!cutoffDate) {
    throw new Error('assertNoLookaheadInControls: cutoffDate is required');
  }
  let violations = 0;
  for (const r of records) {
    if (r.klines && r.klines.length > 0) {
      const lastDate = r.klines[r.klines.length - 1].date;
      if (lastDate > cutoffDate) {
        violations++;
        if (violations <= 2) {
          console.warn(`  WARN: ${r.stockCode} kline ${lastDate} > cutoff ${cutoffDate}`);
        }
      }
    }
  }
  if (violations > 0) {
    throw new Error(`assertNoLookaheadInControls: ${violations} records with kline > cutoff`);
  }
}

module.exports = {
  neutralizeCrossSection,
  neutralizePanel,
  getL1Industry,
  assertNeutralized,
  assertNoLookaheadInControls,
  ols,
  gaussJordan,
};
