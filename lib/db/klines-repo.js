// Kline repository — local SQLite read first, optional online fallback
// Single-stock source lock: same stock + period must not mix sources
// Return format fully consistent with lib/data-sources/dispatcher

import { getDb } from './connection.js';
import { tableForPeriod } from './schema.js';

/**
 * Read klines from local database
 * @param {object} params
 * @param {string} params.code — stock code, e.g. "600519"
 * @param {string} params.market — market '0'(SZ) / '1'(SH)
 * @param {string} [params.period='monthly'] — monthly | weekly | daily | 60min
 * @param {number} [params.limit=60] — take last N bars
 * @param {string|null} [params.cutoffDate=null] — eval mode: only take data before this date
 * @param {Function} [params.onlineFetcher=null] — async (params) => result, online fallback
 * @returns {Promise<{name, code, market, klines:Array, sourceUsed:string, fetchedAt:string}>}
 */
export async function getKlines(params = {}) {
  const { code, market, period = 'monthly', limit = 60, cutoffDate = null, onlineFetcher = null } = params;

  if (!code || !market) {
    throw new Error('klines-repo: code and market are required');
  }

  const db = getDb();
  const table = tableForPeriod(period);

  let query;
  let queryParams;
  if (cutoffDate) {
    query = `SELECT * FROM ${table} WHERE code = ? AND date <= ? ORDER BY date DESC LIMIT ?`;
    queryParams = [code, cutoffDate, limit];
  } else {
    query = `SELECT * FROM ${table} WHERE code = ? ORDER BY date DESC LIMIT ?`;
    queryParams = [code, limit];
  }

  const rows = db.prepare(query).all(...queryParams);

  if (rows.length > 0) {
    const stock = db.prepare('SELECT name FROM stocks WHERE code = ?').get(code);
    const name = stock?.name || code;

    // detect cross-source mixing
    const sources = [...new Set(rows.map((r) => r.source).filter(Boolean))];
    if (sources.length > 1) {
      console.warn(`[klines-repo] WARNING: ${code} ${period} cross-source data detected: ${sources.join(', ')}`);
    }
    const dominantSource = sources.length === 1 ? sources[0] : 'mixed';

    return {
      name,
      code,
      market,
      klines: rows.map(rowToKline).reverse(),
      sourceUsed: sources.length === 1 ? sources[0] : 'local',
      fetchedAt: new Date().toISOString(),
      _sourceBreakdown: sources.length > 1 ? sources : undefined,
    };
  }

  // no local data, try online fallback
  if (onlineFetcher) {
    const onlineResult = await onlineFetcher({ code, market, period, limit });
    if (onlineResult && Array.isArray(onlineResult.klines) && onlineResult.klines.length > 0) {
      const sourceName = onlineResult.sourceUsed || 'online';
      saveKlinesAsync(db, table, code, market, onlineResult.name || code, onlineResult.klines, sourceName);
      return { ...onlineResult, sourceUsed: sourceName };
    }
  }

  return {
    name: code, code, market,
    klines: [],
    sourceUsed: 'local',
    fetchedAt: new Date().toISOString(),
  };
}

/**
 * Query existing data source for a code + period
 * @returns {string|null} data source name, null if no data
 */
export function getExistingSource(db, table, code) {
  const row = db.prepare(
    `SELECT source FROM ${table} WHERE code = ? AND source IS NOT NULL LIMIT 1`
  ).get(code);
  return row ? row.source : null;
}

/**
 * Check cross-source conflict
 * @throws {Error} if data from a different source already exists
 */
export function checkSourceLock(db, table, code, newSource) {
  const existing = getExistingSource(db, table, code);
  if (existing && existing !== newSource) {
    throw new Error(
      `Cross-source contamination: ${code} ${table} has existing data from ${existing}, ` +
      `refuse to write ${newSource} data. Use --force to override (will delete existing first).`
    );
  }
}

// ---- internal save ----

/**
 * Sync write klines to local DB (for db-init batch import)
 * @param {string} sourceName — data source name, e.g. 'baidu'
 */
export function saveKlinesSync(db, table, code, market, name, klines, sourceName = null) {
  if (!klines || klines.length === 0) return;

  // if sourceName not specified, infer from existing data (backward compat)
  if (!sourceName) {
    sourceName = getExistingSource(db, table, code);
  }

  // single-stock source lock check
  if (sourceName) {
    checkSourceLock(db, table, code, sourceName);
  }

  // upsert stock record
  db.prepare(`
    INSERT OR REPLACE INTO stocks (code, market, name, listing_date, delisted, industry, last_updated)
    VALUES (?, ?, ?, ?, 0, NULL, ?)
  `).run(code, market, name, null, new Date().toISOString().slice(0, 10));

  // bulk insert klines (including source column)
  const insertStmt = db.prepare(`
    INSERT OR REPLACE INTO ${table}
    (code, date, open, close, high, low, volume, amount, amplitude, change_percent, change_amount, turnover_rate, source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const insertMany = db.transaction((items) => {
    for (const k of items) {
      insertStmt.run(
        code, k.date,
        k.open, k.close, k.high, k.low,
        k.volume, k.amount,
        k.amplitude, k.changePercent, k.change, k.turnoverRate,
        sourceName,
      );
    }
  });

  insertMany(klines);
}

/**
 * Force-clear existing data for a code + period (used for --force override)
 */
export function clearExistingData(db, table, code) {
  db.prepare(`DELETE FROM ${table} WHERE code = ?`).run(code);
}

/** Async write (non-blocking) */
function saveKlinesAsync(db, table, code, market, name, klines, sourceName) {
  setImmediate(() => {
    try {
      saveKlinesSync(db, table, code, market, name, klines, sourceName);
    } catch (err) {
      console.warn(`[klines-repo] async save failed ${code}: ${err.message}`);
    }
  });
}

// ---- helper functions ----

function rowToKline(row) {
  return {
    date: row.date,
    open: row.open,
    close: row.close,
    high: row.high,
    low: row.low,
    volume: row.volume,
    amount: row.amount,
    amplitude: row.amplitude,
    changePercent: row.change_percent,
    change: row.change_amount,
    turnoverRate: row.turnover_rate,
  };
}
