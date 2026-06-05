// Kline repository — local SQLite read first, optional online fallback
// Single-stock source lock: same stock + period must not mix sources
// Return format fully consistent with lib/data-sources/dispatcher

import { getDb } from './connection.js';
import { tableForPeriod } from './schema.js';

/**
 * Read klines from local database
 * @param {object} params
 * @param {string} params.code — 股票代码，如 "600519"
 * @param {string} params.market — 市场 '0'(深) / '1'(沪)
 * @param {string} [params.period='monthly'] — monthly | weekly | daily | 60min
 * @param {number} [params.limit=60] — 取最近 N 根
 * @param {string|null} [params.cutoffDate=null] — eval 场景：只取该日期之前的数据
 * @param {Function} [params.onlineFetcher=null] — async (params) => result，在线 fallback
 * @returns {Promise<{name, code, market, klines:Array, sourceUsed:string, fetchedAt:string}>}
 */
export async function getKlines(params = {}) {
  const { code, market, period = 'monthly', limit = 60, cutoffDate = null, onlineFetcher = null } = params;

  if (!code || !market) {
    throw new Error('klines-repo: code 和 market 为必填参数');
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

    // 检测是否跨源混用
    const sources = [...new Set(rows.map((r) => r.source).filter(Boolean))];
    if (sources.length > 1) {
      console.warn(`[klines-repo] WARNING: ${code} ${period} 存在跨源数据: ${sources.join(', ')}`);
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

  // 本地无数据，尝试在线 fallback
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
 * 查询某 code + period 的已有数据源
 * @returns {string|null} 数据源名称，无数据时返回 null
 */
export function getExistingSource(db, table, code) {
  const row = db.prepare(
    `SELECT source FROM ${table} WHERE code = ? AND source IS NOT NULL LIMIT 1`
  ).get(code);
  return row ? row.source : null;
}

/**
 * 检查跨源冲突
 * @throws {Error} 如果已有不同源的数据
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

// ---- 内部保存 ----

/**
 * 同步写入 K 线到本地库（用于 db-init 批量导入）
 * @param {string} sourceName — 数据源名称，如 'baidu'
 */
export function saveKlinesSync(db, table, code, market, name, klines, sourceName = null) {
  if (!klines || klines.length === 0) return;

  // 如果未指定 sourceName，从已有数据推断（兼容旧数据）
  if (!sourceName) {
    sourceName = getExistingSource(db, table, code);
  }

  // 单股源锁定检查
  if (sourceName) {
    checkSourceLock(db, table, code, sourceName);
  }

  // upsert stock
  db.prepare(`
    INSERT OR REPLACE INTO stocks (code, market, name, listing_date, delisted, industry, last_updated)
    VALUES (?, ?, ?, ?, 0, NULL, ?)
  `).run(code, market, name, null, new Date().toISOString().slice(0, 10));

  // bulk insert klines（含 source 列）
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
 * 强制清除某 code + period 的已有数据（用于 --force 覆盖）
 */
export function clearExistingData(db, table, code) {
  db.prepare(`DELETE FROM ${table} WHERE code = ?`).run(code);
}

/** 异步写入（不阻塞调用者） */
function saveKlinesAsync(db, table, code, market, name, klines, sourceName) {
  setImmediate(() => {
    try {
      saveKlinesSync(db, table, code, market, name, klines, sourceName);
    } catch (err) {
      console.warn(`[klines-repo] 异步入库失败 ${code}: ${err.message}`);
    }
  });
}

// ---- 辅助函数 ----

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
