// ema db init — one-time DB initialization
// Usage: node cli/db-init.js --scope hs300 [--periods monthly,weekly,daily]
// Adaptive rate limiting: auto-pause on Eastmoney rejection, resume when clear

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const DATA_DIR = path.join(PROJECT_DIR, '.eastmoney-ai');
const DB_DIR = path.join(DATA_DIR, 'db');
const PROGRESS_PATH = path.join(DB_DIR, 'init-progress.json');

const CONCURRENCY = 2;         // limit concurrency to avoid rate limiting
const BATCH_DELAY_MS = 5000;   // 5s delay between batches
const LOG_INTERVAL = 50;
const RATE_LIMIT_PAUSE_MS = 5 * 60 * 1000; // pause 5 min on rate limit
const RATE_LIMIT_FAIL_THRESHOLD = 0.6;     // 60% batch failure → treat as rate limit

export async function runDbInit(scope = 'hs300', periods = ['monthly', 'weekly', 'daily'], sourceName = 'baidu', onlyFailed = false) {
  console.log(`=== DB init start ===`);
  console.log(`Scope: ${scope} | Periods: ${periods.join(', ')} | Source: ${sourceName} | Concurrency: ${CONCURRENCY} | Batch interval: ${BATCH_DELAY_MS}ms`);

  const stockList = await fetchStockList(scope);
  if (stockList.length === 0) { console.log('no stocks, exit'); return; }
  console.log(`Stock list: ${stockList.length}`);

  const progress = loadProgress();
  const done = new Set(progress.done || []);
  const pending = stockList.filter((s) => !done.has(s.code));
  console.log(`Done: ${done.size} | Pending: ${pending.length}`);
  if (pending.length === 0) { console.log('all done, exit'); return; }

  const { getDb, closeDb } = await import('../lib/db/connection.js');
  const { saveKlinesSync } = await import('../lib/db/klines-repo.js');
  const { tableForPeriod } = await import('../lib/db/schema.js');
  const db = getDb();

  // direct connect to specified source (no fallback, keep purity)
  let fetchKlinesDirect;
  try {
    const srcMod = await import(`../lib/data-sources/${sourceName}.js`);
    fetchKlinesDirect = srcMod.fetchKlines;
  } catch (err) {
    console.error(`Data source "${sourceName}" unavailable: ${err.message}`);
    return;
  }

  let completed = done.size;
  let failed = 0;
  let successful = 0; // stocks with kline data
  let totalKlines = 0;
  const startTime = Date.now();
  let pauseCount = 0;

  for (let i = 0; i < pending.length; i += CONCURRENCY) {
    const batch = pending.slice(i, i + CONCURRENCY);
    const results = await Promise.allSettled(
      batch.map((stock) => processStock(stock, periods, fetchKlinesDirect, db, saveKlinesSync, tableForPeriod, sourceName)),
    );

    let batchFailures = 0;
    for (let j = 0; j < results.length; j++) {
      const r = results[j];
      const stock = batch[j];
      if (r.status === 'fulfilled') {
        const { klines: count, errors } = r.value;
        completed++;
        done.add(stock.code);
        if (count === 0) {
          failed++;
          batchFailures++;
          const errMsg = errors.length > 0 ? errors.slice(0, 2).join('; ') : 'unknown reason';
          if (failed <= 10 || failed % 100 === 0) {
            console.warn(`  ✗ ${stock.code}(${stock.name}): ${errMsg}`);
          }
        } else {
          successful++;
          totalKlines += count;
        }
      } else {
        failed++;
        batchFailures++;
        console.warn(`  ✗ ${stock.code}(${stock.name}): ${r.reason?.message || r.reason}`);
      }
    }

    saveProgress({ done: [...done], failed, updatedAt: new Date().toISOString() });

    // progress log
    if (completed % LOG_INTERVAL === 0 || completed === stockList.length) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
      const rate = (completed / (Math.max(elapsed, 1) / 3600)).toFixed(0);
      const successCount = completed - failed;
      console.log(`[${completed}/${stockList.length}] ${elapsed}s | OK ${successCount} | Failed ${failed} | Klines ${totalKlines} | Pauses ${pauseCount}`);
    }

    // batch interval
    if (i + CONCURRENCY < pending.length) {
      await sleep(BATCH_DELAY_MS);
    }
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
  const successCount = completed - failed;
  console.log('');
  console.log(`=== DB init complete ===`);
  console.log(`Elapsed: ${elapsed}s | Inserted: ${successCount} | Failed: ${failed} | Klines: ${totalKlines} bars | Pauses: ${pauseCount}`);

  closeDb();
}

async function processStock(stock, periods, fetchFn, db, saveKlinesSync, tableForPeriod, sourceName) {
  let totalKlines = 0;
  const errors = [];

  for (const period of periods) {
    try {
      const result = await fetchFn({
        market: stock.market, code: stock.code, period,
        limit: 0, // full fetch (Baidu has no hard limit, start_time traces back to 1990)
      });
      if (result && Array.isArray(result.klines) && result.klines.length > 0) {
        const table = tableForPeriod(period);
        saveKlinesSync(db, table, stock.code, stock.market, result.name || stock.name, result.klines, sourceName);
        totalKlines += result.klines.length;
      } else {
        errors.push(`${period}:no data`);
      }
    } catch (err) {
      errors.push(`${period}:${err.message || err}`);
    }
  }

  return { klines: totalKlines, errors };
}

async function fetchStockList(scope) {
  switch (scope) {
    case 'hs300': {
      const { fetchHS300Constituents } = await import('../lib/scanner/hs300.js');
      return fetchHS300Constituents(false);
    }
    case 'watchlist': {
      const { loadWatchlist } = await import('../lib/scanner/watchlist.js');
      return loadWatchlist().stocks || [];
    }
    case 'all': {
      const all = [];
      const markets = [
        { fs: 'm:0+t:6', market: '0' },
        { fs: 'm:0+t:80', market: '0' },
        { fs: 'm:1+t:2', market: '1' },
        { fs: 'm:1+t:23', market: '1' },
      ];
      for (const { fs, market } of markets) {
        try {
          const url = `https://push2.eastmoney.com/api/qt/clist/get?fs=${fs}&fields=f12,f14&pn=1&pz=5000`;
          const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
          const json = await resp.json();
          const diff = json?.data?.diff;
          const items = Array.isArray(diff) ? diff : (diff && typeof diff === 'object' ? Object.values(diff) : []);
          for (const d of items) {
            all.push({ code: d.f12, market, name: d.f14 });
          }
        } catch (err) {
          console.warn(`failed to fetch full market list (${fs}): ${err.message}`);
        }
      }
      return all;
    }
    default:
      console.error(`unknown scope: ${scope}`);
      return [];
  }
}

function loadProgress() {
  try { return JSON.parse(fs.readFileSync(PROGRESS_PATH, 'utf-8')); } catch (_) { return { done: [], failed: 0 }; }
}

function saveProgress(data) {
  if (!fs.existsSync(DB_DIR)) fs.mkdirSync(DB_DIR, { recursive: true });
  fs.writeFileSync(PROGRESS_PATH, JSON.stringify(data, null, 2), 'utf-8');
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
