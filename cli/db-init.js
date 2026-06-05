// ema db init — 一次性建库
// 用法: node cli/db-init.js --scope hs300 [--periods monthly,weekly,daily]
// 自适应限流：检测到东财拒绝后自动暂停，恢复后继续

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..');
const DATA_DIR = path.join(PROJECT_DIR, '.eastmoney-ai');
const DB_DIR = path.join(DATA_DIR, 'db');
const PROGRESS_PATH = path.join(DB_DIR, 'init-progress.json');

const CONCURRENCY = 2;         // 降低并发避免触发限流
const BATCH_DELAY_MS = 5000;   // 批次间 5 秒
const LOG_INTERVAL = 50;
const RATE_LIMIT_PAUSE_MS = 5 * 60 * 1000; // 限流暂停 5 分钟
const RATE_LIMIT_FAIL_THRESHOLD = 0.6;     // 批次 60% 失败 → 视为限流

export async function runDbInit(scope = 'hs300', periods = ['monthly', 'weekly', 'daily'], sourceName = 'baidu', onlyFailed = false) {
  console.log(`=== 建库开始 ===`);
  console.log(`范围: ${scope} | 周期: ${periods.join(', ')} | 源: ${sourceName} | 并发: ${CONCURRENCY} | 批间隔: ${BATCH_DELAY_MS}ms`);

  const stockList = await fetchStockList(scope);
  if (stockList.length === 0) { console.log('无股票，退出'); return; }
  console.log(`股票清单: ${stockList.length} 只`);

  const progress = loadProgress();
  const done = new Set(progress.done || []);
  const pending = stockList.filter((s) => !done.has(s.code));
  console.log(`已完成: ${done.size} | 待处理: ${pending.length}`);
  if (pending.length === 0) { console.log('全部完成，退出'); return; }

  const { getDb, closeDb } = await import('../lib/db/connection.js');
  const { saveKlinesSync } = await import('../lib/db/klines-repo.js');
  const { tableForPeriod } = await import('../lib/db/schema.js');
  const db = getDb();

  // 直连指定源（不降级，保持纯度）
  let fetchKlinesDirect;
  try {
    const srcMod = await import(`../lib/data-sources/${sourceName}.js`);
    fetchKlinesDirect = srcMod.fetchKlines;
  } catch (err) {
    console.error(`数据源 "${sourceName}" 不可用: ${err.message}`);
    return;
  }

  let completed = done.size;
  let failed = 0;
  let successful = 0; // 有 K 线数据的股票数
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
          const errMsg = errors.length > 0 ? errors.slice(0, 2).join('; ') : '不明原因';
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

    // 进度日志
    if (completed % LOG_INTERVAL === 0 || completed === stockList.length) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
      const rate = (completed / (Math.max(elapsed, 1) / 3600)).toFixed(0);
      const successCount = completed - failed;
      console.log(`[${completed}/${stockList.length}] ${elapsed}s | 成功 ${successCount} | 失败 ${failed} | K线 ${totalKlines} | 暂停 ${pauseCount}次`);
    }

    // 批次间隔
    if (i + CONCURRENCY < pending.length) {
      await sleep(BATCH_DELAY_MS);
    }
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
  const successCount = completed - failed;
  console.log('');
  console.log(`=== 建库完成 ===`);
  console.log(`耗时: ${elapsed}s | 成功入库: ${successCount} | 失败: ${failed} | K线: ${totalKlines} 根 | 暂停: ${pauseCount} 次`);

  closeDb();
}

async function processStock(stock, periods, fetchFn, db, saveKlinesSync, tableForPeriod, sourceName) {
  let totalKlines = 0;
  const errors = [];

  for (const period of periods) {
    try {
      const result = await fetchFn({
        market: stock.market, code: stock.code, period,
        limit: 0, // 全量拉取（百度无硬上限，start_time 追溯到 1990）
      });
      if (result && Array.isArray(result.klines) && result.klines.length > 0) {
        const table = tableForPeriod(period);
        saveKlinesSync(db, table, stock.code, stock.market, result.name || stock.name, result.klines, sourceName);
        totalKlines += result.klines.length;
      } else {
        errors.push(`${period}:无数据`);
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
          console.warn(`全市场列表获取失败 (${fs}): ${err.message}`);
        }
      }
      return all;
    }
    default:
      console.error(`未知 scope: ${scope}`);
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
