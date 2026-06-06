// Schedule router — determines what to run based on date
// Exports pure functions for testability; CLI entry calls runScheduledJob

/**
 * Determine what task to run for the current date
 * @param {Date} date
 * @returns {{ runHs300: boolean, runWatchlist: boolean, runDailyReport: boolean, reason: string }}
 */
export function getScheduleForDate(date = new Date()) {
  const dayOfWeek = date.getDay(); // 0=Sun
  const dayOfMonth = date.getDate();
  const weekOfMonth = Math.ceil(dayOfMonth / 7);
  const isHs300Week = weekOfMonth % 2 === 1; // Week 1 and 3

  // Sunday: CSI300 week → HS300+watchlist, watchlist week → watchlist only
  if (dayOfWeek === 0) {
    if (isHs300Week) {
      return { runHs300: true, runWatchlist: true, runDailyReport: true, reason: `周日(第${weekOfMonth}周,HS300周)→沪深300+自选股+日报` };
    }
    return { runHs300: false, runWatchlist: true, runDailyReport: true, reason: `周日(第${weekOfMonth}周,自选股周)→仅自选股+日报` };
  }

  // Other days: only run evaluation collection + review check
  return { runHs300: false, runWatchlist: false, runDailyReport: false, reason: `周${dayOfWeek}→仅evaluation收集+复盘检查` };
}

/**
 * Main scheduler entry
 * @param {object} opts
 * @param {Date} [opts.date] — defaults to today
 * @param {Function} opts.runEvaluation — run evaluation collection
 * @param {Function} opts.runHs300Scan — run CSI300 batch scan
 * @param {Function} opts.runWatchlistScan — run watchlist batch scan
 * @param {Function} opts.generateReport — generate daily report
 */
export async function runScheduledJob(opts) {
  const date = opts.date || new Date();
  const schedule = getScheduleForDate(date);

  console.log(`[scheduled] ${schedule.reason}`);

  // Safety switch
  const safety = loadSafetyConfig();
  if (safety.emergencyStop) {
    console.log('[scheduled] emergencyStop=true, aborting all automated tasks');
    return { status: 'emergency_stop' };
  }
  if (!safety.enabled) {
    console.log('[scheduled] enabled=false, skipping');
    return { status: 'disabled' };
  }

  // 1. Always run evaluation first (free)
  let evalResult = null;
  try {
    evalResult = await opts.runEvaluation();
    console.log(`[scheduled] evaluation: ${evalResult?.newEvals || 0} new entries`);
  } catch (err) {
    console.warn('[scheduled] evaluation failed:', err.message);
  }

  // 2. CSI300 scan
  let hs300Result = null;
  if (schedule.runHs300 && !safety.skipHs300) {
    try {
      hs300Result = await opts.runHs300Scan();
      console.log(`[scheduled] HS300: ${hs300Result?.succeeded || 0}/${hs300Result?.total || 0} succeeded`);
    } catch (err) {
      console.warn('[scheduled] HS300 scan failed:', err.message);
    }
  }

  // 3. Watchlist scan
  let watchlistResult = null;
  if (schedule.runWatchlist && !safety.skipWatchlist) {
    try {
      watchlistResult = await opts.runWatchlistScan();
      console.log(`[scheduled] Watchlist: ${watchlistResult?.succeeded || 0}/${watchlistResult?.total || 0} succeeded`);
    } catch (err) {
      console.warn('[scheduled] Watchlist scan failed:', err.message);
    }
  }

  // 4. Daily report
  if (schedule.runDailyReport && !safety.skipDailyReport) {
    try {
      const allResults = [
        ...(hs300Result?.results || []),
        ...(watchlistResult?.results || []),
      ];
      if (allResults.length > 0) {
        await opts.generateReport(allResults, date);
        console.log('[scheduled] Daily report generated');
      }
    } catch (err) {
      console.warn('[scheduled] Daily report generation failed:', err.message);
    }
  }

  return { status: 'ok', schedule, evalResult, hs300Result, watchlistResult };
}

// ---- Safety switch ----

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const SAFETY_PATH = path.join(PROJECT_DIR, '.eastmoney-ai', 'scheduler-config.json');

const DEFAULT_SAFETY = { enabled: true, skipHs300: false, skipWatchlist: false, skipDailyReport: false, emergencyStop: false };

export function loadSafetyConfig() {
  try { return JSON.parse(fs.readFileSync(SAFETY_PATH, 'utf-8')); } catch (_) { return { ...DEFAULT_SAFETY }; }
}

export function saveSafetyConfig(data) {
  const dir = path.dirname(SAFETY_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(SAFETY_PATH, JSON.stringify(data, null, 2), 'utf-8');
}
