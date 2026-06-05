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

  // 其他天：只做 evaluation 收集 + 复盘检查
  return { runHs300: false, runWatchlist: false, runDailyReport: false, reason: `周${dayOfWeek}→仅evaluation收集+复盘检查` };
}

/**
 * 主调度入口
 * @param {object} opts
 * @param {Date} [opts.date] — 默认今天
 * @param {Function} opts.runEvaluation — 跑 evaluation 收集
 * @param {Function} opts.runHs300Scan — 跑沪深300批量扫描
 * @param {Function} opts.runWatchlistScan — 跑自选股批量扫描
 * @param {Function} opts.generateReport — 生成日报
 */
export async function runScheduledJob(opts) {
  const date = opts.date || new Date();
  const schedule = getScheduleForDate(date);

  console.log(`[scheduled] ${schedule.reason}`);

  // 安全开关
  const safety = loadSafetyConfig();
  if (safety.emergencyStop) {
    console.log('[scheduled] emergencyStop=true，终止所有自动任务');
    return { status: 'emergency_stop' };
  }
  if (!safety.enabled) {
    console.log('[scheduled] enabled=false，跳过');
    return { status: 'disabled' };
  }

  // 1. 总是先跑 evaluation（免费）
  let evalResult = null;
  try {
    evalResult = await opts.runEvaluation();
    console.log(`[scheduled] evaluation: ${evalResult?.newEvals || 0} 条新增`);
  } catch (err) {
    console.warn('[scheduled] evaluation 失败:', err.message);
  }

  // 2. 沪深300扫描
  let hs300Result = null;
  if (schedule.runHs300 && !safety.skipHs300) {
    try {
      hs300Result = await opts.runHs300Scan();
      console.log(`[scheduled] HS300: ${hs300Result?.succeeded || 0}/${hs300Result?.total || 0} 成功`);
    } catch (err) {
      console.warn('[scheduled] HS300 扫描失败:', err.message);
    }
  }

  // 3. 自选股扫描
  let watchlistResult = null;
  if (schedule.runWatchlist && !safety.skipWatchlist) {
    try {
      watchlistResult = await opts.runWatchlistScan();
      console.log(`[scheduled] 自选股: ${watchlistResult?.succeeded || 0}/${watchlistResult?.total || 0} 成功`);
    } catch (err) {
      console.warn('[scheduled] 自选股扫描失败:', err.message);
    }
  }

  // 4. 日报
  if (schedule.runDailyReport && !safety.skipDailyReport) {
    try {
      const allResults = [
        ...(hs300Result?.results || []),
        ...(watchlistResult?.results || []),
      ];
      if (allResults.length > 0) {
        await opts.generateReport(allResults, date);
        console.log('[scheduled] 日报已生成');
      }
    } catch (err) {
      console.warn('[scheduled] 日报生成失败:', err.message);
    }
  }

  return { status: 'ok', schedule, evalResult, hs300Result, watchlistResult };
}

// ---- 安全开关 ----

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
