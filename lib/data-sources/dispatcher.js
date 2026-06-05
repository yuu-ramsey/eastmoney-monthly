// Data source fallback dispatch: eastmoney → sina → tencent
// Uses dynamic import to avoid module-level import caching issues
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const LOG_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'logs');

const SOURCE_PATHS = [
  { name: 'baidu', path: './baidu.js', timeout: 8000 },
  { name: 'sina', path: './sina.js', timeout: 6000 },
  { name: 'tencent', path: './tencent.js', timeout: 6000 },
  { name: 'eastmoney', path: './eastmoney.js', timeout: 8000 },
];

// Degradation counter (prevents overwhelming backup sources during batch scans)
let degradeCounter = 0;
let degradeHour = -1;

function getDegradeCounter() {
  const now = new Date();
  if (now.getHours() !== degradeHour) {
    degradeCounter = 0;
    degradeHour = now.getHours();
  }
  return degradeCounter;
}
function incDegradeCounter() {
  getDegradeCounter();
  degradeCounter++;
}

const MAX_DEGRADES_PER_HOUR = 20;

function ensureLogDir() {
  if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });
}

function logDegrade(succeeded, failedSources) {
  ensureLogDir();
  const today = new Date().toISOString().slice(0, 10);
  const logPath = path.join(LOG_DIR, `degrade-${today}.log`);
  const ts = new Date().toISOString();
  const line = `[${ts}] 降级：${failedSources.map((s) => s.name + '(' + s.reason + ')').join(', ')} → 最终使用 ${succeeded}`;
  fs.appendFileSync(logPath, line + '\n', 'utf-8');
}

/**
 * 验证数据是否可用
 */
function validateData(data, minRatio = 0.5, expectedLimit = 60) {
  if (!data || !Array.isArray(data.klines) || data.klines.length === 0) return false;

  // 至少要有预期数量的 50%
  if (data.klines.length < 12) return false;

  // 降级源数据偏少时 warn 但不拒绝
  if (data.klines.length < expectedLimit * 0.5) {
    console.warn(`[data-source] ${data.sourceUsed} 仅返回 ${data.klines.length} 根K线（期望 ≥${expectedLimit * 0.5}），高级功能可能降级`);
  }

  // 最近一根 K 线日期不早于 90 天前
  const lastDate = data.klines[data.klines.length - 1].date;
  if (lastDate) {
    const d = new Date(lastDate + '-01');
    const daysAgo = (Date.now() - d.getTime()) / 86400000;
    if (daysAgo > 120) return false;
  }

  return true;
}

/**
 * 带降级的 K 线获取
 * @param {object} params - { market, code, period, limit, adjust }
 * @returns {Promise<{name,code,market,klines,sourceUsed,fetchedAt}>}
 */
export async function fetchKlinesWithFallback(params = {}) {
  const failedSources = [];

  for (const source of SOURCE_PATHS) {
    // 降级计数保护
    if (source.name !== 'eastmoney') {
      if (getDegradeCounter() >= MAX_DEGRADES_PER_HOUR) {
        throw new Error(`本小时降级已达 ${MAX_DEGRADES_PER_HOUR} 次，暂停使用备源以防被封`);
      }
    }

    try {
      const mod = await import(source.path);
      const result = await mod.fetchKlines(params);

      if (validateData(result, 0.5, (params.limit || 60))) {
        if (failedSources.length > 0) {
          incDegradeCounter();
          logDegrade(source.name, failedSources);
        }
        return { ...result, degradedFrom: failedSources.map((s) => s.name) };
      }

      failedSources.push({ name: source.name, reason: '数据验证失败' });
    } catch (err) {
      failedSources.push({ name: source.name, reason: err.message || String(err) });
    }
  }

  throw new Error(`所有数据源均失败：${failedSources.map((s) => s.name + '(' + s.reason + ')').join(', ')}`);
}

/**
 * health check
 */
export async function healthCheck() {
  const results = [];
  for (const source of SOURCE_PATHS) {
    const start = Date.now();
    try {
      const mod = await import(source.path);
      const promise = mod.fetchKlines({ market: '1', code: '600522', period: 'monthly', limit: 60 });
      const data = await Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 6000)),
      ]);
      const elapsed = ((Date.now() - start) / 1000).toFixed(1);
      const fields = data.klines.length > 0 ? Object.keys(data.klines[0]).filter((k) => data.klines[0][k] != null).length : 0;
      results.push({ name: source.name, status: 'ok', elapsed: elapsed + 's', klineCount: data.klines.length, fields });
    } catch (err) {
      const elapsed = ((Date.now() - start) / 1000).toFixed(1);
      results.push({ name: source.name, status: 'fail', elapsed: elapsed + 's', error: err.message });
    }
  }
  return results;
}
