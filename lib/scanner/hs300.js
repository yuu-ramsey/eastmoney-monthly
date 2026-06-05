// CSI 300 constituent stock fetch — Sina API, real CSI 300 index constituents
// Auto-refreshes monthly, falls back to cache on failure
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const CACHE_PATH = path.join(PROJECT_DIR, '.eastmoney-ai', 'hs300-cache.json');

const ENDPOINT = 'https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData';
const PAGE_SIZE = 100;
const MAX_PAGES = 4;

/**
 * 拉取沪深300成分股
 * @param {boolean} forceRefresh — 强制刷新，忽略缓存
 * @returns {Array<{code: string, market: string, name: string}>}
 */
export async function fetchHS300Constituents(forceRefresh = false) {
  if (!forceRefresh) {
    const cached = loadCache();
    if (cached && cached.length >= 200 && isCurrentMonth()) {
      return cached;
    }
  }

  try {
    const allStocks = [];

    for (let page = 1; page <= MAX_PAGES; page++) {
      const url = `${ENDPOINT}?page=${page}&num=${PAGE_SIZE}&sort=symbol&asc=1&node=hs300`;
      const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
      if (!resp.ok) throw new Error(`新浪 HTTP ${resp.status}`);
      const text = await resp.text();
      let data;
      try { data = JSON.parse(text); } catch (_) { throw new Error('新浪返回非JSON'); }
      if (!Array.isArray(data) || data.length === 0) break;

      for (const item of data) {
        allStocks.push({
          code: item.code,
          market: /^6/.test(item.code) ? '1' : '0',
          name: item.name || item.code,
        });
      }

      if (data.length < PAGE_SIZE) break;
    }

    if (allStocks.length >= 200) {
      saveCache(allStocks);
      return allStocks;
    }

    throw new Error(`仅获取到 ${allStocks.length} 只，少于 200 下限`);
  } catch (err) {
    console.warn('[hs300] 拉取失败，降级到缓存:', err.message);
  }

  const cached = loadCache();
  return cached || [];
}

function loadCache() {
  try {
    const data = JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8'));
    if (Array.isArray(data.stocks) && data.stocks.length >= 200) return data.stocks;
  } catch (_) {}
  return null;
}

function saveCache(stocks) {
  const dir = path.dirname(CACHE_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(CACHE_PATH, JSON.stringify({ stocks, updatedAt: new Date().toISOString() }, null, 2), 'utf-8');
}

function isCurrentMonth() {
  try {
    const stat = fs.statSync(CACHE_PATH);
    return stat.mtime.toISOString().slice(0, 7) === new Date().toISOString().slice(0, 7);
  } catch (_) { return false; }
}
