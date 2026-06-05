// Watchlist management — read/write .eastmoney-ai/watchlist.json
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const DATA_DIR = path.join(PROJECT_DIR, '.eastmoney-ai');
const WATCHLIST_PATH = path.join(DATA_DIR, 'watchlist.json');

const DEFAULT_WATCHLIST = { stocks: [], maxSize: 50 };

function ensureDir() {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
}

export function loadWatchlist() {
  ensureDir();
  try {
    return JSON.parse(fs.readFileSync(WATCHLIST_PATH, 'utf-8'));
  } catch (_) {
    return { ...DEFAULT_WATCHLIST };
  }
}

function saveWatchlist(data) {
  ensureDir();
  fs.writeFileSync(WATCHLIST_PATH, JSON.stringify(data, null, 2), 'utf-8');
}

export function addStock(code, market, name, tags = []) {
  const wl = loadWatchlist();
  if (wl.stocks.length >= wl.maxSize) {
    throw new Error(`自选股已达上限 ${wl.maxSize} 只，请先删除不再关注的`);
  }
  if (wl.stocks.find((s) => s.code === code)) {
    return { ...wl, alreadyExists: true };
  }
  wl.stocks.push({
    code,
    market: market || (/^6/.test(code) ? '1' : '0'),
    name: name || code,
    addedAt: new Date().toISOString().slice(0, 10),
    tags,
  });
  saveWatchlist(wl);
  return wl;
}

export function removeStock(code) {
  const wl = loadWatchlist();
  wl.stocks = wl.stocks.filter((s) => s.code !== code);
  saveWatchlist(wl);
  return wl;
}

export function getStockList() {
  const wl = loadWatchlist();
  return wl.stocks;
}

// 东财拿股票名
export async function fetchStockName(code) {
  const market = /^6/.test(code) ? '1' : '0';
  try {
    const resp = await fetch(
      `https://push2.eastmoney.com/api/qt/stock/get?secid=${market}.${code}&fields=f57,f58`,
      { signal: AbortSignal.timeout(5000) },
    );
    if (!resp.ok) return code;
    const json = await resp.json();
    return json?.data?.f58 || code;
  } catch (_) {
    return code;
  }
}

export { WATCHLIST_PATH };
