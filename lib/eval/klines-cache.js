// Kline cache layer — deduplicate fetches, reduce API calls
// Only used within eval, does not affect production analysis path
export class KlinesCache {
  #store = new Map();
  #maxAgeMs;
  #hitCount = 0;
  #missCount = 0;

  constructor(maxAgeMs = 24 * 3600 * 1000) {
    this.#maxAgeMs = maxAgeMs;
  }

  /** Generate cache key */
  key(stockCode, market, period, limit) {
    return `${market}.${stockCode}:${period}:${limit}`;
  }

  /** 获取缓存，过期返回null */
  get(stockCode, market, period = 'monthly', limit = 200) {
    const k = this.key(stockCode, market, period, limit);
    const entry = this.#store.get(k);
    if (entry && (Date.now() - entry.ts) < this.#maxAgeMs) {
      this.#hitCount++;
      return entry.klines;
    }
    this.#missCount++;
    return null;
  }

  /** 写入缓存 */
  set(stockCode, market, period, limit, klines) {
    const k = this.key(stockCode, market, period, limit);
    this.#store.set(k, { klines, ts: Date.now() });
  }

  size() { return this.#store.size; }
  hits() { return this.#hitCount; }
  misses() { return this.#missCount; }
  hitRate() { const t = this.#hitCount + this.#missCount; return t > 0 ? (this.#hitCount / t * 100).toFixed(1) + '%' : '0%'; }

  clear() {
    this.#store.clear();
    this.#hitCount = 0;
    this.#missCount = 0;
  }
}

/** 带退避的fetch */
export async function fetchWithRetry(fn, maxRetries = 3) {
  let lastErr;
  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (i < maxRetries) {
        const delay = Math.pow(2, i) * 1000; // 1s, 2s, 4s
        await sleep(delay + Math.random() * 500);
      }
    }
  }
  throw lastErr;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
