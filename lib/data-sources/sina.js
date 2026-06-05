// Sina Finance kline data source (backup)
// Note: amplitude and change% must be computed; no amount/turnoverRate
//
// API limits:
// - datalen valid range 1-240, auto-truncated to actual available when exceeded or for early-listed stocks
// - 600522 (listed 2002) tested: datalen=240 → 240 bars, datalen=500+ → 283 bars (max)
// - Monthly coverage ~20 years, sufficient for regular analysis

const ENDPOINT = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData';

const PERIOD_SCALES = { daily: 240, weekly: 1680, monthly: 7200 };

export const sourceName = 'sina';

/**
 * @param {object} params
 * @returns {Promise<{name,code,market,klines,sourceUsed,fetchedAt}>}
 */
export async function fetchKlines(params = {}) {
  const { market, code, period = 'monthly', limit = 60 } = params;
  const scale = PERIOD_SCALES[period] || 7200;
  const prefix = market === '1' ? 'sh' : 'sz';
  const sinaCode = prefix + code;

  const url = `${ENDPOINT}?symbol=${sinaCode}&scale=${scale}&ma=no&datalen=${limit}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 6000);
  let resp;
  try {
    resp = await fetch(url, { signal: controller.signal, headers: { 'Referer': 'https://finance.sina.com.cn' } });
  } finally {
    clearTimeout(timer);
  }

  if (!resp.ok) throw new Error(`新浪 HTTP ${resp.status}`);
  const raw = await resp.json();
  if (!Array.isArray(raw) || raw.length === 0) throw new Error('新浪返回空数据');

  const klines = [];
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i];
    const open = parseFloat(item.open) || null;
    const close = parseFloat(item.close) || null;
    const high = parseFloat(item.high) || null;
    const low = parseFloat(item.low) || null;
    const volume = parseFloat(item.volume) || null;

    // 计算振幅和涨跌幅
    let amplitude = null;
    let changePercent = null;
    if (i > 0 && raw[i - 1]) {
      const prevClose = parseFloat(raw[i - 1].close);
      if (prevClose && prevClose !== 0 && open && close && high && low) {
        changePercent = +((close - prevClose) / prevClose * 100).toFixed(2);
        amplitude = +((high - low) / prevClose * 100).toFixed(2);
      }
    }

    klines.push({
      date: normalizeDate(item.day, period),
      open, close, high, low, volume,
      amount: null,
      amplitude,
      changePercent,
      change: null,
      turnoverRate: null,
    });
  }

  return {
    name: code,
    code,
    market,
    klines,
    sourceUsed: 'sina',
    fetchedAt: new Date().toISOString(),
  };
}

function normalizeDate(dateStr, period) {
  if (!dateStr) return '';
  if (period === 'monthly') return String(dateStr).slice(0, 7);
  if (period === 'weekly') return String(dateStr).slice(0, 10);
  return String(dateStr).slice(0, 10);
}
