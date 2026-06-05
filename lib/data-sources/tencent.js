// Tencent kline data source (last-resort source)
// No amount/amplitude/changePercent/turnoverRate
//
// API limits:
// - limit param is effective, tested lmt=200 returns 200 monthly bars (since 2009)
// - Monthly earliest reaches approx 2009 (varies by stock), large values auto-truncated

const ENDPOINT = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get';

const PERIOD_KEYS = { daily: 'day', weekly: 'week', monthly: 'month' };
const ADJUST_KEYS = { qfq: 'qfq', hfq: 'hfq', none: '' };

export const sourceName = 'tencent';

/**
 * @param {object} params
 * @returns {Promise<{name,code,market,klines,sourceUsed,fetchedAt}>}
 */
export async function fetchKlines(params = {}) {
  const { market, code, period = 'monthly', limit = 60, adjust = 'qfq' } = params;
  const prefix = market === '1' ? 'sh' : 'sz';
  const marketCode = prefix + code;
  const pkey = PERIOD_KEYS[period] || 'month';
  const akey = ADJUST_KEYS[adjust] || 'qfq';

  const url = `${ENDPOINT}?param=${marketCode},${pkey},,,${limit},${akey}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 6000);
  let resp;
  try {
    resp = await fetch(url, { signal: controller.signal, headers: { 'Referer': 'https://gu.qq.com' } });
  } finally {
    clearTimeout(timer);
  }

  if (!resp.ok) throw new Error(`Tencent HTTP ${resp.status}`);
  const json = await resp.json();

  // 腾讯返回格式: data.{code}.{prefix+period} 或 data.{code}.qfq{period}
  const stockData = json?.data?.[marketCode];
  if (!stockData) throw new Error('腾讯返回结构异常');

  // 腾讯 key 格式：qfqmonth / qfqweek / qfqday
  const dataKey = akey + pkey;
  // also try without adjust prefix
  const dataKeyNoAdj = pkey;
  let rawKlines = stockData[dataKey] || stockData[dataKeyNoAdj];
  // also search any key ending with pkey
  if (!rawKlines) {
    for (const k of Object.keys(stockData)) {
      if (k.endsWith(pkey)) { rawKlines = stockData[k]; break; }
    }
  }
  if (!Array.isArray(rawKlines) || rawKlines.length === 0) throw new Error('腾讯返回空数据');

  const klines = [];
  for (let i = 0; i < rawKlines.length; i++) {
    const row = rawKlines[i];
    // 格式: [date, open, close, high, low, volume]
    const open = parseFloat(row[1]) || null;
    const close = parseFloat(row[2]) || null;
    const high = parseFloat(row[3]) || null;
    const low = parseFloat(row[4]) || null;
    const volume = parseFloat(row[5]) || null;

    let changePercent = null;
    let amplitude = null;
    if (i > 0 && rawKlines[i - 1] && open && close && high && low) {
      const prevClose = parseFloat(rawKlines[i - 1][2]);
      if (prevClose && prevClose !== 0) {
        changePercent = +((close - prevClose) / prevClose * 100).toFixed(2);
        amplitude = +((high - low) / prevClose * 100).toFixed(2);
      }
    }

    klines.push({
      date: normalizeDate(row[0], period),
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
    sourceUsed: 'tencent',
    fetchedAt: new Date().toISOString(),
  };
}

function normalizeDate(dateStr, period) {
  if (!dateStr) return '';
  if (period === 'monthly') return String(dateStr).slice(0, 7);
  if (period === 'weekly') return String(dateStr).slice(0, 10);
  return String(dateStr).slice(0, 10);
}
