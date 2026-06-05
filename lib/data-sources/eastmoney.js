// Eastmoney kline data source (primary, with the most complete fields)
const ENDPOINT = 'https://push2his.eastmoney.com/api/qt/stock/kline/get';

const PERIOD_KLTS = { monthly: '103', weekly: '102', daily: '101' };
const FQT_MAP = { qfq: '1', hfq: '2', none: '0' };

export const sourceName = 'eastmoney';

/**
 * @param {object} params
 * @param {string} params.market - '0'|'1'
 * @param {string} params.code
 * @param {string} [params.period='monthly']
 * @param {number} [params.limit=60]
 * @param {string} [params.adjust='qfq']
 * @returns {Promise<{name,code,market,klines,sourceUsed,fetchedAt}>}
 */
export async function fetchKlines(params = {}) {
  const { market, code, period = 'monthly', limit = 60, adjust = 'qfq' } = params;
  const klt = PERIOD_KLTS[period] || '103';
  const fqt = FQT_MAP[adjust] || '1';

  const url = `${ENDPOINT}?secid=${market}.${code}&klt=${klt}&fqt=${fqt}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&beg=0&end=20500101&lmt=${limit}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);

  let resp;
  try {
    resp = await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }

  if (!resp.ok) throw new Error(`东财 HTTP ${resp.status}`);
  const json = await resp.json();
  const data = json?.data;
  if (!data || !Array.isArray(data.klines) || data.klines.length === 0) {
    throw new Error('东财返回空数据');
  }

  const klines = data.klines.map((line) => {
    const fields = String(line).split(',');
    return {
      date: normalizeDate(fields[0], period),
      open: parseFloat(fields[1]) || null,
      close: parseFloat(fields[2]) || null,
      high: parseFloat(fields[3]) || null,
      low: parseFloat(fields[4]) || null,
      volume: parseFloat(fields[5]) || null,
      amount: parseFloat(fields[6]) || null,
      amplitude: parseFloat(fields[7]) || null,
      changePercent: parseFloat(fields[8]) || null,
      change: parseFloat(fields[9]) || null,
      turnoverRate: parseFloat(fields[10]) || null,
    };
  });

  return {
    name: data.name || code,
    code,
    market,
    klines,
    sourceUsed: 'eastmoney',
    fetchedAt: new Date().toISOString(),
  };
}

function normalizeDate(dateStr, period) {
  if (!dateStr) return '';
  if (period === 'monthly') return String(dateStr).slice(0, 7);
  if (period === 'weekly') return String(dateStr).slice(0, 10);
  return String(dateStr).slice(0, 10);
}
