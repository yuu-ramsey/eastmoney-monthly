// Baidu stock kline data source
// Returns complete fields (including turnover rate/amount), rate limits much lower than eastmoney
// API source: extracted from adata source stock_market_baidu.py

const ENDPOINT = 'https://finance.pae.baidu.com/selfselect/getstockquotation';

const PERIOD_KTYPES = { daily: '1', weekly: '2', monthly: '3' };

export const sourceName = 'baidu';

/**
 * @param {object} params
 * @param {string} params.market — '0'(深) / '1'(沪)
 * @param {string} params.code
 * @param {string} [params.period='monthly']
 * @param {number} [params.limit=60]
 * @returns {Promise<{name,code,market,klines,sourceUsed,fetchedAt}>}
 */
export async function fetchKlines(params = {}) {
  const { market, code, period = 'monthly', limit = 60 } = params;
  const ktype = PERIOD_KTYPES[period] || '3';

  // start_time：limit 为 0 或极大值时取全量（追溯到 1990），否则根据 limit 估算
  const now = new Date();
  const startDate = new Date(now);
  if (limit === 0 || limit >= 1000) {
    startDate.setFullYear(1990); // Full: from 1990 (covers all listed stocks)
  } else if (period === 'daily') {
    startDate.setFullYear(now.getFullYear() - 2);
  } else if (period === 'weekly') {
    startDate.setFullYear(now.getFullYear() - 3);
  } else {
    startDate.setFullYear(now.getFullYear() - 10);
  }
  const startStr = startDate.toISOString().slice(0, 10) + ' 00:00:00';

  const url = `${ENDPOINT}?all=1&isIndex=false&isBk=false&isBlock=false&isFutures=false&isStock=true&newFormat=1&group=quotation_kline_ab&finClientType=pc&code=${code}&start_time=${encodeURIComponent(startStr)}&ktype=${ktype}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  let resp;
  try {
    resp = await fetch(url, {
      signal: controller.signal,
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' },
    });
  } finally {
    clearTimeout(timer);
  }

  if (!resp.ok) throw new Error(`百度 HTTP ${resp.status}`);

  const json = await resp.json();
  if (json.ResultCode !== '0') throw new Error(`百度 API 错误: ResultCode=${json.ResultCode}`);

  const raw = json?.Result?.newMarketData;
  if (!raw || !raw.marketData) throw new Error('百度返回空数据');

  // marketData 是分号分隔的行，每行是逗号分隔的字段
  // headers: [时间戳, 时间, 开盘, 收盘, 成交量, 最高, 最低, 成交额, 涨跌额, 涨跌幅, 换手率, 昨收, ma5均价, ma5成交量, ma10均价, ma10成交量, ma20均价, ma20成交量]
  const rows = raw.marketData.split(';').filter(Boolean);
  const klines = [];

  for (const row of rows) {
    const fields = row.split(',');
    const dateStr = fields[1]; // 原始格式: 2024-01-31
    if (!dateStr) continue;

    const open = parseFloat(fields[2]) || null;
    const close = parseFloat(fields[3]) || null;
    const high = parseFloat(fields[5]) || null;
    const low = parseFloat(fields[6]) || null;
    const volume = parseFloat(fields[4]) || null;
    const amount = parseFloat(fields[7]) || null;
    const change = parseFloat(fields[8]) || null;
    const changePercent = parseFloat(fields[9]) || null;
    const turnoverRate = parseFloat(fields[10]) || null;

    // 计算振幅
    let amplitude = null;
    if (high != null && low != null) {
      const prevClose = parseFloat(fields[11]);
      if (prevClose && prevClose !== 0) {
        amplitude = +((high - low) / prevClose * 100).toFixed(2);
      }
    }

    klines.push({
      date: normalizeDate(dateStr, period),
      open, close, high, low, volume, amount,
      amplitude,
      changePercent,
      change,
      turnoverRate,
    });
  }

  if (klines.length === 0) throw new Error('百度返回空 K 线');

  // limit=0 或极大值 → 返回全部；否则取最近 N 根
  const result = (limit === 0 || limit >= 1000) ? klines : klines.slice(-limit);

  return {
    name: code,
    code,
    market,
    klines: result,
    sourceUsed: 'baidu',
    fetchedAt: new Date().toISOString(),
  };
}

function normalizeDate(dateStr, period) {
  if (!dateStr) return '';
  if (period === 'monthly') return String(dateStr).slice(0, 7);
  return String(dateStr).slice(0, 10);
}
