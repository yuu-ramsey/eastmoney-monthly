// Fetch current financial indicators for a stock (PE/PB/market cap/industry)
// endpoint: https://push2.eastmoney.com/api/qt/stock/get

const ENDPOINT = 'https://push2.eastmoney.com/api/qt/stock/get';

// Eastmoney field number mapping (verified on 600522/600519)
const FIELD_MAP = {
  f43: 'Current price',
  f57: '代码',
  f58: '名称',
  f162: '市盈率TTM',
  f167: '市净率',
  f116: '总市值',
  f117: '流通市值',
  f127: '所属行业',
};

const FIELD_KEYS = Object.keys(FIELD_MAP);

// Eastmoney API unit conversion: some fields return fen/basis points, divide by 100 to standard units
// Verified benchmark: 600519 Moutai 2026-05-15, f43=133295 -> 1332.95 yuan, f162=1532 -> 15.32, f167=616 -> 6.16
const FIELD_DIVISORS = {
  f43: 100,   // Current price: fen -> yuan
  f162: 100,  // PE TTM: basis points -> standard
  f167: 100,  // PB: basis points -> standard
};

function applyDivisor(field, value) {
  if (value == null || typeof value !== 'number') return value;
  const div = FIELD_DIVISORS[field];
  return div ? value / div : value;
}

const TIMEOUT_MS = 8000;

export const getFinancialsTool = {
  name: 'get_financials',
  description: '获取股票的当前财务指标：当前价、市盈率(TTM)、市净率、总市值、流通市值、所属行业。传入 secid 参数（格式如 "1.600519"），返回格式化的财务指标文本。',
  input_schema: {
    type: 'object',
    properties: {
      secid: {
        type: 'string',
        description: '东方财富证券代码标识，格式为 "市场.代码"，例如 "1.600519"（沪市贵州茅台）、"0.300750"（深市宁德时代）',
      },
    },
    required: ['secid'],
  },

  async handler(input) {
    const secid = input?.secid;
    if (!secid || typeof secid !== 'string') {
      return '错误：缺少 secid 参数';
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    let resp;
    try {
      resp = await fetch(
        `${ENDPOINT}?secid=${encodeURIComponent(secid)}&fields=${FIELD_KEYS.join(',')}`,
        { signal: controller.signal },
      );
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') return '错误：获取财务数据超时（8s），请稍后重试';
      return '错误：网络请求失败，无法获取财务数据';
    }
    clearTimeout(timer);

    if (!resp.ok) return `错误：东财接口返回 HTTP ${resp.status}`;

    let json;
    try {
      json = await resp.json();
    } catch (_) {
      return '错误：东财接口返回格式异常';
    }

    const data = json?.data;
    if (!data) return '错误：未找到该股票的财务数据，请检查 secid 是否正确';

    // Format output, mark missing fields as "Data temporarily unavailable"
    const name = data.f58 || '未知';
    const code = data.f57 || secid;
    const parts = [`${name}(${code}) 财务指标:`, ''];

    const price = applyDivisor('f43', data.f43);
    parts.push(`  当前价: ${price != null ? formatNumber(price) + ' 元' : '该数据暂不可用'}`);

    const pe = applyDivisor('f162', data.f162);
    parts.push(`  市盈率(TTM): ${pe != null ? formatNumber(pe) : '该数据暂不可用'}`);

    const pb = applyDivisor('f167', data.f167);
    parts.push(`  市净率: ${pb != null ? formatNumber(pb) : '该数据暂不可用'}`);

    const totalMv = data.f116;
    if (totalMv != null) {
      parts.push(`  总市值: ${formatMarketCap(totalMv)}`);
    } else {
      parts.push('  总市值: 该数据暂不可用');
    }

    const circMv = data.f117;
    if (circMv != null) {
      parts.push(`  流通市值: ${formatMarketCap(circMv)}`);
    } else {
      parts.push('  流通市值: 该数据暂不可用');
    }

    const industry = data.f127;
    parts.push(`  行业: ${industry || '该数据暂不可用'}`);

    return parts.join('\n');
  },
};

// Plain number format (keep 2 decimal places)
function formatNumber(v) {
  const n = Number(v);
  if (isNaN(n)) return String(v);
  return n.toFixed(2);
}

// Market cap formatter (in 100-millions)
function formatMarketCap(v) {
  const n = Number(v);
  if (isNaN(n)) return String(v);
  // Eastmoney market cap unit uncertain (might be yuan), convert by scale
  if (n >= 1e8) return (n / 1e8).toFixed(0) + ' 亿';
  if (n >= 1e4) return (n / 1e4).toFixed(0) + ' 万';
  return n.toFixed(0);
}
