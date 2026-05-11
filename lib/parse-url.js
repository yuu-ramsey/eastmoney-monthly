// 从东方财富个股页 URL 解析市场和股票代码
// 支持形态:
//   https://quote.eastmoney.com/sh600519.html
//   https://quote.eastmoney.com/sz000001.html
//   https://quote.eastmoney.com/sh.600519
//   大小写不敏感
// 返回:{ market: '0' | '1', code: '6 位数字' } | null
//   sh -> '1' (上交所),sz -> '0' (深交所);secid 拼接时用

export function parseStockUrl(url) {
  if (typeof url !== 'string' || url.length === 0) return null;
  // \b 边界 + 可选的点号分隔 + 必须 6 位数字
  const m = url.match(/\b(sh|sz)\.?(\d{6})\b/i);
  if (!m) return null;
  const market = m[1].toLowerCase() === 'sh' ? '1' : '0';
  return { market, code: m[2] };
}
