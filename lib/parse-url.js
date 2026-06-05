// Parse market and stock code from Eastmoney stock page URL
// Supported formats:
//   https://quote.eastmoney.com/sh600519.html
//   https://quote.eastmoney.com/sz000001.html
//   https://quote.eastmoney.com/sh.600519
//   Case-insensitive
// Returns: { market: '0' | '1', code: '6-digit' } | null
//   sh -> '1' (Shanghai Exchange), sz -> '0' (Shenzhen Exchange); used for secid concatenation

export function parseStockUrl(url) {
  if (typeof url !== 'string' || url.length === 0) return null;
  // \b boundary + optional dot separator + required 6 digits
  const m = url.match(/\b(sh|sz)\.?(\d{6})\b/i);
  if (!m) return null;
  const market = m[1].toLowerCase() === 'sh' ? '1' : '0';
  return { market, code: m[2] };
}
