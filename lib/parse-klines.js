// 解析东财月线响应的 klines 字符串数组
// 输入示例:["2024-01-31,1700.00,1750.50,1780.00,1690.00,1234567,12345678.90,5.0,2.5,40.0,1.5", ...]
// 字段顺序:日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
// 返回:对象数组,字段不全或 open/close 非数字的行会被丢弃

export function parseKlines(rawArray) {
  if (!Array.isArray(rawArray)) return [];
  const result = [];
  for (const line of rawArray) {
    if (typeof line !== 'string') continue;
    const parts = line.split(',');
    if (parts.length < 11) continue;
    const k = {
      date: parts[0],
      open: Number(parts[1]),
      close: Number(parts[2]),
      high: Number(parts[3]),
      low: Number(parts[4]),
      volume: Number(parts[5]),
      amount: Number(parts[6]),
      amplitude: Number(parts[7]),
      changePercent: Number(parts[8]),
      changeAmount: Number(parts[9]),
      turnoverRate: Number(parts[10]),
    };
    if (Number.isNaN(k.open) || Number.isNaN(k.close)) continue;
    result.push(k);
  }
  return result;
}
