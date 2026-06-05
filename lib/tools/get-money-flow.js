// Fetch last N months of major fund flow
// endpoint: https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get

const ENDPOINT = 'https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get';

const TIMEOUT_MS = 8000;

export const getMoneyFlowTool = {
  name: 'get_money_flow',
  description: 'Get major capital flow data for the last N months for a stock。传入 secid 和可选的 limit（默认 12 个月），返回每月主力净流入金额表格，用于判断资金面的多空倾向。',
  input_schema: {
    type: 'object',
    properties: {
      secid: {
        type: 'string',
        description: '东方财富证券代码标识，格式为 "市场.代码"，例如 "1.600519"',
      },
      limit: {
        type: 'integer',
        description: '返回的月数，默认 12，最大 60',
      },
    },
    required: ['secid'],
  },

  async handler(input) {
    const secid = input?.secid;
    if (!secid || typeof secid !== 'string') {
      return '错误：缺少 secid 参数';
    }

    const limit = Math.min(Math.max(parseInt(String(input?.limit), 10) || 12, 1), 60);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    let resp;
    try {
      resp = await fetch(
        `${ENDPOINT}?secid=${encodeURIComponent(secid)}&klt=103&lmt=${limit}&fields1=f1,f2,f3,f4,f5,f6,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65`,
        { signal: controller.signal },
      );
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') return '错误：获取资金流数据超时（8s），请稍后重试';
      return '错误：网络请求失败，无法获取资金流数据';
    }
    clearTimeout(timer);

    if (!resp.ok) return `错误：东财接口返回 HTTP ${resp.status}`;

    let json;
    try {
      json = await resp.json();
    } catch (_) {
      return '错误：东财接口返回格式异常';
    }

    const klines = json?.data?.klines;
    if (!klines || klines.length === 0) {
      return '该股票暂无资金流数据（可能上市时间较短或数据源未覆盖）';
    }

    // 解析 K 线字符串（格式：日期,主力净流入,...）
    const rows = [];
    let totalInflow = 0;
    let positiveMonths = 0;

    for (const line of klines) {
      const fields = String(line).split(',');
      const date = fields[0] || '';
      const mainInflow = parseFloat(fields[1]) || 0; // f52: 主力净流入（元）
      const retailInflow = parseFloat(fields[2]) || 0; // f53: 小单净流入（元）

      rows.push({ date: date.slice(0, 7), mainInflow, retailInflow });
      totalInflow += mainInflow;
      if (mainInflow > 0) positiveMonths++;
    }

    // 格式化输出
    const parts = [
      `近 ${rows.length} 个月主力资金流向:`,
      '',
      '| 月份 | 主力净流入 | 方向 |',
      '|------|-----------------|------|',
    ];

    for (const r of rows) {
      const dir = r.mainInflow > 0 ? '流入' : r.mainInflow < 0 ? '流出' : '持平';
      parts.push(`| ${r.date} | ${formatFlow(r.mainInflow)} | ${dir} |`);
    }

    parts.push('');
    parts.push(`资金流统计:`);
    parts.push(`  近 ${rows.length} 个月主力累计净流入: ${formatFlow(totalInflow)}`);
    parts.push(`  主力净流入为正的月份: ${positiveMonths}/${rows.length} (${(positiveMonths / rows.length * 100).toFixed(0)}%)`);

    if (totalInflow > 0) {
      parts.push('  整体判断: 近 N 月主力资金呈净流入态势，资金面偏多');
    } else if (totalInflow < 0) {
      parts.push('  整体判断: 近 N 月主力资金呈净流出态势，资金面偏空');
    } else {
      parts.push('  整体判断: 近 N 月主力资金流入流出基本持平');
    }

    return parts.join('\n');
  },
};

function formatFlow(v) {
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e8) return sign + (abs / 1e8).toFixed(2) + ' 亿';
  if (abs >= 1e4) return sign + (abs / 1e4).toFixed(0);
  return sign + abs.toFixed(0);
}
