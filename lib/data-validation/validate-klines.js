// Kline data health check — field completeness, value ranges, time-series continuity, cross-field consistency, outliers

/**
 * @param {Array} klines - K-line array
 * @param {string} period — 'monthly'|'weekly'|'daily'|'60min'
 * @returns {{ valid: boolean, issues: Array<{field:string, message:string, severity:'warn'|'error'}>, severity: 'ok'|'warn'|'error' }}
 */
export function checkKlines(klines, period = 'monthly') {
  const issues = [];

  if (!Array.isArray(klines) || klines.length === 0) {
    return { valid: false, issues: [{ field: 'klines', message: 'K线数组为空', severity: 'error' }], severity: 'error' };
  }

  if (klines.length < 12) {
    issues.push({ field: 'klines', message: `仅 ${klines.length} 根 K 线，高级分析需要 ≥12 根`, severity: 'warn' });
  }

  let prevDate = null;
  let totalChanges = 0;
  let bigMoves = 0;

  for (let i = 0; i < klines.length; i++) {
    const k = klines[i];
    const idx = `[${i}] ${k.date || '?'}`;

    // 字段完整
    for (const f of ['date', 'open', 'close', 'high', 'low', 'volume']) {
      if (k[f] == null || (typeof k[f] === 'number' && isNaN(k[f]))) {
        issues.push({ field: f, message: `${idx}: ${f} 缺失`, severity: 'error' });
      }
    }

    // 数值范围
    if (k.close != null && k.close <= 0) {
      issues.push({ field: 'close', message: `${idx}: close=${k.close} ≤ 0`, severity: 'error' });
    }
    if (k.volume != null && k.volume < 0) {
      issues.push({ field: 'volume', message: `${idx}: volume=${k.volume} < 0`, severity: 'error' });
    }

    // 跨字段自洽
    if (k.low != null && k.high != null && k.low > k.high) {
      issues.push({ field: 'low/high', message: `${idx}: low=${k.low} > high=${k.high}`, severity: 'error' });
    }
    if (k.open != null && k.close != null && k.high != null && k.low != null) {
      const rangeMin = Math.min(k.open, k.close);
      const rangeMax = Math.max(k.open, k.close);
      if (k.low > rangeMin + 0.001) {
        issues.push({ field: 'low', message: `${idx}: low=${k.low} > min(open,close)=${rangeMin.toFixed(2)}`, severity: 'warn' });
      }
      if (k.high < rangeMax - 0.001) {
        issues.push({ field: 'high', message: `${idx}: high=${k.high} < max(open,close)=${rangeMax.toFixed(2)}`, severity: 'warn' });
      }
    }

    // 异常值检测（单根涨跌幅 > 50%，可能除权未处理）
    if (i > 0 && k.close != null && klines[i - 1].close != null && klines[i - 1].close > 0) {
      const changePct = Math.abs((k.close - klines[i - 1].close) / klines[i - 1].close * 100);
      if (changePct > 50) {
        bigMoves++;
        issues.push({ field: 'close', message: `${idx}: 单期变化 ${changePct.toFixed(1)}% > 50%，可能除权未处理`, severity: 'warn' });
      }
    }

    // 时序连续
    if (prevDate && k.date) {
      const gap = dateGap(prevDate, k.date, period);
      if (gap !== null && gap > 1) {
        issues.push({ field: 'date', message: `${prevDate} → ${k.date}: 缺失 ${gap - 1} 期`, severity: 'warn' });
      }
    }
    prevDate = k.date;
  }

  // 判定整体 severity
  const errors = issues.filter(i => i.severity === 'error');
  const warns = issues.filter(i => i.severity === 'warn');

  let severity = 'ok';
  if (errors.length > 0) severity = 'error';
  else if (warns.length > 0) severity = 'warn';

  // 去重（同类型 issue 限制 5 条）
  const limited = [];
  const counter = {};
  for (const issue of issues) {
    const key = issue.field + ':' + issue.severity;
    counter[key] = (counter[key] || 0) + 1;
    if (counter[key] <= 5) limited.push(issue);
  }

  return {
    valid: severity !== 'error',
    issues: limited,
    severity,
    _summary: { total: issues.length, errors: errors.length, warns: warns.length, bigMoves },
  };
}

/** 计算两个日期之间的周期差距，返回 null 表示无法判断 */
function dateGap(prev, curr, period) {
  if (!prev || !curr) return null;
  if (period === 'monthly') {
    const [py, pm] = prev.split('-').map(Number);
    const [cy, cm] = curr.split('-').map(Number);
    if (!py || !pm || !cy || !cm) return null;
    return (cy - py) * 12 + (cm - pm);
  }
  // daily/weekly: 简单比较日期字符串
  if (prev < curr) return 0; // 有递增即可，不检查精确 gap
  return null;
}

/** 生成 prompt 可注入的 warning 文本 */
export function buildValidationWarning(validation) {
  if (!validation || validation.severity === 'ok') return '';
  const lines = ['⚠ 数据质量警告：'];
  for (const issue of validation.issues) {
    if (issue.severity === 'error') lines.push(`- [错误] ${issue.message}`);
  }
  for (const issue of validation.issues.slice(0, 3)) {
    if (issue.severity === 'warn') lines.push(`- [警告] ${issue.message}`);
  }
  if (validation.issues.length > 3) {
    lines.push(`- ... 共 ${validation.issues.length} 条警告，详情见日志`);
  }
  lines.push('以上数据异常可能影响分析准确性，请酌情处理。');
  return lines.join('\n');
}
