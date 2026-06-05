// Transparent-denominator score calculation
// Always outputs both full (all-denominator) and excl_pf (excluding parse_failed)

function mapSignal(s) {
  if (s === 'strong_bull') return 2;
  if (s === 'bull') return 1;
  if (s === 'neutral') return 0;
  if (s === 'bear') return -1;
  if (s === 'strong_bear') return -2;
  return null;
}

export function scorePrediction(predictedSignal, groundTruth) {
  if (!predictedSignal || !groundTruth) return 0;
  const p = mapSignal(predictedSignal);
  const g = mapSignal(groundTruth);
  if (p === null || g === null) return 0;
  if (p === g) return 1.0;
  if (p * g < 0) {
    return (Math.abs(p) === 2 && Math.abs(g) === 2) ? -1.0 : -0.5;
  }
  if (p === 0 || g === 0) return 0.3;
  return 0.5;
}

/**
 * Transparent score computation, returns both full and excl_pf
 * @param {Array} records — eval jsonl 解析后的记录数组
 * @returns {{ full: Stats, exclPf: Stats }}
 */
export function computeScoreTransparent(records) {
  const valid = records.filter(r => r.score != null);
  const totalFull = records.length;
  const sumFull = records.reduce((s, r) => s + (r.score || 0), 0);

  // 排除 parse_failed (signal === 'parse_failed' 或 predictedSignal === 'parse_failed')
  const nonPF = valid.filter(r => {
    const sig = r.predictedSignal || r.signal || '';
    return sig !== 'parse_failed';
  });
  const totalExclPF = nonPF.length;
  const sumExclPF = nonPF.reduce((s, r) => s + (r.score || 0), 0);

  const full = makeStats(valid, totalFull, sumFull);
  const exclPf = makeStats(nonPF, totalExclPF, sumExclPF);

  return { full, exclPf };
}

function makeStats(records, denominator, scoreSum) {
  const total = denominator;
  const weightedScore = total > 0 ? +(scoreSum / total).toFixed(4) : null;

  let perfect = 0, dirCorrect = 0, dirWrong = 0;
  for (const r of records) {
    if (r.score === 1.0) perfect++;
    if (r.score != null && r.score >= 0.5) dirCorrect++;
    if (r.score != null && r.score < 0) dirWrong++;
  }

  // Signal 分布
  const signalDist = { strong_bull: 0, bull: 0, neutral: 0, bear: 0, strong_bear: 0, parse_failed: 0 };
  for (const r of records) {
    const sig = r.predictedSignal || r.signal || 'parse_failed';
    if (signalDist[sig] !== undefined) signalDist[sig]++;
  }

  return {
    denominator: total,
    weightedScore,
    sum: +scoreSum.toFixed(2),
    perfect: { count: perfect, pct: total > 0 ? (perfect / total * 100).toFixed(1) + '%' : '0%' },
    directionCorrect: { count: dirCorrect, pct: total > 0 ? (dirCorrect / total * 100).toFixed(1) + '%' : '0%' },
    directionWrong: { count: dirWrong, pct: total > 0 ? (dirWrong / total * 100).toFixed(1) + '%' : '0%' },
    signalDistribution: signalDist,
  };
}

/**
 * 格式化对比输出（用于 CLI 报告）
 */
export function formatScoreComparison(result) {
  const { full, exclPf } = result;
  const lines = [
    '=== Score 计算（分母透明） ===',
    '',
    `全量分母: weightedScore=${full.weightedScore}, n=${full.denominator}`,
    `排除PF:   weightedScore=${exclPf.weightedScore}, n=${exclPf.denominator}`,
    `PF数:     ${full.signalDistribution.parse_failed}/${full.denominator}`,
    '',
    '⚠ 报告用 full.weightedScore 为官方数字。exclPf 仅供辅助参考。',
  ];
  return lines.join('\n');
}
