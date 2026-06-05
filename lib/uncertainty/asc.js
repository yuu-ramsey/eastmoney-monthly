// Adaptive Signal Calibration (ASC)
// Calibrate using LLM self-assessed confidence: high-confidence subsets should have higher accuracy
// Output calibration curve + high-conf subset scores

/**
 * Parse confidence field from LLM rawResponse
 * @param {string} rawResponse
 * @returns {'high'|'medium'|'low'|null}
 */
export function parseConfidence(rawResponse) {
  if (!rawResponse) return null;
  try {
    const m = rawResponse.match(/```json\s*([\s\S]*?)```/);
    if (!m) return null;
    const data = JSON.parse(m[1].trim());
    const conf = (data.confidence || '').toLowerCase();
    if (['high', 'medium', 'low'].includes(conf)) return conf;
    return null;
  } catch (_) { return null; }
}

/**
 * Stratify score statistics by confidence
 * @param {Array} records — eval jsonl 解析后的记录数组
 * @returns {{ high: Stats, medium: Stats, low: Stats, overall: Stats }}
 */
export function calibrateByConfidence(records) {
  const valid = records.filter(r => r.score != null && !r.error);
  const bins = { high: [], medium: [], low: [], unparsed: [] };

  for (const r of valid) {
    const conf = parseConfidence(r.rawResponse);
    if (conf && bins[conf]) bins[conf].push(r);
    else bins.unparsed.push(r);
  }

  function stats(arr) {
    if (arr.length === 0) return { n: 0, weightedScore: null, accuracy: null };
    const sum = arr.reduce((s, r) => s + r.score, 0);
    const correct = arr.filter(r => r.score >= 0.5).length;
    const perfect = arr.filter(r => r.score === 1.0).length;
    // Signal distribution
    const sigDist = { strong_bull: 0, bull: 0, neutral: 0, bear: 0, strong_bear: 0 };
    for (const r of arr) {
      const s = r.predictedSignal || r.signal || '?';
      if (sigDist[s] !== undefined) sigDist[s]++;
    }
    return {
      n: arr.length,
      weightedScore: +(sum / arr.length).toFixed(4),
      accuracy: +(correct / arr.length).toFixed(4),
      perfectRate: +(perfect / arr.length).toFixed(4),
      signalDistribution: sigDist,
    };
  }

  return {
    high: stats(bins.high),
    medium: stats(bins.medium),
    low: stats(bins.low),
    unparsed: stats(bins.unparsed),
    overall: stats(valid),
  };
}

/**
 * 格式化校准曲线（markdown表格）
 */
export function formatCalibrationCurve(result) {
  const { high, medium, low, unparsed, overall } = result;
  const lines = [
    '## ASC 校准曲线',
    '',
    '| confidence | n | weightedScore | accuracy | perfect% | strong% | neutral% |',
    '|------------|---|--------------|----------|----------|---------|----------|',
  ];
  for (const [label, s] of [['high', high], ['medium', medium], ['low', low], ['unparsed', unparsed]]) {
    const strongPct = ((s.signalDistribution?.strong_bull || 0) + (s.signalDistribution?.strong_bear || 0)) / Math.max(1, s.n) * 100;
    const neutralPct = (s.signalDistribution?.neutral || 0) / Math.max(1, s.n) * 100;
    lines.push(`| ${label} | ${s.n} | ${s.weightedScore ?? '-'} | ${s.accuracy ? (s.accuracy*100).toFixed(1)+'%' : '-'} | ${s.perfectRate ? (s.perfectRate*100).toFixed(1)+'%' : '-'} | ${strongPct.toFixed(1)}% | ${neutralPct.toFixed(1)}% |`);
  }
  lines.push(`| **overall** | ${overall.n} | ${overall.weightedScore ?? '-'} | ${overall.accuracy ? (overall.accuracy*100).toFixed(1)+'%' : '-'} | ${overall.perfectRate ? (overall.perfectRate*100).toFixed(1)+'%' : '-'} | — | — |`);
  return lines.join('\n');
}

/**
 * 获取 high-conf 子集 score
 */
export function getHighConfScore(result) {
  return result.high.weightedScore;
}
