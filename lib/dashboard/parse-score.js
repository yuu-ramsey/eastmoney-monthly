// Score dashboard — JSON parsing + validation + weighted calculation
// Extract structured score data from end of LLM analysis text

const VALID_SIGNALS = ['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear'];
const VALID_CONFIDENCES = ['high', 'medium', 'low'];
const VALID_TRENDS = ['uptrend', 'downtrend', 'sideways', 'reversing'];

/**
 * Parse the last JSON code block from analysis text
 * @param {string} analysisText
 * @returns {object|null} 解析后的 scoreData，失败返回 null
 */
export function parseScoreBlock(text) {
  if (!text || typeof text !== 'string') return null;

  // 匹配所有 ```json ... ``` 代码块
  const blocks = [];
  const regex = /```json\s*([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    blocks.push(match[1].trim());
  }

  if (blocks.length === 0) return null;

  // 遍历所有 JSON 块，取第一个包含有效 signal 字段的
  //（DeepSeek 有时输出多个 JSON 块，第一个含 score/signal，后面的含 centralZone 等）
  for (const block of blocks) {
    let data;
    try {
      data = JSON.parse(block);
    } catch (_) {
      continue;
    }

    const validation = validateScoreData(data);
    if (validation.valid) {
      return normalizeScoreData(data);
    }
  }

  return null;
}

/**
 * 验证并补全字段
 */
export function validateScoreData(data) {
  const errors = [];
  if (!data || typeof data !== 'object') {
    errors.push('data 不是对象');
    return { valid: false, errors };
  }

  // score
  if (typeof data.score !== 'number' || !Number.isFinite(data.score) || data.score < 0 || data.score > 100) {
    errors.push(`score 无效: ${data.score}`);
  }

  // signal
  if (!VALID_SIGNALS.includes(data.signal)) {
    errors.push(`signal 无效: ${data.signal}`);
  }

  // confidence
  if (!VALID_CONFIDENCES.includes(data.confidence)) {
    errors.push(`confidence 无效: ${data.confidence}`);
  }

  // key_levels
  const kl = data.key_levels;
  if (!kl || typeof kl !== 'object') {
    errors.push('key_levels 缺失');
  } else {
    if (!Array.isArray(kl.support)) errors.push('key_levels.support 不是数组');
    if (!Array.isArray(kl.resistance)) errors.push('key_levels.resistance 不是数组');
    if (typeof kl.stop_loss !== 'number' || !Number.isFinite(kl.stop_loss)) {
      errors.push('key_levels.stop_loss 无效');
    }
  }

  // trend
  if (!VALID_TRENDS.includes(data.trend)) {
    errors.push(`trend 无效: ${data.trend}`);
  }

  // position_percentile
  if (typeof data.position_percentile !== 'number' || !Number.isFinite(data.position_percentile)
    || data.position_percentile < 0 || data.position_percentile > 100) {
    errors.push(`position_percentile 无效: ${data.position_percentile}`);
  }

  // one_line_summary
  if (typeof data.one_line_summary !== 'string' || data.one_line_summary.trim().length === 0) {
    errors.push('one_line_summary 缺失或为空');
  }

  return { valid: errors.length === 0, errors };
}

function normalizeScoreData(data) {
  const kl = data.key_levels || {};
  return {
    score: Math.round(data.score),
    signal: data.signal,
    confidence: data.confidence,
    key_levels: {
      support: Array.isArray(kl.support) ? kl.support.filter((v) => typeof v === 'number') : [],
      resistance: Array.isArray(kl.resistance) ? kl.resistance.filter((v) => typeof v === 'number') : [],
      stop_loss: typeof kl.stop_loss === 'number' ? kl.stop_loss : null,
    },
    trend: data.trend,
    position_percentile: +data.position_percentile.toFixed(1),
    one_line_summary: String(data.one_line_summary || '').slice(0, 40),
  };
}

/**
 * 多模板加权综合分
 * @param {Array} scoreArray — [{ template, scoreData }, ...]
 * @param {object} [weights] — { technical, trend, valuation, sentiment }
 * @returns {number|null}
 */
export function computeWeightedScore(scoreArray, weights = {}) {
  const defaults = { technical: 0.35, trend: 0.25, valuation: 0.20, sentiment: 0.20 };
  const w = { ...defaults, ...weights };

  const valid = scoreArray.filter((s) => s && s.scoreData && typeof s.scoreData.score === 'number');

  // 没有有效数据
  if (valid.length === 0) return null;

  // 有部分缺失 → 重新分配权重
  const availableTemplates = valid.map((s) => s.template);
  const totalWeight = availableTemplates.reduce((sum, t) => sum + (w[t] || 0), 0);

  if (totalWeight === 0) return null;

  let weightedSum = 0;
  for (const s of valid) {
    const tWeight = (w[s.template] || 0) / totalWeight;
    weightedSum += s.scoreData.score * tWeight;
  }

  return +weightedSum.toFixed(1);
}

// 颜色映射（content.js / popup 共用）
export const SIGNAL_COLORS = {
  strong_bull: '#00aa44',
  bull: '#4ec77b',
  neutral: '#888',
  bear: '#e88880',
  strong_bear: '#cc0000',
};

export const SIGNAL_LABELS = {
  strong_bull: '🟢🟢 强看多',
  bull: '🟢 看多',
  neutral: '⚪ 中性',
  bear: '🔴 看空',
  strong_bear: '🔴🔴 强看空',
};

export const CONFIDENCE_LABELS = { high: '置信度：高', medium: '置信度：中', low: '置信度：低' };
export const TREND_LABELS = { uptrend: '上行趋势 ↗', downtrend: '下行趋势 ↘', sideways: '横盘 ↔', reversing: '趋势反转中 ⇄' };
