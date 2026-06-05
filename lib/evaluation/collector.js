// Evaluation data collection (pure numeric computation, no LLM calls, zero cost)
//
// extractJudgment() — extract directional judgment from analysis text
// evaluateOneAnalysis() — evaluate a single entry, return verdict
// evaluateBatch() — batch evaluation, incremental write to storage

/**
 * 从分析文本提取方向判断
 * @param {string} analysisText
 * @returns {'bull'|'bear'|'neutral'|null}
 */
export function extractJudgment(text) {
  if (!text || typeof text !== 'string') return null;

  // Markdown bold: **方向判断**：**中性震荡（略偏空）**
  const md = text.match(/\*\*方向判断\*\*[：:]\s*\*\*([^*]+)\*\*/);
  if (md) return parseDirection(md[1]);

  // 方向判断:【偏多】 format
  const explicit = text.match(/方向判断[：:]\s*【([^】]+)】/);
  if (explicit) return parseDirection(explicit[1]);

  const explicit2 = text.match(/方向判断[：:]\s*(偏多|偏空|中性震荡|中性)/);
  if (explicit2) return parseDirection(explicit2[1]);

  // Fallback: paragraph after "综合方向判断"
  const section = text.match(/综合方向判断[^：:]*[：:]\s*\n?\s*(偏多|偏空|中性震荡|中性)/);
  if (section) return parseDirection(section[1]);

  // Direction in comprehensive conclusion
  const conclusion = text.match(/综合结论[\s\S]{0,200}?(偏多|偏空|中性震荡|中性)/);
  if (conclusion) return parseDirection(conclusion[1]);

  return null;
}

function parseDirection(raw) {
  const s = raw.trim();
  if (s.includes('偏多')) return 'bull';
  if (s.includes('偏空')) return 'bear';
  if (s.includes('中性')) return 'neutral';
  return null;
}

/**
 * 单条评估
 * @param {object} historyEntry — { code, analysis, timestamp, ... }
 * @param {Function} fetchKlines — (market, code) => klines[]
 * @param {Function} fetchIndexKlines — () => klines[]
 * @returns {object} { historyId, code, judgment, verdict, stockReturn, indexReturn, alpha, daysElapsed, timestamp }
 */
export async function evaluateOneAnalysis(entry, fetchKlines, fetchIndexKlines) {
  // Prefer scoreData.signal, fall back to extractJudgment regex
  let judgment = null;
  if (entry.scoreData && entry.scoreData.signal) {
    const s = entry.scoreData.signal;
    if (s === 'strong_bull' || s === 'bull') judgment = 'bull';
    else if (s === 'strong_bear' || s === 'bear') judgment = 'bear';
    else if (s === 'neutral') judgment = 'neutral';
  }
  if (!judgment) {
    judgment = extractJudgment(entry.analysis || '');
  }
  if (!judgment) {
    return {
      historyId: entry.id || entry.timestamp,
      code: entry.code,
      judgment: null,
      verdict: 'no_judgment',
      stockReturn: null, indexReturn: null, alpha: null,
      daysElapsed: null,
      template: entry.template, model: entry.model,
      mode: entry.mode || 'single',
      enableSelfBacktest: entry.enableSelfBacktest, enableThinking: entry.enableThinking,
      timestamp: Date.now(),
    };
  }

  // Fetch latest monthly klines
  let stockKlines = [];
  let indexKlines = [];
  try {
    const parts = (entry.code || '').match(/^(\d{6})$/);
    const market = parts ? (/^6/.test(parts[1]) ? '1' : '0') : '1';
    const code = parts ? parts[1] : entry.code;
    stockKlines = await fetchKlines(market, code) || [];
    indexKlines = await fetchIndexKlines() || [];
  } catch (_) {
    // Skip on fetch failure, don't block overall evaluation
  }

  if (stockKlines.length < 2) {
    return {
      historyId: entry.id || entry.timestamp,
      code: entry.code,
      judgment,
      verdict: 'pending',
      stockReturn: null, indexReturn: null, alpha: null,
      daysElapsed: null, timestamp: Date.now(),
    };
  }

  // Find kline index corresponding to entry.timestamp date
  const entryDate = entry.timestamp
    ? new Date(entry.timestamp).toISOString().slice(0, 7)
    : null;
  let fromIdx = 0;
  if (entryDate) {
    for (let i = 0; i < stockKlines.length; i++) {
      if (String(stockKlines[i].date).slice(0, 7) >= entryDate) { fromIdx = i; break; }
    }
  }

  const toIdx = stockKlines.length - 1;
  const fromClose = stockKlines[fromIdx]?.close;
  const toClose = stockKlines[toIdx]?.close;
  if (!fromClose || !toClose || fromIdx >= toIdx) {
    return {
      historyId: entry.id || entry.timestamp,
      code: entry.code, judgment,
      verdict: 'pending',
      stockReturn: null, indexReturn: null, alpha: null,
      daysElapsed: null, timestamp: Date.now(),
    };
  }

  const stockReturn = ((toClose - fromClose) / fromClose) * 100;
  let indexReturn = null;
  let alpha = null;

  if (indexKlines.length > 0) {
    let idxFrom = 0;
    if (entryDate) {
      for (let i = 0; i < indexKlines.length; i++) {
        if (String(indexKlines[i].date).slice(0, 7) >= entryDate) { idxFrom = i; break; }
      }
    }
    const idxTo = indexKlines.length - 1;
    const idxFromClose = indexKlines[idxFrom]?.close;
    const idxToClose = indexKlines[idxTo]?.close;
    if (idxFromClose && idxToClose && idxFrom < idxTo) {
      indexReturn = ((idxToClose - idxFromClose) / idxFromClose) * 100;
      alpha = stockReturn - indexReturn;
    }
  }

  // Days
  const msPerDay = 86400000;
  const daysElapsed = entry.timestamp
    ? Math.round((Date.now() - entry.timestamp) / msPerDay)
    : null;

  // Verdict
  const verdict = computeVerdict(judgment, stockReturn, alpha, daysElapsed);

  return {
    historyId: entry.id || entry.timestamp,
    code: entry.code,
    judgment,
    verdict,
    stockReturn: +stockReturn.toFixed(2),
    indexReturn: indexReturn != null ? +indexReturn.toFixed(2) : null,
    alpha: alpha != null ? +alpha.toFixed(2) : null,
    daysElapsed,
    // Inherit history dimension fields for computeStats grouping
    template: entry.template,
    model: entry.model,
    mode: entry.mode || 'single',
    enableSelfBacktest: entry.enableSelfBacktest,
    enableThinking: entry.enableThinking,
    timestamp: Date.now(),
  };
}

function computeVerdict(judgment, stockReturn, alpha, daysElapsed) {
  if (daysElapsed != null && daysElapsed < 30) return 'pending';

  const a = alpha || 0;

  if (judgment === 'bull') {
    if (stockReturn > 0 && a > 5) return 'strong_correct';
    if (stockReturn > 0) return 'correct';
    return 'wrong';
  }
  if (judgment === 'bear') {
    if (stockReturn < 0 && a < -5) return 'strong_correct';
    if (stockReturn < 0) return 'correct';
    return 'wrong';
  }
  if (judgment === 'neutral') {
    if (Math.abs(a) < 5) return 'correct';
    if (Math.abs(a) >= 10) return 'wrong';
    return 'neutral'; // uncertain
  }
  return 'unknown';
}

/**
 * 批量评估，增量写入
 * @param {Array} historyEntries
 * @param {object} fetchers — { fetchKlines, fetchIndexKlines }
 * @param {object} storage — { getEvaluations, saveEvaluation }
 * @returns {Array} 新增的 evaluation 记录
 */
export async function evaluateBatch(historyEntries, fetchers, storage) {
  const existing = await storage.getEvaluations();
  const existingIds = new Set(existing.map((e) => e.historyId));

  // Filter: skip already-evaluated + pending (wait for next run)
  const toEvaluate = historyEntries.filter((e) => {
    const hid = e.id || e.timestamp;
    return !existingIds.has(hid);
  });

  const results = [];
  for (const entry of toEvaluate) {
    const record = await evaluateOneAnalysis(entry, fetchers.fetchKlines, fetchers.fetchIndexKlines);
    if (record.verdict !== 'pending') {
      await storage.saveEvaluation(record);
      results.push(record);
    }
  }

  return results;
}
