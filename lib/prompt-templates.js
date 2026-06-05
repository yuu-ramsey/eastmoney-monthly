// Analysis dimension prompt templates (single mode only)
// Old 4 styles (technical/chanlun/value/comprehensive) kept in build-prompt.js, debate mode only
//
// Each template embeds 3 hard constraints:
//   1. All judgments must include specific numbers; vague terms like "relatively high" or "obvious" are prohibited
//   2. Must output counter-viewpoints (2 bullish + 2 bearish)
//   3. Trading suggestions give price ranges only, not single points; include disclaimer at end

const HARD_CONSTRAINTS = (periodLabel, dataWindow, includeClosingBarRule) => `
## Hard Constraints (ALL must be satisfied in this analysis, without exception)

1. **Numerical Evidence**: Every technical judgment must be accompanied by specific numbers (e.g., "MA20=35.03, current price is 8.5% above MA20", "8 out of last 12 ${periodLabel} bars closed bullish"). Qualitative terms like "relatively high," "obvious," "moderately strong," "moderately weak" without numerical support are prohibited.

2. **Counter-Viewpoints**: Fully output 2 bullish arguments + 2 bearish arguments, each supported by corresponding numerical evidence. Bullish and bearish weight must be balanced — do not expand one side while dismissing the other in one sentence.

3. **Trading Suggestions**: Only give price ranges, never a single precise price point. Must end with: "**Disclaimer: This analysis is for research and educational purposes only and does not constitute investment advice. Markets contain unpredictable factors; all conclusions should be independently evaluated against personal risk tolerance.**"

4. **Data Window Annotation**: The K-line data for this analysis contains ${dataWindow.count} ${periodLabel} bars, covering the period from ${dataWindow.startDate} to ${dataWindow.endDate}. All phrasing such as "all-time high/low," "historical first," or "unprecedented" must be explicitly scoped within this window. Use precise language like "the highest/lowest value within these ${dataWindow.count} ${periodLabel} bars." Phrasing that implies unlimited historical scope, such as "ever in history," "never before," or "absolute peak," is prohibited.

5. **No Out-of-Window Data (violation invalidates the analysis)**: Only reference dates and values explicitly appearing in the K-line table above.
   - Before mentioning any specific year/month/value, self-check: "Does this data appear within the ${dataWindow.count} rows of the table?"
   - Years not in the table are absolutely prohibited — especially years earlier than ${dataWindow.startDate} and years later than ${dataWindow.endDate}. The table covers the entire available time range for this analysis; there is no "history beyond the table."
   - Any historical information about this stock seen during training must be ignored; only use the data provided here.
   - If an argument requires data outside the window, rewrite as: "This argument requires earlier data, which is not available in the current data window (${dataWindow.startDate} to ${dataWindow.endDate}); deferred for now."${includeClosingBarRule ? `

6. **Distinguishing Closed K-line Bars**: The last bar in the K-line table may not yet be closed. Determination method: if the last bar's date is not later than today and is more than one full period away from today, it is considered closed; otherwise, it is considered "current ${periodLabel} in-progress data."
   - Primary analysis should be based on "the last closed K-line bar" (i.e., the second-to-last bar if the current period is in-progress, or the last bar if it is already closed).
   - In-progress K-line data may only serve as "real-time reference as of ${dataWindow.endDate}" and cannot be the primary basis for pattern judgments or breakout/breakdown signals.
   - Clearly distinguish in trading suggestions: "confirmed signals" based on closed data, "real-time observation signals" based on in-progress data.` : ''}
7. **Structured Output**: After the analysis body, append a JSON code block (wrapped in \`\`\`json markers) in the following format. All fields are required:

\`\`\`json
{
  "score": integer 0-100, composite score — higher = more bullish, lower = more bearish, 50 = neutral,
  "signal": "strong_bull" | "bull" | "neutral" | "bear" | "strong_bear",
  "confidence": "high" | "medium" | "low",
  "key_levels": {
    "support": [level1, level2],
    "resistance": [level1, level2],
    "stop_loss": stopLossLevel
  },
  "trend": "uptrend" | "downtrend" | "sideways" | "reversing",
  "position_percentile": number 0-100, current price percentile within data window,
  "one_line_summary": "core conclusion within 20 words"
}
\`\`\`

Notes:
- score calculation: strong technicals + low valuation + good trend -> close to 100; opposite -> close to 0
- confidence reflects the LLM's own certainty about its judgment
- one_line_summary must not exceed 20 words
- JSON must be parseable by JSON.parse, no comments
- Field names strictly use lowercase_underscore style

8. **No Self-Computed Technical Indicators**: Mathematical calculations (MA/MACD/RSI/KDJ/BOLL/turnover rate averages/percentiles) have already been completed by the program and filled into the "Pre-Computed Technical Indicators" table below. Your task is to provide comprehensive interpretation based on these facts — do not recalculate. If the numbers you cite differ from the table, it is considered an error.
9. **Signal Trigger Guidance**: The program has identified and listed structured buy/sell signals (see the "Structured Signal Identification" section below). If this section includes "signal guidance," follow its instructions for selecting the signal label:
   - "Sufficient evidence, signal field must be strong_bull" -> you MUST assign strong_bull, downgrading to bull or neutral is prohibited
   - "Sufficient evidence, signal field must be strong_bear" -> you MUST assign strong_bear
   - "Signal field should be at least bull" -> you must assign at least bull, cannot downgrade to neutral
   - strong_bull/strong_bear are not taboo words. Failure to use them when warranted = negligence
10. **Signal Consistency**: The JSON signal field must be consistent with the text conclusion. If your text conclusion states "moderately bullish," "bullish outlook," or "uptrend," the signal cannot be bear or neutral. And vice versa.

11. **Multi-Period Resonance Reversal Constraint** (based on HS300 2018-2025 backtest: three-period full alignment is a trend-exhaustion signal, not trend confirmation):
   - strong_bull resonance (all three periods bullish) = trend-exhaustion warning -> prohibited from assigning strong_bull, at most bull; lean toward neutral (historical: after full-bull alignment, 6-month alpha was 4.31%, underperforming the 7.67% after full-bear alignment)
   - strong_bear resonance (all three periods bearish) = mean-reversion opportunity -> may assign bull or strong_bull (if technicals also support it); do not assign bear or strong_bear
   - partial resonance (2/3 aligned): directional reference but constraint weakened
   - divergent (no resonance): freely judge based on technicals
   - Violating this constraint (assigning strong_bull during strong_bull resonance) = negligence. History confirms that three-period full-bull alignment is the least reliable bullish signal.

12. **Sector Relative Strength Mandatory Reasoning**: If the "HS300 Intra-Sector Alpha" section exists above, you MUST rigorously follow this reasoning process:

    **Alpha Level Classification** (must explicitly cite):
    - alpha > +5pp     -> "Sector-leading strength"
    - -5pp <= alpha <= +5pp -> "Sector-neutral"
    - alpha < -5pp     -> "Sector-lagging weakness"

    **Reasoning Constraints**:
    1. Must explicitly cite the alpha level in the analysis body (verbatim).
    2. The signal direction must be broadly consistent with the alpha level; if opposite, an explicit rebuttal reason must be given in the comprehensive conclusion section.
    3. **strong_bull must be supported by alpha level "Sector-leading strength"** — when alpha is not sector-leading, assigning strong_bull is prohibited.
    4. **strong_bear must be supported by alpha level "Sector-lagging weakness"** — when alpha is not sector-lagging, assigning strong_bear is prohibited.
    5. When alpha level is "Sector-neutral," at most bull or bear may be assigned; strong is prohibited.
    6. Sector alpha is based on HS300 internal benchmark, not the Shenwan all-market index.

13. **Confidence Field Honesty Rule**: The JSON confidence field must truthfully reflect your judgment certainty:
    - When technical indicators contradict each other and signal direction is unclear -> must set confidence="low", signal should lean neutral.
    - When multiple signals align but strength is moderate -> confidence="medium".
    - Only when multiple independent chains of evidence converge on the same direction -> confidence="high".
    - Do not default to confidence="medium" across all outputs.

14. **LSTM Quantitative Signal Constraint** (if "LSTM Daily Prediction Signal" section exists above):
    This signal comes from a daily LSTM model (Test IC=0.10, empirically validated). Reasoning rules:
    - LSTM signal > +0.5 -> quantitative bullish (if your signal contradicts this, you must provide technical rebuttal reasoning).
    - -0.5 <= signal <= +0.5 -> quantitative neutral (LSTM has no clear direction; freely judge based on technicals).
    - signal < -0.5 -> quantitative bearish (if your signal contradicts this, you must provide technical rebuttal reasoning).
    Note: The LSTM signal is not binding on your technical analysis — you may reasonably refute it, but you cannot ignore it.
`;


// [DEPRECATED] Reversal factor signal — flipped negative -19.2% on 24tp pool, withdrawn from runtime. Empty function kept to prevent import errors.
export function buildReversalSignalBlock(_signalData) {
  return '';
}

// Kronos monthly prediction signal — the only direction signal passing 24tp gating (docs/p3-kronos-confirm.md, docs/p3-llm-gate-24tp.md)
export function buildKronosSignalBlock(signalData) {
  if (!signalData || signalData.prediction_6m_pct == null) return '';
  const pct = signalData.prediction_6m_pct;
  const dir = signalData.direction || (pct > 0 ? 'bullish' : pct < 0 ? 'bearish' : 'neutral');
  const sign = pct > 0 ? '+' : '';
  return `
## Kronos Monthly Prediction Signal (the only direction signal passing 24tp hold-out gating)

- Predicted 6-month direction: **${dir}** (predicted change ${sign}${pct.toFixed(1)}%)
- Validation data: unbiased low-position pool, 24 time points, test spread +9.7% CI[+5.1,+15.0], decay=0.80
- LLM technical analysis 24tp direction signal +1.6% CI[-3.4,+6.6] includes 0 (not significant) -> Kronos is the only reliable direction signal

**Usage Guidance**: Kronos is the only direction signal that passed 24tp hold-out validation. LLM technical analysis serves as supplementary interpretation. If Kronos and technicals diverge, prioritize the Kronos direction and use technicals to explain possible reasons.
`;
}

// HS300 Intra-Sector Alpha paragraph formatting
// LSTM daily prediction signal block (with MC Dropout uncertainty)
export function buildLstmSignalBlock(signalData) {
  if (!signalData || signalData.lstm_signal == null) return '';

  // new format with uncertainty data
  if (signalData.uncertainty_level) {
    const sig = signalData.lstm_signal;
    const sigRaw = signalData.lstm_signal_raw != null ? signalData.lstm_signal_raw : sig;
    const direction = sig > 0.01 ? 'Bullish' : sig < -0.01 ? 'Bearish' : 'Neutral';
    const sign = sigRaw > 0 ? '+' : '';
    const confPct = signalData.overall_confidence != null
      ? (signalData.overall_confidence * 100).toFixed(0) + '%'
      : '?';

    return `
## LSTM Quantitative Prediction Signal (with MC Dropout Uncertainty)

| Metric | Value |
|------|------|
| Raw prediction y3 | ${sign}${sigRaw.toFixed(4)} |
| Uncertainty sigma | ${signalData.y3_std.toFixed(4)} |
| Confidence-penalized y3 | ${sig >= 0 ? '+' : ''}${sig.toFixed(4)} |
| Prediction confidence | ${confPct} |
| Uncertainty level | ${signalData.uncertainty_level === 'low' ? 'Low' : signalData.uncertainty_level === 'medium' ? 'Medium' : 'High'} uncertainty |
| Forward prediction y6 | ${signalData.y6_mean != null ? (signalData.y6_mean >= 0 ? '+' : '') + signalData.y6_mean.toFixed(4) + ' +/- ' + signalData.y6_std.toFixed(4) : 'N/A'} |
| MC samples | ${signalData.mc_samples || 50} |

${signalData.uncertainty_desc}

**Usage Guidance**:
${signalData.uncertainty_level === 'low'
    ? '- Model signal has relatively high reliability and can serve as primary quantitative reference for technical analysis. If technicals align, upgrade confidence; if contrary, provide explicit technical rebuttal.'
    : signalData.uncertainty_level === 'medium'
    ? '- Model signal shows divergence; use only as supplementary reference. Your technical analysis judgment should carry more weight than the quantitative signal. If technicals contradict, follow technicals.'
    : '- Model signal shows significant divergence and is unreliable. Use your technical analysis as the sole basis for judgment; this signal is recorded for reference only.'}
- Do not directly copy this signal as your conclusion; you must combine it with independent technical judgment.
`;
  }

  // legacy format: scalar signal only
  const sig = signalData.lstm_signal;
  const strength = sig > 0.5 ? 'Bullish' : sig < -0.5 ? 'Bearish' : 'Neutral';
  const sign = sig > 0 ? '+' : '';
  return `
## LSTM Daily Prediction Signal (Quantitative Model)

| Metric | Value |
|------|------|
| LSTM signal | ${sign}${sig.toFixed(4)} |
| Direction | **${strength}** |

This signal comes from a quantitative model without uncertainty estimation. Reference the signal direction in your analysis, but prioritize technical judgment.
`;
}

export function buildSectorAlphaBlock(alphaData) {
  if (!alphaData || alphaData.hs300_sector_alpha == null) return '';

  const val = alphaData.hs300_sector_alpha;
  const alphaDisplay = val > 0 ? `+${val.toFixed(2)}pp` : `${val.toFixed(2)}pp`;
  const level = val > 5 ? 'Sector-leading strength' : val < -5 ? 'Sector-lagging weakness' : 'Sector-neutral';
  const sectorReturn = alphaData.sector_return;
  const sectorDisplay = sectorReturn > 0 ? `+${sectorReturn.toFixed(2)}%` : `${sectorReturn.toFixed(2)}%`;

  return `
## HS300 Intra-Sector Alpha (12-month lookback)

| Metric | Value |
|------|------|
| Stock alpha | ${alphaDisplay} |
| Level | **${level}** |
| Sector rank | ${alphaData.hs300_sector_rank || '?'}/${alphaData.hs300_sector_total} (top ${alphaData.hs300_sector_percentile || '?'}%) |
| Sector | ${alphaData.sector_name} (${alphaData.sector_code}) |
| Sector benchmark return | ${sectorDisplay} |

(HS300 internal benchmark, not Shenwan all-market)
`;
}

// Pre-computed indicator table formatting
export function buildIndicatorTable(kwi, indicatorsObj) {
  if (!indicatorsObj || !indicatorsObj.ma5 || indicatorsObj.ma5.length === 0) return '';
  const n = Math.min(indicatorsObj.ma5.length, 5);
  const sliceKwi = kwi.slice(-n);
  const heads = ['Date','Close','MA5','MA20','MA60','RSI14','MACD_DIF','K','D','J','BOLL_U','BOLL_M','BOLL_L'];
  const lines = ['\n## Pre-Computed Technical Indicators (program-calculated, cite directly)'];
  lines.push('| ' + heads.join(' | ') + ' |');
  lines.push('|' + heads.map(() => '---').join('|') + '|');
  for (let i = 0; i < n; i++) {
    const k = sliceKwi[i];
    if (!k) continue;
    const row = [
      k.date || '?',
      (k.close || 0).toFixed(2),
      indicatorsObj.ma5[i] != null ? indicatorsObj.ma5[i].toFixed(2) : '-',
      indicatorsObj.ma20[i] != null ? indicatorsObj.ma20[i].toFixed(2) : '-',
      indicatorsObj.ma60[i] != null ? indicatorsObj.ma60[i].toFixed(2) : '-',
      indicatorsObj.rsi14[i] != null ? indicatorsObj.rsi14[i].toFixed(1) : '-',
      indicatorsObj.macd_dif[i] != null ? indicatorsObj.macd_dif[i].toFixed(2) : '-',
      indicatorsObj.kdj_k[i] != null ? indicatorsObj.kdj_k[i].toFixed(2) : '-',
      indicatorsObj.kdj_d[i] != null ? indicatorsObj.kdj_d[i].toFixed(2) : '-',
      indicatorsObj.kdj_j[i] != null ? indicatorsObj.kdj_j[i].toFixed(2) : '-',
      indicatorsObj.boll_upper[i] != null ? indicatorsObj.boll_upper[i].toFixed(2) : '-',
      indicatorsObj.boll_mid[i] != null ? indicatorsObj.boll_mid[i].toFixed(2) : '-',
      indicatorsObj.boll_lower[i] != null ? indicatorsObj.boll_lower[i].toFixed(2) : '-',
    ];
    lines.push('| ' + row.join(' | ') + ' |');
  }
  return lines.join('\n') + '\n';
}

// validation_warning injection
export function buildValidationWarningText(validation) {
  if (!validation || validation.severity === 'ok') return '';
  const lines = ['\n Data health check found issues:'];
  for (const issue of validation.issues.slice(0, 5)) {
    lines.push(`- [${issue.severity}] ${issue.message}`);
  }
  lines.push('Please consider the impact of the above data anomalies on your judgment.\n');
  return lines.join('\n');
}
// Position percentile calculation requirement (shared by technical/trend/sentiment templates)
const PERCENTILE_REQUIREMENT = (periodLabel) => `
**Position Percentile Calculation**: You must calculate the current price's historical position percentile.
Formula: percentile = (current close - lowest price in last N ${periodLabel} bars) / (highest price in last N ${periodLabel} bars - lowest price in last N ${periodLabel} bars) x 100%
N = the actual number of K-line bars in the data. Result format: "Current price is at the XX.X percentile of the last {N} ${periodLabel} bars (i.e., higher than XX.X% of historical price ranges)."
`;

const TEMPLATES = {
  technical: {
    label: 'Technical',
    build(periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
      return `## Analysis Task: ${periodLabel} Technical Analysis
${HARD_CONSTRAINTS(periodLabel, dataWindow, includeClosingBarRule)}
${PERCENTILE_REQUIREMENT(periodLabel)}

Please analyze the following ${periodLabel} technical indicators:

1. **Moving Average System**: Current values and alignment relationship (bullish/bearish/crossover) of MA5/MA20/MA60. For each MA, provide the specific deviation percentage from the current price.

2. **K-line Patterns**: The real body length, shadow positions of the last 3 ${periodLabel} bars, and whether any key reversal patterns have appeared (e.g., hammer, engulfing, doji, etc.). Each pattern judgment must cite specific dates and price levels.

3. **Support and Resistance**: Identify 3-5 key price levels from the K-line data and annotate their source (e.g., "MA60=XX.XX", "${periodLabel} low of YYYY-MM XX.XX", "central hub upper bound XX.XX"), providing an effectiveness assessment for each level.

4. **MACD State**: Current DIF/DEA values, histogram trend, and whether any divergence signals are present. Must provide specific numerical values.

5. **Comprehensive Judgment**:
   - Direction judgment: choose one of [Moderately Bullish / Moderately Bearish / Neutral Range-bound], with numerical evidence
   - Key verification levels: 1-2 specific price levels, "if breaks above X, confirms bullish; if breaks below Y, shifts to bearish"
   - Counter-viewpoints: 2 bullish + 2 bearish, balanced in weight

Output in Markdown format.`;
    },
  },

  trend: {
    label: 'Trend',
    build(periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
      return `## Analysis Task: ${periodLabel} Medium-to-Long-Term Trend Assessment
${HARD_CONSTRAINTS(periodLabel, dataWindow, includeClosingBarRule)}
${PERCENTILE_REQUIREMENT(periodLabel)}

Please analyze the trend state at the ${periodLabel} level:

1. **Trend Direction Classification**: Choose one of [Clear Uptrend / Moderate Uptrend / Sideways Consolidation / Moderate Downtrend / Clear Downtrend / Trend Reversal]. Must cite MA alignment and price structure as evidence.

2. **K-line Statistics**: Calculate the bullish/bearish close ratio across the last N ${periodLabel} bars, the longest consecutive bullish/bearish streak, and the average gain/loss. Numbers must be derived by counting each bar individually.

3. **Momentum Assessment**: Based on MACD histogram changes and gain/loss volatility, determine whether current momentum is accelerating/decelerating/exhausting. Provide specific numerical comparisons (e.g., average histogram over first 3 ${unitLabel} vs. last 3 ${unitLabel}).

4. **Trend Maturity**: How many ${periodLabel} bars has the current trend been running (counted from the most recent clear inflection point)? Are there trend-exhaustion signals (e.g., consecutive volume contraction, narrowing price range, MA convergence)?

5. **Comprehensive Judgment**:
   - Trend phase: choose one of [Early Trend / Mid-Trend / Late Trend / Trend Unclear]
   - Direction judgment: choose one of [Moderately Bullish / Moderately Bearish / Neutral]
   - Trend key level: at what price point would the trend change if broken?
   - Counter-viewpoints: 2 bullish + 2 bearish, balanced in weight

Output in Markdown format.`;
    },
  },

  valuation: {
    label: 'Valuation',
    build(periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
      return `## Analysis Task: ${periodLabel} Price History — Long-Term Value Perspective
${HARD_CONSTRAINTS(periodLabel, dataWindow, includeClosingBarRule)}

Note: This dimension makes relative value judgments based on historical price percentiles and long-term MA positions. When analyzing valuation, you should actively call the get_financials tool to obtain current PE/PB/market cap/sector and other fundamental indicators, and use them to assess whether valuation is within a reasonable range.

Please analyze from a long-term value perspective:

1. **Historical Position**: Calculate the current price's position percentile within the last N ${periodLabel} bars of data (same formula as above), explicitly providing the percentile number.

2. **Long-Term Trend Segmentation**: Divide the last N ${periodLabel} bars into "rally phase / decline phase / consolidation phase," marking start/end dates and gain/loss for each phase. Determine which phase the current price is in.

3. **Extreme Position Signals**: Check whether the current price is near historical extremes (within +/-10% of highest/lowest, near long-term MA support levels, etc.), providing specific deviation percentages.

4. **Long-Term MA Alignment**: The evolution of MA5/MA20/MA60 long-term alignment (dates and directions of the last 3 crossovers), assessing whether the MA system is healthy.

5. **Valuation Stage Classification**: Synthesizing price historical percentile, MA positions, and PE/PB fundamentals obtained via tools, choose one of [Bottom Recovery / Mid-Value Restoration / Approaching Fair Value / Overvalued Zone]. All judgments must cite specific prices and percentile numbers.

6. **Comprehensive Judgment**:
   - Time horizon: explicitly state the time scale of this judgment (quarterly / annual)
   - Direction judgment: choose one of [Moderately Bullish / Moderately Bearish / Neutral]
   - Counter-viewpoints: 2 bullish + 2 bearish, balanced in weight

Output in Markdown format.`;
    },
  },

  sentiment: {
    label: 'Sentiment',
    build(periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
      return `## Analysis Task: ${periodLabel} Market Sentiment and Volume-Price Analysis
${HARD_CONSTRAINTS(periodLabel, dataWindow, includeClosingBarRule)}
${PERCENTILE_REQUIREMENT(periodLabel)}

Please analyze market sentiment and volume-price relationships at the ${periodLabel} level:

1. **Volume-Price Coordination Analysis**: Check volume-price relationship pair by pair for the last 5 ${periodLabel} bars (surging volume + rising price / shrinking volume + rising price / surging volume + falling price / shrinking volume + falling price), and calculate the match ratio. Must provide the volume-price state for each ${periodLabel} bar.

2. **Turnover Rate Analysis** (if turnover data is provided): Mean, standard deviation, and current percentile of turnover rate across the last N ${periodLabel} bars. Assess whether market participation is rising/declining/stable. If turnover rate data is missing, skip this item and annotate at the end: "Turnover rate data missing; sentiment analysis downgraded."

3. **Amplitude Analysis**: The trend of amplitude changes (expanding/contracting) across the last N ${periodLabel} bars, and the current amplitude's historical percentile. High amplitude indicates strong divergence; low amplitude indicates directional energy building.

4. **Sentiment Deviation**: The percentage deviation of current price from MA20 and from MA60. Larger deviation = more extreme sentiment. Provide historical comparison (current deviation's rank in history).

5. **Trap Signal Detection**: Check for signals such as "price rising with shrinking volume" (bull trap), "price falling with shrinking volume" (bear exhaustion), "surging volume with stagnant price" (distribution suspicion).

6. **Comprehensive Judgment**:
   - Sentiment classification: choose one of [Optimistic / Mildly Optimistic / Neutral / Mildly Pessimistic / Pessimistic]
   - Direction judgment: choose one of [Moderately Bullish / Moderately Bearish / Neutral Range-bound]
   - Counter-viewpoints: 2 bullish + 2 bearish, balanced in weight

Output in Markdown format.`;
    },
  },
};

// Multi-time-window normalized return block
export function buildNormalizedReturnBlock(nrData) {
  if (!nrData || !nrData.windows || nrData.windows.length === 0) return '';

  const rows = nrData.windows.map((w) => {
    const retStr = (w.retPct >= 0 ? '+' : '') + w.retPct.toFixed(2) + '%';
    const volStr = w.volPct.toFixed(2) + '%';
    const ratioStr = (w.ratio >= 0 ? '+' : '') + w.ratio.toFixed(3);
    const pctStr = 'P' + w.percentile.toFixed(0);
    return `| ${w.label} | ${retStr} | ${volStr} | ${ratioStr} | ${pctStr} |`;
  }).join('\n');

  const trendLabel = nrData.trend === 'improving'
    ? 'Improving (risk-adjusted return is getting better)'
    : nrData.trend === 'deteriorating'
    ? 'Deteriorating (risk-adjusted return is worsening)'
    : 'Stable';

  return `
## Multi-Time-Window Normalized Returns (program-calculated)

Efficiency Ratio = window return / annualized volatility. Directly comparable across windows — higher positive values = better risk-adjusted return.

| Time Window | Return | Ann. Vol | Efficiency | Hist. Pct |
|----------|--------|----------|--------|----------|
${rows}

**Best Window**: ${nrData.bestWindow}, Composite Trend: **${trendLabel}**

**Usage Guidance**:
- Efficiency Ratio is a cross-period comparable "risk-adjusted return" metric; it is not a directional judgment.
- Windows with positive efficiency and percentile > P70 -> favorable risk/reward at that time scale, can serve as position time-horizon anchor reference.
- "Improving" efficiency trend -> recent risk-adjusted return is improving; overall confidence can be moderately upgraded.
- "Deteriorating" efficiency trend -> recent risk-adjusted return is worsening; prudence should be reflected in the comprehensive conclusion.
- This data is for decision reference only; it does not directly dictate buy/sell direction.
`;
}

// Available tool instructions (only appended to prompt when Anthropic provider has tool_use enabled)
export function buildToolInstructions(secid) {
  return `
## Available Tools (Anthropic provider only)

You can use the following tools to supplement analysis data. When you need information not present in the table, proactively call tools rather than filling in from memory.

### get_financials
Fetch current PE/PB/market cap/sector and other financial indicators.
- Parameter: \`secid\` = \`"${secid}"\`
- The Valuation template **encourages calling** this tool to supplement fundamental data.

### get_money_flow
Fetch major capital flow direction over the last N months.
- Parameters: \`secid\` = \`"${secid}"\`, \`limit\` = number of months (default 12)
- The Sentiment template **encourages calling** this tool to supplement the capital flow dimension.

Usage principles:
1. Only call when the analysis requires this data and it is not in the K-line table.
2. Technical and Trend templates call on demand; avoid excessive calls.
3. After calling, incorporate the tool's real returned data into the analysis, replacing memory-based completion.
4. This stock's secid is \`"${secid}"\` — use this value directly when calling tools.`;
}

// default template key
export const DEFAULT_TEMPLATE = 'technical';

// template label mapping (for popup use)
export const TEMPLATE_LABELS = {};
for (const [key, tpl] of Object.entries(TEMPLATES)) {
  TEMPLATE_LABELS[key] = tpl.label;
}

/**
 * Get the prompt task text for a template, with periodLabel/unitLabel directly interpolated via template strings.
 * @param {string} templateKey - 'technical' | 'trend' | 'valuation' | 'sentiment'
 * @param {string} periodLabel - Monthly/Weekly/Daily
 * @param {string} unitLabel - Month/Week/Day
 * @returns {string} prompt task text
 */
export function buildTemplatePrompt(templateKey, periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
  const tpl = TEMPLATES[templateKey];
  if (!tpl) {
    return buildTemplatePrompt(DEFAULT_TEMPLATE, periodLabel, unitLabel, dataWindow, includeClosingBarRule);
  }
  return tpl.build(periodLabel, unitLabel, dataWindow, includeClosingBarRule);
}
