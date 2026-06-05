// prompt assembly, two entry points:
//   buildPromptByTemplate() -> for single mode (new 4-dimension templates)
//   buildPrompt()          -> for debate mode (legacy 4 styles, kept for compat)

import { buildTemplatePrompt, DEFAULT_TEMPLATE, buildToolInstructions, buildIndicatorTable, buildValidationWarningText, buildSectorAlphaBlock, buildLstmSignalBlock, buildKronosSignalBlock, buildNormalizedReturnBlock } from './prompt-templates.js';
import { calcNormalizedReturns } from './multi-period/normalized-return.js';
import { calculateAll, tailIndicators } from './indicators/calculate.js';
import { checkKlines } from './data-validation/validate-klines.js';
import { generateSignalSummary, formatSignalSummary, buildSignalGuidance } from './signals/summary.js';
// resonance functions dynamically imported to avoid service worker loading Node native modules (better-sqlite3)

const STRUCTURED_OUTPUT_BLOCK = `## Structured Data Output (must strictly comply)

At the end of the analysis report, append a JSON block (wrapped in \`\`\` markers), machine-readable, for internal tool validation only — does not affect the main report content:

\`\`\`json
{
  "period": "monthly|weekly|daily",
  "centralZone": {
    "lower": <lower bound price, to 2 decimal places>,
    "upper": <upper bound price, to 2 decimal places>,
    "exists": <true|false, whether a clear central hub is identified in this period>
  },
  "keySupport": [<3-5 key support levels, from strongest to weakest>],
  "keyResistance": [<3-5 key resistance levels, from weakest to strongest>],
  "trend": "up|down|sideways|reversal_top|reversal_bottom"
}
\`\`\`

Notes:
- This JSON is for tool validation only; do not explain it or add text like "below is the JSON output."
- Price levels must be directly cited from K-line data (MA60, prior high, prior low, etc. with specific values).
- When centralZone.exists = false, lower/upper may be null.
- The multi-period resonance mode does not require this JSON block (it is already a composite output).`;

export { STRUCTURED_OUTPUT_BLOCK };

const DISCLAIMER = '\n\n> This analysis is for research and educational purposes only and does not constitute investment advice. Markets contain unpredictable factors; all conclusions should be independently evaluated against personal risk tolerance.';

export const PERSONAL_DECISION_BLOCK = `\n\n## Personal Decision Perspective (for the holder's personal reference only)

Based on the above analysis, provide a clear directional judgment and action framework:

1. **Current Position Classification**: Explicitly choose one of [Entry Zone / Elevated Zone / Critical Decision Zone / High-Risk Zone / Not Recommended for Participation], with 1-2 sentences of rationale.

2. **Non-Holder Recommendation**: Choose one of [Aggressively Buy / Build Position in Tranches / Wait for Pullback / Observe / Avoid], with rationale.

3. **Existing Holder Recommendation**: Choose one of [Add / Hold / Reduce / Liquidate], with rationale.

4. **Key Stop-Loss Level**: Must provide a specific price to 2 decimal places (e.g., 35.03, not "around 35" or "35-36"). The price must be directly cited from K-line data (MA60, prior low, central hub lower bound, strong support level, etc. with specific values) + response strategy if breached.

5. **Key Add/Reduce Levels**: Must provide specific prices (to 2 decimal places); ranges are not allowed. Each price must be annotated with two things:
   a. Data source for the price (e.g., "MA60=35.03" / "2026-03-20 weekly low 28.10" / "weekly central hub lower bound 30.20")
   b. Action plan when triggered (e.g., "add 30% position," "reduce 20%")
   WRONG example: 35-36 yuan (daily MA60, shallow pullback) -> add base position
   CORRECT example:
       Add Level 1: 35.03 (daily MA60 current value, shallow pullback level) -> add 20% position
       Add Level 2: 28.10 (2026-03 weekly prior low + weekly central hub lower bound, deep retracement level) -> add 30% position
       Reduce Level 1: 42.90 (2025-09 monthly prior high, first resistance) -> reduce 20%
       Reduce Level 2: 45.95 (2026-01 monthly absolute high, second resistance) -> reduce 30%

6. **Holding Time Expectation**: Must strictly match the current analysis period; must not be shorter than one K-line bar's duration:
     - Monthly analysis: choose one of [3-6 months / 6-12 months / 1+ year]
     - Weekly analysis: choose one of [1-3 months / 3-6 months / 6-12 months]
     - Daily analysis: choose one of [1-4 weeks / 1-3 months / 3-6 months]
     - Multi-period resonance analysis: choose one of [3-6 months / 6-12 months] (synthesizing all periods)

7. **One-Sentence Core Logic**: Summarize "why I recommend this" in one sentence.

8. **Relative Valuation Assessment**: Based on event information in the additional context (earnings announcements/research report titles), provide a relative valuation assessment:
   - Prioritize using "earnings announcements/research reports" content from the event feed, citing specific YoY net profit/EPS figures.
   - If earnings information is insufficient, explicitly state "Insufficient fundamental information; judgment based on technicals only."
   - Output format: choose one of [Clearly Undervalued / Reasonably Low / Fair / Reasonably High / Clearly Overvalued / Insufficient Fundamental Information to Judge].
   - Strictly prohibited: applying CAPM, DDM, dividend discount models, or any other academic financial models — for A-share individual stocks, these models produce excessive data noise and false precision. Only make relative judgments based on actual announcements and price levels.

9. **Recommended Position Size**: Choose one of [Stay in Cash / Trial 5-10% / Light 10-25% / Half 25-50% / Heavy 50-75% / Full 75-100%], with rationale:
   - Current position classification (Critical Decision Zone -> trial; Entry Zone -> light to half; High-Risk Zone -> stay in cash)
   - Multi-period resonance state (triple resonance bullish -> upgrade; chaotic signals -> downgrade)
   - Valuation assessment (clearly undervalued -> upgrade; clearly overvalued -> downgrade)
   - Strictly prohibited: applying Kelly Criterion — win rate and risk/reward ratio are both estimates; calculating "precise position sizing" on top of estimates produces false precision.

10. **Entry Strategy**: Derive entry rhythm from items 8 and 9:
    - Choose one of [Single Entry / 2-Tranche Build / 3-Tranche Build / Wait for Pullback then Unified Entry / No Entry].
    - Trigger conditions for each tranche must be explicit (e.g., "Tranche 1: trial 10% at current price 38.87; Tranche 2: add 15% at 35.03; Tranche 3: add 25% at 28.10").
    - Prices must be precise to 2 decimal places and consistent with item 5's add levels.

This section is for the tool owner's personal decision support only and does not constitute investment advice to any third party. Past performance does not guarantee future results. Actual actions should be based on independent judgment and personal risk tolerance.`;

const OUTPUT_WEIGHT_GUIDANCE = `## Output Weight Requirements

Historical data is the foundation of judgment, but the output structure must center on "current position + next key observation points." Specific requirements:

- Descriptions involving historical key points (e.g., multi-year highs/bottoms) must be limited to 2 sentences per point, and must explicitly state how they support the "current judgment" — do not state them in isolation.
- The "Comprehensive Conclusion" section must be placed at the end of the report and should account for 20%-30% of the total word count — it is the section users read first.
- Each intermediate analysis subsection (e.g., MA, MACD, central hub identification, etc.) must state its core conclusion in the opening sentence before expanding on evidence. Do not make the user read the entire subsection to understand the conclusion.
- Strictly prohibited: listing irrelevant historical events "to pad the output." Every historical reference must serve the current judgment.
- **Numerical Evidence Hard Constraint**: Every technical judgment must be accompanied by specific numerical evidence (e.g., "current price 38.87 is 11.0% above MA60=35.03"). Qualitative descriptions such as "on the high side," "on the low side," "strong," or "weak" without numerical support are not allowed. Judgments without numerical support are considered invalid.
- **Counter-Viewpoints Mandatory**: Each analysis subsection must include a "Counter-Viewpoints / Risk Notes" segment listing 1-2 key counter-arguments or risk scenarios where the current judgment could be wrong — do not skip.
- **Position Percentile Calculation**: Must explicitly calculate and provide the current price's historical position percentile. Formula: percentile = (current close - N-period lowest) / (N-period highest - N-period lowest) x 100%. N is the actual length of the provided K-line data. The result must be written as "Current price is at the XX percentile of the last {N} {PERIOD} bars (i.e., higher than XX% of historical price ranges)." Any position descriptions (high/low/mid) in analysis subsections must reference this percentile value.
`;

const TASK_TECHNICAL = OUTPUT_WEIGHT_GUIDANCE + `
Please analyze:
1. Current {PERIOD} position (historical high / low / continuation zone)
2. Moving average system state (MA5/MA20/MA60 bull/bear alignment)
3. MACD indicator state (DIF/DEA position relationship, histogram trend)
4. Volume-price relationship (volume coordination with gain/loss)
5. Key support and resistance levels (provide specific prices)
6. {PERIOD}-level trend assessment

7. Comprehensive Conclusion:
   - Current directional judgment: explicitly choose one of [Moderately Bullish / Moderately Bearish / Neutral Range-bound], with one sentence stating the primary rationale.
   - Signal resonance analysis: list all mutually confirming signals across indicators (e.g., bullish MA alignment + aligned MACD direction + supportive volume-price), and mutually contradictory signals (e.g., bullish MAs but contracting volume).
   - Key observation levels: provide 1-2 specific prices as subsequent verification levels, stating "if breaks above X, confirms bullish; if breaks below Y, shifts to bearish."
   - Do not directly say "recommend buying/selling/holding." The conclusion tone should be "if X occurs, then it validates Y judgment," not "you should do X."
8. Counter-Viewpoints / Risk Notes: List 1-2 key risk scenarios where the current judgment could be wrong (e.g., MA false breakout, MACD flattening failure, volume-price divergence misjudgment). Each risk must include the observation conditions that would trigger it.`;

const CHANLUN_STRICTNESS = `## Chanlun System Strictness Requirements

This analysis must strictly adhere to the original Chanlun framework, avoiding the following common errors:

**Regarding Strokes and Segments:**
- A stroke is confirmed by two opposing fractals before and after. A secondary test (e.g., a second retest near a prior low) is a sub-level fluctuation within the same stroke, NOT the start of a new stroke. When determining a new stroke, a clear opposing fractal must be identified.
- If two adjacent lows (or highs) are very close and there is no clear opposing movement between them, they should be treated as the same bottom (or top) area — do not split into two strokes.

**Regarding Central Hub:**
- A {PERIOD} central hub must be defined by "overlapping sub-level movement segments." Do NOT directly use the current period's highest/lowest values as ZG/ZD.
- ZG = the lowest high among the sub-level downward segments; ZD = the highest low among the sub-level upward segments.
- At least 3 sub-level movement segments overlapping within the range are required to form a central hub. If the current data cannot clearly identify 3 overlapping segments, explicitly state "central hub not yet clearly formed" — do not force one.
- After a central hub is effectively broken, 3 new overlapping segments are required before calling it "new central hub under construction." When there is only a single segment departing, it should be described as "an upward segment leaving the old central hub (no central hub)."

**Regarding the Three Types of Buy/Sell Points:**
- Type 3 buy point requires: "after leaving the central hub, the pullback low does not enter the central hub range."
- "Does not enter" must be clear — the pullback low must be at least 3% above ZG to qualify as an effective Type 3 buy point. If the pullback low is close to ZG (within 1% difference), it is "approaching ZG but not confirmed as fully detached" and should be classified as "pending confirmation" rather than "established."
- Similarly, a Type 3 sell point requires "the bounce high does not enter the central hub range," with symmetric criteria.

**Regarding Divergence:**
- Trend divergence requires the premise of "two non-overlapping central hubs in the same direction." If there is only one central hub or none, do not call it "trend divergence" — only "range-bound divergence."
- MACD strength comparison must provide specific numerical values (DIF peak or cumulative HIST) — do not use qualitative descriptions like "clearly smaller."

**Honest Uncertainty Annotation:**
- If the data is insufficient for a strict Chanlun judgment (e.g., {PERIOD} data cannot identify sub-level movement segments), must explicitly state "cannot strictly determine X with current data; only a rough estimate based on Y is provided." Do not fabricate conclusions to achieve output completeness.
`;



const TASK_CHANLUN = OUTPUT_WEIGHT_GUIDANCE + CHANLUN_STRICTNESS + `
Please strictly analyze according to the original Chanlun framework. The use of non-Chanlun terminology such as "golden cross," "death cross," "overbought," or "oversold" is prohibited.

1. Strokes and Segments: Identify recent strokes and segments on the {PERIOD} K-line chart, marking the dates and corresponding prices of the last 3 key turning points (top fractal or bottom fractal endpoints).
2. Central Hub: Determine whether a {PERIOD}-level central hub currently exists (at least 3 overlapping sub-level movement segments). If it exists, provide the ZG (central hub upper bound) and ZD (central hub lower bound) prices.
3. Movement Type: Based on the central hub, determine whether the current movement is a trending movement or a range-bound movement. If trending, indicate the direction (up/down) and which central hub number the current price is in.
4. Divergence: Take two consecutive same-direction movement segments, compare their corresponding MACD red/green histogram cumulative areas, and determine whether trend divergence or range-bound divergence is present. If area differences are not significant, explicitly state "no divergence signal."
5. Buy/Sell Points: Based on the above analysis, determine one by one whether any of the three types of buy/sell points exist, providing specific dates and prices:
   - Type 1 buy/sell point (trend divergence point)
   - Type 2 buy/sell point (pullback to central hub without breaching ZG/ZD)
   - Type 3 buy/sell point (leaving central hub then retracing without re-entering)
   If a certain type of buy/sell point does not exist, explicitly state "no Type X buy/sell point at present."

6. Comprehensive Conclusion:
   - Current Chanlun structure classification: explicitly choose one of [Uptrend / Downtrend / Range-bound / Trend Exhaustion / Central Hub Under Construction]
   - Three types of buy/sell points status summary: one-sentence summary of which points are confirmed, which are pending, and which have been invalidated.
   - Key observation levels: provide "central hub upper/lower bounds" or "segment endpoints" as structural change verification levels.
   - Do not directly say "recommend buying/selling." The tone of Chanlun output should be structural — "if retracement to X holds without breaking, Type 3 buy point is established," not "recommend buying at X."

7. Counter-Viewpoints / Risk Notes: List 1-2 key risk scenarios where the current Chanlun structure judgment could be wrong (e.g., central hub identification error, divergence signal destroyed by minor-to-major reversal, segment division error). Each risk must include specific observation conditions.`;

const TASK_VALUE = OUTPUT_WEIGHT_GUIDANCE + `
Please analyze from a valuation and long-term trend perspective:
1. Current {PERIOD} percentile within the historical price range (rough estimate)
2. Long-term trend direction (up/down/sideways); do not over-interpret single-{UNIT} moves.
3. Long-term alignment state of the moving average system (MA5/MA20/MA60)
4. Whether there are obvious extreme position signals (near historical highs, long-term MA support, etc.)

5. Comprehensive Conclusion:
   - Current valuation stage classification: explicitly choose one of [Bottom Recovery / Mid-Value Restoration / Approaching Fair Value / Overvalued Zone]
   - Core long-term investment perspective: based on historical percentile, fundamental changes (if supported by announcements/research reports), and long-term trend maturity, provide a directional view.
   - Time horizon note: explicitly state whether this judgment's time scale is quarterly or annual.
   - Do not directly give "recommend buying/selling." The tone should be "from a long-term value perspective, the current position is closer to X than Y."
6. Counter-Viewpoints / Risk Notes: List 1-2 key risk scenarios where the current valuation judgment could be wrong (e.g., value trap, fundamental regime change). Each risk must include specific observation conditions.`;

const TASK_COMPREHENSIVE = OUTPUT_WEIGHT_GUIDANCE + `
Please provide a comprehensive analysis integrating technical analysis and valuation perspectives:

[Technical]
1. Current {PERIOD} position (historical high / low / continuation zone)
2. Moving average system state (MA5/MA20/MA60 bull/bear alignment)
3. MACD indicator state
4. Key support and resistance levels (provide specific prices)

[Valuation & Long-Term Perspective]
5. Current {PERIOD} percentile within the historical price range
6. Long-term trend direction; be careful not to over-interpret single-{UNIT} fluctuations.

7. Comprehensive Conclusion: integrating technical and value perspectives:
   - Technical signal: one-sentence summary
   - Value signal: one-sentence summary
   - Whether they resonate: explicitly state "Resonance Bullish / Resonance Bearish / Divergent / One Side Dominant"
   - Comprehensive directional judgment: choose one of [Moderately Bullish / Moderately Bearish / Neutral / Inconsistent Signals]
   - Key observation levels: provide one technical and one value verification level.
   - Do not give "recommend buying/selling."
8. Counter-Viewpoints / Risk Notes: List 1-2 key risk scenarios where the current comprehensive judgment could be wrong (e.g., technical-value resonance failure). Each risk must include specific observation conditions.`;

const TASK_MAP = {
  technical: TASK_TECHNICAL,
  chanlun: TASK_CHANLUN,
  value: TASK_VALUE,
  comprehensive: TASK_COMPREHENSIVE,
};

const PERIOD_LABELS = { monthly: 'Monthly', weekly: 'Weekly', daily: 'Daily' };
const COUNT_LABELS = { monthly: 'months', weekly: 'weeks', daily: 'days' };
const UNIT_LABELS  = { monthly: 'month',   weekly: 'week', daily: 'day' };

/**
 * Generate K-line table string (with header), for single analysis and Agent reuse
 */
export function buildKlineTable(klines) {
  const fmt = (v) => (v == null || Number.isNaN(v) ? '-' : Number(v).toFixed(2));
  const fmtVol = (v) => (v == null || Number.isNaN(v) ? '-' : String(v));

  const header = 'Date\tOpen\tClose\tHigh\tLow\tVolume\tChange%\tMA5\tMA20\tMA60\tMACD-DIF\tMACD-DEA\tMACD-HIST\tTurnover';
  const rows = klines.map((k) => [
    k.date,
    fmt(k.open),
    fmt(k.close),
    fmt(k.high),
    fmt(k.low),
    fmtVol(k.volume),
    fmt(k.changePercent),
    fmt(k.ma5),
    fmt(k.ma20),
    fmt(k.ma60),
    fmt(k.dif),
    fmt(k.dea),
    fmt(k.hist),
    fmt(k.turnoverRate),
  ].join('\t'));
  return [header, ...rows].join('\n');
}

// ---- index comparison section (percentage calculation, no longer outputs full K-line table) ----

/**
 * Generate HS300 comparison paragraph
 * @param {Object|null} indexData - { name, klines }, null returns ''
 * @param {Array} stockKlines - stock klines array
 * @param {string} periodLabel - Monthly/Weekly/Daily
 * @returns {string}
 */
function buildIndexBlock(indexData, stockKlines, periodLabel) {
  if (!indexData || !Array.isArray(indexData.klines) || indexData.klines.length === 0) return '';

  const idxK = indexData.klines;
  const N = Math.min(stockKlines.length, idxK.length);

  const stockFirst = stockKlines[stockKlines.length - N].close;
  const stockLast = stockKlines[stockKlines.length - 1].close;
  const idxFirst = idxK[idxK.length - N].close;
  const idxLast = idxK[idxK.length - 1].close;

  const stockChange = ((stockLast - stockFirst) / stockFirst * 100);
  const idxChange = ((idxLast - idxFirst) / idxFirst * 100);
  const diff = stockChange - idxChange;

  return `
## Cross-Sectional Comparison (vs. HS300 over same ${periodLabel} period)

- Stock performance over last ${N} ${periodLabel} bars: **${stockChange >= 0 ? '+' : ''}${stockChange.toFixed(2)}%**
- HS300 over same period: **${idxChange >= 0 ? '+' : ''}${idxChange.toFixed(2)}%**
- Outperformance/Underperformance vs. benchmark: **${diff >= 0 ? '+' : ''}${diff.toFixed(2)} percentage points**

Please reference the above comparison data in each analysis subsection and the comprehensive conclusion to assess the stock's relative strength vs. the broad market.
`;
}

// ---- new template system entry (for single mode) ----

/**
 * Build prompt based on new 4-dimension templates (single mode only)
 */
export async function buildPromptByTemplate({ templateKey = DEFAULT_TEMPLATE, name, code, market, klines, period = 'monthly', provider = 'anthropic', extraContext, decisionMode = false, indexData = null, resonance = null, sectorAlphaData = null, lstmSignalData = null, kronosSignalData = null }) {
  if (!Array.isArray(klines) || klines.length === 0) {
    throw new Error('klines is empty, cannot construct prompt');
  }

  const periodLabel = PERIOD_LABELS[period] || PERIOD_LABELS.monthly;
  const countLabel = COUNT_LABELS[period] || COUNT_LABELS.monthly;
  const unitLabel  = UNIT_LABELS[period]  || UNIT_LABELS.monthly;

  // data window (for 4th hard constraint)
  const dataWindow = {
    count: klines.length,
    startDate: klines[0].date,
    endDate: klines[klines.length - 1].date,
  };

  // 6th hard constraint: only monthly/weekly enable distinguishing closed bars
  const includeClosingBarRule = period === 'monthly' || period === 'weekly';

  // data health check
  const validation = checkKlines(klines, period);
  const validationWarning = buildValidationWarningText(validation);

  // pre-compute technical indicators
  const indicators = calculateAll(klines);
  const tail = tailIndicators(indicators, 5);
  const indicatorTable = buildIndicatorTable(klines, tail);

  // structured signal identification
  const signalSummary = generateSignalSummary(klines, indicators);
  const signalText = formatSignalSummary(signalSummary);
  const signalGuidance = buildSignalGuidance(signalSummary);

  // get template task text
  const task = buildTemplatePrompt(templateKey, periodLabel, unitLabel, dataWindow, includeClosingBarRule) + DISCLAIMER;

  const table = buildKlineTable(klines);

  // additional context
  let contextBlock = '';
  const events = (extraContext && Array.isArray(extraContext.events)) ? extraContext.events : [];
  if (events.length > 0) {
    const eventLines = events.map((e) => `- ${e.date}  ${e.type}: ${e.title || ''}`).join('\n');
    contextBlock = `
## Additional Context

### Recent Key Events (from Eastmoney page "Event Feed", last ${events.length} items)
${eventLines}

These event details may help explain K-line anomalies, but the analysis should still center on technical/valuation/Chanlun framework. Do not over-rely on a single research report or announcement for conclusions.
`;
  }

  // self-backtest calibration block
  if (extraContext && extraContext.backtestBlock) {
    contextBlock += '\n' + extraContext.backtestBlock + '\n';
  }

  // index comparison (percentage format)
  const indexBlock = buildIndexBlock(indexData, klines, periodLabel);


  // HS300 intra-industry Alpha (excess return vs industry benchmark)
  const sectorAlphaBlock = buildSectorAlphaBlock(sectorAlphaData);
  // LSTM daily prediction signal
  const lstmBlock = buildLstmSignalBlock(lstmSignalData);
  const kronosBlock = buildKronosSignalBlock(kronosSignalData);
  // multi-time-window normalized returns
  const nrData = calcNormalizedReturns(klines, period);
  const normalizedReturnBlock = buildNormalizedReturnBlock(nrData);
  let prompt = `You are an A-share technical analyst. Below is ${name}(${code}) ${periodLabel} data for the last ${klines.length} ${countLabel} (forward-adjusted, including MA5/MA20/MA60 and MACD indicators):
${table}
${contextBlock}
${indexBlock}
${sectorAlphaBlock}
${kronosBlock}
${lstmBlock}
${normalizedReturnBlock}
${task}
${provider === 'anthropic' && market ? buildToolInstructions(`${market}.${code}`) : ''}

Output in Markdown format.`;

  if (decisionMode) {
    prompt += PERSONAL_DECISION_BLOCK;
  }

  prompt += '\n' + indicatorTable;
  prompt += '\n' + signalText;
  if (signalGuidance) prompt += '\n' + signalGuidance;

  // multi-period resonance
  if (resonance) {
    const { formatResonanceSummary, buildResonanceConstraint } = await import('./multi-period/resonance.js');
    prompt += '\n' + formatResonanceSummary(resonance);
    prompt += '\n' + buildResonanceConstraint(resonance, signalSummary);
  }
  prompt += '\n' + validationWarning;
  prompt += '\n\n' + STRUCTURED_OUTPUT_BLOCK;

  return prompt;
}

// ---- legacy style entry (kept for debate mode) ----

export function buildPrompt({ name, code, klines, style = 'technical', period = 'monthly', provider = 'anthropic', extraContext, decisionMode = false, indexData = null }) {
  if (!Array.isArray(klines) || klines.length === 0) {
    throw new Error('klines is empty, cannot construct prompt');
  }

  const periodLabel = PERIOD_LABELS[period] || PERIOD_LABELS.monthly;
  const countLabel = COUNT_LABELS[period] || COUNT_LABELS.monthly;
  const unitLabel  = UNIT_LABELS[period]  || UNIT_LABELS.monthly;

  const rawTask = TASK_MAP[style] || TASK_TECHNICAL;
  const task = rawTask.replace(/\{PERIOD\}/g, periodLabel).replace(/\{UNIT\}/g, unitLabel) + DISCLAIMER;

  const table = buildKlineTable(klines);

  // additional context
  let contextBlock = '';
  const events = (extraContext && Array.isArray(extraContext.events)) ? extraContext.events : [];
  if (events.length > 0) {
    const eventLines = events.map((e) => `- ${e.date}  ${e.type}: ${e.title || ''}`).join('\n');
    contextBlock = `
## Additional Context

### Recent Key Events (from Eastmoney page "Event Feed", last ${events.length} items)
${eventLines}

These event details may help explain K-line anomalies, but the analysis should still center on technical/valuation/Chanlun framework. Do not over-rely on a single research report or announcement for conclusions.
`;
  }

  // index comparison (percentage format)
  const indexBlock = buildIndexBlock(indexData, klines, periodLabel);

  let prompt = `You are an A-share technical analyst. Below is ${name}(${code}) ${periodLabel} data for the last ${klines.length} ${countLabel} (forward-adjusted, including MA5/MA20/MA60 and MACD indicators):
${table}
${contextBlock}
${indexBlock}
${task}

Output in Markdown format.`;

  if (decisionMode) {
    prompt += PERSONAL_DECISION_BLOCK;
  }

  prompt += '\n\n' + STRUCTURED_OUTPUT_BLOCK;

  return prompt;
}

/**
 * Multi-period resonance analysis prompt
 */
export function buildMultiPeriodPrompt({
  name, code,
  monthlyKlines, weeklyKlines, dailyKlines,
  style = 'technical', provider = 'anthropic', extraContext, decisionMode = false, indexData = null,
}) {
  const periodLabel = 'Multi-Period Resonance (Monthly+Weekly+Daily)';

  const events = (extraContext && Array.isArray(extraContext.events)) ? extraContext.events : [];
  let contextBlock = '';
  if (events.length > 0) {
    const eventLines = events.map((e) => `- ${e.date}  ${e.type}: ${e.title || ''}`).join('\n');
    contextBlock = `
## Additional Context
${eventLines}
`;
  }

  let prompt = `You are an A-share technical analyst. Below is multi-period K-line data for ${name}(${code}). Please perform a comprehensive analysis based on the "multi-period resonance" principle.

## Monthly Data (long-term trend assessment)
${buildKlineTable(monthlyKlines)}

## Weekly Data (mid-term structure assessment)
${buildKlineTable(weeklyKlines)}

## Daily Data (short-term entry assessment)
${buildKlineTable(dailyKlines)}
${contextBlock}
${buildIndexBlock(indexData, monthlyKlines, 'Monthly')}
## Multi-Period Resonance Analysis Task

**Important Constraints:**
- Every technical judgment must be accompanied by specific numerical evidence (e.g., "Monthly MA60=XX, current price is XX% above MA60"). Qualitative descriptions alone are not allowed.
- Must calculate and explicitly provide the current price's historical position percentile for each period. Formula: percentile = (current close - N-period lowest) / (N-period highest - N-period lowest) x 100%.
- Each analysis subsection must include counter-viewpoints / risk notes.

Output in the following structure (each level states the conclusion first, then expands):

### I. Monthly Trend Positioning (long-term, determines the overall direction)
- Current monthly trend nature: uptrend / downtrend / range-bound / trend exhaustion
- Monthly-level key central hub / support-resistance levels
- Long-term directional judgment: moderately bullish / moderately bearish / neutral

### II. Weekly Structure Positioning (mid-term, determines operational rhythm)
- Current weekly central hub identification (if any)
- Weekly-level buy/sell point status
- Whether it resonates with the monthly direction

### III. Daily Entry Assessment (short-term, determines specific actions)
- Current daily position: overbought / oversold / neutral
- Short-term MACD / MA signals
- Whether it resonates with weekly and monthly

### IV. Multi-Period Resonance Conclusion
- Three-period resonance state: choose one of [Triple Resonance Bullish / Triple Resonance Bearish / Monthly+Weekly Bullish, Daily Bearish (pullback entry opportunity) / Monthly Bearish, Weekly+Daily Bullish (bounce, do not chase) / Chaotic Signals]
- Different resonance states correspond to different operational approaches

### V. Comprehensive Conclusion
${styleConclusion(style)}

### VI. Counter-Viewpoints / Risk Notes
- List 1-2 key risk scenarios where the multi-period resonance judgment could be wrong.
- Each risk must include specific observation conditions (e.g., "if weekly breaks below X, the resonance logic is invalidated").

Output in Markdown format.`;

  if (decisionMode) {
    prompt += PERSONAL_DECISION_BLOCK;
  }

  return prompt;
}

function styleConclusion(style) {
  if (style === 'chanlun') {
    return '- Comprehensive directional judgment: based on multi-period Chanlun structure, choose one of [Moderately Bullish / Moderately Bearish / Neutral]\n- Multi-period buy/sell point status summary\n- Key observation levels (integrating three-period levels)\n- Risk statement';
  }
  if (style === 'value') {
    return '- Comprehensive directional judgment: based on multi-period valuation and long-term trend, choose one of [Moderately Bullish / Moderately Bearish / Neutral]\n- Core long-term perspective judgment\n- Key observation levels\n- Risk statement';
  }
  if (style === 'comprehensive') {
    return '- Technical signal summary (three periods)\n- Value signal summary\n- Resonance assessment\n- Comprehensive directional judgment\n- Key observation levels\n- Risk statement';
  }
  // technical
  return '- Current directional judgment: explicitly choose one of [Moderately Bullish / Moderately Bearish / Neutral Range-bound]\n- Signal resonance analysis (three-period signals confirming/contradicting each other)\n- Key observation levels\n- Risk statement';
}
