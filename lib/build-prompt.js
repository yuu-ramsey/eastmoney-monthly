// prompt 拼装，两个入口：
//   buildPromptByTemplate() → single 模式用（新 4 维度模板）
//   buildPrompt()          → debate 模式用（旧 4 风格，保留兼容）

import { buildTemplatePrompt, DEFAULT_TEMPLATE } from './prompt-templates.js';

const STRUCTURED_OUTPUT_BLOCK = `## 结构化数据输出（必须严格遵守）

在分析报告末尾，附加一个 JSON 块（用 \`\`\` 包裹），机器可读，仅供工具内部校验使用，不影响主报告内容：

\`\`\`json
{
  "period": "monthly|weekly|daily",
  "centralZone": {
    "lower": <下沿价位，精确到 2 位小数>,
    "upper": <上沿价位，精确到 2 位小数>,
    "exists": <true|false，当前周期是否识别出明确中枢>
  },
  "keySupport": [<3-5 个关键支撑位，从强到弱>],
  "keyResistance": [<3-5 个关键阻力位，从弱到强>],
  "trend": "up|down|sideways|reversal_top|reversal_bottom"
}
\`\`\`

注意：
- 这个 JSON 是工具校验用，不要解释它，不要说"以下是 JSON 输出"等多条文字
- 价位必须从 K 线数据里直接引用（MA60、前高、前低等具体数值）
- centralZone.exists = false 时 lower/upper 可以为 null
- 多周期共振模式不需要这个 JSON 块（它已是综合输出）`;

export { STRUCTURED_OUTPUT_BLOCK };

const DISCLAIMER = '\n\n> 本分析仅供研究学习使用，不构成投资建议。市场存在不可预测因素，任何结论都需结合个人风险承受能力独立判断。';

export const PERSONAL_DECISION_BLOCK = `\n\n## 个人决策视角（仅供持有者本人参考）

基于上述分析，给出明确的方向判断和操作思路：

1. **当前位置定性**：在【可介入区 / 偏高位 / 关键决断位 / 高风险区 / 不建议参与】中明确选一个，给出 1-2 句依据
2. **未持仓建议**：在【积极买入 / 分批建仓 / 等回调 / 观望 / 回避】中选一个，给出依据
3. **已持仓建议**：在【加仓 / 持有 / 减仓 / 清仓】中选一个，给出依据
4. **关键止损位**：必须给一个精确到 2 位小数的具体价位（如 35.03 而不是"35 附近"或"35-36"），价位必须从 K 线数据里直接引用（MA60、前低、中枢下沿、强支撑位等具体数值）+ 跌破时的应对策略

5. **关键加仓 / 减仓位**：必须给具体价位（精确到 2 位小数），不允许给区间。每个价位必须标注两件事：
   a. 价位的数据来源（如"MA60=35.03" / "2026-03-20 周线低点 28.10" / "周线中枢下沿 30.20"）
   b. 触及时的处理思路（如"加 30% 仓位"、"减 20%"）
   ❌ 错误示例：35-36 元（日线 MA60，浅回调）→ 加底仓
   ✅ 正确示例：
       加仓位 1：35.03（日线 MA60 当前值，浅回调位）→ 加 20% 仓位
       加仓位 2：28.10（2026-03 周线前低 + 周线中枢下沿，深度回踩位）→ 加 30% 仓位
       减仓位 1：42.90（2025-09 月线前高，第一阻力）→ 减 20%
       减仓位 2：45.95（2026-01 月线绝对高点，第二阻力）→ 减 30%

6. **持有时间预期**：必须严格匹配当前分析周期，不允许短于一根 K 线的时长：
     - 月线分析：在【3-6 个月 / 6-12 个月 / 1 年以上】中选一个
     - 周线分析：在【1-3 个月 / 3-6 个月 / 6-12 个月】中选一个
     - 日线分析：在【1-4 周 / 1-3 个月 / 3-6 个月】中选一个
     - 多周期共振分析：在【3-6 个月 / 6-12 个月】中选一个（综合所有周期）

7. **核心逻辑一句话**：用一句话总结"我为什么这么建议"

8. **相对估值判断**：基于附加上下文中的事件信息（业绩公告/研报标题），给出相对估值判断：
   - 优先使用大事提醒中的"业绩公告/研报"内容，引用具体净利润同比/EPS 数值
   - 如果业绩信息不足，明确说"基本面信息不足，仅基于技术面判断"
   - 输出格式：在【明显低估 / 合理偏低 / 合理 / 合理偏高 / 明显高估 / 基本面信息不足无法判断】中选一个
   - 严禁套用 CAPM、DDM、股息折现等学术模型——A 股个股层面这些模型数据噪音太大，输出会有虚假精确感。只做基于实际公告和价位的相对判断

9. **建议仓位**：在【空仓观望 / 试仓 5-10% / 轻仓 10-25% / 半仓 25-50% / 重仓 50-75% / 满仓 75-100%】中选一个，给出依据：
   - 当前位置定性（关键决断位→试仓；可介入区→轻仓到半仓；高风险区→空仓）
   - 多周期共振状态（三共振多→仓位上调；信号紊乱→仓位下调）
   - 估值判断（明显低估→仓位上调；明显高估→仓位下调）
   - 严禁套用凯利公式——胜率和盈亏比都是估算值，建立在估算基础上算出"精确仓位"是虚假精确

10. **入场策略**：根据第 8、9 项推导入场节奏：
    - 在【一次性入场 / 分 2 批建仓 / 分 3 批建仓 / 等回调统一建仓 / 不建仓】中选一个
    - 每批的触发条件必须明确（如"第一批在当前价 38.87 试仓 10%；第二批在 35.03 加 15%；第三批在 28.10 加 25%"）
    - 价位必须精确到 2 位小数，跟第 5 项的加仓位保持一致

⚠️ 本段输出仅供工具持有者本人决策辅助，不构成对任何第三方的投资建议。历史不代表未来，实际操作请结合个人风险承受能力独立判断。`;

const OUTPUT_WEIGHT_GUIDANCE = `## 输出权重要求

历史数据是判断的基础，但输出结构必须以"当前位置 + 下一步关键观察点"为主。具体要求：

- 涉及历史关键点（如多年前的高点/底部）的描述，每个要点不超过 2 句话，且必须明确说明它对"当前判断"的支撑作用，不要孤立陈述。
- "综合结论"小节必须放在报告末尾，且字数应占整份报告的 20%-30%，是用户最优先阅读的部分。
- 中间各分析小节（如均线、MACD、中枢识别等）的核心结论必须放在小节开头一句话点明，再展开论据。不要让用户读完整段才知道结论是什么。
- 严格禁止"为了凑数"列出无关历史事件。每一条历史引用必须服务于当前判断。
- **数字依据硬约束**：每个技术判断必须附带具体数字依据（如"当前价 38.87 高于 MA60=35.03，偏离 11.0%"），不允许仅使用"偏高""偏低""强势""弱势"等定性描述。无数字支撑的判断视为无效。
- **反方观点必答**：每个分析小节必须包含"反方观点/风险提示"，列出当前判断可能出错的 1-2 个关键反方论据或风险场景，不能跳过。
- **位置百分位计算**：必须明确计算并给出当前价格的历史位置百分位。公式：百分位 = (当前收盘价 - N期最低价) ÷ (N期最高价 - N期最低价) × 100%。N 取所给 K 线数据的实际长度。结果必须写明如"当前价格处于近{N}根{PERIOD}的第 XX 百分位（即高于历史 XX% 的价格区间）"。各分析小节中凡是提到"高位/低位/中位"等位置判断，必须引用这个百分位数值。
`;

const TASK_TECHNICAL = OUTPUT_WEIGHT_GUIDANCE + `
请用中文分析:
1. 当前{PERIOD}位置(历史高位/低位/中继区)
2. 均线系统状态(MA5/MA20/MA60 多空排列)
3. MACD 指标状态(DIF/DEA 位置关系、柱状线变化趋势)
4. 量价关系(成交量与涨跌幅配合情况)
5. 关键支撑位和压力位(给出具体价格)
6. {PERIOD}级别趋势判断

7. 综合结论:
   - 当前方向判断:在【偏多 / 偏空 / 中性震荡】中明确选一个,并用一句话说明主要依据
   - 数据共振分析:列出当前所有指标中互相印证的信号(如均线多头排列 + MACD 方向一致 + 量价配合),以及互相矛盾的信号(如均线偏多但成交量萎缩)
   - 关键观察价位:给出 1-2 个具体价格作为后续走势验证位,说明"若突破 X 则验证偏多,若跌破 Y 则转为偏空"
   - 不要直接说"建议买入/卖出/持有"。结论的语气应该是"如果出现 X 则验证 Y 判断",而不是"应该做 X"
	8. 反方观点/风险提示:列出当前判断可能出错的 1-2 个关键风险场景(如均线假突破、MACD 钝化失效、量价背离误判),每个风险必须附带触发该风险的观测条件`;

const CHANLUN_STRICTNESS = `## 缠论体系严格性要求

本次分析必须严格遵守缠论原文体系，避免以下常见错误：

**关于笔与线段：**
- 笔由前后两个相反的分型确认，二次探底（如二次回踩前低附近）属于同一笔内的次级别波动，不是新笔的起点。判定新笔时必须找到明确的反向分型。
- 如果两个相邻低点（或高点）非常接近且中间没有明确反向走势，应视为同一底部（或顶部）区域，不要拆成两笔。

**关于中枢：**
- {PERIOD}中枢必须由"次级别走势段重叠"定义，不能直接拿当前周期的最高最低值作为 ZG/ZD。
- ZG = 各次级别下跌段的最低高点，ZD = 各次级别上涨段的最高低点。
- 至少需要 3 段次级别走势在区间内重叠才能成立中枢。如果当前数据无法清晰识别 3 段重叠，应明确说明"中枢尚未明确形成"，不要硬凑。
- 中枢被有效突破后，必须有新的 3 段重叠才能称"新中枢构建中"。仅有单段离开走势时，应称为"离开旧中枢的上行段（无中枢）"。

**关于三类买卖点：**
- 第三类买点要求："离开中枢后，回踩低点不进入中枢区间"。
- "不进入"必须明显，回踩低点至少要在 ZG 上方 3% 以上的位置才算有效三买。如果回踩低点贴近 ZG（差距 1% 以内），属于"接近 ZG 但未确认完全脱离"，应判定为"待确认"而非"已成立"。
- 同理三类卖点要求"反弹高点不进入中枢区间"，标准对称。

**关于背驰：**
- 趋势背驰要求"两个同向不重叠的中枢"前提。如果当前只有一个中枢或没有中枢，不能称"趋势背驰"，只能称"盘整背驰"。
- MACD 力度比较必须给出具体数值（DIF 峰值或 HIST 累计），不要只用"明显小于"这种定性描述。

**诚实标注不确定性：**
- 如果数据不足以做出严格的缠论判断（如{PERIOD}数据不够识别次级别走势段），必须明确说"在当前数据下无法严格判定 X，仅基于 Y 给出粗略推测"，不要为了输出完整性硬编结论。
`;



const TASK_CHANLUN = OUTPUT_WEIGHT_GUIDANCE + CHANLUN_STRICTNESS + `
请严格按缠论(缠中说禅)原文体系分析,禁止使用"金叉""死叉""超买""超卖"等非缠论术语。

1. 笔与线段:识别{PERIOD} K 线图上最近的笔和线段,标出最近 3 个关键转折点(顶分型或底分型端点)的日期和对应价格。
2. 中枢:判断当前是否存在{PERIOD}级别中枢(至少 3 段次级别走势重叠区间)。若存在,给出中枢的 ZG(中枢上沿)和 ZD(中枢下沿)价格。
3. 走势类型:基于中枢,判断当前是趋势走势还是盘整走势。若是趋势,指明方向(向上/向下)以及当前处于第几个中枢。
4. 背驰:取前后两段同向走势段,比较其对应的 MACD 红绿柱面积之和,判断是否构成趋势背驰或盘整背驰。若面积差异不显著,明确说明"无背驰信号"。
5. 买卖点:基于上述分析,逐一判定是否存在三类买卖点,给出具体日期和价格:
   - 第一类买卖点(趋势背驰点)
   - 第二类买卖点(回拉中枢不破 ZG/ZD)
   - 第三类买卖点(离开中枢后回抽不进中枢)
   若某类买卖点不存在,明确说明"暂无 X 类买卖点"。

6. 综合结论:
   - 当前缠论结构定性:在【上涨趋势 / 下跌趋势 / 盘整 / 趋势末端 / 中枢构建中】中明确选一个
   - 三类买卖点状态汇总:用一句话总结哪些买卖点已确认、哪些待确认、哪些已失效
   - 关键观察位:给出"中枢上沿/下沿"或"线段端点"作为结构变化的验证位
   - 不要直接说"建议买入/卖出"。缠论的输出语气应该是结构性的——"若回踩 X 不破,三类买点确立",而不是"建议在 X 买入"

	7. 反方观点/风险提示:列出当前缠论结构判断可能出错的 1-2 个关键风险场景(如中枢识别错误、背驰信号被小转大破坏、线段划分偏差),每个风险必须附带具体观测条件`;

const TASK_VALUE = OUTPUT_WEIGHT_GUIDANCE + `
请从估值与长期趋势视角分析:
1. 当前{PERIOD}处于历史价格区间的分位(大致估算)
2. 长期趋势方向(上升/下降/横盘),不要过度解读单{UNIT}涨跌幅
3. 均线系统(MA5/MA20/MA60)的长期排列状态
4. 是否有明显的极端位置信号(历史高点附近、长期均线支撑等)

5. 综合结论:
   - 当前估值阶段定性:在【底部修复 / 价值回归中段 / 接近合理估值 / 高估区间】中明确选一个
   - 长期投资视角的核心判断:基于历史分位、基本面变化(如有公告/研报佐证)、长期趋势完整度,给出方向性观点
   - 时间维度提示:明确说明这个判断的时间尺度是季度级别还是年度级别
   - 不要直接给"建议买入/卖出"。语气应该是"从长期价值视角看,当前更接近 X 而非 Y"
	6. 反方观点/风险提示:列出当前估值判断可能出错的 1-2 个关键风险场景(如估值陷阱、基本面突变),每个风险必须附带具体观测条件`;

const TASK_COMPREHENSIVE = OUTPUT_WEIGHT_GUIDANCE + `
请综合技术分析与估值视角分析:

【技术面】
1. 当前{PERIOD}位置(历史高位/低位/中继区)
2. 均线系统状态(MA5/MA20/MA60 多空排列)
3. MACD 指标状态
4. 关键支撑位和压力位(给出具体价格)

【估值与长期视角】
5. 当前{PERIOD}处于历史价格区间的分位
6. 长期趋势方向,注意不要过度解读单{UNIT}波动

7. 综合结论:整合技术面和价值面:
   - 技术面信号:一句话总结
   - 价值面信号:一句话总结
   - 两者是否共振:明确说明"共振看多 / 共振看空 / 互相背离 / 一方主导"
   - 综合方向判断:在【偏多 / 偏空 / 中性 / 信号不一致】中选一个
   - 关键观察价位:给一个技术面和一个价值面的验证位
   - 不要给"建议买入/卖出"
	8. 反方观点/风险提示:列出当前综合判断可能出错的 1-2 个关键风险场景(如技术面与价值面共振失效),每个风险必须附带具体观测条件`;

const TASK_MAP = {
  technical: TASK_TECHNICAL,
  chanlun: TASK_CHANLUN,
  value: TASK_VALUE,
  comprehensive: TASK_COMPREHENSIVE,
};

const PERIOD_LABELS = { monthly: '月线', weekly: '周线', daily: '日线' };
const COUNT_LABELS = { monthly: '个月', weekly: '周', daily: '日' };
const UNIT_LABELS  = { monthly: '月',   weekly: '周', daily: '日' };

/**
 * 生成 K 线表格字符串（含表头），供单次分析和 Agent 复用
 */
export function buildKlineTable(klines) {
  const fmt = (v) => (v == null || Number.isNaN(v) ? '-' : Number(v).toFixed(2));
  const fmtVol = (v) => (v == null || Number.isNaN(v) ? '-' : String(v));

  const header = '日期\t开盘\t收盘\t最高\t最低\t成交量\t涨跌幅\tMA5\tMA20\tMA60\tMACD-DIF\tMACD-DEA\tMACD-HIST\t换手率';
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

// ---- 大盘对照段（百分比计算，不再输出完整 K 线表格） ----

/**
 * 生成沪深300对比段落
 * @param {Object|null} indexData - { name, klines }，null 则返回 ''
 * @param {Array} stockKlines - 个股 klines 数组
 * @param {string} periodLabel - 月线/周线/日线
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
## 横向对照（沪深300同期${periodLabel}）

- 个股近 ${N} 根${periodLabel}涨跌幅：**${stockChange >= 0 ? '+' : ''}${stockChange.toFixed(2)}%**
- 沪深300同期：**${idxChange >= 0 ? '+' : ''}${idxChange.toFixed(2)}%**
- 跑赢/跑输大盘：**${diff >= 0 ? '+' : ''}${diff.toFixed(2)} 个百分点**

请在各分析小节和综合结论中引用上述对比数据，判断个股相对大盘的强弱。
`;
}

// ---- 新模板系统入口（single 模式用） ----

/**
 * 基于新 4 维度模板 build prompt（仅 single 模式）
 */
export function buildPromptByTemplate({ templateKey = DEFAULT_TEMPLATE, name, code, klines, period = 'monthly', provider = 'anthropic', extraContext, decisionMode = false, indexData = null }) {
  if (!Array.isArray(klines) || klines.length === 0) {
    throw new Error('klines 为空,无法构造 prompt');
  }

  const periodLabel = PERIOD_LABELS[period] || PERIOD_LABELS.monthly;
  const countLabel = COUNT_LABELS[period] || COUNT_LABELS.monthly;
  const unitLabel  = UNIT_LABELS[period]  || UNIT_LABELS.monthly;

  // 数据窗口（第 4 条硬约束用）
  const dataWindow = {
    count: klines.length,
    startDate: klines[0].date,
    endDate: klines[klines.length - 1].date,
  };

  // 第 6 条硬约束：仅月线/周线启用区分已收盘 K 线
  const includeClosingBarRule = period === 'monthly' || period === 'weekly';

  // 取模板任务文本
  const task = buildTemplatePrompt(templateKey, periodLabel, unitLabel, dataWindow, includeClosingBarRule) + DISCLAIMER;

  const table = buildKlineTable(klines);

  // 附加上下文
  let contextBlock = '';
  const events = (extraContext && Array.isArray(extraContext.events)) ? extraContext.events : [];
  if (events.length > 0) {
    const eventLines = events.map((e) => `- ${e.date}  ${e.type}：${e.title || ''}`).join('\n');
    contextBlock = `
## 附加上下文

### 近期重要事件（来自东方财富页面"大事提醒"，最近 ${events.length} 条）
${eventLines}

这些事件信息可帮助解释 K 线异动，但分析仍以技术面/估值/缠论体系为主，不要过度依赖单条研报或公告下定论。
`;
  }

  // 大盘对照（百分比格式）
  const indexBlock = buildIndexBlock(indexData, klines, periodLabel);

  let prompt = `你是 A 股技术分析师。以下是 ${name}(${code}) 近 ${klines.length} ${countLabel}的${periodLabel}数据(前复权,含 MA5/MA20/MA60 及 MACD 指标):
${table}
${contextBlock}
${indexBlock}
${task}

用 Markdown 格式输出。`;

  if (decisionMode) {
    prompt += PERSONAL_DECISION_BLOCK;
  }

  prompt += '\n\n' + STRUCTURED_OUTPUT_BLOCK;

  return prompt;
}

// ---- 旧风格入口（debate 模式保留） ----

export function buildPrompt({ name, code, klines, style = 'technical', period = 'monthly', provider = 'anthropic', extraContext, decisionMode = false, indexData = null }) {
  if (!Array.isArray(klines) || klines.length === 0) {
    throw new Error('klines 为空,无法构造 prompt');
  }

  const periodLabel = PERIOD_LABELS[period] || PERIOD_LABELS.monthly;
  const countLabel = COUNT_LABELS[period] || COUNT_LABELS.monthly;
  const unitLabel  = UNIT_LABELS[period]  || UNIT_LABELS.monthly;

  const rawTask = TASK_MAP[style] || TASK_TECHNICAL;
  const task = rawTask.replace(/\{PERIOD\}/g, periodLabel).replace(/\{UNIT\}/g, unitLabel) + DISCLAIMER;

  const table = buildKlineTable(klines);

  // 附加上下文
  let contextBlock = '';
  const events = (extraContext && Array.isArray(extraContext.events)) ? extraContext.events : [];
  if (events.length > 0) {
    const eventLines = events.map((e) => `- ${e.date}  ${e.type}：${e.title || ''}`).join('\n');
    contextBlock = `
## 附加上下文

### 近期重要事件（来自东方财富页面"大事提醒"，最近 ${events.length} 条）
${eventLines}

这些事件信息可帮助解释 K 线异动，但分析仍以技术面/估值/缠论体系为主，不要过度依赖单条研报或公告下定论。
`;
  }

  // 大盘对照（百分比格式）
  const indexBlock = buildIndexBlock(indexData, klines, periodLabel);

  let prompt = `你是 A 股技术分析师。以下是 ${name}(${code}) 近 ${klines.length} ${countLabel}的${periodLabel}数据(前复权,含 MA5/MA20/MA60 及 MACD 指标):
${table}
${contextBlock}
${indexBlock}
${task}

用 Markdown 格式输出。`;

  if (decisionMode) {
    prompt += PERSONAL_DECISION_BLOCK;
  }

  prompt += '\n\n' + STRUCTURED_OUTPUT_BLOCK;

  return prompt;
}

/**
 * 多周期共振分析 prompt
 */
export function buildMultiPeriodPrompt({
  name, code,
  monthlyKlines, weeklyKlines, dailyKlines,
  style = 'technical', provider = 'anthropic', extraContext, decisionMode = false, indexData = null,
}) {
  const periodLabel = '多周期共振（月+周+日）';

  const events = (extraContext && Array.isArray(extraContext.events)) ? extraContext.events : [];
  let contextBlock = '';
  if (events.length > 0) {
    const eventLines = events.map((e) => `- ${e.date}  ${e.type}：${e.title || ''}`).join('\n');
    contextBlock = `
## 附加上下文
${eventLines}
`;
  }

  let prompt = `你是 A 股技术分析师。下面是 ${name}(${code}) 的多周期 K 线数据，请基于"多周期共振"原则做综合分析。

## 月线数据（长期趋势判断）
${buildKlineTable(monthlyKlines)}

## 周线数据（中期结构判断）
${buildKlineTable(weeklyKlines)}

## 日线数据（短期入场判断）
${buildKlineTable(dailyKlines)}
${contextBlock}
${buildIndexBlock(indexData, monthlyKlines, '月线')}
## 多周期共振分析任务

**重要约束：**
- 每个技术判断必须附带具体数字依据（如"月线 MA60=XX，当前价高于 MA60 XX%"），不允许仅使用定性描述
- 必须计算并显式给出当前价格在每个周期的历史位置百分位。公式：百分位 = (当前收盘价 - N期最低价) ÷ (N期最高价 - N期最低价) × 100%
- 每个分析小节必须包含反方观点/风险提示

按以下结构输出（每个层级先给结论再展开）：

### 一、月线趋势定位（长期，决定大方向）
- 当前月线趋势性质：上涨趋势/下跌趋势/盘整/趋势末端
- 月线级别关键中枢/支撑压力位
- 长期方向判断：偏多/偏空/中性

### 二、周线结构定位（中期，决定操作节奏）
- 当前周线中枢识别（如有）
- 周线级别买卖点状态
- 与月线方向是否共振

### 三、日线入场判断（短期，决定具体动作）
- 当前日线位置：超买/超卖/中性
- 短期 MACD/均线信号
- 与周线、月线是否共振

### 四、多周期共振结论
- 三周期共振状态：【三共振多 / 三共振空 / 月周多日空（回调中可介入）/ 月空周日多（反弹勿追）/ 信号紊乱】中选一个
- 不同共振状态对应不同的操作思路

### 五、综合结论
${styleConclusion(style)}

### 六、反方观点/风险提示
- 列出多周期共振判断可能出错的 1-2 个关键风险场景
- 每个风险必须附带具体观测条件（如"若周线跌破 X 则共振逻辑失效"）

用 Markdown 格式输出。`;

  if (decisionMode) {
    prompt += PERSONAL_DECISION_BLOCK;
  }

  return prompt;
}

function styleConclusion(style) {
  if (style === 'chanlun') {
    return '- 综合方向判断：基于多周期缠论结构，在【偏多/偏空/中性】中选一个\n- 多周期买卖点状态汇总\n- 关键观察价位（整合三周期价位）\n- 风险声明';
  }
  if (style === 'value') {
    return '- 综合方向判断：基于多周期估值与长期趋势，在【偏多/偏空/中性】中选一个\n- 长期视角核心判断\n- 关键观察价位\n- 风险声明';
  }
  if (style === 'comprehensive') {
    return '- 技术面信号总结（三周期）\n- 价值面信号总结\n- 共振判断\n- 综合方向判断\n- 关键观察价位\n- 风险声明';
  }
  // technical
  return '- 当前方向判断：在【偏多 / 偏空 / 中性震荡】中明确选一个\n- 数据共振分析（三周期信号互相印证/矛盾）\n- 关键观察价位\n- 风险声明';
}
