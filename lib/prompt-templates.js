// 分析维度 prompt 模板（仅 single 模式使用）
// 旧 4 风格 (technical/chanlun/value/comprehensive) 保留在 build-prompt.js，仅 debate 模式用
//
// 每个模板内嵌 3 条硬约束：
//   1. 所有判断附具体数字，禁止"较高""明显"等模糊表述
//   2. 必须输出反方观点（2 条看多 + 2 条看空）
//   3. 操作建议只给区间，不给单点；末尾写免责声明

const HARD_CONSTRAINTS = (periodLabel, dataWindow, includeClosingBarRule) => `
## 硬约束（本次分析必须全部遵守，缺一不可）

1. **数字依据**：每个技术判断必须附带具体数字（如"MA20=35.03，当前价高于 MA20 8.5%"、"近 12 根${periodLabel}中 8 根收阳"），禁止使用"较高""明显""偏强""偏弱"等无数字支撑的定性词。

2. **反方观点**：完整输出 2 条看多论点 + 2 条看空论点，每条必须有对应数字依据。看多和看空的权重需对等，不能一边长篇一边一句话带过。

3. **操作建议**：只给价格区间，不给单一精确价位。末尾必须写上"**免责声明：本分析仅供研究学习，不构成投资建议。市场存在不可预测因素，任何结论都需结合个人风险承受能力独立判断。**"

4. **数据窗口标注**：本次分析的 K 线数据共 ${dataWindow.count} 根${periodLabel}，覆盖时间范围为 ${dataWindow.startDate} 至 ${dataWindow.endDate}。所有"历史极值""历史首次""前所未有"类措辞必须明确限定在该窗口内，改用"近 ${dataWindow.count} 根${periodLabel}区间内的最高/最低值"等精确表述。禁止使用"历史上""从未""绝对峰值"等暗示无限时间跨度的措辞。

5. **禁止表外数据（违反此条视为分析失效）**：仅可引用上方 K 线表格中明确出现的日期和数值。
   - 提及任何具体年份/月份/数值前，必须自检"该数据是否在表格的 ${dataWindow.count} 行中"
   - 不在表格内的年份绝对禁止出现——尤其是早于 ${dataWindow.startDate} 的年份、以及晚于 ${dataWindow.endDate} 的年份。表格覆盖的就是本次分析的全部可用时间范围，不存在"表格之外的历史"
   - 训练时见过的该股票历史信息一律忽略，仅使用本次提供的数据
   - 如必须引用窗口外数据论证，改写为"该论证需要更早期数据，本数据窗口（${dataWindow.startDate} 至 ${dataWindow.endDate}）不包含，暂不展开"${includeClosingBarRule ? `

6. **区分已收盘 K 线**：K 线表格最后一根可能尚未到期收盘。判断方式：如果最后一根 K 线的日期不晚于今日且距今日超过一个完整周期，则视为已收盘；否则视为"当${periodLabel}在途数据"。
   - 主要分析依据基于"已收盘的最后一根 K 线"（即倒数第二根如果当周期在途，或倒数第一根如果已收盘）
   - 在途 K 线数据仅可作为"截至 ${dataWindow.endDate} 的实时参考"，不能作为形态判断、突破/跌破信号的主依据
   - 操作建议中明确区分："已确认信号"基于已收盘数据，"实时观察信号"基于在途数据` : ''}
`;

// 位置百分位计算要求（技术面/趋势/情绪面模板共用）
const PERCENTILE_REQUIREMENT = (periodLabel) => `
**位置百分位计算**：必须计算当前价格的历史位置百分位。
公式：百分位 = (当前收盘价 - 近 N 根${periodLabel}最低价) ÷ (近 N 根${periodLabel}最高价 - 近 N 根${periodLabel}最低价) × 100%
N = 数据中实际 K 线条数。结果格式："当前价格处于近{N}根${periodLabel}的第 XX.X 百分位（高于历史 XX.X% 的价格区间）"
`;

const TEMPLATES = {
  technical: {
    label: '技术面',
    build(periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
      return `## 分析任务：${periodLabel}技术面分析
${HARD_CONSTRAINTS(periodLabel, dataWindow, includeClosingBarRule)}
${PERCENTILE_REQUIREMENT(periodLabel)}

请用中文分析以下${periodLabel}技术指标：

1. **均线系统**：MA5/MA20/MA60 当前值及排列关系（多头/空头/交叉）。每条均线给出与当前价的具体偏离百分比。

2. **K 线形态**：最近 3 根${periodLabel}的实体长短、影线位置、是否出现关键反转形态（如锤子线、吞没、十字星等）。每个形态判断必须引用具体日期和价位。

3. **支撑与压力**：从 K 线数据中识别 3-5 个关键价位并标注来源（如"MA60=XX.XX"、"2025-09 ${periodLabel}低点 XX.XX"、"中枢上沿 XX.XX"），给出每个价位的有效性评估。

4. **MACD 状态**：DIF/DEA 当前值、柱状线变化趋势、是否出现背离信号。必须给出具体数值。

5. **综合判断**：
   - 方向判断：在【偏多 / 偏空 / 中性震荡】中选一个，附数字依据
   - 关键验证位：1-2 个具体价位，"若突破 X 则验证偏多，若跌破 Y 则转为偏空"
   - 反方观点：2 条看多 + 2 条看空，对等展开

用 Markdown 格式输出。`;
    },
  },

  trend: {
    label: '趋势判断',
    build(periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
      return `## 分析任务：${periodLabel}中长期趋势判断
${HARD_CONSTRAINTS(periodLabel, dataWindow, includeClosingBarRule)}
${PERCENTILE_REQUIREMENT(periodLabel)}

请用中文分析${periodLabel}级别的趋势状态：

1. **趋势方向定性**：在【明确上涨趋势 / 温和上涨 / 横盘震荡 / 温和下跌 / 明确下跌趋势 / 趋势转折中】中选一个。必须引用均线排列和价格结构作为依据。

2. **K 线统计**：统计近 N 根${periodLabel}的收阳/收阴比例、连续阳线/阴线的最长连数、平均涨跌幅。数字必须逐根统计后给出。

3. **动能评估**：基于 MACD 柱状线变化、涨跌幅波动率，判断当前动能是加速/减速/衰竭。给出具体数值对比（如前 3${unitLabel} hist 均值 vs 后 3${unitLabel} hist 均值）。

4. **趋势完整性**：当前趋势已运行多少根${periodLabel}（从最近一次明确转折点起算），是否有趋势末端信号（如连续缩量、涨跌幅收窄、均线粘合）。

5. **综合判断**：
   - 趋势阶段：在【趋势初期 / 趋势中段 / 趋势末端 / 趋势不明】中选一个
   - 方向判断：在【偏多 / 偏空 / 中性】中选一个
   - 趋势关键位：若跌破/突破哪个点位趋势会改变
   - 反方观点：2 条看多 + 2 条看空，对等展开

用 Markdown 格式输出。`;
    },
  },

  valuation: {
    label: '估值面',
    build(periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
      return `## 分析任务：${periodLabel}价格历史长期价值视角
${HARD_CONSTRAINTS(periodLabel, dataWindow, includeClosingBarRule)}

注：本维度基于价格历史分位和长期均线位置做相对价值判断，不涉及 PE/PB 等基本面估值指标。

请用中文从长期价值视角分析：

1. **历史位置**：计算当前价格在近 N 根${periodLabel}数据中的位置百分位（公式同上），明确给出百分位数字。

2. **长期趋势分期**：把近 N 根${periodLabel}数据分为"上涨段/下跌段/横盘段"，标注每段的起止日期和涨跌幅。判断当前处于哪个阶段。

3. **极端位置信号**：检查当前价是否接近历史极值（最高/最低 ±10% 内、长期均线支撑位等），给出具体偏离度。

4. **均线长期排列**：MA5/MA20/MA60 长期排列状态演变（最近 3 次交叉日期和方向），判断均线系统是否健康。

5. **估值阶段定性**：综合价格历史分位和均线位置，在【底部修复 / 价值回归中段 / 接近合理估值 / 高估区间】中选一个。所有判断必须引用具体价格和分位数字，不引用 PE/PB 等未提供数据。

6. **综合判断**：
   - 时间维度：明确说明本判断的时间尺度（季度级别 / 年度级别）
   - 方向判断：在【偏多 / 偏空 / 中性】中选一个
   - 反方观点：2 条看多 + 2 条看空，对等展开

用 Markdown 格式输出。`;
    },
  },

  sentiment: {
    label: '情绪面',
    build(periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
      return `## 分析任务：${periodLabel}市场情绪与量价分析
${HARD_CONSTRAINTS(periodLabel, dataWindow, includeClosingBarRule)}
${PERCENTILE_REQUIREMENT(periodLabel)}

请用中文分析${periodLabel}级别的市场情绪和量价关系：

1. **量价配合分析**：逐对检查最近 5 根${periodLabel}的量价关系（放量上涨/缩量上涨/放量下跌/缩量下跌），统计匹配比例。必须给出每根${periodLabel}的量价状态。

2. **换手率分析**：近 N 根${periodLabel}换手率的均值、标准差、当前值所处百分位。判断市场参与度是上升/下降/稳定。

3. **振幅分析**：近 N 根${periodLabel}的振幅变化趋势（扩大/收窄），当前振幅在历史中的百分位。高振幅代表分歧大，低振幅代表方向蓄势。

4. **情绪偏离度**：当前价与 MA20 的偏离百分比、与 MA60 的偏离百分比。偏离度越大情绪越极端。给出历史对比（当前偏离度在历史中的排名）。

5. **陷阱信号检测**：检查是否出现"价涨量缩"（多头陷阱）、"价跌量缩"（空头衰竭）、"放量滞涨"（出货嫌疑）等信号。

6. **综合判断**：
   - 情绪定性：在【乐观 / 中性偏乐观 / 中性 / 中性偏悲观 / 悲观】中选一个
   - 方向判断：在【偏多 / 偏空 / 中性震荡】中选一个
   - 反方观点：2 条看多 + 2 条看空，对等展开

用 Markdown 格式输出。`;
    },
  },
};

// 默认模板 key
export const DEFAULT_TEMPLATE = 'technical';

// 模板标签映射（popup 用）
export const TEMPLATE_LABELS = {};
for (const [key, tpl] of Object.entries(TEMPLATES)) {
  TEMPLATE_LABELS[key] = tpl.label;
}

/**
 * 获取模板的 prompt 任务文本，periodLabel/unitLabel 已通过模板字符串直接插入。
 * @param {string} templateKey - 'technical' | 'trend' | 'valuation' | 'sentiment'
 * @param {string} periodLabel - 月线/周线/日线
 * @param {string} unitLabel - 月/周/日
 * @returns {string} prompt 任务文本
 */
export function buildTemplatePrompt(templateKey, periodLabel, unitLabel, dataWindow, includeClosingBarRule) {
  const tpl = TEMPLATES[templateKey];
  if (!tpl) {
    return buildTemplatePrompt(DEFAULT_TEMPLATE, periodLabel, unitLabel, dataWindow, includeClosingBarRule);
  }
  return tpl.build(periodLabel, unitLabel, dataWindow, includeClosingBarRule);
}
