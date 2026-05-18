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
7. **结构化输出**：分析正文结束后，必须在末尾追加以下格式的 JSON 代码块（用 \`\`\`json 标记），所有字段必填：

\`\`\`json
{
  "score": 0-100 的整数，综合评分，越高越看多，越低越看空，50 为中性,
  "signal": "strong_bull" | "bull" | "neutral" | "bear" | "strong_bear",
  "confidence": "high" | "medium" | "low",
  "key_levels": {
    "support": [价位1, 价位2],
    "resistance": [价位1, 价位2],
    "stop_loss": 止损位
  },
  "trend": "uptrend" | "downtrend" | "sideways" | "reversing",
  "position_percentile": 0-100 之间的数字，当前价在数据窗口的百分位,
  "one_line_summary": "20 字以内核心结论"
}
\`\`\`

注意：
- score 计算：技术面强 + 估值低 + 趋势好 → 接近 100；反之接近 0
- confidence 反映 LLM 自己对判断的把握程度
- one_line_summary 不超过 20 字
- JSON 必须能被 JSON.parse 解析，不要带注释
- 字段名严格小写下划线风格

8. **禁止自算技术指标**：数学计算（MA/MACD/RSI/KDJ/BOLL/换手率均值/百分位）已由程序完成并填入下方"预计算技术指标"表格。你的任务是基于这些事实做综合解读，不需要自己重新计算。如果你引用的数字与表格不一致，视为错误。
9. **信号触发指引**：程序已识别并列出结构化买卖信号（见下方"结构化信号识别"段）。若本段出现"信号指引"，按其指示选择 signal 标签：   - "证据充分，signal 字段应判 strong_bull" → 你必须判 strong_bull，禁止退到 bull 或 neutral   - "证据充分，signal 字段应判 strong_bear" → 你必须判 strong_bear   - "signal 字段应至少判 bull" → 你最弱判 bull，不能退到 neutral   - strong_bull/strong_bear 不是禁忌词。该用时不用 = 失职10. **signal 一致性**：JSON 的 signal 字段必须与正文的文字结论一致。如果你在正文结论里写"偏多"、"看涨"、"上涨趋势"，signal 不能是 bear 或 neutral。反之亦然。

11. **多周期共振反向约束**（基于 HS300 2018-2025 回测：三周期全同向是趋势末端信号，非趋势确认）：
   - strong_bull 共振（三周期全偏多）= 趋势末端警告 → 禁止判 strong_bull，最多判 bull；倾向判 neutral（历史：全偏多后 6 月 alpha 4.31%，跑输全偏空后 7.67%）
   - strong_bear 共振（三周期全偏空）= 均值回归机会 → 可判 bull 或 strong_bull（若技术面也支持）；不要判 bear 或 strong_bear
   - partial 共振（2/3 同向）：方向参考但规则弱化
   - divergent（无共振）：按技术面自由判断
   - 违反此约束（strong_bull 共振时判 strong_bull）= 失职。历史证实三周期全偏多是最不可靠的看多信号

	12. **行业相对强弱强制推理**：若上方存在"HS300 行业内 Alpha"段，你必须严格执行以下推理流程：

		**alpha 等级判定**（必须明确引用）：
		- alpha > +5pp     → "行业内强势"
		- -5pp ≤ alpha ≤ +5pp → "行业内中性"
		- alpha < -5pp     → "行业内弱势"

		**推理约束**：
		1. 必须在分析正文中明确引用 alpha 等级（原文复述等级名称）
		2. signal 方向必须与 alpha 等级整体一致；若相反，必须在综合结论段给出显式反驳理由
		3. **strong_bull 必须有 alpha 等级"强势"支持**，alpha 非强势时禁止判 strong_bull
		4. **strong_bear 必须有 alpha 等级"弱势"支持**，alpha 非弱势时禁止判 strong_bear
		5. alpha 等级"中性"时，最多判 bull 或 bear，不得判 strong
		6. 行业 alpha 是 hs300 内部基准，不是申万全市场指数

		13. **confidence 字段诚实法则**：JSON 的 confidence 字段必须真实反映你的判断确定性：
		- 当技术指标相互矛盾、信号方向不明确时 → 必须设 confidence="low"，signal 倾向 neutral
		- 当多个信号一致但强度一般时 → confidence="medium"
		- 仅当多条独立证据链一致指向同一方向时 → confidence="high"
		- 不得在所有输出中默认使用 confidence="medium"
`;


// HS300 行业内 Alpha 段落格式化
export function buildSectorAlphaBlock(alphaData) {
  if (!alphaData || alphaData.hs300_sector_alpha == null) return '';

  const val = alphaData.hs300_sector_alpha;
  const alphaDisplay = val > 0 ? `+${val.toFixed(2)}pp` : `${val.toFixed(2)}pp`;
  const level = val > 5 ? '行业内强势' : val < -5 ? '行业内弱势' : '行业内中性';
  const sectorReturn = alphaData.sector_return;
  const sectorDisplay = sectorReturn > 0 ? `+${sectorReturn.toFixed(2)}%` : `${sectorReturn.toFixed(2)}%`;

  return `
## HS300 行业内 Alpha（12 月 lookback）

| 指标 | 数值 |
|------|------|
| 个股 alpha | ${alphaDisplay} |
| 等级 | **${level}** |
| 行业排名 | ${alphaData.hs300_sector_rank || '?'}/${alphaData.hs300_sector_total}（前 ${alphaData.hs300_sector_percentile || '?'}%）|
| 行业 | ${alphaData.sector_name}（${alphaData.sector_code}）|
| 行业基准涨幅 | ${sectorDisplay} |

（hs300 内部基准，非申万全市场）
`;
}

// 预计算指标表格格式化
export function buildIndicatorTable(kwi, indicatorsObj) {
  if (!indicatorsObj || !indicatorsObj.ma5 || indicatorsObj.ma5.length === 0) return '';
  const n = Math.min(indicatorsObj.ma5.length, 5);
  const sliceKwi = kwi.slice(-n);
  const heads = ['日期','收盘','MA5','MA20','MA60','RSI14','MACD_DIF','K','D','J','BOLL上','BOLL中','BOLL下'];
  const lines = ['\n## 预计算技术指标（程序完成，直接引用）'];
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

// validation_warning 注入
export function buildValidationWarningText(validation) {
  if (!validation || validation.severity === 'ok') return '';
  const lines = ['\n⚠️ 本次数据健康检查发现问题：'];
  for (const issue of validation.issues.slice(0, 5)) {
    lines.push(`- [${issue.severity}] ${issue.message}`);
  }
  lines.push('请在分析时考虑以上数据异常对判断的影响。\n');
  return lines.join('\n');
}
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

注：本维度基于价格历史分位和长期均线位置做相对价值判断。分析估值面时，应主动调用 get_financials 工具获取当前 PE/PB/市值/行业等基本面指标，据此判断估值是否处于合理区间。

请用中文从长期价值视角分析：

1. **历史位置**：计算当前价格在近 N 根${periodLabel}数据中的位置百分位（公式同上），明确给出百分位数字。

2. **长期趋势分期**：把近 N 根${periodLabel}数据分为"上涨段/下跌段/横盘段"，标注每段的起止日期和涨跌幅。判断当前处于哪个阶段。

3. **极端位置信号**：检查当前价是否接近历史极值（最高/最低 ±10% 内、长期均线支撑位等），给出具体偏离度。

4. **均线长期排列**：MA5/MA20/MA60 长期排列状态演变（最近 3 次交叉日期和方向），判断均线系统是否健康。

5. **估值阶段定性**：综合价格历史分位、均线位置和工具获取的 PE/PB 等基本面数据，在【底部修复 / 价值回归中段 / 接近合理估值 / 高估区间】中选一个。所有判断必须引用具体价格和分位数字。

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

2. **换手率分析**（若提供换手率数据）：近 N 根${periodLabel}换手率的均值、标准差、当前值所处百分位。判断市场参与度是上升/下降/稳定。若换手率数据缺失，跳过此项，并在分析末尾标注"换手率数据缺失，情绪面分析降级"。

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

// 可用工具说明（仅 Anthropic provider 启用 tool_use 时附加到 prompt）
export function buildToolInstructions(secid) {
  return `
## 可用工具（仅 Anthropic provider）

你可以使用以下工具补充分析所需数据。当你需要表格中未包含的信息时，主动调用工具获取，而不是凭记忆补全。

### get_financials
获取当前 PE/PB/市值/行业等财务指标。
- 参数：\`secid\` = \`"${secid}"\`
- 估值面模板**鼓励调用**本工具补足基本面数据

### get_money_flow
获取近 N 月主力资金流向。
- 参数：\`secid\` = \`"${secid}"\`, \`limit\` = 月数（默认 12）
- 情绪面模板**鼓励调用**本工具补足资金维度

使用原则：
1. 仅在分析需要这些数据但 K 线表格中没有时调用
2. 技术面和趋势模板按需调用，避免过度调用
3. 调用后将工具返回的真实数据写入分析，替代凭记忆补全
4. 本股票的 secid 是 \`"${secid}"\`，调用工具时直接使用这个值`;
}

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