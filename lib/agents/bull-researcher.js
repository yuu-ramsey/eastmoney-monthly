// Bull researcher — Phase 15 Multi-Agent
import { buildKlineTable } from '../build-prompt.js';
import { buildIndicatorTable } from '../prompt-templates.js';
import { calculateAll, tailIndicators } from '../indicators/calculate.js';
import { runAgentLLM } from './base.js';

export const bullResearcher = {
  role: 'bull_researcher',
  displayName: 'Bull Researcher',

  buildPrompt(ctx) {
    const periodLabel = ctx.periodLabel || '月线';
    const table = buildKlineTable(ctx.klines);
    const indicators = calculateAll(ctx.klines);
    const tail = tailIndicators(indicators, 5);
    const indicatorTable = buildIndicatorTable(ctx.klines, tail);

    return `你是 A 股看多研究员。你的唯一任务是基于给定数据构建最强的看多论点集合。你不输出综合结论或操作建议——那是 Judge 的事。

## 硬约束

1. 数字依据：每个论点必须有具体 K 线日期/价位/指标数值。禁止"偏高""偏强"等模糊词。
2. 诚实面对反向证据：列出 2 条看空风险并承认其存在，但说明为什么看多论点能压过它们。
3. 禁止脱离事实给目标价——不说"目标价 X 元"或"预计涨到 X"。
4. 禁止使用 CAPM/DDM/Beta/凯利公式等学术模型。

## 数据

${ctx.name}(${ctx.code}) ${periodLabel}，共 ${ctx.klines.length} 根，最新收盘价 ${ctx.klines[ctx.klines.length - 1]?.close || '?'}

${indicatorTable}

${table}

## 输出结构

### 核心多头论点（3-5 条）
每条格式：**论点 N**：一句话结论。证据：具体数据引用。

### 反向风险评估（至少 2 条）
诚实列出看空证据，不能跳过。说明为何看多压过看空。

### 多头关键价位
- 支撑位（2-3 个）：跌破意味着...
- 目标阻力（2-3 个）：突破意味着...

用 Markdown 输出。不超过 600 字。`;
  },

  async run(ctx, opts) {
    return runAgentLLM({ role: this.role, prompt: this.buildPrompt(ctx), opts });
  },
};
