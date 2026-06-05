// Bear researcher — Phase 15 Multi-Agent
import { buildKlineTable } from '../build-prompt.js';
import { buildIndicatorTable } from '../prompt-templates.js';
import { calculateAll, tailIndicators } from '../indicators/calculate.js';
import { runAgentLLM } from './base.js';

export const bearResearcher = {
  role: 'bear_researcher',
  displayName: 'Bear Researcher',

  buildPrompt(ctx) {
    const periodLabel = ctx.periodLabel || 'Monthly';
    const table = buildKlineTable(ctx.klines);
    const indicators = calculateAll(ctx.klines);
    const tail = tailIndicators(indicators, 5);
    const indicatorTable = buildIndicatorTable(ctx.klines, tail);

    return `你是 A 股看空研究员。你的唯一任务是基于给定数据构建最强的看空论点集合。你不输出综合结论或操作建议——那是 Judge 的事。

## 硬约束

1. 数字依据：每个论点必须有具体 K 线日期/价位/指标数值。禁止"偏低""偏弱"等模糊词。
2. 诚实面对反向证据：列出 2 条看多证据并承认其存在，但说明为什么看空论点能压过它们。
3. 禁止脱离事实给跌幅预测——不说"预计跌到 X 元"或"跌幅 X%"。
4. 禁止使用 CAPM/DDM/Beta/凯利公式等学术模型。

## 数据

${ctx.name}(${ctx.code}) ${periodLabel}，共 ${ctx.klines.length} 根，最新收盘价 ${ctx.klines[ctx.klines.length - 1]?.close || '?'}

${indicatorTable}

${table}

## 输出结构

### 核心空头论点（3-5 条）
每条格式：**论点 N**：一句话结论。证据：具体数据引用。

### 反向风险评估（至少 2 条）
诚实列出看多证据，不能跳过。说明为何看空压过看多。

### 空头关键价位
- 阻力位（2-3 个）：突破意味着...
- 目标支撑（2-3 个）：跌破意味着...

用 Markdown 输出。不超过 600 字。`;
  },

  async run(ctx, opts) {
    return runAgentLLM({ role: this.role, prompt: this.buildPrompt(ctx), opts });
  },
};
