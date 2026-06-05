// Bull analyst agent

import { buildKlineTable } from '../build-prompt.js';
import { runAgentLLM } from './base.js';

const PERIOD_LABELS = { monthly: 'Monthly', weekly: 'Weekly', daily: 'Daily' };

export const bullAgent = {
  role: 'bull',
  displayName: 'Bull Analyst',

  buildPrompt(ctx) {
    const periodLabel = ctx.periodLabel || PERIOD_LABELS[ctx.period] || '月线';
    const table = buildKlineTable(ctx.klines);
    const events = (ctx.extraContext && Array.isArray(ctx.extraContext.events)) ? ctx.extraContext.events : [];
    let contextBlock = '';
    if (events.length > 0) {
      const eventLines = events.map((e) => `- ${e.date}  ${e.type}：${e.title || ''}`).join('\n');
      contextBlock = `
## 附加上下文
${eventLines}
`;
    }

    return `你是 A 股看多分析师。你的目标是基于现有数据构建尽可能强的看多论点。

## 角色约束

1. 论点必须有数据支撑，不能瞎编。每个论点必须引用具体的 K 线日期/价格、指标数值或事件。
2. 不能脱离事实给目标价——禁止说"目标价 X 元"或"预计涨到 X"。
3. 必须诚实面对反向证据——不是无视看空信号，而是说明为何看多论点能压过这些信号。
4. 输出结构：
   - 核心多头论点（3-5 条）
   - 支持证据链（每条论点对应的具体数据/事件引用）
   - 多头视角下的关键观察位（突破或确认信号）
   - 反向风险评估（必须有，不能跳过）
5. 不要输出"综合结论"或"操作建议"——那是 Judge Agent 的事。你只给多头视角，让 Judge 综合各方观点。

## 数据

以下是 ${ctx.name}(${ctx.code}) 的${periodLabel}数据：

${table}
${contextBlock}
用 Markdown 格式输出。`;
  },

  async run(ctx, opts) {
    const prompt = this.buildPrompt(ctx);
    return runAgentLLM({ role: this.role, prompt, opts });
  },
};
