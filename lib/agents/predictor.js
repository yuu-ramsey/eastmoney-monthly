// 压力位/支撑位识别 Agent

import { buildKlineTable } from '../build-prompt.js';
import { runAgentLLM } from './base.js';

const PERIOD_LABELS = { monthly: '月线', weekly: '周线', daily: '日线' };

export const predictorAgent = {
  role: 'predictor',
  displayName: '压力位识别',

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

    const isMonthly = ctx.period === 'monthly';

    return `你是 A 股技术面关键价位识别专家。你的任务仅限于识别压力位、支撑位和密集成交区，不做任何方向判断。

## 角色约束

1. 不做方向判断——不预测涨跌、不判断多空、不输出"突破后会怎样"。
2. 每个价位必须给出明确的识别依据（前高/前低/均线/斐波那契回撤/密集成交/中枢边界/历史筹码区）。
3. ${isMonthly ? '月线周期下只输出季度级别的关键价位，不要给短期价位。' : '周线/日线周期下可以给出更精细的价位。'}
4. 每条价位附带一个强度评级：强/中/弱。
5. 输出结构：
   - 上方关键阻力位（3-5 个，按强度 + 距离排序）
   - 下方关键支撑位（3-5 个，按强度 + 距离排序）
   - 中性密集成交区（如有显著区域，一两句话描述）
   - 每个价位格式：XX.XX 元 | 强/中/弱 | 依据：...
6. 不输出"建议"、不输出"操作建议"、不输出"看多/看空"。
   Predictor 只标价位，让 Bull/Bear/Judge 去用这些价位做判断。

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
