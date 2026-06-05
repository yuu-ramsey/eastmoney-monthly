// Sector analyst — Phase 15 Multi-Agent
// Displays sector alpha data, no directional constraints, pure information layer
import { buildSectorAlphaBlock } from '../prompt-templates.js';
import { runAgentLLM } from './base.js';

export const sectorAgent = {
  role: 'sector_agent',
  displayName: 'Sector Analyst',

  buildPrompt(ctx) {
    const periodLabel = ctx.periodLabel || '月线';
    const sectorBlock = ctx.sectorAlphaData ? buildSectorAlphaBlock(ctx.sectorAlphaData) : '(无行业数据)';

    return `你是 A 股行业分析员。你的任务是在行业背景下解读个股的相对位置，不做方向判断。

## 硬约束

1. 引用行业 alpha 数据时，必须使用原文的具体数字
2. 行业 alpha 是 hs300 内部基准（相对同行业成分股），不是 vs 申万全市场
3. 不输出"建议买入/卖出"或方向判断（偏多/偏空）
4. 行业 alpha 仅展示个股的行业内相对强弱，不直接指导操作方向
5. 不输出综合结论

## 数据

${ctx.name}(${ctx.code}) ${periodLabel}，最新收盘价 ${ctx.klines[ctx.klines.length - 1]?.close || '?'}

${sectorBlock}

## 输出结构

### 行业位置
- 个股在行业内的相对强弱（引用 alpha 等级和具体数值）
- 行业排名解读

### 行业背景参考
- 行业基准涨跌幅的解读
- 若 alpha 绝对值大（|hs300_sector_alpha|≥10pp），说明可能原因（不保证因果）

### 行业内对比注意事项
- 指出该信息的时间窗口限制（12 月 lookback）
- 说明行业 alpha 不代表未来方向（仅反映过去 12 月差异）

用 Markdown 输出。不超过 300 字。`;
  },

  async run(ctx, opts) {
    return runAgentLLM({ role: this.role, prompt: this.buildPrompt(ctx), opts });
  },
};
