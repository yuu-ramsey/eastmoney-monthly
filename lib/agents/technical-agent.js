// Technical analyst — Phase 15 Multi-Agent
// Pure technical: indicators + signals, no directional judgment
import { buildKlineTable } from '../build-prompt.js';
import { buildIndicatorTable } from '../prompt-templates.js';
import { calculateAll, tailIndicators } from '../indicators/calculate.js';
import { generateSignalSummary, formatSignalSummary } from '../signals/summary.js';
import { runAgentLLM } from './base.js';

export const technicalAgent = {
  role: 'technical_agent',
  displayName: 'Technical Analyst',

  buildPrompt(ctx) {
    const periodLabel = ctx.periodLabel || 'Monthly';
    const table = buildKlineTable(ctx.klines);
    const indicators = calculateAll(ctx.klines);
    const tail = tailIndicators(indicators, 5);
    const indicatorTable = buildIndicatorTable(ctx.klines, tail);
    const signalSummary = generateSignalSummary(ctx.klines, indicators);
    const signalText = formatSignalSummary(signalSummary);

    return `你是 A 股技术分析员。你的任务是纯粹的技术面解读，不做方向判断。Bull/Bear 会分别负责看多/看空论证，Judge 负责综合——你只提供技术事实清单。

## 硬约束

1. 所有陈述必须有具体数字（价位/指标值/日期）
2. 不自算指标——预计算表格已提供所有技术指标值
3. 不输出"建议买入/卖出"或方向判断（偏多/偏空）
4. 不输出综合结论

## 数据

${ctx.name}(${ctx.code}) ${periodLabel}，共 ${ctx.klines.length} 根，最新收盘价 ${ctx.klines[ctx.klines.length - 1]?.close || '?'}

${indicatorTable}

${signalText}

${table}

## 输出结构

### 均线系统状态
- MA5/MA20/MA60 排列关系（多头/空头/交叉）、各均线当前值与价格偏离%
- MA20 斜率方向（过去 5${periodLabel === '月线' ? '月' : periodLabel === '周线' ? '周' : '日'}）
- 最近一次均线交叉日期和方向

### MACD 状态
- DIF/DEA/HIST 当前值 + 近 3${periodLabel === '月线' ? '月' : periodLabel === '周线' ? '周' : '日'}变化趋势
- 是否有背离信号（必须给出具体日期和价位）

### 关键价位
- 支撑位（3-5 个）：标注来源（均线/前低/中枢）
- 阻力位（3-5 个）：标注来源（均线/前高/中枢）

### 量价特征
- 近${Math.min(5, ctx.klines.length)}根${periodLabel === '月线' ? '月线' : periodLabel}的量价关系（放量上涨/缩量下跌等）
- 换手率趋势（若有）

用 Markdown 列表输出。不超过 500 字。`;
  },

  async run(ctx, opts) {
    return runAgentLLM({ role: this.role, prompt: this.buildPrompt(ctx), opts });
  },
};
