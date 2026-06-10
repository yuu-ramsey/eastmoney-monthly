// Synthesis judge agent

import { runAgentLLM } from './base.js';
import { PERSONAL_DECISION_BLOCK } from '../build-prompt.js';

export const judgeAgent = {
  role: 'judge',
  displayName: '综合裁判',

  buildPrompt(ctx) {
    const periodLabel = ctx.periodLabel || '月线';
    const latestKline = ctx.klines && ctx.klines.length > 0 ? ctx.klines[ctx.klines.length - 1] : null;
    const latestClose = latestKline ? latestKline.close : '未知';

    // Append context summary
    const events = (ctx.extraContext && Array.isArray(ctx.extraContext.events)) ? ctx.extraContext.events : [];
    let contextSummary = '';
    if (events.length > 0) {
      const eventLines = events.slice(0, 10).map((e) => `- ${e.date} ${e.type}：${e.title || ''}`).join('\n');
      contextSummary = `\n大事提醒（最近 ${events.length} 条）：\n${eventLines}`;
    }

    const bullText = ctx.partials?.bull?.text || '【Bull Agent 调用失败】';
    const bearText = ctx.partials?.bear?.text || '【Bear Agent 调用失败】';
    const predictorText = ctx.partials?.predictor?.text || '【Predictor Agent 调用失败】';

    let prompt = `你是综合裁判 Agent，负责仲裁三方分析的输出，给出最终的综合判断。

[Bull Agent 输出]
${bullText}

[Bear Agent 输出]
${bearText}

[Predictor Agent 输出]
${predictorText}

[基础事实参考]
股票：${ctx.name}(${ctx.code})
周期：${periodLabel}
最新收盘价：${latestClose}${contextSummary}

## 你的任务

作为裁判，你不重新做技术分析。你的任务是评估和综合 Bull/Bear/Predictor 三方的输出，得出一个高质量的最终报告。具体要求：

1. **论点对比**：
   - 列出 Bull 和 Bear 各自最强的 2-3 条论点
   - 不要简单复述，要点出每条论点的核心数据依据

2. **论点扎实度评估**：
   - 哪一方的论点更扎实？标准是：数据更硬（具体数字 vs 模糊描述）、逻辑链更通顺、反向风险评估更诚实
   - 不要做"和稀泥"的中性评估。如果一方明显更扎实，明确指出
   - 但也要诚实标注"这是基于当前数据下的判断，不能作为投资决策依据"

3. **综合方向判断**：
   在【偏多 / 偏空 / 中性 / 信号不一致】中明确选一个，给出依据
   - "中性" 适用于 Bull 和 Bear 论点都不弱、互相抵消的场景
   - "信号不一致" 适用于不同维度信号矛盾（如基本面强但技术弱）
   - 不要选"中性"作为偷懒选项——只在真正势均力敌时使用

4. **关键观察价位**：
   - 整合 Predictor 的价位识别 + Bull/Bear 提到的关键位
   - 优先列 Predictor 标注为"强"的价位
   - 给每个价位附上"如果触及，意味着什么"

5. **风险声明**：
   > 本分析仅供研究学习使用，不构成投资建议。市场存在不可预测因素，任何结论都需结合个人风险承受能力独立判断。

## 输出禁止项

- 不要重新做笔/中枢/MACD 的技术分析（三方已做过，重复浪费 token）
- 不要给出"建议买入/卖出/持有"
- 不要使用"建议加仓"、"建议减仓"、"建议止损"等操作性语言
- 不要给出具体仓位百分比建议
- 不要试图"调和"明显矛盾的观点——分歧本身就是有价值的信息

## 输出结构

用 Markdown，约 800-1200 字。结构如下：

# ${ctx.name}(${ctx.code}) 综合裁判报告

## 多空双方核心论点对比
## 论点扎实度评估
## 综合方向判断
## 关键观察价位
## 风险声明`;

    if (ctx.decisionMode) {
      prompt += PERSONAL_DECISION_BLOCK;
    }

    return prompt;
  },

  async run(ctx, opts) {
    const prompt = this.buildPrompt(ctx);
    return runAgentLLM({ role: this.role, prompt, opts });
  },
};
