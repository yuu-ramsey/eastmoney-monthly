// Synthesis judge — Phase 15 Multi-Agent
// Reads Bull/Bear/Technical/Sector four agent outputs → comprehensive judgment
import { runAgentLLM } from './base.js';

export const judgeAgent = {
  role: 'judge',
  displayName: 'Synthesis Judge',

  buildPrompt(ctx) {
    const periodLabel = ctx.periodLabel || '月线';
    const latestClose = ctx.klines?.[ctx.klines.length - 1]?.close || '?';

    const bull = ctx.partials?.bull_researcher?.text || '【Bull 调用失败】';
    const bear = ctx.partials?.bear_researcher?.text || '【Bear 调用失败】';
    const tech = ctx.partials?.technical_agent?.text || '【Technical 调用失败】';
    const sector = ctx.partials?.sector_agent?.text || '【Sector 调用失败】';

    return `你是综合裁判。阅读以下四个研究员的输出，给出最终综合判断。

## 四个研究员的输出

### Bull Researcher（看多研究员）
${bull}

### Bear Researcher（看空研究员）
${bear}

### Technical Agent（技术分析员）
${tech}

### Sector Agent（行业分析员）
${sector}

## 基础事实
股票：${ctx.name}(${ctx.code}) | ${periodLabel} | 最新收盘价：${latestClose}

## 你的任务

1. **多空论点对比**：Bull 和 Bear 各自最强的 2 条论点，标注每条的数据支撑强度（强/中/弱）
2. **技术面一致性检查**：Technical Agent 的均线/MACD/量价特征是否与 Bull/Bear 某一方更吻合？
3. **行业背景检查**：Sector Agent 的行业 alpha 数据是否与 Bull/Bear 某一方的论点冲突？
4. **综合方向判断**：在【偏多 / 偏空 / 中性 / 信号不一致】中明确选一个
   - "中性"只能用于 Bull 和 Bear 论点都不弱且完全抵消时
   - "信号不一致"用于多空逻辑链均成立但方向冲突
5. **关键价位总结**：整合各 agent 提到的价位，标注来源

## 输出格式

### 多空论点对比
### 技术面一致性
### 行业背景
### 综合方向判断
### 关键价位
### 风险声明

并在末尾追加结构化 JSON：
\`\`\`json
{ "signal": "bull|bear|neutral|strong_bull|strong_bear", "confidence": "high|medium|low", "one_line_summary": "20 字以内" }
\`\`\`

## 禁止

- 不重新做技术分析（Technical Agent 已做）
- 不直接抄某个 agent 的结论——必须有自己的对比和综合逻辑
- 不输出操作建议`;
  },

  async run(ctx, opts) {
    return runAgentLLM({ role: this.role, prompt: this.buildPrompt(ctx), opts });
  },
};
