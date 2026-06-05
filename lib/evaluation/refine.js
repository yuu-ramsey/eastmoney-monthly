// User review + Claude Opus refinement
// No automatic prompt changes; all modifications require manual user approval before Claude generates change plans

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { estimateDeepSeekCost } from './cost-guard.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_DIR = path.resolve(__dirname, '..', '..');
const REVIEW_DIR = path.join(PROJECT_DIR, '.eastmoney-ai', 'reviews');

// Anthropic 定价（USD → CNY，汇率 7.2）
const CLAUDE_PRICING = {
  'claude-opus-4-7': { input: 15 * 7.2, output: 75 * 7.2 },
  'claude-sonnet-4-6': { input: 3 * 7.2, output: 15 * 7.2 },
};

function estimateClaudeCost(model, usage) {
  const p = CLAUDE_PRICING[model] || CLAUDE_PRICING['claude-opus-4-7'];
  return (usage.inputTokens / 1_000_000) * p.input + (usage.outputTokens / 1_000_000) * p.output;
}

/**
 * Parse user review section from draft
 * @param {string} draftPath
 * @returns {{ approved: Array, rejected: Array, modified: Array }}
 */
export function parseUserReview(draftPath) {
  const content = fs.readFileSync(draftPath, 'utf-8');
  const reviewSection = content.split('## 用户审核区')[1] || '';

  const approved = [];
  const rejected = [];
  const modified = [];

  const lines = reviewSection.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('- [') && !trimmed.startsWith('* [')) continue;

    const suggestion = extractSuggestionText(trimmed);

    if (trimmed.includes('[x]') || trimmed.includes('[X]')) {
      if (trimmed.includes('通过')) approved.push(suggestion);
      else if (trimmed.includes('拒绝')) rejected.push(suggestion);
      else if (trimmed.includes('修改')) modified.push({ original: suggestion, note: extractModifyNote(trimmed) });
    }
  }

  return { approved, rejected, modified };
}

function extractSuggestionText(line) {
  const match = line.match(/建议\s*\d+[：:]\s*(.+?)(?:\s*\[.*\]\s*)?$/);
  return match ? match[1].trim() : line.replace(/^[-*]\s*\[.\]\s*/, '').replace(/\s*\[.*\]\s*$/, '').trim();
}

function extractModifyNote(line) {
  const match = line.match(/修改\s*[：:]\s*(.+)/);
  return match ? match[1].trim() : '';
}

/**
 * Claude Opus refinement
 * @param {object} opts
 * @param {Array} opts.approvedSuggestions
 * @param {string} opts.draftContent
 * @param {Function} opts.callClaude — (prompt, model, apiKey) => { text, usage }
 * @param {string} opts.apiKey
 * @param {Function} opts.recordSpending
 * @returns {{ refinedPath: string, cost: number }}
 */
export async function refineWithClaude(opts) {
  const { approvedSuggestions, draftContent, callClaude, apiKey, recordSpending } = opts;

  if (approvedSuggestions.length === 0) {
    throw new Error('没有审核通过的建议，无需精修');
  }

  const model = 'claude-opus-4-7';
  const approvedText = approvedSuggestions.map((s, i) => `${i + 1}. ${s}`).join('\n');

  const prompt = `你是一位 prompt 工程专家。以下是基于历史 evaluation 数据得出的诊断和用户审核通过的改进建议。请输出具体的代码改动：

【原始诊断】
${draftContent.slice(0, 3000)}

【用户通过的建议】
${approvedText}

请输出：
1. 具体改哪些文件、哪些行、新文本是什么（diff 风格）
2. 改动的理论依据（引用具体 evaluation 数据）
3. 改动后的预期 evaluation 变化
4. 必要的回归测试用例建议（test/*.test.js）

格式要求：
- 用 markdown，代码块标注语言
- diff 风格：- 旧行 / + 新行
- 每条改动独立成节，便于用户挑选执行`;

  const result = await callClaude(prompt, model, apiKey);
  const cost = result.usage
    ? estimateClaudeCost(model, result.usage)
    : 0;

  if (recordSpending) recordSpending(cost);

  const today = new Date().toISOString().slice(0, 10);
  const refinedPath = path.join(REVIEW_DIR, `refined-${today}.md`);

  const content = `# Claude Opus 精修报告 ${today}

> 基于草稿 draft-${today}.md 的用户审核结果
> 本次精修成本：¥${cost.toFixed(4)}

## 通过审核的建议
${approvedText}

## Claude Opus 改动方案
${result.text}
`;

  fs.writeFileSync(refinedPath, content, 'utf-8');
  return { refinedPath, cost };
}
