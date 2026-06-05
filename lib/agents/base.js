// Agent abstraction layer: interface definitions + LLM call factory

import { getProvider } from '../llm/index.js';
import { estimateCost } from '../llm/pricing.js';

/**
 * @typedef {Object} AgentContext
 * @property {string} name         - Stock name
 * @property {string} code
 * @property {string} period       - monthly/weekly/daily
 * @property {string} periodLabel  - Period label (Monthly/Weekly/Daily)
 * @property {Array}  klines       - klinesWithMA 数组
 * @property {Object} extraContext - 资金流 + 事件
 *
 * @typedef {Object} AgentResult
 * @property {string} role         - 'bull' | 'bear' | 'predictor' | 'judge'
 * @property {string} text
 * @property {Object|null} usage   - {inputTokens, outputTokens}
 * @property {number} cost
 * @property {number} durationMs
 *
 * @typedef {Object} Agent
 * @property {string} role
 * @property {string} displayName
 * @property {(ctx: AgentContext) => string} buildPrompt
 */

/**
 * Agent 通用 LLM 调用工厂
 */
export async function runAgentLLM({ role, prompt, opts }) {
  const startTime = Date.now();
  const provider = getProvider(opts.provider);
  const result = await provider.call(prompt, {
    model: opts.model,
    apiKey: opts.apiKey,
    maxTokens: opts.maxTokens,
  });
  const cost = result.usage ? estimateCost(opts.provider, opts.model, result.usage) : 0;
  return {
    role,
    text: result.text,
    usage: result.usage || null,
    cost,
    durationMs: Date.now() - startTime,
  };
}
