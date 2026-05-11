// 各 provider 各模型的定价表（2026-04 官方价格）
// 单位统一:元人民币 / 1M token
// 汇率取稳定值 7.2（UI 估算用，非精确值）
const USD_TO_CNY = 7.2;

export const PRICING = {
  anthropic: {
    'claude-sonnet-4-6':   { input:  3.0 * USD_TO_CNY, output: 15.0 * USD_TO_CNY },
    'claude-opus-4-7':     { input: 15.0 * USD_TO_CNY, output: 75.0 * USD_TO_CNY },
    // 未匹配模型回退到 sonnet 价格
  },
  deepseek: {
    'deepseek-chat':       { input:  1.0, output:  4.0 },   // V3.2 通用
    'deepseek-reasoner':   { input:  4.0, output: 16.0 },   // R1 推理
    // 未匹配回退到 deepseek-chat 价格
  },
};

/**
 * 估算一次调用的成本（元人民币）
 * @param {string} providerId
 * @param {string} model
 * @param {{inputTokens: number, outputTokens: number}|null} usage
 * @returns {number}
 */
export function estimateCost(providerId, model, usage) {
  if (!usage) return 0;
  const table = PRICING[providerId];
  if (!table) return 0;
  const price = table[model] || Object.values(table)[0];
  if (!price) return 0;
  return (
    (usage.inputTokens / 1_000_000) * price.input +
    (usage.outputTokens / 1_000_000) * price.output
  );
}
