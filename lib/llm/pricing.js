// Pricing table by provider and model (2026-04 official prices)
// Common unit: CNY / 1M tokens
// Exchange rate fixed at 7.2 (for UI estimates, not precise)
const USD_TO_CNY = 7.2;

export const PRICING = {
  anthropic: {
    'claude-sonnet-4-6':   { input:  3.0 * USD_TO_CNY, output: 15.0 * USD_TO_CNY },
    'claude-opus-4-7':     { input: 15.0 * USD_TO_CNY, output: 75.0 * USD_TO_CNY },
    // Unmatched models fall back to sonnet pricing
  },
  deepseek: {
    'deepseek-chat':       { input:  1.0, output:  4.0 },   // V3.2 general
    'deepseek-reasoner':   { input:  4.0, output: 16.0 },   // R1 reasoning
    // Unmatched models fall back to deepseek-chat pricing
  },
};

/**
 * Estimate cost of a single API call (CNY)
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
