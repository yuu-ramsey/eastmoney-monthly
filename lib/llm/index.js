import { anthropicProvider } from './anthropic.js';
import { deepseekProvider } from './deepseek.js';

const PROVIDERS = {
  anthropic: anthropicProvider,
  deepseek: deepseekProvider,
};

export function getProvider(id) {
  const p = PROVIDERS[id];
  if (!p) throw new Error(`Unknown LLM provider: ${id}`);
  return p;
}

export function listProviders() {
  return Object.values(PROVIDERS);
}
