import { test } from 'node:test';
import assert from 'node:assert/strict';
import { estimateCost } from '../lib/llm/pricing.js';

test('estimateCost: Anthropic sonnet correct calculation', () => {
  const cost = estimateCost('anthropic', 'claude-sonnet-4-6', {
    inputTokens: 1_000_000,
    outputTokens: 1_000_000,
  });
  // input: 3.0 * 7.2 = 21.6, output: 15.0 * 7.2 = 108
  assert.ok(Math.abs(cost - (21.6 + 108)) < 0.001, `expected 129.6, got ${cost}`);
});

test('estimateCost: DeepSeek chat correct calculation', () => {
  const cost = estimateCost('deepseek', 'deepseek-chat', {
    inputTokens: 1_000_000,
    outputTokens: 500_000,
  });
  assert.equal(cost, 1.0 + 2.0); // input 1.0/M + output 4.0/M*0.5 = 1+2 = 3
});

test('estimateCost: unknown model falls back to provider first price', () => {
  const cost = estimateCost('anthropic', 'unknown-model-xyz', {
    inputTokens: 1_000_000,
    outputTokens: 1_000_000,
  });
  // falls back to claude-sonnet-4-6
  assert.ok(cost > 0);
});

test('estimateCost: usage is null returns 0', () => {
  assert.equal(estimateCost('anthropic', 'claude-sonnet-4-6', null), 0);
});

test('estimateCost: inputTokens=0 does not throw', () => {
  const cost = estimateCost('deepseek', 'deepseek-chat', {
    inputTokens: 0,
    outputTokens: 1000,
  });
  assert.equal(cost, 4.0 * 1000 / 1_000_000);
});

test('estimateCost: outputTokens=0 does not throw', () => {
  const cost = estimateCost('deepseek', 'deepseek-chat', {
    inputTokens: 5000,
    outputTokens: 0,
  });
  assert.equal(cost, 1.0 * 5000 / 1_000_000);
});

test('estimateCost: unknown provider returns 0', () => {
  assert.equal(estimateCost('unknown', 'm', { inputTokens: 100, outputTokens: 100 }), 0);
});
