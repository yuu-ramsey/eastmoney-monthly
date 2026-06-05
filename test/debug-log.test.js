// debug:lastAnalysis record test
// 只能在 Node 测试框架中模拟 chrome.storage.local
import { test } from 'node:test';
import assert from 'node:assert/strict';

// Mock chrome API
const storage = new Map();
globalThis.chrome = {
  storage: {
    local: {
      get: async (keys) => {
        if (Array.isArray(keys)) {
          const result = {};
          for (const k of keys) {
            if (storage.has(k)) result[k] = storage.get(k);
          }
          return result;
        }
        return storage.has(keys) ? { [keys]: storage.get(keys) } : {};
      },
      set: async (items) => {
        for (const [k, v] of Object.entries(items)) {
          storage.set(k, v);
        }
      },
      remove: async (keys) => {
        for (const k of keys) storage.delete(k);
      },
    },
  },
  runtime: {
    sendMessage: async () => {},
  },
  alarms: {
    create: () => {},
    clear: () => {},
  },
  tabs: {
    sendMessage: async () => {},
  },
};

test('debug:lastAnalysis 写入', async () => {
  storage.clear();
  const record = {
    timestamp: Date.now(),
    code: '600522', name: '中天科技',
    template: 'valuation', provider: 'anthropic', model: 'claude-sonnet-4-6',
    settings: { enableSelfBacktest: true, enableThinking: false },
    fullPrompt: '# 分析提示词...',
    toolCalls: [{ name: 'get_financials', input: '{"secid":"1.600522"}', result: 'PE=25.3', durationMs: 234 }],
    rawResponse: '## 分析结果...',
    usage: { inputTokens: 5000, outputTokens: 1500 },
    cost: { cny: 0.05 },
    durationMs: 4500,
  };

  await chrome.storage.local.set({ 'debug:lastAnalysis': record });

  const items = await chrome.storage.local.get(['debug:lastAnalysis']);
  assert.ok(items['debug:lastAnalysis']);
  assert.equal(items['debug:lastAnalysis'].code, '600522');
  assert.equal(items['debug:lastAnalysis'].toolCalls.length, 1);
});

test('debug:lastAnalysis 覆盖旧记录', async () => {
  storage.clear();
  await chrome.storage.local.set({ 'debug:lastAnalysis': { code: 'old' } });
  await chrome.storage.local.set({ 'debug:lastAnalysis': { code: 'new' } });

  const items = await chrome.storage.local.get(['debug:lastAnalysis']);
  assert.equal(items['debug:lastAnalysis'].code, 'new');
});

test('debug:lastAnalysis 不含 API key', async () => {
  storage.clear();
  const record = {
    fullPrompt: '分析任务：...',
    rawResponse: '分析结果...',
    settings: { apiKey: undefined },
  };
  await chrome.storage.local.set({ 'debug:lastAnalysis': record });

  const items = await chrome.storage.local.get(['debug:lastAnalysis']);
  const saved = items['debug:lastAnalysis'];
  // fullPrompt 不包含任何 key 相关信息
  assert.ok(!String(saved.fullPrompt).includes('sk-ant'));
  assert.ok(!String(saved.fullPrompt).includes('sk-'));
  // settings 里没有 apiKey
  assert.equal(saved.settings.apiKey, undefined);
});

test('debug:lastAnalysis 手动清空', async () => {
  storage.clear();
  await chrome.storage.local.set({ 'debug:lastAnalysis': { code: 'test' } });
  let items = await chrome.storage.local.get(['debug:lastAnalysis']);
  assert.ok(items['debug:lastAnalysis']);

  await chrome.storage.local.remove(['debug:lastAnalysis']);
  items = await chrome.storage.local.get(['debug:lastAnalysis']);
  assert.equal(items['debug:lastAnalysis'], undefined);
});

test('debug:lastAnalysis toolCalls 为空时不报错', async () => {
  storage.clear();
  const record = {
    toolCalls: [],
    fullPrompt: 'prompt',
    rawResponse: 'response',
  };
  await chrome.storage.local.set({ 'debug:lastAnalysis': record });
  const items = await chrome.storage.local.get(['debug:lastAnalysis']);
  assert.deepEqual(items['debug:lastAnalysis'].toolCalls, []);
});
