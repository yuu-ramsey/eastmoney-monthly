// LLM provider 层 + 定价测试
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { getProvider, listProviders } from '../lib/llm/index.js';
import { estimateCost, PRICING } from '../lib/llm/pricing.js';

// ---- provider 注册 ----

test('getProvider: 返回正确实例', () => {
  const ap = getProvider('anthropic');
  assert.equal(ap.id, 'anthropic');
  assert.equal(ap.displayName, 'Anthropic Claude');
  assert.equal(typeof ap.call, 'function');

  const dp = getProvider('deepseek');
  assert.equal(dp.id, 'deepseek');
  assert.equal(dp.displayName, 'DeepSeek');
  assert.equal(typeof dp.call, 'function');
});

test('getProvider: 未知 id 抛错', () => {
  assert.throws(() => getProvider('openai'), /未知的 LLM 提供商/);
  assert.throws(() => getProvider(''), /未知的 LLM 提供商/);
});

test('listProviders: 返回数组长度为 2', () => {
  const list = listProviders();
  assert.equal(list.length, 2);
  const ids = list.map((p) => p.id).sort();
  assert.deepEqual(ids, ['anthropic', 'deepseek']);
});

// ---- mock fetch 工具 ----

const originalFetch = globalThis.fetch;
let mockResponse = null;

function mockFetch(status, body) {
  mockResponse = { status, ok: status >= 200 && status < 300, json: () => Promise.resolve(body), text: () => Promise.resolve(JSON.stringify(body)) };
  globalThis.fetch = () => Promise.resolve(mockResponse);
}

function mockFetchThrow() {
  globalThis.fetch = () => Promise.reject(new Error('fetch failed'));
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---- Anthropic call ----

test('anthropic.call: 正常响应返回 text + usage', async () => {
  mockFetch(200, {
    content: [{ type: 'text', text: '这是一份技术分析报告' }],
    usage: { input_tokens: 5000, output_tokens: 1500 },
  });
  const provider = getProvider('anthropic');
  const result = await provider.call('prompt text', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-ant-test',
    maxTokens: 4000,
  });
  assert.equal(result.text, '这是一份技术分析报告');
  assert.equal(result.usage.inputTokens, 5000);
  assert.equal(result.usage.outputTokens, 1500);
});

test('anthropic.call: 401 抛 key 无效', async () => {
  mockFetch(401, { error: { message: 'invalid' } });
  const provider = getProvider('anthropic');
  await assert.rejects(
    () => provider.call('x', { model: 'm', apiKey: 'bad', maxTokens: 100 }),
    /API key 无效/,
  );
});

test('anthropic.call: 429 抛限流', async () => {
  mockFetch(429, {});
  const provider = getProvider('anthropic');
  await assert.rejects(
    () => provider.call('x', { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /触发 Anthropic Claude 限流/,
  );
});

test('anthropic.call: 500 抛服务过载', async () => {
  mockFetch(500, {});
  const provider = getProvider('anthropic');
  await assert.rejects(
    () => provider.call('x', { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /服务过载/,
  );
});

test('anthropic.call: fetch 抛错时返回网络错误', async () => {
  mockFetchThrow();
  const provider = getProvider('anthropic');
  await assert.rejects(
    () => provider.call('x', { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /网络错误/,
  );
});

test('anthropic.call: 响应无 usage 字段返回 0', async () => {
  mockFetch(200, {
    content: [{ type: 'text', text: 'OK' }],
    usage: { input_tokens: 0, output_tokens: 0 },
  });
  const provider = getProvider('anthropic');
  const result = await provider.call('x', { model: 'm', apiKey: 'k', maxTokens: 100 });
  assert.equal(result.usage.inputTokens, 0);
  assert.equal(result.usage.outputTokens, 0);
});

test('anthropic.call: 非法类型抛 TypeError', async () => {
  const provider = getProvider('anthropic');
  await assert.rejects(
    () => provider.call(123, { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /TypeError.*string 或 messages 数组/,
  );
  await assert.rejects(
    () => provider.call({ foo: 'bar' }, { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /TypeError.*string 或 messages 数组/,
  );
  await assert.rejects(
    () => provider.call(null, { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /TypeError.*string 或 messages 数组/,
  );
});

test('anthropic.call: messages 数组模式', async () => {
  mockFetch(200, {
    content: [{ type: 'text', text: '多轮回复' }],
    usage: { input_tokens: 6000, output_tokens: 2000 },
  });
  const provider = getProvider('anthropic');
  const result = await provider.call([
    { role: 'user', content: '第一问' },
    { role: 'assistant', content: '第一答' },
    { role: 'user', content: '追问' },
  ], { model: 'claude-sonnet-4-6', apiKey: 'sk-ant-test', maxTokens: 4000 });
  assert.equal(result.text, '多轮回复');
  assert.equal(result.usage.inputTokens, 6000);
});

// ---- DeepSeek call ----

test('deepseek.call: 正常响应返回 text + usage', async () => {
  mockFetch(200, {
    choices: [{ message: { content: 'DeepSeek 分析结果' } }],
    usage: { prompt_tokens: 3000, completion_tokens: 800 },
  });
  const provider = getProvider('deepseek');
  const result = await provider.call('prompt', {
    model: 'deepseek-chat',
    apiKey: 'sk-test',
    maxTokens: 4000,
  });
  assert.equal(result.text, 'DeepSeek 分析结果');
  assert.equal(result.usage.inputTokens, 3000);
  assert.equal(result.usage.outputTokens, 800);
});

test('deepseek.call: 401 抛 key 无效', async () => {
  mockFetch(401, { error: { message: 'invalid api key' } });
  const provider = getProvider('deepseek');
  await assert.rejects(
    () => provider.call('x', { model: 'm', apiKey: 'bad', maxTokens: 100 }),
    /API key 无效/,
  );
});

test('deepseek.call: 429 抛限流', async () => {
  mockFetch(429, {});
  const provider = getProvider('deepseek');
  await assert.rejects(
    () => provider.call('x', { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /触发 DeepSeek 限流/,
  );
});

test('deepseek.call: 500 抛服务过载', async () => {
  mockFetch(500, {});
  const provider = getProvider('deepseek');
  await assert.rejects(
    () => provider.call('x', { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /服务过载/,
  );
});

test('deepseek.call: fetch 抛错时返回网络错误', async () => {
  mockFetchThrow();
  const provider = getProvider('deepseek');
  await assert.rejects(
    () => provider.call('x', { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /网络错误/,
  );
});

test('deepseek.call: 响应无 usage 返回 0', async () => {
  mockFetch(200, {
    choices: [{ message: { content: 'OK' } }],
  });
  const provider = getProvider('deepseek');
  const result = await provider.call('x', { model: 'm', apiKey: 'k', maxTokens: 100 });
  assert.equal(result.usage.inputTokens, 0);
  assert.equal(result.usage.outputTokens, 0);
});

test('deepseek.call: 非法类型抛 TypeError', async () => {
  const provider = getProvider('deepseek');
  await assert.rejects(
    () => provider.call(456, { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /TypeError.*string 或 messages 数组/,
  );
  await assert.rejects(
    () => provider.call(true, { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /TypeError.*string 或 messages 数组/,
  );
  await assert.rejects(
    () => provider.call(null, { model: 'm', apiKey: 'k', maxTokens: 100 }),
    /TypeError.*string 或 messages 数组/,
  );
});

test('deepseek.call: messages 数组模式', async () => {
  mockFetch(200, {
    choices: [{ message: { content: 'DeepSeek 多轮回复' } }],
    usage: { prompt_tokens: 4000, completion_tokens: 1000 },
  });
  const provider = getProvider('deepseek');
  const result = await provider.call([
    { role: 'user', content: '问' },
    { role: 'assistant', content: '答' },
    { role: 'user', content: '再问' },
  ], { model: 'deepseek-chat', apiKey: 'sk-test', maxTokens: 4000 });
  assert.equal(result.text, 'DeepSeek 多轮回复');
  assert.equal(result.usage.inputTokens, 4000);
});

// ---- 两家 usage 结构一致 ----

test('两家 provider 返回的 usage.inputTokens / outputTokens 字段一致', async () => {
  mockFetch(200, {
    content: [{ type: 'text', text: 'A' }],
    usage: { input_tokens: 100, output_tokens: 200 },
  });
  const a = await getProvider('anthropic').call('x', { model: 'm', apiKey: 'k', maxTokens: 100 });
  assert.ok(typeof a.usage.inputTokens === 'number');
  assert.ok(typeof a.usage.outputTokens === 'number');

  mockFetch(200, {
    choices: [{ message: { content: 'D' } }],
    usage: { prompt_tokens: 300, completion_tokens: 400 },
  });
  const d = await getProvider('deepseek').call('x', { model: 'm', apiKey: 'k', maxTokens: 100 });
  assert.ok(typeof d.usage.inputTokens === 'number');
  assert.ok(typeof d.usage.outputTokens === 'number');
});
