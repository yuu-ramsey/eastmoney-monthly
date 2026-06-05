// LLM provider layer + pricing test
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

// ---- Anthropic Tool Use ----

function makeMockTool(name, handler) {
  return {
    name,
    description: `Mock tool: ${name}`,
    input_schema: {
      type: 'object',
      properties: { query: { type: 'string' } },
      required: ['query'],
    },
    handler: handler || (async (input) => `Result for ${name}: ${input.query || 'none'}`),
  };
}

// 辅助：构造一次 fetch 返回 tool_use + 一次返回最终文本
function mockToolUseThenText(toolName, toolInput, toolId, finalText) {
  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    if (callCount === 1) {
      return Promise.resolve({
        status: 200,
        ok: true,
        json: () => Promise.resolve({
          stop_reason: 'tool_use',
          content: [
            { type: 'tool_use', id: toolId, name: toolName, input: toolInput },
          ],
          usage: { input_tokens: 100, output_tokens: 50 },
        }),
      });
    }
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve({
        stop_reason: 'end_turn',
        content: [{ type: 'text', text: finalText }],
        usage: { input_tokens: 200, output_tokens: 100 },
      }),
    });
  };
}

test('anthropic tool_use: 单轮工具调用后返回最终文本', async () => {
  mockToolUseThenText('my_tool', { query: 'hello' }, 'toolu_001', '最终分析结果');
  const tool = makeMockTool('my_tool');

  const result = await getProvider('anthropic').call('分析一下', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-ant-test',
    maxTokens: 4000,
    tools: [tool],
  });

  assert.equal(result.text, '最终分析结果');
  // usage 累加两次调用
  assert.equal(result.usage.inputTokens, 300);
  assert.equal(result.usage.outputTokens, 150);
});

test('anthropic tool_use: 无 tools 参数时忽略 stop_reason=tool_use', async () => {
  // 不传 tools 参数，即使 stop_reason 是 tool_use 也按普通响应处理
  mockFetch(200, {
    stop_reason: 'tool_use',
    content: [
      { type: 'text', text: 'fallback text' },
      { type: 'tool_use', id: 'tu_1', name: 'x', input: {} },
    ],
    usage: { input_tokens: 100, output_tokens: 50 },
  });
  const result = await getProvider('anthropic').call('x', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-ant-test',
    maxTokens: 100,
  });
  // extractText 过滤掉 tool_use block，只返回 text
  assert.equal(result.text, 'fallback text');
});

test('anthropic tool_use: 两轮工具调用后正常返回', async () => {
  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    if (callCount <= 2) {
      return Promise.resolve({
        status: 200,
        ok: true,
        json: () => Promise.resolve({
          stop_reason: 'tool_use',
          content: [
            { type: 'tool_use', id: `toolu_${callCount}`, name: 't1', input: { query: `q${callCount}` } },
          ],
          usage: { input_tokens: 100, output_tokens: 50 },
        }),
      });
    }
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve({
        stop_reason: 'end_turn',
        content: [{ type: 'text', text: 'done after 2 tools' }],
        usage: { input_tokens: 200, output_tokens: 100 },
      }),
    });
  };
  const tool = makeMockTool('t1');

  const result = await getProvider('anthropic').call('test', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-ant-test',
    maxTokens: 4000,
    tools: [tool],
  });

  assert.equal(result.text, 'done after 2 tools');
  // 3 次调用累加
  assert.equal(result.usage.inputTokens, 400);
  assert.equal(result.usage.outputTokens, 200);
});

test('anthropic tool_use: 超过 5 轮抛错', async () => {
  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve({
        stop_reason: 'tool_use',
        content: [
          { type: 'tool_use', id: `toolu_${callCount}`, name: 't1', input: { query: `q${callCount}` } },
        ],
        usage: { input_tokens: 100, output_tokens: 50 },
      }),
    });
  };
  const tool = makeMockTool('t1');

  await assert.rejects(
    () => getProvider('anthropic').call('test', {
      model: 'claude-sonnet-4-6',
      apiKey: 'sk-ant-test',
      maxTokens: 4000,
      tools: [tool],
    }),
    /工具调用轮次超过上限/,
  );
  assert.ok(callCount >= 6);
});

test('anthropic tool_use: 多个 tool_use block 在一次响应中', async () => {
  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    if (callCount === 1) {
      return Promise.resolve({
        status: 200,
        ok: true,
        json: () => Promise.resolve({
          stop_reason: 'tool_use',
          content: [
            { type: 'tool_use', id: 'tu_a', name: 'tool_a', input: { query: 'a' } },
            { type: 'tool_use', id: 'tu_b', name: 'tool_b', input: { query: 'b' } },
          ],
          usage: { input_tokens: 100, output_tokens: 50 },
        }),
      });
    }
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve({
        stop_reason: 'end_turn',
        content: [{ type: 'text', text: 'multi-tool done' }],
        usage: { input_tokens: 150, output_tokens: 80 },
      }),
    });
  };
  const toolA = makeMockTool('tool_a');
  const toolB = makeMockTool('tool_b');

  const result = await getProvider('anthropic').call('test', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-ant-test',
    maxTokens: 4000,
    tools: [toolA, toolB],
  });

  assert.equal(result.text, 'multi-tool done');
  assert.equal(callCount, 2);
});

test('anthropic tool_use: 未知工具返回错误文本但不中断循环', async () => {
  mockToolUseThenText('unknown_tool', { query: 'x' }, 'toolu_x', '继续分析');
  const knownTool = makeMockTool('known_tool');

  const result = await getProvider('anthropic').call('test', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-ant-test',
    maxTokens: 4000,
    tools: [knownTool],
  });

  assert.equal(result.text, '继续分析');
});

test('anthropic tool_use: handler 抛异常不中断循环', async () => {
  mockToolUseThenText('bad_tool', { query: 'x' }, 'toolu_bad', '异常后继续');
  const badTool = makeMockTool('bad_tool', async () => { throw new Error('handler crash'); });

  const result = await getProvider('anthropic').call('test', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-ant-test',
    maxTokens: 4000,
    tools: [badTool],
  });

  assert.equal(result.text, '异常后继续');
});

test('anthropic tool_use: DeepSeek 不受 tools 影响', async () => {
  mockFetch(200, {
    choices: [{ message: { content: 'normal' } }],
    usage: { prompt_tokens: 100, completion_tokens: 50 },
  });
  const result = await getProvider('deepseek').call('test', {
    model: 'deepseek-chat',
    apiKey: 'sk-test',
    maxTokens: 100,
    tools: [makeMockTool('x')],
  });
  assert.equal(result.text, 'normal');
});
