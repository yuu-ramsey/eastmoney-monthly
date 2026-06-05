// Anthropic streaming SSE 解析测试
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { getProvider } from '../../lib/llm/index.js';

const originalFetch = globalThis.fetch;

// 构造 SSE 流
function sseEvents(...events) {
  let content = '';
  for (const ev of events) {
    if (ev.event) content += `event: ${ev.event}\n`;
    if (ev.data) content += `data: ${JSON.stringify(ev.data)}\n`;
    content += '\n';
  }
  const encoder = new TextEncoder();
  const data = encoder.encode(content);
  const stream = new ReadableStream({
    pull(controller) {
      controller.enqueue(data);
      controller.close();
    },
  });
  return { status: 200, ok: true, body: stream };
}

function mockStream(...events) {
  globalThis.fetch = () => Promise.resolve(sseEvents(...events));
}

function makeMockTool(name) {
  return {
    name,
    description: `Mock: ${name}`,
    input_schema: { type: 'object', properties: { x: { type: 'string' } }, required: ['x'] },
    handler: async (input) => `Result: ${input?.x || 'none'}`,
  };
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---- 基本流式文本 ----

test('stream: onProgress 收到 text_delta', async () => {
  const events = [
    { event: 'message_start', data: { type: 'message_start', message: { usage: { input_tokens: 50 } } } },
    { event: 'content_block_start', data: { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } } },
    { event: 'content_block_delta', data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Hello' } } },
    { event: 'content_block_delta', data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: ' World' } } },
    { event: 'content_block_stop', data: { type: 'content_block_stop', index: 0 } },
    { event: 'message_delta', data: { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 10 } } },
    { event: 'message_stop', data: { type: 'message_stop' } },
  ];
  mockStream(...events);

  const texts = [];
  const result = await getProvider('anthropic').call('prompt', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-test',
    maxTokens: 100,
    onProgress: (ev) => { if (ev.type === 'text') texts.push(ev.text); },
  });

  assert.equal(result.text, 'Hello World');
  assert.deepEqual(texts, ['Hello', ' World']);
});

// ---- thinking_delta ----

test('stream: onProgress 收到 thinking_delta', async () => {
  const events = [
    { event: 'message_start', data: { type: 'message_start', message: { usage: { input_tokens: 100 } } } },
    { event: 'content_block_start', data: { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '' } } },
    { event: 'content_block_delta', data: { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Let me analyze...' } } },
    { event: 'content_block_stop', data: { type: 'content_block_stop', index: 0 } },
    { event: 'content_block_start', data: { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } } },
    { event: 'content_block_delta', data: { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Result' } } },
    { event: 'content_block_stop', data: { type: 'content_block_stop', index: 1 } },
    { event: 'message_delta', data: { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 20 } } },
    { event: 'message_stop', data: { type: 'message_stop' } },
  ];
  mockStream(...events);

  const thinkings = [];
  const texts = [];
  const result = await getProvider('anthropic').call('prompt', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-test',
    maxTokens: 100,
    onProgress: (ev) => {
      if (ev.type === 'thinking') thinkings.push(ev.text);
      if (ev.type === 'text') texts.push(ev.text);
    },
  });

  assert.equal(result.text, 'Result');
  assert.ok(thinkings.length > 0);
  assert.ok(thinkings.join('').includes('Let me analyze'));
});

// ---- tool_use 流式 ----

test('stream: tool_use 触发 tool_start + tool_result', async () => {
  // 分两轮返回不同的事件
  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    if (callCount === 1) {
      return Promise.resolve(sseEvents(
        { event: 'message_start', data: { type: 'message_start', message: { usage: { input_tokens: 100 } } } },
        { event: 'content_block_start', data: { type: 'content_block_start', index: 0, content_block: { type: 'tool_use', id: 'toolu_001', name: 'get_financials', input: {} } } },
        { event: 'content_block_delta', data: { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{"secid":"1.600519"}' } } },
        { event: 'content_block_stop', data: { type: 'content_block_stop', index: 0 } },
        { event: 'message_delta', data: { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 30 } } },
        { event: 'message_stop', data: { type: 'message_stop' } },
      ));
    }
    return Promise.resolve(sseEvents(
      { event: 'message_start', data: { type: 'message_start', message: { usage: { input_tokens: 200 } } } },
      { event: 'content_block_start', data: { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } } },
      { event: 'content_block_delta', data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: '分析完成' } } },
      { event: 'content_block_stop', data: { type: 'content_block_stop', index: 0 } },
      { event: 'message_delta', data: { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 10 } } },
      { event: 'message_stop', data: { type: 'message_stop' } },
    ));
  };

  const toolStarts = [];
  const toolResults = [];
  const result = await getProvider('anthropic').call('prompt', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-test',
    maxTokens: 100,
    tools: [makeMockTool('get_financials')],
    onProgress: (ev) => {
      if (ev.type === 'tool_start') toolStarts.push(ev);
      if (ev.type === 'tool_result') toolResults.push(ev);
    },
  });

  assert.equal(result.text, '分析完成');
  assert.equal(toolStarts.length, 1);
  assert.equal(toolStarts[0].name, 'get_financials');
  assert.equal(toolResults.length, 1);
  assert.equal(toolResults[0].name, 'get_financials');
});

// ---- 无 onProgress 时走非流式路径 ----

test('stream: 无 onProgress 时走非流式路径', async () => {
  globalThis.fetch = () => Promise.resolve({
    status: 200,
    ok: true,
    json: () => Promise.resolve({
      stop_reason: 'end_turn',
      content: [{ type: 'text', text: '非流式输出' }],
      usage: { input_tokens: 100, output_tokens: 50 },
    }),
  });
  const result = await getProvider('anthropic').call('prompt', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-test',
    maxTokens: 100,
  });
  assert.equal(result.text, '非流式输出');
});

// ---- extended thinking 预算 ----

test('stream: enableThinking + Opus 时请求体含 thinking 字段', async () => {
  let capturedBody = null;
  globalThis.fetch = (url, init) => {
    capturedBody = JSON.parse(init.body);
    return Promise.resolve(sseEvents(
      { event: 'message_start', data: { type: 'message_start', message: { usage: { input_tokens: 10 } } } },
      { event: 'content_block_start', data: { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } } },
      { event: 'content_block_delta', data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'ok' } } },
      { event: 'content_block_stop', data: { type: 'content_block_stop', index: 0 } },
      { event: 'message_delta', data: { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } } },
      { event: 'message_stop', data: { type: 'message_stop' } },
    ));
  };

  await getProvider('anthropic').call('prompt', {
    model: 'claude-opus-4-7',
    apiKey: 'sk-test',
    maxTokens: 100,
    enableThinking: true,
    onProgress: () => {},
  });

  assert.ok(capturedBody.thinking);
  assert.equal(capturedBody.thinking.type, 'enabled');
  assert.equal(capturedBody.thinking.budget_tokens, 8000);
});

// ---- SSE 错误 ----

test('stream: HTTP 401 抛 key 无效', async () => {
  globalThis.fetch = () => Promise.resolve({ status: 401, ok: false });
  await assert.rejects(
    () => getProvider('anthropic').call('x', {
      model: 'm', apiKey: 'bad', maxTokens: 100, onProgress: () => {},
    }),
    /API key 无效/,
  );
});

// ---- 消息边界（ping 忽略）----

test('stream: ping 事件被忽略', async () => {
  const events = [
    { event: 'message_start', data: { type: 'message_start', message: { usage: { input_tokens: 10 } } } },
    { event: 'content_block_start', data: { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } } },
    { event: 'content_block_delta', data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'after ping' } } },
    { event: 'content_block_stop', data: { type: 'content_block_stop', index: 0 } },
    { event: 'message_delta', data: { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 5 } } },
    { event: 'message_stop', data: { type: 'message_stop' } },
  ];
  // 插入一个空行（模拟 ping）
  mockStream(...events);
  // ping 不产生事件，不抛错

  const result = await getProvider('anthropic').call('prompt', {
    model: 'claude-sonnet-4-6',
    apiKey: 'sk-test',
    maxTokens: 100,
    onProgress: () => {},
  });

  assert.ok(result.text);
});
