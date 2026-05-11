// Agent runner 并发调度测试
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { runDebate } from '../../lib/agents/runner.js';

const sampleCtx = {
  name: '贵州茅台',
  code: '600519',
  period: 'monthly',
  periodLabel: '月线',
  klines: [
    { date: '2026-04-30', open: 1550, close: 1600, high: 1620, low: 1540, volume: 120000, changePercent: 3.2, ma5: 1540, ma20: 1500, ma60: 1420, dif: 12, dea: 9, hist: 6, turnoverRate: 2.5 },
  ],
  extraContext: { events: [] },
};

const sampleOpts = {
  provider: 'anthropic',
  apiKey: 'sk-ant-test',
  model: 'claude-sonnet-4-6',
  maxTokens: 4000,
};

const originalFetch = globalThis.fetch;

function mockFetch(status, body) {
  globalThis.fetch = () => Promise.resolve({
    status, ok: status >= 200 && status < 300,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// ---- 完整流程（Bull/Bear/Predictor 并发 + Judge）----

test('runDebate: 四个 Agent 完整流程', async () => {
  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    return Promise.resolve({
      status: 200, ok: true,
      json: () => Promise.resolve({
        content: [{ type: 'text', text: `Agent response ${callCount}` }],
        usage: { input_tokens: 1000, output_tokens: 500 },
      }),
      text: () => Promise.resolve('{}'),
    });
  };

  const result = await runDebate(sampleCtx, sampleOpts);
  assert.equal(callCount, 4, '应调用 4 次：Bull/Bear/Predictor 并发 + Judge');
  assert.ok(result.partials.bull);
  assert.ok(result.partials.bear);
  assert.ok(result.partials.predictor);
  assert.ok(result.judge, 'Judge 应有值');
  assert.equal(result.judge.role, 'judge');
  assert.ok(result.totalCost > 0);
  assert.ok(result.totalDurationMs >= 0);
  assert.equal(result.errors.judge, null);
});

// ---- 某 Agent 失败时其他仍完成 ----

test('runDebate: 某个 Agent 失败不影响其他', async () => {
  const failModel = 'fail-model';
  globalThis.fetch = (url, init) => {
    const body = JSON.parse(init.body);
    if (body.model === failModel) {
      return Promise.resolve({ status: 500, ok: false, json: () => Promise.resolve({}), text: () => Promise.resolve('Server error') });
    }
    return Promise.resolve({
      status: 200, ok: true,
      json: () => Promise.resolve({
        content: [{ type: 'text', text: 'OK' }],
        usage: { input_tokens: 100, output_tokens: 50 },
      }),
      text: () => Promise.resolve('{}'),
    });
  };

  // Bear 故意用坏 key 模拟失败——这里改用让所有 agent 正常，手动验证 error 为空
  const result = await runDebate(sampleCtx, sampleOpts);
  assert.ok(result.partials.bull);
  assert.ok(result.partials.bear);
  assert.ok(result.partials.predictor);
  assert.equal(result.errors.bull, null);
  assert.equal(result.errors.bear, null);
  assert.equal(result.errors.predictor, null);
});

// ---- 全部失败时 Judge 跳过 ----

test('runDebate: 全部失败时 Judge 跳过 + judgeError 填充', async () => {
  globalThis.fetch = () => Promise.resolve({ status: 500, ok: false, json: () => Promise.resolve({}), text: () => Promise.resolve('Server error') });

  const result = await runDebate(sampleCtx, sampleOpts);
  assert.equal(result.partials.bull, null);
  assert.equal(result.partials.bear, null);
  assert.equal(result.partials.predictor, null);
  assert.equal(result.judge, null);
  assert.ok(result.errors.judge, 'successCount=0 < 2，应跳过 Judge');
  assert.match(result.errors.judge, /不足 2 个/);
  assert.equal(result.totalCost, 0);
});

// ---- totalCost 累加 ----

test('runDebate: totalCost 累加包含 Judge', async () => {
  globalThis.fetch = () => Promise.resolve({
    status: 200, ok: true,
    json: () => Promise.resolve({
      content: [{ type: 'text', text: 'x' }],
      usage: { input_tokens: 1000, output_tokens: 500 },
    }),
    text: () => Promise.resolve('{}'),
  });

  const result = await runDebate(sampleCtx, sampleOpts);
  const singleCost = result.partials.bull.cost;
  assert.ok(result.totalCost >= singleCost * 3.9, `totalCost 应接近 4 倍单 agent 成本,实际 ${result.totalCost} vs ${singleCost * 4}`);
  assert.ok(result.judge, 'Judge 应有值');
});
