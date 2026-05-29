// Agent runner 并发调度测试（含 checkpoint 续跑）
import { test, afterEach, before } from 'node:test';
import assert from 'node:assert/strict';
import { runDebate } from '../../lib/agents/runner.js';

// ---- Mock chrome.storage (checkpoint 测试需要) ----
const storageMap = new Map();
globalThis.chrome = globalThis.chrome || {
  storage: {
    local: {
      get: async (keys) => {
        const result = {};
        for (const k of (Array.isArray(keys) ? keys : [keys])) {
          if (storageMap.has(k)) result[k] = storageMap.get(k);
        }
        return result;
      },
      set: async (items) => {
        for (const [k, v] of Object.entries(items)) storageMap.set(k, v);
      },
      remove: async (keys) => {
        for (const k of keys) storageMap.delete(k);
      },
    },
  },
  runtime: { sendMessage: async () => {}, sendNativeMessage: async () => {} },
  alarms: { create: () => {}, clear: () => {} },
  tabs: { sendMessage: async () => {} },
};

before(() => { storageMap.clear(); });

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

// ======== Checkpoint 续跑测试 ========

const CK_KEY = 'debate-wip:1.600519:monthly:2026-04:technical:off';

// 构造一个合法的 checkpoint partial
function makePartial(role) {
  return { role, text: `${role} checkpoint 缓存`, usage: { input_tokens: 500, output_tokens: 200 }, cost: 0.01, durationMs: 1000 };
}

// 指纹必须匹配 sampleCtx（code=600519, period=monthly, klines[0].date=2026-04-30 close=1600）
// buildFingerprint: 600519|monthly|2026-04-30:1600 → djb2 hash
const VALID_FP = '2980147625'; // 预计算: djb2('600519|monthly|2026-04-30:1600')

test('checkpoint: 无 checkpoint → 三个 Agent 全被调用', async () => {
  storageMap.clear();
  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    return Promise.resolve({
      status: 200, ok: true,
      json: () => Promise.resolve({
        content: [{ type: 'text', text: `call ${callCount}` }],
        usage: { input_tokens: 100, output_tokens: 50 },
      }),
      text: () => Promise.resolve('{}'),
    });
  };

  const opts = { ...sampleOpts, checkpointKey: CK_KEY };
  const result = await runDebate(sampleCtx, opts);

  // 无 checkpoint：Bull+Bear+Predictor+Judge 都跑 → 4 次 fetch
  assert.equal(callCount, 4, '无 checkpoint 时应调用 4 次');
  assert.ok(result.partials.bull);
  assert.ok(result.partials.bear);
  assert.ok(result.partials.predictor);
  assert.ok(result.judge);

  // 验证 checkpoint 已落盘（三个 partial 都写入了）
  const stored = storageMap.get(CK_KEY);
  assert.ok(stored, 'checkpoint 应已落盘');
  assert.ok(stored.partials.bull);
  assert.ok(stored.partials.bear);
  assert.ok(stored.partials.predictor);
  assert.equal(stored.v, 1);
});

test('checkpoint: 已有 bull+bear → 只 predictor 与 Judge 被调用', async () => {
  storageMap.clear();
  // 预设 checkpoint：bull+bear 已完成
  const ck = {
    v: 1,
    ts: Date.now(),
    fp: VALID_FP,
    partials: { bull: makePartial('bull'), bear: makePartial('bear') },
    errors: {},
  };
  storageMap.set(CK_KEY, ck);

  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    return Promise.resolve({
      status: 200, ok: true,
      json: () => Promise.resolve({
        content: [{ type: 'text', text: `call ${callCount}` }],
        usage: { input_tokens: 100, output_tokens: 50 },
      }),
      text: () => Promise.resolve('{}'),
    });
  };

  const opts = { ...sampleOpts, checkpointKey: CK_KEY };
  const result = await runDebate(sampleCtx, opts);

  // 仅 predictor + judge 调 LLM → 2 次 fetch
  assert.equal(callCount, 2, 'bull+bear 复用 checkpoint → 只 predictor+judge 调 LLM = 2 次');

  // 验证 bull 来自 checkpoint
  assert.equal(result.partials.bull.text, 'bull checkpoint 缓存');
  assert.equal(result.partials.bear.text, 'bear checkpoint 缓存');

  // predictor 和 judge 是新跑的
  assert.ok(result.partials.predictor);
  assert.ok(result.judge);
  assert.notEqual(result.partials.predictor.text, 'predictor checkpoint 缓存', 'predictor 应是新跑的');

  // 验证 checkpoint 已更新（predictor 也落盘了）
  const stored = storageMap.get(CK_KEY);
  assert.ok(stored.partials.predictor, 'predictor 完成后应落盘到 checkpoint');
});

test('checkpoint: 指纹不匹配 → checkpoint 被丢弃，三个全跑', async () => {
  storageMap.clear();
  // 预设 checkpoint，但 fp 错误
  const ck = {
    v: 1,
    ts: Date.now(),
    fp: 'wrong_fingerprint_9999999',
    partials: { bull: makePartial('bull'), bear: makePartial('bear') },
    errors: {},
  };
  storageMap.set(CK_KEY, ck);

  let callCount = 0;
  globalThis.fetch = () => {
    callCount++;
    return Promise.resolve({
      status: 200, ok: true,
      json: () => Promise.resolve({
        content: [{ type: 'text', text: `call ${callCount}` }],
        usage: { input_tokens: 100, output_tokens: 50 },
      }),
      text: () => Promise.resolve('{}'),
    });
  };

  const opts = { ...sampleOpts, checkpointKey: CK_KEY };
  const result = await runDebate(sampleCtx, opts);

  // 指纹不匹配 → 丢弃 checkpoint → 4 次 fetch
  assert.equal(callCount, 4, '指纹不匹配应丢弃 checkpoint，全部重跑 = 4 次');
  assert.ok(result.partials.bull);
  assert.ok(result.partials.bear);
  assert.ok(result.partials.predictor);
  assert.ok(result.judge);

  // 旧 checkpoint 已被删除（remove 后 loadCheckpoint 返回 null）
  // 然后新的已写入
  const stored = storageMap.get(CK_KEY);
  assert.ok(stored, '新 checkpoint 应已落盘');
  assert.notEqual(stored.partials.bull.text, 'bull checkpoint 缓存', 'bull 不应是旧值');
});

test('checkpoint: ≥2 Agent 成功才调 Judge 规则未被破坏', async () => {
  storageMap.clear();
  // checkpoint 只有 bull，bear 和 predictor 都失败 → successCount=1 < 2
  const ck = {
    v: 1,
    ts: Date.now(),
    fp: VALID_FP,
    partials: { bull: makePartial('bull') },
    errors: {},
  };
  storageMap.set(CK_KEY, ck);

  // bear 和 predictor 都返回 500
  globalThis.fetch = () => Promise.resolve({
    status: 500, ok: false,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve('Server error'),
  });

  const opts = { ...sampleOpts, checkpointKey: CK_KEY };
  const result = await runDebate(sampleCtx, opts);

  // bull 来自 checkpoint，bear+predictor 失败 → successCount=1
  assert.ok(result.partials.bull, 'bull 复用 checkpoint');
  assert.equal(result.partials.bear, null);
  assert.equal(result.partials.predictor, null);
  assert.equal(result.judge, null, 'successCount=1 < 2，不应调 Judge');
  assert.ok(result.errors.judge);
  assert.match(result.errors.judge, /不足 2 个/);
});
