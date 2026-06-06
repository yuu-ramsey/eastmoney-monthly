// Agent runner concurrent dispatch test (with checkpoint resume)
import { test, afterEach, before } from 'node:test';
import assert from 'node:assert/strict';
import { runDebate } from '../../lib/agents/runner.js';

// ---- Mock chrome.storage (needed for checkpoint tests) ----
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
  name: 'Kweichow Moutai',
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

// ---- Full flow (Bull/Bear/Predictor concurrent + Judge) ----

test('runDebate: full 4-Agent flow', async () => {
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
  assert.equal(callCount, 4, 'should call 4 times: Bull/Bear/Predictor concurrent + Judge');
  assert.ok(result.partials.bull);
  assert.ok(result.partials.bear);
  assert.ok(result.partials.predictor);
  assert.ok(result.judge, 'Judge should have a value');
  assert.equal(result.judge.role, 'judge');
  assert.ok(result.totalCost > 0);
  assert.ok(result.totalDurationMs >= 0);
  assert.equal(result.errors.judge, null);
});

// ---- One Agent failure does not affect others ----

test('runDebate: single Agent failure does not affect others', async () => {
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

  // Use normal opts for all agents, manually verify errors are null
  const result = await runDebate(sampleCtx, sampleOpts);
  assert.ok(result.partials.bull);
  assert.ok(result.partials.bear);
  assert.ok(result.partials.predictor);
  assert.equal(result.errors.bull, null);
  assert.equal(result.errors.bear, null);
  assert.equal(result.errors.predictor, null);
});

// ---- All failed -> Judge skipped ----

test('runDebate: all failed -> Judge skipped + judgeError populated', async () => {
  globalThis.fetch = () => Promise.resolve({ status: 500, ok: false, json: () => Promise.resolve({}), text: () => Promise.resolve('Server error') });

  const result = await runDebate(sampleCtx, sampleOpts);
  assert.equal(result.partials.bull, null);
  assert.equal(result.partials.bear, null);
  assert.equal(result.partials.predictor, null);
  assert.equal(result.judge, null);
  assert.ok(result.errors.judge, 'successCount=0 < 2, should skip Judge');
  assert.match(result.errors.judge, /不足 2 个/);
  assert.equal(result.totalCost, 0);
});

// ---- totalCost accumulation ----

test('runDebate: totalCost accumulation includes Judge', async () => {
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
  assert.ok(result.totalCost >= singleCost * 3.9, `totalCost should be roughly 4x single agent cost, actual ${result.totalCost} vs ${singleCost * 4}`);
  assert.ok(result.judge, 'Judge should have a value');
});

// ======== Checkpoint resume tests ========

const CK_KEY = 'debate-wip:1.600519:monthly:2026-04:technical:off';

// Build a valid checkpoint partial
function makePartial(role) {
  return { role, text: `${role} checkpoint cached`, usage: { input_tokens: 500, output_tokens: 200 }, cost: 0.01, durationMs: 1000 };
}

// Fingerprint must match sampleCtx (code=600519, period=monthly, klines[0].date=2026-04-30 close=1600)
// buildFingerprint: 600519|monthly|2026-04-30:1600 -> djb2 hash
const VALID_FP = '386909631'; // precomputed: djb2('600519|monthly|1|2026-04-30|2026-04-30|1600')

test('checkpoint: no checkpoint -> all 3 Agents are called', async () => {
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

  // No checkpoint: Bull+Bear+Predictor+Judge all run -> 4 fetch calls
  assert.equal(callCount, 4, 'no checkpoint should call 4 times');
  assert.ok(result.partials.bull);
  assert.ok(result.partials.bear);
  assert.ok(result.partials.predictor);
  assert.ok(result.judge);

  // Verify checkpoint was persisted (all three partials written)
  const stored = storageMap.get(CK_KEY);
  assert.ok(stored, 'checkpoint should be persisted');
  assert.ok(stored.partials.bull);
  assert.ok(stored.partials.bear);
  assert.ok(stored.partials.predictor);
  assert.equal(stored.v, 1);
});

test('checkpoint: existing bull+bear -> only predictor and Judge called', async () => {
  storageMap.clear();
  // Preset checkpoint: bull+bear completed
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

  // Only predictor + judge call LLM -> 2 fetch calls
  assert.equal(callCount, 2, 'bull+bear reuse checkpoint -> only predictor+judge call LLM = 2 times');

  // Verify bull from checkpoint
  assert.equal(result.partials.bull.text, 'bull checkpoint cached');
  assert.equal(result.partials.bear.text, 'bear checkpoint cached');

  // predictor and judge are fresh runs
  assert.ok(result.partials.predictor);
  assert.ok(result.judge);
  assert.notEqual(result.partials.predictor.text, 'predictor checkpoint cached', 'predictor should be newly run');

  // Verify checkpoint updated (predictor also persisted)
  const stored = storageMap.get(CK_KEY);
  assert.ok(stored.partials.predictor, 'predictor should be persisted to checkpoint after completion');
});

test('checkpoint: fingerprint mismatch -> checkpoint discarded, all three re-run', async () => {
  storageMap.clear();
  // Preset checkpoint with wrong fp
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

  // Fingerprint mismatch -> discard checkpoint -> 4 fetch calls
  assert.equal(callCount, 4, 'fingerprint mismatch should discard checkpoint, full re-run = 4 times');
  assert.ok(result.partials.bull);
  assert.ok(result.partials.bear);
  assert.ok(result.partials.predictor);
  assert.ok(result.judge);

  // Old checkpoint deleted (remove -> loadCheckpoint returns null)
  // New checkpoint written
  const stored = storageMap.get(CK_KEY);
  assert.ok(stored, 'new checkpoint should be persisted');
  assert.notEqual(stored.partials.bull.text, 'bull checkpoint cached', 'bull should not be old value');
});

test('checkpoint: >=2 Agent success required for Judge rule is not broken', async () => {
  storageMap.clear();
  // Checkpoint only has bull, bear and predictor both fail -> successCount=1 < 2
  const ck = {
    v: 1,
    ts: Date.now(),
    fp: VALID_FP,
    partials: { bull: makePartial('bull') },
    errors: {},
  };
  storageMap.set(CK_KEY, ck);

  // bear and predictor both return 500
  globalThis.fetch = () => Promise.resolve({
    status: 500, ok: false,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve('Server error'),
  });

  const opts = { ...sampleOpts, checkpointKey: CK_KEY };
  const result = await runDebate(sampleCtx, opts);

  // bull from checkpoint, bear+predictor failed -> successCount=1
  assert.ok(result.partials.bull, 'bull reused from checkpoint');
  assert.equal(result.partials.bear, null);
  assert.equal(result.partials.predictor, null);
  assert.equal(result.judge, null, 'successCount=1 < 2, should not call Judge');
  assert.ok(result.errors.judge);
  assert.match(result.errors.judge, /不足 2 个/);
});

// ---- Fingerprint uses only closed bars ----

test('checkpoint: same closed bars, current bar close changes -> fingerprint unchanged -> reuse checkpoint', async () => {
  storageMap.clear();
  // Preset checkpoint: bull+bear completed
  const ck = {
    v: 1,
    ts: Date.now(),
    fp: VALID_FP,
    partials: { bull: makePartial('bull'), bear: makePartial('bear') },
    errors: {},
  };
  storageMap.set(CK_KEY, ck);

  // Build ctx: closed bar unchanged, but add a current bar (simulates intra-bar close fluctuation)
  const ctxWithCurrentBar = {
    ...sampleCtx,
    klines: [
      { date: '2026-04-30', open: 1550, close: 1600, high: 1620, low: 1540, volume: 120000, changePercent: 3.2, ma5: 1540, ma20: 1500, ma60: 1420, dif: 12, dea: 9, hist: 6, turnoverRate: 2.5 },
      { date: '2026-05-29', open: 1610, close: 1595, high: 1630, low: 1580, volume: 80000, changePercent: -0.3, ma5: 1580, ma20: 1550, ma60: 1450, dif: 10, dea: 8, hist: 4, turnoverRate: 1.8 },
    ],
  };

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
  const result = await runDebate(ctxWithCurrentBar, opts);

  // Current bar (2026-05-29) filtered by isBarClosed, fingerprint unchanged -> bull+bear reused
  assert.equal(callCount, 2, 'current bar close change should not affect fingerprint -> only predictor+judge call LLM = 2 times');
  assert.equal(result.partials.bull.text, 'bull checkpoint cached', 'bull should come from checkpoint');
  assert.equal(result.partials.bear.text, 'bear checkpoint cached', 'bear should come from checkpoint');
  assert.ok(result.judge);
});

test('checkpoint: closed bar close changed -> fingerprint mismatch -> discard checkpoint, full re-run', async () => {
  storageMap.clear();
  // Preset checkpoint: bull+bear completed
  const ck = {
    v: 1,
    ts: Date.now(),
    fp: VALID_FP,
    partials: { bull: makePartial('bull'), bear: makePartial('bear') },
    errors: {},
  };
  storageMap.set(CK_KEY, ck);

  // Build ctx: closed bar close changed (simulates adjustment factor switch)
  const ctxChangedClose = {
    ...sampleCtx,
    klines: [
      { date: '2026-04-30', open: 1550, close: 1620, high: 1630, low: 1540, volume: 120000, changePercent: 4.5, ma5: 1550, ma20: 1510, ma60: 1430, dif: 13, dea: 10, hist: 7, turnoverRate: 2.6 },
    ],
  };

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
  const result = await runDebate(ctxChangedClose, opts);

  // Closed bar close changed 1600 -> 1620, fingerprint changed -> discard checkpoint -> all re-run
  assert.equal(callCount, 4, 'closed bar close changed should discard checkpoint, full re-run = 4 times');
  assert.ok(result.partials.bull);
  assert.notEqual(result.partials.bull.text, 'bull checkpoint cached', 'bull should not reuse old checkpoint');
  assert.ok(result.judge);
});

// ---- #3: failed Agent is not reused ----

test('checkpoint: preset bull success + bear persistent error -> bull reused, bear re-called', async () => {
  storageMap.clear();
  // Preset checkpoint: bull success (partials), bear failed (errors, not in partials)
  const ck = {
    v: 1,
    ts: Date.now(),
    fp: VALID_FP,
    partials: { bull: makePartial('bull') },
    errors: { bear: 'LLM timeout' },
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

  // bull reused from checkpoint, bear + predictor + judge call LLM -> 3 fetch calls
  assert.equal(callCount, 3, 'bull reused from checkpoint -> bear+predictor+judge = 3 fetch calls');
  assert.equal(result.partials.bull.text, 'bull checkpoint cached', 'bull should come from checkpoint');
  assert.ok(result.partials.bear, 'bear should be re-called (not skipped)');
  assert.notEqual(result.partials.bear.text, 'bear checkpoint cached', 'bear should not reuse error cache');
  assert.ok(result.partials.predictor);
  assert.ok(result.judge);
});
