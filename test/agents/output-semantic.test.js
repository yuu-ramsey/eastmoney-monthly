// Agent output semantic regression test: mock LLM provider returns fixed text, verify processing flow + report field population
// Does not test LLM output quality, only tests "if LLM outputs X, can the extension correctly handle X"

import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { runDebate } from '../../lib/agents/runner.js';
import { judgeAgent } from '../../lib/agents/judge.js';
import { buildPrompt, buildMultiPeriodPrompt } from '../../lib/build-prompt.js';

// ---- Shared test data ----

const sampleKlines = [
  { date: '2026-04-30', open: 1550, close: 1600, high: 1620, low: 1540, volume: 120000, changePercent: 3.2, ma5: 1540, ma20: 1500, ma60: 1420, dif: 12, dea: 9, hist: 6, turnoverRate: 2.5 },
];

const debateCtx = {
  name: 'Kweichow Moutai',
  code: '600519',
  period: 'monthly',
  periodLabel: '月线',
  klines: sampleKlines,
  extraContext: { events: [] },
};

const debateOpts = {
  provider: 'anthropic',
  apiKey: 'sk-ant-test',
  model: 'claude-sonnet-4-6',
  maxTokens: 4000,
};

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

// Helper: mock fetch that returns different text per call order
function mockFetchSequential(texts) {
  let idx = 0;
  globalThis.fetch = () => {
    const text = idx < texts.length ? texts[idx] : (texts[texts.length - 1] || 'default');
    idx++;
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve({
        content: [{ type: 'text', text }],
        usage: { input_tokens: 100, output_tokens: 50 },
      }),
      text: () => Promise.resolve('{}'),
    });
  };
}

// Helper: mock fetch that can fail/succeed per call order
function mockFetchWithFails(results) {
  let idx = 0;
  globalThis.fetch = () => {
    const r = idx < results.length ? results[idx] : (results[results.length - 1] || { ok: true });
    idx++;
    if (!r.ok) {
      return Promise.resolve({ status: r.status || 500, ok: false, json: () => Promise.resolve({}), text: () => Promise.resolve(r.error || 'error') });
    }
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve({
        content: [{ type: 'text', text: r.text || 'default' }],
        usage: { input_tokens: 100, output_tokens: 50 },
      }),
      text: () => Promise.resolve('{}'),
    });
  };
}

// ============================================================
// Scenario 1: Bull Agent mock returns text with 5 numbered arguments
// ============================================================
test('Scenario 1: Bull Agent mock returns 5 numbered arguments -> debate.partials.bull.text contains arguments 1-5', async () => {
  mockFetchSequential([
    '## 核心多头论点\n1. 均线多头排列形成金叉支撑\n2. MACD 金叉确认中期趋势向上\n3. 成交量放大配合价涨量增\n4. 前高压力位突破后转为支撑\n5. 月线级别趋势线完好向上',
    '## 核心空头论点\n1. x\n2. y\n3. z',
    '阻力位 100.00 / 支撑位 90.00',
  ]);

  const result = await runDebate(debateCtx, debateOpts);
  assert.ok(result.partials.bull, 'Bull Agent should have output');
  const text = result.partials.bull.text;

  for (let i = 1; i <= 5; i++) {
    assert.match(text, new RegExp(`^${i}\\.`, 'm'), `should contain argument ${i}`);
  }
});

// ============================================================
// Scenario 2: Bear Agent mock returns text with 5 numbered arguments
// ============================================================
test('Scenario 2: Bear Agent mock returns 5 numbered arguments -> debate.partials.bear.text contains arguments 1-5', async () => {
  mockFetchSequential([
    '## 核心多头论点\n1. 多\n2. 多\n3. 多',
    '## 核心空头论点\n1. 前高压力位尚未有效突破\n2. MACD 红柱缩短动能衰减\n3. 成交量持续萎缩量价背离\n4. 均线乖离率过大有回调需求\n5. 估值偏高缺乏安全边际',
    '阻力位 100.00 / 支撑位 90.00',
  ]);

  const result = await runDebate(debateCtx, debateOpts);
  assert.ok(result.partials.bear, 'Bear Agent should have output');
  const text = result.partials.bear.text;

  for (let i = 1; i <= 5; i++) {
    assert.match(text, new RegExp(`^${i}\\.`, 'm'), `should contain argument ${i}`);
  }
});

// ============================================================
// Scenario 3: Judge Agent mock returns complete report -> verify main output correctly extracted
// ============================================================
test('Scenario 3: Judge Agent mock complete report -> result.judge.text contains synthesis report structure', async () => {
  const judgeReport = `# 贵州茅台(600519) 综合裁判报告

## 多空双方核心论点对比
- Bull 最强论点：均线多头排列，MACD 金叉确认
- Bear 最强论点：接近前高压力，成交量萎缩

## 论点扎实度评估
Bull 方论点更扎实，数据引用具体，逻辑链完整。

## 综合方向判断
偏多 —— 多头信号更强，空头论点缺乏足够数据支撑。

## 关键观察价位
- 上方阻力：1650（前高）
- 下方支撑：1500（MA60）

## 风险声明
> 本分析仅供研究学习使用，不构成投资建议。`;

  mockFetchSequential([
    'bull text',
    'bear text',
    'predictor text',
    judgeReport,
  ]);

  const result = await runDebate(debateCtx, debateOpts);
  assert.ok(result.judge, 'Judge should have output');
  assert.match(result.judge.text, /综合裁判报告/);
  assert.match(result.judge.text, /多空双方核心论点对比/);
  assert.match(result.judge.text, /综合方向判断/);
  assert.match(result.judge.text, /偏多/);
  assert.match(result.judge.text, /风险声明/);
  assert.match(result.judge.text, /不构成投资建议/);
});

// ============================================================
// Scenario 4: Judge prompt construction correctly injects Bull/Bear/Predictor text
// ============================================================
test('Scenario 4: Judge prompt injects Bull/Bear/Predictor text at template positions', () => {
  const ctx = {
    ...debateCtx,
    partials: {
      bull:      { role: 'bull',      text: '【BULL_UNIQUE_MARKER_12345】' },
      bear:      { role: 'bear',      text: '【BEAR_UNIQUE_MARKER_67890】' },
      predictor: { role: 'predictor', text: '【PREDICTOR_UNIQUE_MARKER_ABCDE】' },
    },
  };

  const prompt = judgeAgent.buildPrompt(ctx);

  // All three Agent texts are present in the prompt
  assert.match(prompt, /【BULL_UNIQUE_MARKER_12345】/);
  assert.match(prompt, /【BEAR_UNIQUE_MARKER_67890】/);
  assert.match(prompt, /【PREDICTOR_UNIQUE_MARKER_ABCDE】/);

  // Position verification: Bull before Bear, Bear before Predictor
  const bullIdx = prompt.indexOf('【BULL_UNIQUE_MARKER_12345】');
  const bearIdx = prompt.indexOf('【BEAR_UNIQUE_MARKER_67890】');
  const predIdx = prompt.indexOf('【PREDICTOR_UNIQUE_MARKER_ABCDE】');
  assert.ok(bullIdx < bearIdx, 'Bull should be before Bear');
  assert.ok(bearIdx < predIdx, 'Bear should be before Predictor');

  // Structure labels present
  assert.match(prompt, /\[Bull Agent 输出\]/);
  assert.match(prompt, /\[Bear Agent 输出\]/);
  assert.match(prompt, /\[Predictor Agent 输出\]/);
});

// ============================================================
// Scenario 5: decisionMode=true appends PERSONAL_DECISION_BLOCK at end of prompt
// ============================================================
test('Scenario 5: Judge decisionMode=true -> prompt end contains personal decision perspective', () => {
  const ctx = {
    ...debateCtx,
    decisionMode: true,
    partials: {
      bull:      { role: 'bull',      text: 'bull output' },
      bear:      { role: 'bear',      text: 'bear output' },
      predictor: { role: 'predictor', text: 'pred output' },
    },
  };

  const prompt = judgeAgent.buildPrompt(ctx);

  assert.match(prompt, /个人决策视角/);
  assert.match(prompt, /关键止损位/);
  assert.match(prompt, /关键加仓.*减仓位/);
  assert.match(prompt, /持有时间预期/);
  assert.match(prompt, /仅供持有者本人/);
  assert.match(prompt, /建议仓位/);
});

// ============================================================
// Scenario 6: successCount < 2 -> Judge skipped + errors.judge populated
// ============================================================
test('Scenario 6: Only 1 Agent succeeds -> Judge skipped, errors.judge contains "less than 2"', async () => {
  mockFetchWithFails([
    { ok: false, status: 500, error: 'bull fail' },    // Bull failed
    { ok: false, status: 500, error: 'bear fail' },    // Bear failed
    { ok: true,  text: 'predictor success' },           // Predictor succeeded -> successCount=1
  ]);

  const result = await runDebate(debateCtx, debateOpts);

  // Verify successCount = 1
  const successCount = [result.partials.bull, result.partials.bear, result.partials.predictor]
    .filter((p) => p !== null).length;
  assert.ok(successCount < 2, `successCount should be <2, actual ${successCount}`);

  assert.equal(result.judge, null, 'Judge should be null');
  assert.ok(result.errors.judge, 'errors.judge should be populated');
  assert.match(String(result.errors.judge), /不足 2 个/);

  // Verify failed Agents have error records
  assert.ok(result.errors.bull, 'Bull error should have record');
  assert.ok(result.errors.bear, 'Bear error should have record');
  assert.equal(result.errors.predictor, null, 'Predictor should not have error');
});

// ============================================================
// Scenario 7: Predictor mock returns text with specific price levels -> verify number format
// ============================================================
test('Scenario 7: Predictor mock returns price level list -> output contains >=3 XX.XX format numbers', async () => {
  mockFetchSequential([
    'bull text',
    'bear text',
    `## 上方关键阻力位
- 65.50 元 | 强 | 依据：2025-09 月线前高
- 62.30 元 | 中 | 依据：MA60 当前值
- 60.00 元 | 弱 | 依据：心理整数关口

## 下方关键支撑位
- 55.20 元 | 强 | 依据：2026-03 月线前低
- 52.10 元 | 中 | 依据：周线中枢下沿
- 50.00 元 | 强 | 依据：月线中枢下沿`,
  ]);

  const result = await runDebate(debateCtx, debateOpts);
  assert.ok(result.partials.predictor, 'Predictor should have output');
  const text = result.partials.predictor.text;

  const priceMatches = text.match(/\d+\.\d{2}/g) || [];
  assert.ok(priceMatches.length >= 3, `should have at least 3 XX.XX format prices, actual ${priceMatches.length}: ${JSON.stringify(priceMatches)}`);

  // Extra verification of Predictor structure keywords
  assert.match(text, /阻力位/);
  assert.match(text, /支撑位/);
});

// ============================================================
// Scenario 8: decisionMode=false -> PERSONAL_DECISION_BLOCK does not appear
// ============================================================
test('Scenario 8: Judge decisionMode=false -> prompt does not contain personal decision perspective', () => {
  const ctx = {
    ...debateCtx,
    decisionMode: false,
    partials: {
      bull:      { role: 'bull',      text: 'bull' },
      bear:      { role: 'bear',      text: 'bear' },
      predictor: { role: 'predictor', text: 'pred' },
    },
  };

  const prompt = judgeAgent.buildPrompt(ctx);

  assert.ok(!/个人决策视角/.test(prompt), 'should not contain "personal decision perspective"');
  assert.ok(!/关键止损位/.test(prompt), 'should not contain "key stop-loss level"');
  assert.ok(!/持有时间预期/.test(prompt), 'should not contain "expected holding period"');
  assert.ok(!/建议仓位/.test(prompt), 'should not contain "suggested position size"');
});

// ============================================================
// Scenario 9: 4 styles each contain unique keywords
// ============================================================
test('Scenario 9: buildPrompt 4 styles each have unique keywords', () => {
  const technical    = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'technical' });
  const chanlun      = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun' });
  const value        = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'value' });
  const comprehensive = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'comprehensive' });

  // technical: moving averages + MACD
  assert.match(technical, /均线/, 'technical should contain "MA"');
  assert.match(technical, /MACD/, 'technical should contain "MACD"');

  // chanlun: pivot + stroke + segment
  assert.match(chanlun, /中枢/, 'chanlun should contain "pivot"');
  assert.match(chanlun, /笔/, 'chanlun should contain "stroke"');
  assert.match(chanlun, /线段/, 'chanlun should contain "segment"');

  // value: percentile + valuation
  assert.match(value, /分位/, 'value should contain "percentile"');
  assert.match(value, /估值/, 'value should contain "valuation"');

  // comprehensive: synthesis + resonance
  assert.match(comprehensive, /综合/, 'comprehensive should contain "synthesis"');
  assert.match(comprehensive, /共振/, 'comprehensive should contain "resonance"');

  // All 4 styles contain STRUCTURED_OUTPUT_BLOCK
  const styles = { technical, chanlun, value, comprehensive };
  for (const [name, prompt] of Object.entries(styles)) {
    assert.match(prompt, /结构化数据输出/, `${name} should contain "structured data output" section`);
    assert.match(prompt, /centralZone/, `${name} should contain centralZone`);
    assert.match(prompt, /keySupport/, `${name} should contain keySupport`);
    assert.match(prompt, /keyResistance/, `${name} should contain keyResistance`);
  }
});

// ============================================================
// Scenario 9b: multi-period resonance mode does NOT contain STRUCTURED_OUTPUT_BLOCK
// ============================================================
test('Scenario 9b: buildMultiPeriodPrompt does not contain structured data output section', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台',
    code: '600519',
    monthlyKlines: sampleKlines,
    weeklyKlines: sampleKlines,
    dailyKlines: sampleKlines,
  });
  assert.ok(!/结构化数据输出/.test(out), 'multi-period mode should not contain "structured data output"');
  assert.ok(!/centralZone/.test(out), 'multi-period mode should not contain centralZone');
});

// ============================================================
// Scenario 10: multi-period resonance prompt contains 5 section structure keywords
// ============================================================
test('Scenario 10: buildMultiPeriodPrompt contains 5 section structure keywords', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台',
    code: '600519',
    monthlyKlines: sampleKlines,
    weeklyKlines: sampleKlines,
    dailyKlines: sampleKlines,
  });

  assert.match(out, /月线趋势定位/, 'should contain "monthly trend positioning"');
  assert.match(out, /周线结构定位/, 'should contain "weekly structure positioning"');
  assert.match(out, /日线入场判断/, 'should contain "daily entry judgment"');
  assert.match(out, /多周期共振结论/, 'should contain "multi-period resonance conclusion"');
  assert.match(out, /综合结论/, 'should contain "synthesis conclusion"');
});
