// Agent 输出语义级回归测试：mock LLM provider 返回固定文本，验证处理流程 + 报告字段填充
// 不测 LLM 输出质量，只测"假设 LLM 输出 X，扩展能否正确处理 X"

import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { runDebate } from '../../lib/agents/runner.js';
import { judgeAgent } from '../../lib/agents/judge.js';
import { buildPrompt, buildMultiPeriodPrompt } from '../../lib/build-prompt.js';

// ---- 共享测试数据 ----

const sampleKlines = [
  { date: '2026-04-30', open: 1550, close: 1600, high: 1620, low: 1540, volume: 120000, changePercent: 3.2, ma5: 1540, ma20: 1500, ma60: 1420, dif: 12, dea: 9, hist: 6, turnoverRate: 2.5 },
];

const debateCtx = {
  name: '贵州茅台',
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

// 助⼿：按调用顺序返回不同文本的 mock fetch
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

// 助⼿：按调用顺序可选择失败/成功的 mock fetch
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
// 场景 1：Bull Agent mock 返回含 5 个编号论点的文本
// ============================================================
test('场景1: Bull Agent mock 返回 5 个论点编号 → debate.partials.bull.text 含论点1-5', async () => {
  mockFetchSequential([
    '## 核心多头论点\n1. 均线多头排列形成金叉支撑\n2. MACD 金叉确认中期趋势向上\n3. 成交量放大配合价涨量增\n4. 前高压力位突破后转为支撑\n5. 月线级别趋势线完好向上',
    '## 核心空头论点\n1. x\n2. y\n3. z',
    '阻力位 100.00 / 支撑位 90.00',
  ]);

  const result = await runDebate(debateCtx, debateOpts);
  assert.ok(result.partials.bull, 'Bull Agent 应有输出');
  const text = result.partials.bull.text;

  for (let i = 1; i <= 5; i++) {
    assert.match(text, new RegExp(`^${i}\\.`, 'm'), `应包含论点${i}`);
  }
});

// ============================================================
// 场景 2：Bear Agent mock 返回含 5 个编号论点的文本
// ============================================================
test('场景2: Bear Agent mock 返回 5 个论点编号 → debate.partials.bear.text 含论点1-5', async () => {
  mockFetchSequential([
    '## 核心多头论点\n1. 多\n2. 多\n3. 多',
    '## 核心空头论点\n1. 前高压力位尚未有效突破\n2. MACD 红柱缩短动能衰减\n3. 成交量持续萎缩量价背离\n4. 均线乖离率过大有回调需求\n5. 估值偏高缺乏安全边际',
    '阻力位 100.00 / 支撑位 90.00',
  ]);

  const result = await runDebate(debateCtx, debateOpts);
  assert.ok(result.partials.bear, 'Bear Agent 应有输出');
  const text = result.partials.bear.text;

  for (let i = 1; i <= 5; i++) {
    assert.match(text, new RegExp(`^${i}\\.`, 'm'), `应包含论点${i}`);
  }
});

// ============================================================
// 场景 3：Judge Agent mock 返回完整报告 → 验证主输出正确提取
// ============================================================
test('场景3: Judge Agent mock 完整报告 → result.judge.text 包含综合裁判报告结构', async () => {
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
  assert.ok(result.judge, 'Judge 应有输出');
  assert.match(result.judge.text, /综合裁判报告/);
  assert.match(result.judge.text, /多空双方核心论点对比/);
  assert.match(result.judge.text, /综合方向判断/);
  assert.match(result.judge.text, /偏多/);
  assert.match(result.judge.text, /风险声明/);
  assert.match(result.judge.text, /不构成投资建议/);
});

// ============================================================
// 场景 4：Judge prompt 构造时 Bull/Bear/Predictor 文本被正确注入
// ============================================================
test('场景4: Judge prompt 中 Bull/Bear/Predictor 文本按模板位置注入', () => {
  const ctx = {
    ...debateCtx,
    partials: {
      bull:      { role: 'bull',      text: '【BULL_UNIQUE_MARKER_12345】' },
      bear:      { role: 'bear',      text: '【BEAR_UNIQUE_MARKER_67890】' },
      predictor: { role: 'predictor', text: '【PREDICTOR_UNIQUE_MARKER_ABCDE】' },
    },
  };

  const prompt = judgeAgent.buildPrompt(ctx);

  // 三个 Agent 文本都在 prompt 中
  assert.match(prompt, /【BULL_UNIQUE_MARKER_12345】/);
  assert.match(prompt, /【BEAR_UNIQUE_MARKER_67890】/);
  assert.match(prompt, /【PREDICTOR_UNIQUE_MARKER_ABCDE】/);

  // 位置验证：Bull 在 Bear 之前，Bear 在 Predictor 之前
  const bullIdx = prompt.indexOf('【BULL_UNIQUE_MARKER_12345】');
  const bearIdx = prompt.indexOf('【BEAR_UNIQUE_MARKER_67890】');
  const predIdx = prompt.indexOf('【PREDICTOR_UNIQUE_MARKER_ABCDE】');
  assert.ok(bullIdx < bearIdx, 'Bull 应在 Bear 之前');
  assert.ok(bearIdx < predIdx, 'Bear 应在 Predictor 之前');

  // 结构标签存在
  assert.match(prompt, /\[Bull Agent 输出\]/);
  assert.match(prompt, /\[Bear Agent 输出\]/);
  assert.match(prompt, /\[Predictor Agent 输出\]/);
});

// ============================================================
// 场景 5：decisionMode=true 时 prompt 末尾追加 PERSONAL_DECISION_BLOCK
// ============================================================
test('场景5: Judge decisionMode=true → prompt 末尾含个人决策视角', () => {
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
// 场景 6：successCount < 2 时 Judge 跳过 + errors.judge 填充
// ============================================================
test('场景6: 仅 1 个 Agent 成功 → Judge 跳过, errors.judge 含"不足 2 个"', async () => {
  mockFetchWithFails([
    { ok: false, status: 500, error: 'bull fail' },    // Bull 失败
    { ok: false, status: 500, error: 'bear fail' },    // Bear 失败
    { ok: true,  text: 'predictor success' },           // Predictor 成功 → successCount=1
  ]);

  const result = await runDebate(debateCtx, debateOpts);

  // 验证 successCount = 1
  const successCount = [result.partials.bull, result.partials.bear, result.partials.predictor]
    .filter((p) => p !== null).length;
  assert.ok(successCount < 2, `successCount 应为 <2，实际 ${successCount}`);

  assert.equal(result.judge, null, 'Judge 应为 null');
  assert.ok(result.errors.judge, 'errors.judge 应填充');
  assert.match(String(result.errors.judge), /不足 2 个/);

  // 验证失败的 Agent 有错误记录
  assert.ok(result.errors.bull, 'Bull 错误应有记录');
  assert.ok(result.errors.bear, 'Bear 错误应有记录');
  assert.equal(result.errors.predictor, null, 'Predictor 不应有错误');
});

// ============================================================
// 场景 7：Predictor mock 返回含具体价位的文本 → 验证数字格式
// ============================================================
test('场景7: Predictor mock 返回价位列表 → 输出包含 ≥3 处 XX.XX 格式数字', async () => {
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
  assert.ok(result.partials.predictor, 'Predictor 应有输出');
  const text = result.partials.predictor.text;

  const priceMatches = text.match(/\d+\.\d{2}/g) || [];
  assert.ok(priceMatches.length >= 3, `应至少 3 处 XX.XX 格式价位，实际 ${priceMatches.length} 处：${JSON.stringify(priceMatches)}`);

  // 额外验证 Predictor 结构关键词
  assert.match(text, /阻力位/);
  assert.match(text, /支撑位/);
});

// ============================================================
// 场景 8：decisionMode=false 时 PERSONAL_DECISION_BLOCK 不出现
// ============================================================
test('场景8: Judge decisionMode=false → prompt 不含个人决策视角', () => {
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

  assert.ok(!/个人决策视角/.test(prompt), '不应含"个人决策视角"');
  assert.ok(!/关键止损位/.test(prompt), '不应含"关键止损位"');
  assert.ok(!/持有时间预期/.test(prompt), '不应含"持有时间预期"');
  assert.ok(!/建议仓位/.test(prompt), '不应含"建议仓位"');
});

// ============================================================
// 场景 9：4 种风格各含独有关键词
// ============================================================
test('场景9: buildPrompt 4 种风格各有独有关键词', () => {
  const technical    = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'technical' });
  const chanlun      = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun' });
  const value        = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'value' });
  const comprehensive = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'comprehensive' });

  // technical：均线 + MACD
  assert.match(technical, /均线/, 'technical 应含"均线"');
  assert.match(technical, /MACD/, 'technical 应含"MACD"');

  // chanlun：中枢 + 笔 + 线段
  assert.match(chanlun, /中枢/, 'chanlun 应含"中枢"');
  assert.match(chanlun, /笔/, 'chanlun 应含"笔"');
  assert.match(chanlun, /线段/, 'chanlun 应含"线段"');

  // value：分位 + 估值
  assert.match(value, /分位/, 'value 应含"分位"');
  assert.match(value, /估值/, 'value 应含"估值"');

  // comprehensive：综合 + 共振
  assert.match(comprehensive, /综合/, 'comprehensive 应含"综合"');
  assert.match(comprehensive, /共振/, 'comprehensive 应含"共振"');

  // 4 种风格均包含 STRUCTURED_OUTPUT_BLOCK
  const styles = { technical, chanlun, value, comprehensive };
  for (const [name, prompt] of Object.entries(styles)) {
    assert.match(prompt, /结构化数据输出/, `${name} 应含"结构化数据输出"段落`);
    assert.match(prompt, /centralZone/, `${name} 应含 centralZone`);
    assert.match(prompt, /keySupport/, `${name} 应含 keySupport`);
    assert.match(prompt, /keyResistance/, `${name} 应含 keyResistance`);
  }
});

// ============================================================
// 场景 9b：多周期共振模式不含 STRUCTURED_OUTPUT_BLOCK
// ============================================================
test('场景9b: buildMultiPeriodPrompt 不含结构化数据输出段落', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台',
    code: '600519',
    monthlyKlines: sampleKlines,
    weeklyKlines: sampleKlines,
    dailyKlines: sampleKlines,
  });
  assert.ok(!/结构化数据输出/.test(out), '多周期模式不应含"结构化数据输出"');
  assert.ok(!/centralZone/.test(out), '多周期模式不应含 centralZone');
});

// ============================================================
// 场景 10：多周期共振 prompt 包含 5 段结构关键词
// ============================================================
test('场景10: buildMultiPeriodPrompt 包含 5 段结构关键词', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台',
    code: '600519',
    monthlyKlines: sampleKlines,
    weeklyKlines: sampleKlines,
    dailyKlines: sampleKlines,
  });

  assert.match(out, /月线趋势定位/, '应含"月线趋势定位"');
  assert.match(out, /周线结构定位/, '应含"周线结构定位"');
  assert.match(out, /日线入场判断/, '应含"日线入场判断"');
  assert.match(out, /多周期共振结论/, '应含"多周期共振结论"');
  assert.match(out, /综合结论/, '应含"综合结论"');
});
