// Self-backtest function test
import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { calculateActualReturn, buildSelfCalibrationBlock } from '../lib/self-backtest.js';

// ---- calculateActualReturn 数值测试 ----

test('calculateActualReturn: 上涨场景', () => {
  const klines = [
    { date: '2024-01', close: 10.0 },
    { date: '2024-02', close: 12.0 },
  ];
  const result = calculateActualReturn(klines, 0, 1, null);
  assert.equal(result.fromDate, '2024-01');
  assert.equal(result.toDate, '2024-02');
  assert.ok(Math.abs(result.stockReturn - 20.0) < 0.01, `应为 +20%，实际 ${result.stockReturn}`);
  assert.equal(result.indexReturn, null);
  assert.equal(result.alpha, null);
});

test('calculateActualReturn: 下跌场景', () => {
  const klines = [
    { date: '2024-01', close: 20.0 },
    { date: '2024-06', close: 15.0 },
  ];
  const result = calculateActualReturn(klines, 0, 1, null);
  assert.ok(Math.abs(result.stockReturn - (-25.0)) < 0.01, `应为 -25%，实际 ${result.stockReturn}`);
});

test('calculateActualReturn: 含沪深300 alpha 计算', () => {
  const klines = [
    { date: '2024-01', close: 10.0 },
    { date: '2024-06', close: 13.0 }, // +30%
  ];
  const indexKlines = [
    { date: '2024-01', close: 3000 },
    { date: '2024-06', close: 3150 }, // +5%
  ];
  const result = calculateActualReturn(klines, 0, 1, indexKlines);
  assert.equal(result.stockReturn, 30.0);
  assert.ok(Math.abs(result.indexReturn - 5.0) < 0.01);
  assert.ok(Math.abs(result.alpha - 25.0) < 0.01, `alpha 应为 25%，实际 ${result.alpha}`);
});

test('calculateActualReturn: 负 alpha', () => {
  const klines = [
    { date: '2024-01', close: 10.0 },
    { date: '2024-06', close: 10.5 }, // +5%
  ];
  const indexKlines = [
    { date: '2024-01', close: 3000 },
    { date: '2024-06', close: 3300 }, // +10%
  ];
  const result = calculateActualReturn(klines, 0, 1, indexKlines);
  assert.ok(Math.abs(result.alpha - (-5.0)) < 0.01);
});

test('calculateActualReturn: 沪深300不匹配时返回null', () => {
  const klines = [
    { date: '2024-01', close: 10.0 },
    { date: '2024-06', close: 12.0 },
  ];
  // 沪深300日期完全不重叠
  const indexKlines = [
    { date: '2025-01', close: 4000 },
    { date: '2025-02', close: 4100 },
  ];
  const result = calculateActualReturn(klines, 0, 1, indexKlines);
  assert.equal(result.indexReturn, null);
  assert.equal(result.alpha, null);
});

test('calculateActualReturn: 无效索引抛错', () => {
  const klines = [
    { date: '2024-01', close: 10.0 },
    { date: '2024-02', close: 12.0 },
  ];
  assert.throws(() => calculateActualReturn(klines, 0, 3, null), /无效索引/);
  assert.throws(() => calculateActualReturn(klines, 1, 0, null), /无效索引/);
});

test('calculateActualReturn: 沪深300按月匹配容差', () => {
  // 个股日期有日，沪深300只有月
  const klines = [
    { date: '2024-01-15', close: 10.0 },
    { date: '2024-06-20', close: 12.0 },
  ];
  const indexKlines = [
    { date: '2023-12-25', close: 2900 },
    { date: '2024-01-10', close: 3000 },
    { date: '2024-06-15', close: 3150 },
    { date: '2024-07-01', close: 3200 },
  ];
  const result = calculateActualReturn(klines, 0, 1, indexKlines);
  assert.ok(result.indexReturn != null);
  assert.ok(result.alpha != null);
});

// ---- buildSelfCalibrationBlock 格式测试 ----

test('buildSelfCalibrationBlock: 空数组返回空字符串', () => {
  assert.equal(buildSelfCalibrationBlock([]), '');
  assert.equal(buildSelfCalibrationBlock(null), '');
});

test('buildSelfCalibrationBlock: 包含标题和必要元素', () => {
  const results = [{
    date: '2024-01',
    judgment: '偏多',
    keyLevels: [35.03, 28.10],
    actualReturn: { toDate: '2025-01', stockReturn: 15.5, indexReturn: 8.0, alpha: 7.5 },
  }];
  const block = buildSelfCalibrationBlock(results);
  assert.ok(block.includes('## 历史自我校准'));
  assert.ok(block.includes('2024-01'));
  assert.ok(block.includes('偏多'));
  assert.ok(block.includes('35.03'));
  assert.ok(block.includes('28.10'));
  assert.ok(block.includes('涨 15.50%'));
  assert.ok(block.includes('alpha = +7.50%'));
});

test('buildSelfCalibrationBlock: 两个时点的回测', () => {
  const results = [
    { date: '2023-05', judgment: '中性', keyLevels: [], actualReturn: { toDate: '2024-05', stockReturn: -5.2, indexReturn: 2.0, alpha: -7.2 } },
    { date: '2024-05', judgment: '偏空', keyLevels: [18.50], actualReturn: { toDate: '2025-05', stockReturn: -12.8, indexReturn: -3.0, alpha: -9.8 } },
  ];
  const block = buildSelfCalibrationBlock(results);
  assert.ok(block.includes('2023-05'));
  assert.ok(block.includes('2024-05'));
  assert.ok(block.includes('中性'));
  assert.ok(block.includes('偏空'));
  assert.ok(block.includes('跌 5.20%'));
  assert.ok(block.includes('跌 12.80%'));
});

test('buildSelfCalibrationBlock: 无大盘数据时跳过alpha', () => {
  const results = [{
    date: '2024-01',
    judgment: '偏多',
    keyLevels: [],
    actualReturn: { toDate: '2025-01', stockReturn: 10.0, indexReturn: null, alpha: null },
  }];
  const block = buildSelfCalibrationBlock(results);
  assert.ok(block.includes('无大盘对照数据'));
});

test('buildSelfCalibrationBlock: 包含置信度提醒', () => {
  const results = [{
    date: '2024-01', judgment: '偏多', keyLevels: [],
    actualReturn: { toDate: '2025-01', stockReturn: 10.0, indexReturn: 5.0, alpha: 5.0 },
  }];
  const block = buildSelfCalibrationBlock(results);
  assert.ok(block.includes('置信度'));
  assert.ok(block.includes('可能存在偏差'));
});

// ---- 数据不足降级 ----

test('runHistoricalAnalysis: cutoffIndex<12 抛错', async () => {
  const { runHistoricalAnalysis } = await import('../lib/self-backtest.js');
  try {
    await runHistoricalAnalysis([], 5, 'technical', 'anthropic', {});
    assert.fail('应抛错');
  } catch (err) {
    assert.ok(err.message.includes('数据不足'));
  }
});

// ---- extractJudgment 内部逻辑（通过 runHistoricalAnalysis 间接测试） ----

test('extractJudgment: 模型输出包含"偏多"', async () => {
  // mock provider.call 返回含"偏多"的文本
  const originalFetch = globalThis.fetch;
  const { getProvider } = await import('../lib/llm/index.js');

  globalThis.fetch = () => Promise.resolve({
    status: 200,
    ok: true,
    json: () => Promise.resolve({
      stop_reason: 'end_turn',
      content: [{ type: 'text', text: '综合结论：偏多。关键价位 42.90 和 35.03。' }],
      usage: { input_tokens: 100, output_tokens: 50 },
    }),
  });

  try {
    const { runHistoricalAnalysis } = await import('../lib/self-backtest.js');
    // 构造最少 12 根 K 线数据
    const klines = Array.from({ length: 12 }, (_, i) => ({
      date: `2025-${String(i + 1).padStart(2, '0')}`,
      open: 10 + i, close: 10 + i + 0.5, high: 11 + i, low: 9 + i,
      volume: 1000, changePercent: 5, turnoverRate: 2,
      ma5: null, ma20: null, ma60: null, dif: null, dea: null, hist: null,
    }));

    const result = await runHistoricalAnalysis(klines, 12, 'technical', 'anthropic', {
      apiKey: 'sk-test', name: '测试', code: '000001',
    });

    assert.equal(result.judgment, '偏多');
    assert.ok(result.keyLevels.length > 0);
    assert.ok(result.keyLevels.includes(42.90));
    assert.ok(result.keyLevels.includes(35.03));
  } finally {
    globalThis.fetch = originalFetch;
  }
});
