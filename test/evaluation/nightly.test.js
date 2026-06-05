// nightly 测试
import { test } from 'node:test';
import assert from 'node:assert/strict';

test('nightly: mock 链路（不调真实 API）', async () => {
  // mock DeepSeek
  let callCount = 0;
  const mockDeepSeek = async (prompt) => {
    callCount++;
    return {
      text: '建议1: 优化技术面模板的均线描述\n建议2: 增加估值数据获取\n建议3: 调整反向观点权重',
      usage: { inputTokens: 5000, outputTokens: 2000 },
    };
  };

  // mock K 线
  const mockKlines = (() => {
    const data = [];
    for (let i = 0; i < 24; i++) {
      data.push({ date: `2024-${String(i + 1).padStart(2, '0')}`, close: 10 + i * 0.5 });
    }
    return data;
  })();

  const mockIndexKlines = (() => {
    const data = [];
    for (let i = 0; i < 24; i++) {
      data.push({ date: `2024-${String(i + 1).padStart(2, '0')}`, close: 3000 + i * 10 });
    }
    return data;
  })();

  // mock history
  const historyEntries = Array.from({ length: 5 }, (_, i) => ({
    id: `h${i}`,
    code: '600522',
    name: '中天科技',
    analysis: i % 3 === 0 ? '方向判断：【偏多】' : (i % 3 === 1 ? '方向判断：【偏空】' : '方向判断：【中性震荡】'),
    timestamp: Date.now() - (50 + i * 10) * 86400000,
    template: 'technical',
    model: 'claude-sonnet-4-6',
    mode: 'single',
    enableSelfBacktest: false,
    enableThinking: false,
    prompt: `prompt #${i}`,
  }));

  // 不 import nightly 直接测核心逻辑 — 只测 evaluateBatch + computeStats + pickFailureCases
  const { evaluateBatch } = await import('../../lib/evaluation/collector.js');
  const { computeStats, pickFailureCases } = await import('../../lib/evaluation/draft-review.js');

  // 内存 storage
  const memEvals = [];
  const storage = {
    async getEvaluations() { return [...memEvals]; },
    async saveEvaluation(r) { memEvals.push(r); },
  };

  const results = await evaluateBatch(historyEntries, {
    fetchKlines: async () => mockKlines,
    fetchIndexKlines: async () => mockIndexKlines,
  }, storage);

  assert.ok(results.length > 0);
  assert.ok(memEvals.length > 0);

  const { stats } = computeStats(memEvals);
  assert.ok(stats.length > 0);

  const failures = pickFailureCases(memEvals, historyEntries, 5);
  // failures may be empty if no |alpha|>10 wrong, which is fine
  assert.ok(Array.isArray(failures));

  // Mock generateDraftReview flow
  const { generateDraftReview } = await import('../../lib/evaluation/draft-review.js');
  const { draftPath, cost } = await generateDraftReview({
    evaluations: memEvals,
    historyEntries,
    callDeepSeek: mockDeepSeek,
  });

  assert.ok(draftPath.includes('draft-'));
  assert.ok(cost > 0);
  assert.equal(callCount, 1);
});
