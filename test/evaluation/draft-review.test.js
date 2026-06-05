// draft-review 测试
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeStats, pickFailureCases } from '../../lib/evaluation/draft-review.js';

// ---- computeStats ----

test('computeStats: 按维度分组统计', () => {
  const evals = [
    { template: 'technical', model: 'claude-sonnet-4-6', mode: 'single', enableSelfBacktest: false, enableThinking: false, verdict: 'correct', alpha: 5.2 },
    { template: 'technical', model: 'claude-sonnet-4-6', mode: 'single', enableSelfBacktest: false, enableThinking: false, verdict: 'wrong', alpha: -12 },
    { template: 'valuation', model: 'claude-opus-4-7', mode: 'single', enableSelfBacktest: true, enableThinking: true, verdict: 'strong_correct', alpha: 18 },
  ];

  const { stats } = computeStats(evals);
  assert.equal(stats.length, 2); // 两个分组

  const techGroup = stats.find((s) => s.template === 'technical');
  assert.equal(techGroup.total, 2);
  assert.equal(techGroup.correct, 1);
  assert.equal(techGroup.wrong, 1);
  assert.equal(techGroup.accuracy, 50);
});

test('computeStats: 空数组', () => {
  const { stats } = computeStats([]);
  assert.deepEqual(stats, []);
});

// ---- pickFailureCases ----

test('pickFailureCases: 按 |alpha| 降序取 top N', () => {
  const evals = [
    { historyId: 'a', verdict: 'wrong', alpha: -25, code: 'A', template: 'technical', judgment: 'bull', stockReturn: -20 },
    { historyId: 'b', verdict: 'wrong', alpha: -15, code: 'B', template: 'valuation', judgment: 'bull', stockReturn: -10 },
    { historyId: 'c', verdict: 'wrong', alpha: -8, code: 'C', template: 'trend', judgment: 'bear', stockReturn: 10 }, // |alpha|<10 不入选
    { historyId: 'd', verdict: 'correct', alpha: -20, code: 'D', template: 'technical', judgment: 'bull', stockReturn: 5 }, // not wrong
  ];

  const history = [
    { id: 'a', prompt: 'prompt A', analysis: 'analysis A' },
    { id: 'b', prompt: 'prompt B', analysis: 'analysis B' },
  ];

  const failures = pickFailureCases(evals, history, 3);
  assert.equal(failures.length, 2); // 只有 a 和 b 满足 wrong + |alpha|>10
  assert.equal(failures[0].code, 'A'); // alpha=-25 最大
  assert.equal(failures[1].code, 'B');
});

test('pickFailureCases: 关联 history 数据', () => {
  const evals = [
    { historyId: 'h1', verdict: 'wrong', alpha: -30, code: 'X', template: 'technical', judgment: 'bull', stockReturn: -25 },
  ];
  const history = [
    { id: 'h1', prompt: '# 完整 prompt...', analysis: '# 完整分析...' },
  ];
  const failures = pickFailureCases(evals, history, 5);
  assert.equal(failures[0].promptSnippet, '# 完整 prompt...');
  assert.equal(failures[0].analysisSnippet, '# 完整分析...');
});
