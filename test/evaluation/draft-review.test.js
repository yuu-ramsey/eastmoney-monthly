// draft-review test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeStats, pickFailureCases } from '../../lib/evaluation/draft-review.js';

// ---- computeStats ----

test('computeStats: group stats by dimension', () => {
  const evals = [
    { template: 'technical', model: 'claude-sonnet-4-6', mode: 'single', enableSelfBacktest: false, enableThinking: false, verdict: 'correct', alpha: 5.2 },
    { template: 'technical', model: 'claude-sonnet-4-6', mode: 'single', enableSelfBacktest: false, enableThinking: false, verdict: 'wrong', alpha: -12 },
    { template: 'valuation', model: 'claude-opus-4-7', mode: 'single', enableSelfBacktest: true, enableThinking: true, verdict: 'strong_correct', alpha: 18 },
  ];

  const { stats } = computeStats(evals);
  assert.equal(stats.length, 2); // two groups

  const techGroup = stats.find((s) => s.template === 'technical');
  assert.equal(techGroup.total, 2);
  assert.equal(techGroup.correct, 1);
  assert.equal(techGroup.wrong, 1);
  assert.equal(techGroup.accuracy, 50);
});

test('computeStats: empty array', () => {
  const { stats } = computeStats([]);
  assert.deepEqual(stats, []);
});

// ---- pickFailureCases ----

test('pickFailureCases: top N by |alpha| descending', () => {
  const evals = [
    { historyId: 'a', verdict: 'wrong', alpha: -25, code: 'A', template: 'technical', judgment: 'bull', stockReturn: -20 },
    { historyId: 'b', verdict: 'wrong', alpha: -15, code: 'B', template: 'valuation', judgment: 'bull', stockReturn: -10 },
    { historyId: 'c', verdict: 'wrong', alpha: -8, code: 'C', template: 'trend', judgment: 'bear', stockReturn: 10 }, // |alpha|<10 excluded
    { historyId: 'd', verdict: 'correct', alpha: -20, code: 'D', template: 'technical', judgment: 'bull', stockReturn: 5 }, // not wrong
  ];

  const history = [
    { id: 'a', prompt: 'prompt A', analysis: 'analysis A' },
    { id: 'b', prompt: 'prompt B', analysis: 'analysis B' },
  ];

  const failures = pickFailureCases(evals, history, 3);
  assert.equal(failures.length, 2); // only a and b satisfy wrong + |alpha|>10
  assert.equal(failures[0].code, 'A'); // alpha=-25 is max
  assert.equal(failures[1].code, 'B');
});

test('pickFailureCases: correlate with history data', () => {
  const evals = [
    { historyId: 'h1', verdict: 'wrong', alpha: -30, code: 'X', template: 'technical', judgment: 'bull', stockReturn: -25 },
  ];
  const history = [
    { id: 'h1', prompt: '# full prompt...', analysis: '# full analysis...' },
  ];
  const failures = pickFailureCases(evals, history, 5);
  assert.equal(failures[0].promptSnippet, '# full prompt...');
  assert.equal(failures[0].analysisSnippet, '# full analysis...');
});
