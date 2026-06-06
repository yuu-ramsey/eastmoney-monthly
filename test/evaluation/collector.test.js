// collector test - extractJudgment / evaluateOneAnalysis / evaluateBatch
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { extractJudgment, evaluateOneAnalysis, evaluateBatch } from '../../lib/evaluation/collector.js';

// ---- extractJudgment ----

test('extractJudgment: direction judgment:【偏多】', () => {
  assert.equal(extractJudgment('方向判断：【偏多】基于均线多头排列'), 'bull');
  assert.equal(extractJudgment('方向判断:【偏空】跌破 MA60'), 'bear');
  assert.equal(extractJudgment('方向判断：【中性震荡】方向不明'), 'neutral');
});

test('extractJudgment: direction judgment: 偏多 (no brackets)', () => {
  assert.equal(extractJudgment('方向判断：偏多'), 'bull');
  assert.equal(extractJudgment('方向判断: 偏空'), 'bear');
});

test('extractJudgment: composite direction judgment section', () => {
  assert.equal(extractJudgment('综合方向判断：\n偏多\n基于均线'), 'bull');
  assert.equal(extractJudgment('综合方向判断:\n偏空\n跌破支撑'), 'bear');
});

test('extractJudgment: match in synthesis conclusion', () => {
  assert.equal(extractJudgment('综合结论：当前价格…建议偏多操作'), 'bull');
});

test('extractJudgment: null/empty/no match', () => {
  assert.equal(extractJudgment(null), null);
  assert.equal(extractJudgment(''), null);
  assert.equal(extractJudgment('这是一段没有方向判断的分析'), null);
});

// ---- evaluateOneAnalysis ----

test('evaluateOneAnalysis: bull + up + alpha>5 -> strong_correct', async () => {
  const entry = { id: 'h1', code: '600522', analysis: '方向判断：【偏多】', timestamp: Date.now() - 40 * 86400000 };
  const klines = [{ date: '2026-04', close: 10 }, { date: '2026-05', close: 13 }]; // +30%
  const idxKlines = [{ date: '2026-04', close: 3000 }, { date: '2026-05', close: 3060 }]; // +2%
  const result = await evaluateOneAnalysis(entry,
    async () => klines,
    async () => idxKlines,
  );
  assert.equal(result.judgment, 'bull');
  assert.equal(result.verdict, 'strong_correct');
  assert.ok(result.alpha > 5);
});

test('evaluateOneAnalysis: bull + small gain -> correct', async () => {
  const entry = { id: 'h2', code: '000001', analysis: '方向判断：【偏多】', timestamp: Date.now() - 40 * 86400000 };
  const klines = [{ date: '2026-04', close: 10 }, { date: '2026-05', close: 10.3 }]; // +3%
  const idxKlines = [{ date: '2026-04', close: 3000 }, { date: '2026-05', close: 3000 }]; // 0%
  const result = await evaluateOneAnalysis(entry, async () => klines, async () => idxKlines);
  assert.equal(result.verdict, 'correct');
});

test('evaluateOneAnalysis: bull + down -> wrong', async () => {
  const entry = { id: 'h3', code: '600000', analysis: '方向判断：【偏多】', timestamp: Date.now() - 40 * 86400000 };
  const klines = [{ date: '2026-04', close: 10 }, { date: '2026-05', close: 9 }]; // -10%
  const result = await evaluateOneAnalysis(entry, async () => klines, async () => []);
  assert.equal(result.verdict, 'wrong');
});

test('evaluateOneAnalysis: bear + big drop + alpha<-5 -> strong_correct', async () => {
  const entry = { id: 'h4', code: '600000', analysis: '方向判断：【偏空】', timestamp: Date.now() - 40 * 86400000 };
  const klines = [{ date: '2026-04', close: 10 }, { date: '2026-05', close: 7 }]; // -30%
  const idxKlines = [{ date: '2026-04', close: 3000 }, { date: '2026-05', close: 3060 }]; // +2%
  const result = await evaluateOneAnalysis(entry, async () => klines, async () => idxKlines);
  assert.equal(result.verdict, 'strong_correct');
  assert.ok(result.alpha < -5);
});

test('evaluateOneAnalysis: neutral + |alpha|<5 -> correct', async () => {
  const entry = { id: 'h5', code: '600000', analysis: '方向判断：【中性震荡】', timestamp: Date.now() - 40 * 86400000 };
  const klines = [{ date: '2026-04', close: 10 }, { date: '2026-05', close: 10.2 }]; // +2%
  const idxKlines = [{ date: '2026-04', close: 3000 }, { date: '2026-05', close: 3030 }]; // +1%
  const result = await evaluateOneAnalysis(entry, async () => klines, async () => idxKlines);
  assert.equal(result.verdict, 'correct');
});

test('evaluateOneAnalysis: daysElapsed<30 -> pending', async () => {
  const entry = { id: 'h6', code: '600000', analysis: '方向判断：【偏多】', timestamp: Date.now() - 10 * 86400000 };
  const klines = [{ date: '2026-05', close: 10 }, { date: '2026-05', close: 13 }];
  const result = await evaluateOneAnalysis(entry, async () => klines, async () => []);
  assert.equal(result.verdict, 'pending');
});

test('evaluateOneAnalysis: no judgment -> no_judgment', async () => {
  const entry = { id: 'h7', code: '600000', analysis: '纯描述无判断', timestamp: Date.now() - 40 * 86400000 };
  const result = await evaluateOneAnalysis(entry, async () => [], async () => []);
  assert.equal(result.verdict, 'no_judgment');
});

test('evaluateOneAnalysis: fetch failure does not block', async () => {
  const entry = { id: 'h8', code: '600000', analysis: '方向判断：【偏多】', timestamp: Date.now() - 40 * 86400000 };
  const result = await evaluateOneAnalysis(entry,
    async () => { throw new Error('network'); },
    async () => { throw new Error('network'); },
  );
  assert.ok(['pending', 'no_judgment'].includes(result.verdict));
});

// ---- evaluateBatch ----

test('evaluateBatch: skip already evaluated', async () => {
  const storage = {
    evaluations: [{ historyId: 'existing' }],
    async getEvaluations() { return this.evaluations; },
    async saveEvaluation(r) { this.evaluations.push(r); },
  };

  const entries = [
    { id: 'existing', code: '600000', analysis: '方向判断：【偏多】', timestamp: Date.now() - 40 * 86400000 },
    { id: 'new', code: '600522', analysis: '方向判断：【偏空】', timestamp: Date.now() - 40 * 86400000 },
  ];

  const klines = [{ date: '2026-04', close: 10 }, { date: '2026-05', close: 9 }];
  const idxKlines = [{ date: '2026-04', close: 3000 }, { date: '2026-05', close: 3100 }];

  const results = await evaluateBatch(entries, {
    fetchKlines: async () => klines,
    fetchIndexKlines: async () => idxKlines,
  }, storage);

  assert.ok(results.length >= 1);
  const newOne = results.find((r) => r.historyId === 'new');
  assert.ok(newOne);
  assert.equal(newOne.judgment, 'bear');
});
