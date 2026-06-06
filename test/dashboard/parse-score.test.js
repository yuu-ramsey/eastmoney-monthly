// parseScoreBlock / validateScoreData / computeWeightedScore test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseScoreBlock, validateScoreData, computeWeightedScore } from '../../lib/dashboard/parse-score.js';

// ---- parseScoreBlock ----

test('parse standard JSON block', () => {
  const text = `## 分析结果
  方向判断：【偏多】...
  \`\`\`json
  {"score":72,"signal":"bull","confidence":"high","key_levels":{"support":[35.03,28.10],"resistance":[42.90,45.95],"stop_loss":33.50},"trend":"uptrend","position_percentile":45.2,"one_line_summary":"均线多头排列，MACD金叉"}
  \`\`\``;
  const data = parseScoreBlock(text);
  assert.ok(data);
  assert.equal(data.score, 72);
  assert.equal(data.signal, 'bull');
  assert.equal(data.confidence, 'high');
  assert.deepEqual(data.key_levels.support, [35.03, 28.10]);
  assert.equal(data.key_levels.stop_loss, 33.50);
  assert.equal(data.trend, 'uptrend');
  assert.equal(data.position_percentile, 45.2);
});

test('LLM forgot JSON -> null', () => {
  assert.equal(parseScoreBlock('plain text analysis, no JSON'), null);
  assert.equal(parseScoreBlock(null), null);
  assert.equal(parseScoreBlock(''), null);
});

test('JSON field missing -> null', () => {
  const text = '```json\n{"score":50}\n```';
  assert.equal(parseScoreBlock(text), null);
});

test('JSON field out of range -> null', () => {
  assert.equal(parseScoreBlock('```json\n{"score":-1,"signal":"bull","confidence":"high","key_levels":{"support":[1],"resistance":[2],"stop_loss":3},"trend":"uptrend","position_percentile":50,"one_line_summary":"x"}\n```'), null);
  assert.equal(parseScoreBlock('```json\n{"score":101,"signal":"bull","confidence":"high","key_levels":{"support":[1],"resistance":[2],"stop_loss":3},"trend":"uptrend","position_percentile":50,"one_line_summary":"x"}\n```'), null);
});

test('signal typo -> null', () => {
  const text = '```json\n{"score":50,"signal":"bullish","confidence":"high","key_levels":{"support":[1],"resistance":[2],"stop_loss":3},"trend":"uptrend","position_percentile":50,"one_line_summary":"x"}\n```';
  assert.equal(parseScoreBlock(text), null);
});

test('multiple JSON blocks -> pick last', () => {
  const text = '```json\n{"score":30}\n```\n中间文字\n```json\n{"score":72,"signal":"bull","confidence":"high","key_levels":{"support":[35,28],"resistance":[42],"stop_loss":33},"trend":"uptrend","position_percentile":45,"one_line_summary":"看好"}\n```';
  const data = parseScoreBlock(text);
  assert.equal(data.score, 72);
});

test('JSON with unicode escapes', () => {
  const text = '```json\n{"score":68,"signal":"bull","confidence":"medium","key_levels":{"support":[35.03],"resistance":[42.90],"stop_loss":33.50},"trend":"uptrend","position_percentile":55,"one_line_summary":"\\u5747\\u7ebf\\u591a\\u5934"}\n```';
  const data = parseScoreBlock(text);
  assert.ok(data);
  assert.equal(data.score, 68);
});

test('JSON key with extra spaces still parses', () => {
  const text = '```json\n{ "score" : 55 , "signal":"neutral","confidence":"low","key_levels":{"support":[10],"resistance":[20],"stop_loss":9},"trend":"sideways","position_percentile":50,"one_line_summary":"震荡" }\n```';
  const data = parseScoreBlock(text);
  assert.equal(data.score, 55);
});

// ---- validateScoreData ----

test('validateScoreData: complete valid data', () => {
  const { valid } = validateScoreData({
    score: 72, signal: 'bull', confidence: 'high',
    key_levels: { support: [1, 2], resistance: [3], stop_loss: 4 },
    trend: 'uptrend', position_percentile: 50, one_line_summary: 'OK',
  });
  assert.equal(valid, true);
});

test('validateScoreData: support not array', () => {
  const { valid, errors } = validateScoreData({
    score: 50, signal: 'neutral', confidence: 'medium',
    key_levels: { support: 'not_array', resistance: [1], stop_loss: 2 },
    trend: 'sideways', position_percentile: 50, one_line_summary: 'X',
  });
  assert.equal(valid, false);
  assert.ok(errors.some((e) => e.includes('support')));
});

// ---- computeWeightedScore ----

test('computeWeightedScore: standard weights', () => {
  const arr = [
    { template: 'technical', scoreData: { score: 80 } },
    { template: 'trend', scoreData: { score: 60 } },
    { template: 'valuation', scoreData: { score: 70 } },
    { template: 'sentiment', scoreData: { score: 50 } },
  ];
  // 80*0.35 + 60*0.25 + 70*0.20 + 50*0.20 = 28+15+14+10 = 67
  const result = computeWeightedScore(arr);
  assert.ok(result > 65 && result < 69);
});

test('computeWeightedScore: missing template reallocates weight', () => {
  const arr = [
    { template: 'technical', scoreData: { score: 80 } },
    { template: 'trend', scoreData: { score: 60 } },
  ];
  // 仅 tech+trend: 80*0.35/0.60 + 60*0.25/0.60 = 46.7+25 = 71.7
  const result = computeWeightedScore(arr);
  assert.ok(result > 70 && result < 73);
});

test('computeWeightedScore: all null -> null', () => {
  assert.equal(computeWeightedScore([]), null);
  assert.equal(computeWeightedScore([{ template: 'technical', scoreData: null }]), null);
});
