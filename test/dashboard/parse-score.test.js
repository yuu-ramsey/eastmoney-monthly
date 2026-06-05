// parseScoreBlock / validateScoreData / computeWeightedScore test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseScoreBlock, validateScoreData, computeWeightedScore } from '../../lib/dashboard/parse-score.js';

// ---- parseScoreBlock ----

test('解析标准 JSON 块', () => {
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

test('LLM 忘写 JSON → null', () => {
  assert.equal(parseScoreBlock('纯文本分析，没有 JSON'), null);
  assert.equal(parseScoreBlock(null), null);
  assert.equal(parseScoreBlock(''), null);
});

test('JSON 字段缺失 → null', () => {
  const text = '```json\n{"score":50}\n```';
  assert.equal(parseScoreBlock(text), null);
});

test('JSON 字段超范围 → null', () => {
  assert.equal(parseScoreBlock('```json\n{"score":-1,"signal":"bull","confidence":"high","key_levels":{"support":[1],"resistance":[2],"stop_loss":3},"trend":"uptrend","position_percentile":50,"one_line_summary":"x"}\n```'), null);
  assert.equal(parseScoreBlock('```json\n{"score":101,"signal":"bull","confidence":"high","key_levels":{"support":[1],"resistance":[2],"stop_loss":3},"trend":"uptrend","position_percentile":50,"one_line_summary":"x"}\n```'), null);
});

test('signal 拼错 → null', () => {
  const text = '```json\n{"score":50,"signal":"bullish","confidence":"high","key_levels":{"support":[1],"resistance":[2],"stop_loss":3},"trend":"uptrend","position_percentile":50,"one_line_summary":"x"}\n```';
  assert.equal(parseScoreBlock(text), null);
});

test('多个 JSON 块取最后一个', () => {
  const text = '```json\n{"score":30}\n```\n中间文字\n```json\n{"score":72,"signal":"bull","confidence":"high","key_levels":{"support":[35,28],"resistance":[42],"stop_loss":33},"trend":"uptrend","position_percentile":45,"one_line_summary":"看好"}\n```';
  const data = parseScoreBlock(text);
  assert.equal(data.score, 72);
});

test('包含中文转义符的 JSON', () => {
  const text = '```json\n{"score":68,"signal":"bull","confidence":"medium","key_levels":{"support":[35.03],"resistance":[42.90],"stop_loss":33.50},"trend":"uptrend","position_percentile":55,"one_line_summary":"\\u5747\\u7ebf\\u591a\\u5934"}\n```';
  const data = parseScoreBlock(text);
  assert.ok(data);
  assert.equal(data.score, 68);
});

test('JSON 键带多余空格仍解析', () => {
  const text = '```json\n{ "score" : 55 , "signal":"neutral","confidence":"low","key_levels":{"support":[10],"resistance":[20],"stop_loss":9},"trend":"sideways","position_percentile":50,"one_line_summary":"震荡" }\n```';
  const data = parseScoreBlock(text);
  assert.equal(data.score, 55);
});

// ---- validateScoreData ----

test('validateScoreData: 完整有效数据', () => {
  const { valid } = validateScoreData({
    score: 72, signal: 'bull', confidence: 'high',
    key_levels: { support: [1, 2], resistance: [3], stop_loss: 4 },
    trend: 'uptrend', position_percentile: 50, one_line_summary: 'OK',
  });
  assert.equal(valid, true);
});

test('validateScoreData: support 非数组', () => {
  const { valid, errors } = validateScoreData({
    score: 50, signal: 'neutral', confidence: 'medium',
    key_levels: { support: 'not_array', resistance: [1], stop_loss: 2 },
    trend: 'sideways', position_percentile: 50, one_line_summary: 'X',
  });
  assert.equal(valid, false);
  assert.ok(errors.some((e) => e.includes('support')));
});

// ---- computeWeightedScore ----

test('computeWeightedScore: 标准权重', () => {
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

test('computeWeightedScore: 缺失模板重新分配权重', () => {
  const arr = [
    { template: 'technical', scoreData: { score: 80 } },
    { template: 'trend', scoreData: { score: 60 } },
  ];
  // 仅 tech+trend: 80*0.35/0.60 + 60*0.25/0.60 = 46.7+25 = 71.7
  const result = computeWeightedScore(arr);
  assert.ok(result > 70 && result < 73);
});

test('computeWeightedScore: 全部 null → null', () => {
  assert.equal(computeWeightedScore([]), null);
  assert.equal(computeWeightedScore([{ template: 'technical', scoreData: null }]), null);
});
