// Judge Agent test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { judgeAgent } from '../../lib/agents/judge.js';

const sampleCtx = {
  name: 'Kweichow Moutai',
  code: '600519',
  period: 'monthly',
  periodLabel: '月线',
  klines: [
    { date: '2026-04-30', open: 1550, close: 1600, high: 1620, low: 1540, volume: 120000, changePercent: 3.2, ma5: 1540, ma20: 1500, ma60: 1420, dif: 12, dea: 9, hist: 6, turnoverRate: 2.5 },
  ],
  extraContext: { events: [{ date: '04-30', type: '研报', title: '测试' }] },
  partials: {
    bull: { role: 'bull', text: '多头论点：均线多头排列，MACD金叉，量价齐升。' },
    bear: { role: 'bear', text: '空头论点：接近前高压力位，成交量萎缩，估值偏高。' },
    predictor: { role: 'predictor', text: '阻力位：1650（前高）、1700（历史高点）。支撑位：1500（MA60）、1450（前低）。' },
  },
};

test('judge.buildPrompt: includes all three agent texts', () => {
  const prompt = judgeAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /多头论点/);     // Bull output content
  assert.match(prompt, /空头论点/);     // Bear output content
  assert.match(prompt, /阻力位.*1650/); // Predictor output content
});

test('judge.buildPrompt: includes "do not redo technical analysis" constraint', () => {
  const prompt = judgeAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /不要重新做.*技术分析|不重新做.*技术分析/);
});

test('judge.buildPrompt: includes composite directional judgment options', () => {
  const prompt = judgeAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /偏多/);
  assert.match(prompt, /偏空/);
  assert.match(prompt, /信号不一致/);
});

test('judge.buildPrompt: includes risk disclaimer', () => {
  const prompt = judgeAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /不构成投资建议/);
});

test('judge.buildPrompt: includes prohibition of action language', () => {
  const prompt = judgeAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /不要给出.*建议买入/);
  assert.match(prompt, /不要使用.*建议加仓/);
});

test('judge.buildPrompt: includes stock name and latest closing price', () => {
  const prompt = judgeAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /贵州茅台/);
  assert.match(prompt, /600519/);
  assert.match(prompt, /1600/);
});

test('judge.buildPrompt: shows failure markers when agents fail', () => {
  const failCtx = { ...sampleCtx, partials: { bull: null, bear: null, predictor: null } };
  const prompt = judgeAgent.buildPrompt(failCtx);
  assert.match(prompt, /Bull Agent 调用失败/);
  assert.match(prompt, /Bear Agent 调用失败/);
  assert.match(prompt, /Predictor Agent 调用失败/);
});
