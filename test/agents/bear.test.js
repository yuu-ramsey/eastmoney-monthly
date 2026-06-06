// Bear Agent test - symmetric to bull.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bearAgent } from '../../lib/agents/bear.js';

const sampleCtx = {
  name: 'Kweichow Moutai',
  code: '600519',
  period: 'monthly',
  periodLabel: '月线',
  klines: [
    { date: '2026-03-31', open: 1500, close: 1550, high: 1580, low: 1490, volume: 100000, changePercent: 3, ma5: 1520, ma20: 1480, ma60: 1400, dif: 10, dea: 8, hist: 4, turnoverRate: 2 },
    { date: '2026-04-30', open: 1550, close: 1600, high: 1620, low: 1540, volume: 120000, changePercent: 3.2, ma5: 1540, ma20: 1500, ma60: 1420, dif: 12, dea: 9, hist: 6, turnoverRate: 2.5 },
  ],
  extraContext: { events: [{ date: '04-30', type: '研报', title: '测试标题' }] },
};

test('bear.buildPrompt: includes bearish role instructions', () => {
  const prompt = bearAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /看空分析师/);
  assert.match(prompt, /看空论点/);
  assert.match(prompt, /空头视角/);
});

test('bear.buildPrompt: includes K-line table', () => {
  const prompt = bearAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /2026-03-31/);
  assert.match(prompt, /MACD-DIF/);
});

test('bear.buildPrompt: includes extra context', () => {
  const prompt = bearAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /测试标题/);
  assert.match(prompt, /附加上下文/);
});

test('bear.buildPrompt: does not include comprehensive conclusion or action advice', () => {
  const prompt = bearAgent.buildPrompt(sampleCtx);
  const afterProhibition = prompt.replace(/不要输出.*综合结论.*操作建议.*\n*/g, '');
  assert.ok(!/综合结论/.test(afterProhibition));
  assert.ok(!/操作建议/.test(afterProhibition));
});

test('bear.buildPrompt: includes reverse risk assessment requirement', () => {
  const prompt = bearAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /反向风险/);
});

test('bear.buildPrompt: includes "honestly confront opposing evidence" directive', () => {
  const prompt = bearAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /诚实面对/);
});

test('bear.buildPrompt: prohibits giving downside predictions without factual basis', () => {
  const prompt = bearAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /不能脱离事实/);
});
