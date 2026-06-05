import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bullAgent } from '../../lib/agents/bull.js';

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

test('bull.buildPrompt: 包含看多角色指令', () => {
  const prompt = bullAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /看多分析师/);
  assert.match(prompt, /看多论点/);
  assert.match(prompt, /多头视角/);
});

test('bull.buildPrompt: 包含 K 线表格', () => {
  const prompt = bullAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /2026-03-31/);
  assert.match(prompt, /1500\.00/);
  assert.match(prompt, /MACD-DIF/);
});

test('bull.buildPrompt: 包含附加上下文', () => {
  const prompt = bullAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /测试标题/);
  assert.match(prompt, /附加上下文/);
});

test('bull.buildPrompt: 不包含综合结论和操作建议', () => {
  const prompt = bullAgent.buildPrompt(sampleCtx);
  // 扣除禁止声明段落后再检查
  const afterProhibition = prompt.replace(/不要输出.*综合结论.*操作建议.*\n*/g, '');
  assert.ok(!/综合结论/.test(afterProhibition), 'bull 分析任务中不应要求综合结论');
  assert.ok(!/操作建议/.test(afterProhibition), 'bull 分析任务中不应要求操作建议');
});

test('bull.buildPrompt: 包含反向风险评估要求', () => {
  const prompt = bullAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /反向风险/);
});

test('bull.buildPrompt: 包含诚实面对反向证据', () => {
  const prompt = bullAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /诚实面对/);
});

test('bull.buildPrompt: 禁止脱离事实给目标价', () => {
  const prompt = bullAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /不能脱离事实/);
});
