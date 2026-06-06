// Predictor Agent test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { predictorAgent } from '../../lib/agents/predictor.js';

const sampleCtx = {
  name: 'Kweichow Moutai',
  code: '600519',
  period: 'monthly',
  periodLabel: '月线',
  klines: [
    { date: '2026-03-31', open: 1500, close: 1550, high: 1580, low: 1490, volume: 100000, changePercent: 3, ma5: 1520, ma20: 1480, ma60: 1400, dif: 10, dea: 8, hist: 4, turnoverRate: 2 },
  ],
  extraContext: { events: [] },
};

test('predictor.buildPrompt: 包含角色指令', () => {
  const prompt = predictorAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /压力位.*支撑位|支撑.*压力/);
  assert.match(prompt, /密集成交区/);
});

test('predictor.buildPrompt: 包含不做方向判断约束', () => {
  const prompt = predictorAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /不做.*方向.*判断|不做.*方向判断/);
  assert.match(prompt, /不预测涨跌/);
});

test('predictor.buildPrompt: contains K-line table', () => {
  const prompt = predictorAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /2026-03-31/);
  assert.match(prompt, /MACD-DIF/);
});

test('predictor.buildPrompt: does not contain trading advice or bullish/bearish view', () => {
  const prompt = predictorAgent.buildPrompt(sampleCtx);
  // 扣除禁止声明后再检查
  const cleaned = prompt.replace(/不输出.*操作建议.*\n*/g, '').replace(/不输出.*看多.*看空.*\n*/g, '');
  assert.ok(!/操作建议/.test(cleaned), 'predictor should not have trading advice');
  assert.ok(!/看多/.test(cleaned), 'predictor should not say bullish');
  assert.ok(!/看空/.test(cleaned), 'predictor should not say bearish');
});

test('predictor.buildPrompt: monthly period mentions quarterly level', () => {
  const prompt = predictorAgent.buildPrompt(sampleCtx);
  assert.match(prompt, /季度级别/);
});

test('predictor.buildPrompt: daily period does not have quarterly constraint', () => {
  const dailyCtx = { ...sampleCtx, period: 'daily', periodLabel: '日线' };
  const prompt = predictorAgent.buildPrompt(dailyCtx);
  assert.ok(!/季度级别/.test(prompt), 'daily should not have quarterly level constraint');
  assert.match(prompt, /更精细/);
});
