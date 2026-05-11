// Judge decisionMode 测试
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { judgeAgent } from '../../lib/agents/judge.js';

const sampleCtx = {
  name: '贵州茅台',
  code: '600519',
  period: 'monthly',
  periodLabel: '月线',
  klines: [{ date: '2026-04-30', open: 1550, close: 1600, high: 1620, low: 1540, volume: 120000, changePercent: 3.2, ma5: 1540, ma20: 1500, ma60: 1420, dif: 12, dea: 9, hist: 6, turnoverRate: 2.5 }],
  extraContext: { events: [] },
  partials: {
    bull: { role: 'bull', text: '多头论点' },
    bear: { role: 'bear', text: '空头论点' },
    predictor: { role: 'predictor', text: '阻力位 1650' },
  },
};

test('judge: decisionMode=true 时包含个人决策视角', () => {
  const prompt = judgeAgent.buildPrompt({ ...sampleCtx, decisionMode: true });
  assert.match(prompt, /个人决策视角/);
  assert.match(prompt, /关键止损位/);
  assert.match(prompt, /持有时间预期/);
  assert.match(prompt, /仅供持有者本人/);
});

test('judge: decisionMode=false 时不包含决策段', () => {
  const prompt = judgeAgent.buildPrompt(sampleCtx);
  assert.ok(!/个人决策视角/.test(prompt));
  assert.ok(!/关键止损位/.test(prompt));
});
