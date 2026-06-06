// Multi-period resonance prompt test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildMultiPeriodPrompt } from '../lib/build-prompt.js';

const sampleKlines = [
  { date: '2026-04-30', open: 1550, close: 1600, high: 1620, low: 1540, volume: 120000, changePercent: 3.2, ma5: 1540, ma20: 1500, ma60: 1420, dif: 12, dea: 9, hist: 6, turnoverRate: 2.5 },
];

test('buildMultiPeriodPrompt: includes three period headers', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台', code: '600519',
    monthlyKlines: sampleKlines, weeklyKlines: sampleKlines, dailyKlines: sampleKlines,
  });
  assert.match(out, /月线数据.*长期趋势判断/);
  assert.match(out, /周线数据.*中期结构判断/);
  assert.match(out, /日线数据.*短期入场判断/);
});

test('buildMultiPeriodPrompt: includes multi-period resonance instructions', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台', code: '600519',
    monthlyKlines: sampleKlines, weeklyKlines: sampleKlines, dailyKlines: sampleKlines,
  });
  assert.match(out, /多周期共振/);
  assert.match(out, /三共振多.*三共振空.*月周多日空.*月空周日多.*信号紊乱/);
});

test('buildMultiPeriodPrompt: includes five-section structure', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台', code: '600519',
    monthlyKlines: sampleKlines, weeklyKlines: sampleKlines, dailyKlines: sampleKlines,
  });
  assert.match(out, /月线趋势定位/);
  assert.match(out, /周线结构定位/);
  assert.match(out, /日线入场判断/);
  assert.match(out, /多周期共振结论/);
  assert.match(out, /综合结论/);
});

test('buildMultiPeriodPrompt: decisionMode=true appends decision block', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台', code: '600519',
    monthlyKlines: sampleKlines, weeklyKlines: sampleKlines, dailyKlines: sampleKlines,
    decisionMode: true,
  });
  assert.match(out, /个人决策视角/);
  assert.match(out, /关键止损位/);
});

test('buildMultiPeriodPrompt: decisionMode=false does not append decision block', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台', code: '600519',
    monthlyKlines: sampleKlines, weeklyKlines: sampleKlines, dailyKlines: sampleKlines,
  });
  assert.ok(!/个人决策视角/.test(out));
});

test('buildMultiPeriodPrompt: 4 styles have different comprehensive conclusions', () => {
  const styles = ['technical', 'chanlun', 'value', 'comprehensive'];
  const results = new Set();
  for (const style of styles) {
    const out = buildMultiPeriodPrompt({
      name: 'X', code: '000001',
      monthlyKlines: sampleKlines, weeklyKlines: sampleKlines, dailyKlines: sampleKlines,
      style,
    });
    // Extract content after "五、综合结论"
    const idx = out.indexOf('### 五、综合结论');
    results.add(out.slice(idx));
  }
  assert.equal(results.size, 4, 'comprehensive conclusions of the four styles should differ from each other');
});

test('buildMultiPeriodPrompt: includes extra context', () => {
  const out = buildMultiPeriodPrompt({
    name: '茅台', code: '600519',
    monthlyKlines: sampleKlines, weeklyKlines: sampleKlines, dailyKlines: sampleKlines,
    extraContext: { events: [{ date: '04-30', type: '研报', title: '测试' }] },
  });
  assert.match(out, /附加上下文/);
  assert.match(out, /测试/);
});
