// HARD_CONSTRAINTS clauses 4/5/6 + buildTemplatePrompt parameter pass-through test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildTemplatePrompt, DEFAULT_TEMPLATE } from '../lib/prompt-templates.js';

// mock data window
const DW = { count: 60, startDate: '2021-01', endDate: '2026-05' };
const TEMPLATE_KEYS = ['technical', 'trend', 'valuation', 'sentiment'];

// ---- Clause 4: data window annotation ----

test('clause 4: all 4 templates include data window annotation', () => {
  for (const key of TEMPLATE_KEYS) {
    const text = buildTemplatePrompt(key, '月线', '月', DW, true);
    assert.ok(text.includes('数据窗口标注'), `${key}: missing "数据窗口标注" heading`);
    assert.ok(text.includes(`共 60 根月线`), `${key}: missing count`);
    assert.ok(text.includes('2021-01'), `${key}: missing startDate`);
    assert.ok(text.includes('2026-05'), `${key}: missing endDate`);
    assert.ok(text.includes('禁止使用"历史上""从未""绝对峰值"'), `${key}: missing prohibition of unlimited timeframe language`);
  }
});

test('clause 4: dataWindow renders correctly with different values', () => {
  const dw2 = { count: 20, startDate: '2024-08', endDate: '2026-03' };
  const text = buildTemplatePrompt('technical', '周线', '周', dw2, true);
  assert.ok(text.includes('共 20 根周线'));
  assert.ok(text.includes('2024-08'));
  assert.ok(text.includes('2026-03'));
  assert.ok(text.includes('近 20 根周线区间内的最高/最低值'));
});

// ---- Clause 5: prohibition of out-of-table data ----

test('clause 5: all 4 templates include out-of-table data prohibition', () => {
  for (const key of TEMPLATE_KEYS) {
    const text = buildTemplatePrompt(key, '月线', '月', DW, true);
    assert.ok(text.includes('禁止表外数据（违反此条视为分析失效）'), `${key}: missing heading`);
    assert.ok(text.includes('仅可引用上方 K 线表格中明确出现的日期和数值'), `${key}: missing core prohibition`);
    assert.ok(text.includes('必须自检'), `${key}: missing self-check requirement`);
    assert.ok(text.includes('训练时见过的该股票历史信息一律忽略'), `${key}: missing ignore-training-data directive`);
    assert.ok(text.includes(`本数据窗口（${DW.startDate} 至 ${DW.endDate}）不包含`), `${key}: missing specific window dates`);
  }
});

// ---- Clause 6: distinguishing closed K-lines ----

test('clause 6: monthly/weekly include closed K-line distinction', () => {
  for (const period of ['月线', '周线']) {
    const unit = period === '月线' ? '月' : '周';
    const text = buildTemplatePrompt('technical', period, unit, DW, true);
    assert.ok(text.includes('区分已收盘 K 线'), `${period}: missing "区分已收盘 K 线" heading`);
    assert.ok(text.includes('尚未到期收盘'), `${period}: missing closing judgment description`);
    assert.ok(text.includes(`截至 ${DW.endDate} 的实时参考`), `${period}: missing endDate reference`);
    assert.ok(text.includes('已确认信号'), `${period}: missing "已确认信号" wording`);
    assert.ok(text.includes('实时观察信号'), `${period}: missing "实时观察信号" wording`);
  }
});

test('clause 6: not output when includeClosingBarRule=false', () => {
  const text = buildTemplatePrompt('technical', '日线', '日', DW, false);
  assert.ok(!text.includes('区分已收盘 K 线'), 'daily should not include clause 6');
});

test('clause 6: all templates in daily period exclude clause 6', () => {
  for (const key of TEMPLATE_KEYS) {
    const text = buildTemplatePrompt(key, '日线', '日', DW, false);
    assert.ok(!text.includes('区分已收盘 K 线'), `${key}/daily should not include clause 6`);
  }
});

test('clause 6: all templates with monthly/weekly and includeClosingBarRule=true include clause 6', () => {
  for (const key of TEMPLATE_KEYS) {
    const text = buildTemplatePrompt(key, '月线', '月', DW, true);
    assert.ok(text.includes('区分已收盘 K 线'), `${key}/monthly should include clause 6`);
  }
});

// ---- Original 3 constraints still present ----

test('original 3 constraints still present', () => {
  const text = buildTemplatePrompt('technical', '月线', '月', DW, true);
  assert.ok(text.includes('数字依据'), 'missing clause 1 "数字依据"');
  assert.ok(text.includes('反方观点'), 'missing clause 2 "反方观点"');
  assert.ok(text.includes('操作建议'), 'missing clause 3 "操作建议"');
});

// ---- Unknown key fallback ----

test('unknown templateKey falls back to DEFAULT_TEMPLATE with parameter pass-through', () => {
  const text = buildTemplatePrompt('nonexistent', '周线', '周', DW, true);
  // falls back to technical, should include all 6 clauses
  assert.ok(text.includes('分析任务：周线技术面分析'));
  assert.ok(text.includes('数据窗口标注'));
  assert.ok(text.includes('禁止表外数据'));
  assert.ok(text.includes('区分已收盘 K 线'));
});

// ---- 4 templates have correct analysis task headings ----

test('4 templates have correct analysis task headings', () => {
  const keywords = {
    technical: '技术面分析',
    trend: '中长期趋势判断',
    valuation: '价格历史长期价值视角',
    sentiment: '市场情绪与量价分析',
  };
  for (const [key, kw] of Object.entries(keywords)) {
    const text = buildTemplatePrompt(key, '月线', '月', DW, true);
    assert.ok(text.includes(kw), `${key}: heading does not contain "${kw}"`);
  }
});

// ---- Valuation template no longer self-contradictory ----

test('valuation template allows use of PE/PB fundamental data', () => {
  const text = buildTemplatePrompt('valuation', '月线', '月', DW, true);
  assert.ok(text.includes('PE/PB'), 'valuation template should mention PE/PB');
  assert.ok(text.includes('基本面指标'), 'valuation template should mention fundamental indicators');
  assert.ok(text.includes('get_financials'), 'valuation template should mention get_financials tool');
  assert.ok(!text.includes('不涉及 PE/PB'), 'valuation template should no longer say "不涉及 PE/PB"');
  assert.ok(!text.includes('不引用 PE/PB'), 'valuation template should no longer say "不引用 PE/PB"');
});
