// HARD_CONSTRAINTS clauses 4/5/6 + buildTemplatePrompt parameter pass-through test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildTemplatePrompt, DEFAULT_TEMPLATE } from '../lib/prompt-templates.js';

// mock 数据窗口
const DW = { count: 60, startDate: '2021-01', endDate: '2026-05' };
const TEMPLATE_KEYS = ['technical', 'trend', 'valuation', 'sentiment'];

// ---- 第 4 条：数据窗口标注 ----

test('第4条: 4 个模板均包含数据窗口标注', () => {
  for (const key of TEMPLATE_KEYS) {
    const text = buildTemplatePrompt(key, '月线', '月', DW, true);
    assert.ok(text.includes('数据窗口标注'), `${key}: 缺少"数据窗口标注"标题`);
    assert.ok(text.includes(`共 60 根月线`), `${key}: 缺少 count`);
    assert.ok(text.includes('2021-01'), `${key}: 缺少 startDate`);
    assert.ok(text.includes('2026-05'), `${key}: 缺少 endDate`);
    assert.ok(text.includes('禁止使用"历史上""从未""绝对峰值"'), `${key}: 缺少禁止无限时间措辞`);
  }
});

test('第4条: dataWindow 不同值正确渲染', () => {
  const dw2 = { count: 20, startDate: '2024-08', endDate: '2026-03' };
  const text = buildTemplatePrompt('technical', '周线', '周', dw2, true);
  assert.ok(text.includes('共 20 根周线'));
  assert.ok(text.includes('2024-08'));
  assert.ok(text.includes('2026-03'));
  assert.ok(text.includes('近 20 根周线区间内的最高/最低值'));
});

// ---- 第 5 条：禁止表外数据 ----

test('第5条: 4 个模板均包含禁止表外数据', () => {
  for (const key of TEMPLATE_KEYS) {
    const text = buildTemplatePrompt(key, '月线', '月', DW, true);
    assert.ok(text.includes('禁止表外数据（违反此条视为分析失效）'), `${key}: 缺少标题`);
    assert.ok(text.includes('仅可引用上方 K 线表格中明确出现的日期和数值'), `${key}: 缺少核心禁令`);
    assert.ok(text.includes('必须自检'), `${key}: 缺少自检要求`);
    assert.ok(text.includes('训练时见过的该股票历史信息一律忽略'), `${key}: 缺少忽略训练数据指令`);
    assert.ok(text.includes(`本数据窗口（${DW.startDate} 至 ${DW.endDate}）不包含`), `${key}: 缺少具体窗口日期`);
  }
});

// ---- 第 6 条：区分已收盘 K 线 ----

test('第6条: monthly/weekly 包含区分已收盘 K 线', () => {
  for (const period of ['月线', '周线']) {
    const unit = period === '月线' ? '月' : '周';
    const text = buildTemplatePrompt('technical', period, unit, DW, true);
    assert.ok(text.includes('区分已收盘 K 线'), `${period}: 缺少"区分已收盘 K 线"标题`);
    assert.ok(text.includes('尚未到期收盘'), `${period}: 缺少收盘判断说明`);
    assert.ok(text.includes(`截至 ${DW.endDate} 的实时参考`), `${period}: 缺少 endDate 引用`);
    assert.ok(text.includes('已确认信号'), `${period}: 缺少"已确认信号"措辞`);
    assert.ok(text.includes('实时观察信号'), `${period}: 缺少"实时观察信号"措辞`);
  }
});

test('第6条: includeClosingBarRule=false 时不输出', () => {
  const text = buildTemplatePrompt('technical', '日线', '日', DW, false);
  assert.ok(!text.includes('区分已收盘 K 线'), '日线不应包含第 6 条');
});

test('第6条: 所有模板 daily 周期均不含第 6 条', () => {
  for (const key of TEMPLATE_KEYS) {
    const text = buildTemplatePrompt(key, '日线', '日', DW, false);
    assert.ok(!text.includes('区分已收盘 K 线'), `${key}/日线不应含第 6 条`);
  }
});

test('第6条: 所有模板 monthly/weekly 且 includeClosingBarRule=true 均含第 6 条', () => {
  for (const key of TEMPLATE_KEYS) {
    const text = buildTemplatePrompt(key, '月线', '月', DW, true);
    assert.ok(text.includes('区分已收盘 K 线'), `${key}/月线 应含第 6 条`);
  }
});

// ---- 旧 3 条约束仍存在 ----

test('旧 3 条约束仍存在', () => {
  const text = buildTemplatePrompt('technical', '月线', '月', DW, true);
  assert.ok(text.includes('数字依据'), '缺少第 1 条"数字依据"');
  assert.ok(text.includes('反方观点'), '缺少第 2 条"反方观点"');
  assert.ok(text.includes('操作建议'), '缺少第 3 条"操作建议"');
});

// ---- 未知 key 回退 ----

test('未知 templateKey 回退到 DEFAULT_TEMPLATE 且参数透传', () => {
  const text = buildTemplatePrompt('nonexistent', '周线', '周', DW, true);
  // 回退到 technical，应包含全部 6 条
  assert.ok(text.includes('分析任务：周线技术面分析'));
  assert.ok(text.includes('数据窗口标注'));
  assert.ok(text.includes('禁止表外数据'));
  assert.ok(text.includes('区分已收盘 K 线'));
});

// ---- 4 个模板分析任务标题正确 ----

test('4 个模板分析任务标题正确', () => {
  const keywords = {
    technical: '技术面分析',
    trend: '中长期趋势判断',
    valuation: '价格历史长期价值视角',
    sentiment: '市场情绪与量价分析',
  };
  for (const [key, kw] of Object.entries(keywords)) {
    const text = buildTemplatePrompt(key, '月线', '月', DW, true);
    assert.ok(text.includes(kw), `${key}: 标题不含"${kw}"`);
  }
});

// ---- 估值模板不再自相矛盾 ----

test('估值模板允许使用 PE/PB 等基本面数据', () => {
  const text = buildTemplatePrompt('valuation', '月线', '月', DW, true);
  assert.ok(text.includes('PE/PB'), '估值模板应提及 PE/PB');
  assert.ok(text.includes('基本面指标'), '估值模板应提及基本面指标');
  assert.ok(text.includes('get_financials'), '估值模板应提及 get_financials 工具');
  assert.ok(!text.includes('不涉及 PE/PB'), '估值模板不应再说"不涉及 PE/PB"');
  assert.ok(!text.includes('不引用 PE/PB'), '估值模板不应再说"不引用 PE/PB"');
});
