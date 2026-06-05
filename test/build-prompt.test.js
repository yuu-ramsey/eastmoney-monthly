import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildPrompt } from '../lib/build-prompt.js';

const sampleKlines = [
  { date: '2024-01-31', open: 1, close: 2, high: 3, low: 0.5, volume: 100, changePercent: 5, ma5: null, ma20: null, ma60: null, dif: null, dea: null, hist: null, turnoverRate: 1.2 },
  { date: '2024-02-29', open: 2, close: 2.5, high: 3.5, low: 1.5, volume: 200, changePercent: 25, ma5: 2.25, ma20: null, ma60: null, dif: -0.5, dea: null, hist: null, turnoverRate: 2.5 },
];

test('buildPrompt: includes stock name, code, date, headers, end requirements', () => {
  const out = buildPrompt({ name: 'Kweichow Moutai', code: '600519', klines: sampleKlines });
  assert.match(out, /贵州茅台/);
  assert.match(out, /600519/);
  assert.match(out, /2024-01-31/);
  assert.match(out, /2024-02-29/);
  assert.match(out, /MA5/);
  assert.match(out, /MA20/);
  assert.match(out, /MA60/);
  assert.match(out, /MACD-DIF/);
  assert.match(out, /MACD-DEA/);
  assert.match(out, /MACD-HIST/);
  assert.match(out, /换手率/);
  assert.match(out, /Markdown/);
  assert.match(out, /综合结论/);
});

test('buildPrompt: 空 klines 抛错', () => {
  assert.throws(() => buildPrompt({ name: 'X', code: '000001', klines: [] }));
  assert.throws(() => buildPrompt({ name: 'X', code: '000001', klines: null }));
});

test('buildPrompt: MA 为 null 显示为 -', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: [sampleKlines[0]] });
  const dataLine = out.split('\n').find((l) => l.startsWith('2024-01-31'));
  assert.ok(dataLine);
  const dashCount = (dataLine.match(/\t-/g) || []).length;
  assert.ok(dashCount >= 3, `期望至少 3 个 - ,实际 ${dashCount} 个;行内容:${dataLine}`);
});

test('buildPrompt: 数字保留 2 位小数', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: [sampleKlines[0]] });
  assert.match(out, /1\.00/);
  assert.match(out, /2\.00/);
});

test('buildPrompt: 换手率出现在数据行', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines });
  assert.match(out, /1\.20/);
  assert.match(out, /2\.50/);
});

// ---- 分析风格 ----

test('buildPrompt: 默认风格为 technical', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines });
  assert.match(out, /均线系统状态/);
});

test('buildPrompt: 4 种 style 输出不同的分析任务段落', () => {
  const technical = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'technical' });
  const chanlun = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun' });
  const value = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'value' });
  const comprehensive = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'comprehensive' });

  // 4 种输出的"分析任务"段落彼此不同
  assert.notEqual(technical, chanlun);
  assert.notEqual(technical, value);
  assert.notEqual(technical, comprehensive);
  assert.notEqual(chanlun, value);
  assert.notEqual(chanlun, comprehensive);
  assert.notEqual(value, comprehensive);
});

test('buildPrompt: chanlun 风格包含缠论体系关键词,禁止非缠论术语', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun' });
  assert.match(out, /中枢/);
  assert.match(out, /背驰/);
  assert.match(out, /笔.*线段|线段.*笔/);
  assert.match(out, /ZG/);
  assert.match(out, /ZD/);
  assert.match(out, /暂无.*买卖点|暂无.*类买卖点/);
  assert.match(out, /MACD.*红.*绿.*柱.*面积/i);
  // 禁止非缠论术语——但 prompt 禁止声明本身会提及这些术语名，扣除后再检查
  const afterHeader = out.replace(/禁止使用.*?术语[。.]?\n*/gs, '');
  assert.ok(!/金叉/.test(afterHeader), 'chanlun 风格分析任务中不应出现"金叉"');
  assert.ok(!/死叉/.test(afterHeader), 'chanlun 风格分析任务中不应出现"死叉"');
  assert.ok(!/超买/.test(afterHeader), 'chanlun 风格分析任务中不应出现"超买"');
  assert.ok(!/超卖/.test(afterHeader), 'chanlun 风格分析任务中不应出现"超卖"');
  // 不应有"操作建议"
  assert.ok(!/操作建议/.test(out), 'chanlun 风格不应有"操作建议"');
});

test('buildPrompt: chanlun 风格严格性约束包含 3%/3 段重叠/诚实标注', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun' });

  assert.match(out, /二次探底.*同一笔|二次探底.*次级别波动/);
  assert.match(out, /3%.*以上/, '应包含三买 3% 以上约束');
  assert.match(out, /3 段.*重叠/, '应包含 3 段重叠约束');
  assert.match(out, /诚实标注不确定性/);
  assert.match(out, /无法严格判定.*粗略推测/);
});

test('buildPrompt: chanlun + 任意 provider 输出一致', () => {
  const outDefault = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun' });
  const outAnthropic = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun', provider: 'anthropic' });
  const outDeepseek = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun', provider: 'deepseek' });
  assert.equal(outDefault, outAnthropic);
  assert.equal(outDefault, outDeepseek);
});

test('buildPrompt: value 风格包含否定提示(不要/单月),适配周期', () => {
  const outMonthly = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'value', period: 'monthly' });
  assert.match(outMonthly, /不要.*单月/);
  assert.match(outMonthly, /单月涨跌幅/);

  const outDaily = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'value', period: 'daily' });
  assert.match(outDaily, /不要.*单日/);
  assert.match(outDaily, /单日涨跌幅/);

  const outWeekly = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'value', period: 'weekly' });
  assert.match(outWeekly, /不要.*单周/);
  assert.match(outWeekly, /单周涨跌幅/);
});

test('buildPrompt: comprehensive 风格包含技术和估值两个视角', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'comprehensive' });
  assert.match(out, /技术面/);
  assert.match(out, /估值/);
});

test('buildPrompt: 未知 style 回退到 technical', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'unknown' });
  assert.match(out, /均线系统状态/);
});

// ---- extraContext ----

test('buildPrompt: extraContext 为 undefined 时向后兼容', () => {
  const without = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines });
  const withEmpty = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, extraContext: {} });
  assert.equal(without, withEmpty);
});

test('buildPrompt: extraContext.events 为空时不输出附加段落', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, extraContext: { events: [] } });
  assert.ok(!/附加上下文/.test(out), '不应出现附加上下文标题');
  assert.ok(!/大事提醒/.test(out), '不应出现大事提醒');
});

test('buildPrompt: extraContext.events 有数据时正确格式化', () => {
  const events = [
    { date: '04-30', type: '研报', title: '测试研报标题' },
    { date: '04-29', type: '公告', title: '测试公告标题' },
  ];
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, extraContext: { events } });
  assert.match(out, /附加上下文/);
  assert.match(out, /大事提醒/);
  assert.match(out, /04-30.*研报.*测试研报标题/);
  assert.match(out, /04-29.*公告.*测试公告标题/);
  assert.match(out, /不要过度依赖单条研报或公告下定论/);
  assert.match(out, /最近 2 条/);
});

// ---- 多周期 ----

test('buildPrompt: 三种 period 表头措辞正确切换', () => {
  const monthly = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, period: 'monthly' });
  assert.match(monthly, /个月的月线数据/);

  const weekly = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, period: 'weekly' });
  assert.match(weekly, /周的周线数据/);

  const daily = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, period: 'daily' });
  assert.match(daily, /日的日线数据/);
});

test('buildPrompt: chanlun 三种周期的中枢级别正确切换', () => {
  const monthly = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun', period: 'monthly' });
  assert.match(monthly, /月线.*K 线图/);
  assert.match(monthly, /月线级别中枢/);

  const weekly = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun', period: 'weekly' });
  assert.match(weekly, /周线.*K 线图/);
  assert.match(weekly, /周线级别中枢/);

  const daily = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'chanlun', period: 'daily' });
  assert.match(daily, /日线.*K 线图/);
  assert.match(daily, /日线级别中枢/);
});

test('buildPrompt: comprehensive 三种周期的波动词汇正确切换', () => {
  const monthly = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'comprehensive', period: 'monthly' });
  assert.match(monthly, /单月波动/);

  const daily = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style: 'comprehensive', period: 'daily' });
  assert.match(daily, /单日波动/);
});

test('buildPrompt: 各风格各周期均不残留 {PERIOD} 或 {UNIT} 占位符', () => {
  const styles = ['technical', 'chanlun', 'value', 'comprehensive'];
  const periods = ['monthly', 'weekly', 'daily'];
  for (const style of styles) {
    for (const period of periods) {
      const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style, period });
      assert.ok(!/\{PERIOD\}/.test(out), `${style}/${period} 不应残留 {PERIOD}`);
      assert.ok(!/\{UNIT\}/.test(out), `${style}/${period} 不应残留 {UNIT}`);
    }
  }
});

// ---- 综合结论 + 风险声明 ----

test('buildPrompt: 4 种风格均包含综合结论小节', () => {
  const styles = ['technical', 'chanlun', 'value', 'comprehensive'];
  for (const style of styles) {
    const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style });
    assert.match(out, /综合结论/, `${style} 应包含"综合结论"`);
  }
});

test('buildPrompt: 4 种风格均包含风险声明', () => {
  const styles = ['technical', 'chanlun', 'value', 'comprehensive'];
  for (const style of styles) {
    const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style });
    assert.match(out, /本分析仅供研究学习使用，不构成投资建议/);
    assert.match(out, /市场存在不可预测因素/);
  }
});

test('buildPrompt: 4 种风格均禁止直接给买入/卖出建议', () => {
  const styles = ['technical', 'chanlun', 'value', 'comprehensive'];
  for (const style of styles) {
    const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style });
    assert.ok(
      /不要.*建议买入.*卖出/.test(out),
      `${style} 应包含"不要给建议买入/卖出"类指令`,
    );
  }
});

// ---- 输出权重要求 ----

test('buildPrompt: 4 种风格均包含输出权重要求段落', () => {
  const styles = ['technical', 'chanlun', 'value', 'comprehensive'];
  for (const style of styles) {
    const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style });
    assert.match(out, /输出权重要求/, `${style} 应包含"输出权重要求"`);
    assert.match(out, /当前位置.*下一步关键观察点/, `${style} 应包含当前位置+下一步指引`);
    assert.match(out, /综合结论.*小节.*放在报告末尾/, `${style} 应包含综合结论位置要求`);
    assert.match(out, /小节开头.*点明/, `${style} 应包含结论先行要求`);
    assert.match(out, /严格禁止.*为了凑数/, `${style} 应包含禁止凑数要求`);
  }
});

// ---- decisionMode ----

test('buildPrompt: decisionMode=true 时包含个人决策视角', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, decisionMode: true });
  assert.match(out, /个人决策视角/);
  assert.match(out, /关键止损位/);
  assert.match(out, /关键加仓.*减仓位/);
  assert.match(out, /持有时间预期/);
  assert.match(out, /仅供持有者本人/);
});

test('buildPrompt: decisionMode=false（默认）时不包含决策段', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines });
  assert.ok(!/个人决策视角/.test(out));
  assert.ok(!/关键止损位/.test(out));
  assert.ok(!/持有时间预期/.test(out));
});

test('buildPrompt: decisionMode + 各风格均正常拼装', () => {
  const styles = ['technical', 'chanlun', 'value', 'comprehensive'];
  for (const style of styles) {
    const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, style, decisionMode: true });
    assert.match(out, /个人决策视角/);
    assert.match(out, /不构成投资建议/); // DISCLAIMER 也在
  }
});

test('buildPrompt: PERSONAL_DECISION_BLOCK 按周期区分持有时间约束', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, decisionMode: true });
  assert.match(out, /严格匹配当前分析周期/);
  assert.match(out, /月线分析.*3-6 个月.*6-12 个月.*1 年以上/);
  assert.match(out, /周线分析.*1-3 个月.*3-6 个月.*6-12 个月/);
  assert.match(out, /日线分析.*1-4 周.*1-3 个月.*3-6 个月/);
  assert.match(out, /多周期共振分析.*3-6 个月.*6-12 个月/);
});

// ---- 价位精确化 + 估值仓位 ----

test('buildPrompt: PERSONAL_DECISION_BLOCK 要求价位精确到 2 位小数', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, decisionMode: true });
  assert.match(out, /精确到 2 位小数/);
  assert.match(out, /不允许给区间/);
});

test('buildPrompt: PERSONAL_DECISION_BLOCK 禁止套用学术模型', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, decisionMode: true });
  assert.match(out, /严禁套用 CAPM、DDM/);
  assert.match(out, /严禁套用凯利公式/);
});

test('buildPrompt: PERSONAL_DECISION_BLOCK 包含估值维度', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, decisionMode: true });
  assert.match(out, /相对估值判断/);
  assert.match(out, /明显低估/);
  assert.match(out, /明显高估/);
  assert.match(out, /基本面信息不足无法判断/);
});

test('buildPrompt: PERSONAL_DECISION_BLOCK 包含建议仓位 6 选项', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, decisionMode: true });
  assert.match(out, /建议仓位/);
  assert.match(out, /空仓观望/);
  assert.match(out, /试仓 5-10%/);
  assert.match(out, /轻仓 10-25%/);
  assert.match(out, /半仓 25-50%/);
  assert.match(out, /重仓 50-75%/);
  assert.match(out, /满仓 75-100%/);
});

test('buildPrompt: PERSONAL_DECISION_BLOCK 包含入场策略 5 选项', () => {
  const out = buildPrompt({ name: 'X', code: '000001', klines: sampleKlines, decisionMode: true });
  assert.match(out, /入场策略/);
  assert.match(out, /一次性入场/);
  assert.match(out, /分 2 批建仓/);
  assert.match(out, /分 3 批建仓/);
  assert.match(out, /等回调统一建仓/);
  assert.match(out, /不建仓/);
});
