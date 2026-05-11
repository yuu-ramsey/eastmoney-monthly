import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  HISTORY_KEY,
  MAX_HISTORY_ITEMS,
  MAX_HISTORY_BYTES,
  generateHistoryId,
  trimHistory,
  formatHistoryDate,
  historyToMarkdown,
  checkCapacity,
} from '../lib/history.js';

// ---- 常量 ----

test('HISTORY_KEY 正确', () => {
  assert.equal(HISTORY_KEY, 'history');
});

test('MAX_HISTORY_ITEMS = 100', () => {
  assert.equal(MAX_HISTORY_ITEMS, 100);
});

test('MAX_HISTORY_BYTES = 9MB', () => {
  assert.equal(MAX_HISTORY_BYTES, 9 * 1024 * 1024);
});

// ---- generateHistoryId ----

test('generateHistoryId: 格式 h_<timestamp>_<random>', () => {
  const id = generateHistoryId();
  assert.ok(/^h_\d{13}_[a-z0-9]{6}$/.test(id), `实际: ${id}`);
});

test('generateHistoryId: 连续两次不重复', () => {
  const ids = new Set(Array.from({ length: 20 }, () => generateHistoryId()));
  assert.equal(ids.size, 20);
});

// ---- trimHistory ----

test('trimHistory: 不超限时不改变原数组内容', () => {
  const list = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
  const result = trimHistory([...list]);
  assert.equal(result.length, 3);
  assert.deepEqual(result[0], { id: 'a' });
});

test('trimHistory: 超过上限时删最旧（头部）', () => {
  const list = Array.from({ length: 105 }, (_, i) => ({ id: String(i) }));
  const result = trimHistory(list);
  assert.equal(result.length, 100);
  // 删了前 5 条
  assert.equal(result[0].id, '5');
  assert.equal(result[99].id, '104');
});

test('trimHistory: 空数组不变', () => {
  assert.equal(trimHistory([]).length, 0);
});

test('trimHistory: 自定义 maxItems', () => {
  const list = [{ id: '1' }, { id: '2' }, { id: '3' }];
  const result = trimHistory(list, 2);
  assert.equal(result.length, 2);
  assert.equal(result[0].id, '2');
});

// ---- formatHistoryDate ----

test('formatHistoryDate: 正常格式化', () => {
  const result = formatHistoryDate(Date.UTC(2026, 4, 10, 12, 0, 0));
  assert.equal(result, '2026-05-10');
});

test('formatHistoryDate: 单数月份补零', () => {
  // 用中午 12 点 UTC 避免任何时区偏移导致的日期变化
  const result = formatHistoryDate(Date.UTC(2026, 0, 5, 12, 0, 0));
  assert.equal(result, '2026-01-05');
});

// ---- historyToMarkdown ----

test('historyToMarkdown: 基本字段正确', () => {
  const entry = {
    name: '贵州茅台',
    code: '600519',
    provider: 'anthropic',
    model: 'claude-sonnet-4-6',
    template: 'technical',
    timestamp: Date.UTC(2026, 4, 10, 12, 0, 0),
    analysis: '## 趋势判断\n\n多头排列。\n\n## 风险\n\n注意回调。',
    conversationHistory: null,
  };
  const md = historyToMarkdown(entry);
  assert.ok(md.includes('# 贵州茅台 (600519)'));
  assert.ok(md.includes('- 时间: 2026-05-10'));
  assert.ok(md.includes('- Provider: Claude / claude-sonnet-4-6'));
  assert.ok(md.includes('- 分析维度: 技术面'));
  assert.ok(md.includes('## 分析结果'));
  assert.ok(md.includes('## 趋势判断'));
  // 无追问
  assert.ok(!md.includes('追问记录'));
});

test('historyToMarkdown: DeepSeek provider 正确标注', () => {
  const entry = {
    name: '测试',
    code: '000001',
    provider: 'deepseek',
    model: 'deepseek-chat',
    template: 'trend',
    timestamp: Date.now(),
    analysis: '分析内容',
    conversationHistory: null,
  };
  const md = historyToMarkdown(entry);
  assert.ok(md.includes('- Provider: DeepSeek / deepseek-chat'));
  assert.ok(md.includes('- 分析维度: 趋势判断'));
});

test('historyToMarkdown: conversationHistory 有追问时包含追问段', () => {
  const entry = {
    name: '测试',
    code: '000001',
    provider: 'anthropic',
    model: 'claude-sonnet-4-6',
    template: 'technical',
    timestamp: Date.now(),
    analysis: '分析内容',
    conversationHistory: [
      { role: 'user', content: '完整 prompt...' },
      { role: 'assistant', content: '分析结果...' },
      { role: 'user', content: 'MACD 背离怎么看？' },
      { role: 'assistant', content: '当前 MACD 无背离信号。' },
    ],
  };
  const md = historyToMarkdown(entry);
  assert.ok(md.includes('## 追问记录'));
  assert.ok(md.includes('### 追问'));
  assert.ok(md.includes('MACD 背离怎么看？'));
  assert.ok(md.includes('### 回复'));
  assert.ok(md.includes('无背离信号'));
});

test('historyToMarkdown: conversationHistory 仅 2 条时不输出追问段', () => {
  const entry = {
    name: '测试',
    code: '000001',
    provider: 'anthropic',
    model: 'claude-sonnet-4-6',
    template: 'technical',
    timestamp: Date.now(),
    analysis: '分析内容',
    conversationHistory: [
      { role: 'user', content: 'prompt' },
      { role: 'assistant', content: '分析' },
    ],
  };
  const md = historyToMarkdown(entry);
  assert.ok(!md.includes('追问记录'));
});

test('historyToMarkdown: 缺失字段不抛错', () => {
  const md = historyToMarkdown({});
  assert.ok(md.includes('# ?'));
  assert.ok(md.includes('?'));
});

test('historyToMarkdown: 未知 template 显示原值', () => {
  const entry = {
    name: '测试',
    code: '000001',
    provider: 'anthropic',
    model: 'm',
    template: 'unknown',
    timestamp: Date.now(),
    analysis: '',
    conversationHistory: null,
  };
  const md = historyToMarkdown(entry);
  assert.ok(md.includes('- 分析维度: unknown'));
});

// ---- checkCapacity ----

test('checkCapacity: 条数不超限', () => {
  const list = Array.from({ length: 50 }, (_, i) => ({ id: String(i), analysis: 'x'.repeat(100) }));
  const result = checkCapacity(list);
  assert.equal(result.trimmed, false);
});

test('checkCapacity: 条数超限', () => {
  const list = Array.from({ length: 105 }, (_, i) => ({ id: String(i) }));
  const result = checkCapacity(list);
  assert.equal(result.trimmed, true);
  assert.ok(result.reason.includes('条数超限'));
});

test('checkCapacity: 体积超限', () => {
  const list = Array.from({ length: 5 }, (_, i) => ({
    id: String(i),
    analysis: 'x'.repeat(2 * 1024 * 1024), // 2MB each
  }));
  const result = checkCapacity(list);
  assert.equal(result.trimmed, true);
  assert.ok(result.reason.includes('体积超限'));
});

test('checkCapacity: 自定义参数', () => {
  const list = [{ id: '1' }, { id: '2' }, { id: '3' }];
  const result = checkCapacity(list, 2, 999999);
  assert.equal(result.trimmed, true);
  assert.ok(result.reason.includes('条数超限'));
});
