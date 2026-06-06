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

// ---- Constants ----

test('HISTORY_KEY is correct', () => {
  assert.equal(HISTORY_KEY, 'history');
});

test('MAX_HISTORY_ITEMS = 100', () => {
  assert.equal(MAX_HISTORY_ITEMS, 100);
});

test('MAX_HISTORY_BYTES = 9MB', () => {
  assert.equal(MAX_HISTORY_BYTES, 9 * 1024 * 1024);
});

// ---- generateHistoryId ----

test('generateHistoryId: format h_<timestamp>_<random>', () => {
  const id = generateHistoryId();
  assert.ok(/^h_\d{13}_[a-z0-9]{6}$/.test(id), `actual: ${id}`);
});

test('generateHistoryId: no duplicates in consecutive calls', () => {
  const ids = new Set(Array.from({ length: 20 }, () => generateHistoryId()));
  assert.equal(ids.size, 20);
});

// ---- trimHistory ----

test('trimHistory: does not modify content when under limit', () => {
  const list = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
  const result = trimHistory([...list]);
  assert.equal(result.length, 3);
  assert.deepEqual(result[0], { id: 'a' });
});

test('trimHistory: removes oldest (head) when over limit', () => {
  const list = Array.from({ length: 105 }, (_, i) => ({ id: String(i) }));
  const result = trimHistory(list);
  assert.equal(result.length, 100);
  // removed first 5 entries
  assert.equal(result[0].id, '5');
  assert.equal(result[99].id, '104');
});

test('trimHistory: empty array unchanged', () => {
  assert.equal(trimHistory([]).length, 0);
});

test('trimHistory: custom maxItems', () => {
  const list = [{ id: '1' }, { id: '2' }, { id: '3' }];
  const result = trimHistory(list, 2);
  assert.equal(result.length, 2);
  assert.equal(result[0].id, '2');
});

// ---- formatHistoryDate ----

test('formatHistoryDate: normal formatting', () => {
  const result = formatHistoryDate(Date.UTC(2026, 4, 10, 12, 0, 0));
  assert.equal(result, '2026-05-10');
});

test('formatHistoryDate: zero-pads single-digit month', () => {
  // Use noon UTC to avoid any timezone offset date changes
  const result = formatHistoryDate(Date.UTC(2026, 0, 5, 12, 0, 0));
  assert.equal(result, '2026-01-05');
});

// ---- historyToMarkdown ----

test('historyToMarkdown: basic fields are correct', () => {
  const entry = {
    name: 'Kweichow Moutai',
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
  // no follow-up
  assert.ok(!md.includes('追问记录'));
});

test('historyToMarkdown: DeepSeek provider correctly labeled', () => {
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

test('historyToMarkdown: includes follow-up section when conversationHistory has follow-ups', () => {
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

test('historyToMarkdown: does not output follow-up section when conversationHistory has only 2 entries', () => {
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

test('historyToMarkdown: missing fields do not throw error', () => {
  const md = historyToMarkdown({});
  assert.ok(md.includes('# ?'));
  assert.ok(md.includes('?'));
});

test('historyToMarkdown: unknown template displays original value', () => {
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

test('checkCapacity: item count not over limit', () => {
  const list = Array.from({ length: 50 }, (_, i) => ({ id: String(i), analysis: 'x'.repeat(100) }));
  const result = checkCapacity(list);
  assert.equal(result.trimmed, false);
});

test('checkCapacity: item count over limit', () => {
  const list = Array.from({ length: 105 }, (_, i) => ({ id: String(i) }));
  const result = checkCapacity(list);
  assert.equal(result.trimmed, true);
  assert.ok(result.reason.includes('条数超限'));
});

test('checkCapacity: size over limit', () => {
  const list = Array.from({ length: 5 }, (_, i) => ({
    id: String(i),
    analysis: 'x'.repeat(2 * 1024 * 1024), // 2MB each
  }));
  const result = checkCapacity(list);
  assert.equal(result.trimmed, true);
  assert.ok(result.reason.includes('体积超限'));
});

test('checkCapacity: custom params', () => {
  const list = [{ id: '1' }, { id: '2' }, { id: '3' }];
  const result = checkCapacity(list, 2, 999999);
  assert.equal(result.trimmed, true);
  assert.ok(result.reason.includes('条数超限'));
});
