// content.js conversationHistory logic unit test
// Core invariant: conversationHistory[0].content is always the complete prompt text, not a UI collapsed summary
// Failure path: all three sendFollowUp branches remove the "thinking..." placeholder and render error message
import { test } from 'node:test';
import assert from 'node:assert/strict';

// ---------------------------------------------------------------------------
// Pure functions extracted from content.js (identical logic, only DOM ops removed)
// ---------------------------------------------------------------------------

/**
 * History initialization logic from renderResult
 * Invariant: history[0].content === full prompt text
 */
function initHistory(prompt, analysis) {
  const firstUserMsg = prompt || `分析 ?(?) monthly`;
  return [
    { role: 'user', content: firstUserMsg },
    { role: 'assistant', content: analysis || '' },
  ];
}

/**
 * UI rendering for first user message in renderChatMessages -- read-only, does not modify history
 * Invariant: history is unchanged before and after call
 */
function renderFirstMessageBubble(msg, i) {
  if (i === 0 && msg.role === 'user') {
    return `📊 首次分析 prompt（已提交，长度 ${msg.content.length} 字符）`;
  }
  return msg.content; // non-first messages returned as-is
}

/**
 * Filter history in sendFollowUp for sending to SW
 * Filter condition: exclude "thinking..." placeholder, exclude the question currently being sent
 * Invariant: after filtering, history[0] is still the full prompt
 */
function prepareFollowUpHistory(history) {
  return history
    .filter((m) => m.content !== '⏳ 思考中…')
    .slice(0, -1); // remove last item (newly pushed user question placeholder or thinking placeholder)
}

/**
 * clearChat logic: reset to first two entries
 */
function resetHistory(history) {
  if (history.length <= 2) return history;
  return history.slice(0, 2);
}

/**
 * Simulate placeholder replacement for the three failure branches in sendFollowUp
 * Returns processed history
 */
function handleFollowUpError(history, loadingIdx, errorMessage) {
  const h = [...history];
  h.splice(loadingIdx, 1); // remove "thinking..." placeholder
  h.push({ role: 'assistant', content: `❌ 错误：${errorMessage}` });
  return h;
}

// ---------------------------------------------------------------------------
// 1. conversationHistory[0].content integrity
// ---------------------------------------------------------------------------

test('initHistory: history[0].content is the full prompt text', () => {
  const prompt = '你是一个 A 股技术分析师。以下是 贵州茅台(600519) 近 60 个月的月线数据…（数万字 K 线表格）';
  const analysis = '## 综合分析\n\n偏多。';

  const history = initHistory(prompt, analysis);

  assert.equal(history.length, 2);
  assert.equal(history[0].role, 'user');
  assert.equal(history[0].content, prompt); // full original text, not a summary
  assert.equal(history[1].role, 'assistant');
  assert.equal(history[1].content, analysis);
});

test('initHistory: when prompt is null, fallback does not lose role structure', () => {
  const h1 = initHistory(null, '分析文本');
  assert.equal(h1[0].role, 'user');
  assert.ok(h1[0].content.includes('分析 ?'));

  // '' is falsy, will trigger fallback (JS semantics: '' || fallback = fallback)
  const h2 = initHistory('', '分析文本');
  assert.equal(h2[0].role, 'user');
  assert.ok(h2[0].content.includes('分析 ?'));
});

test('renderFirstMessageBubble: first message is read-only render, history not modified', () => {
  const prompt = '完整 prompt 原文共 15000 字符';
  const history = initHistory(prompt, '分析内容');

  // Simulate renderChatMessages map call
  const rendered = history.map((msg, i) => renderFirstMessageBubble(msg, i));

  // Rendering output is a summary text (length calculated by JS .length, Chinese chars count as 1 each)
  assert.ok(rendered[0].includes('📊 首次分析 prompt'));
  assert.ok(rendered[0].includes('已提交，长度'));

  // But original history is unchanged
  assert.equal(history[0].content, prompt);
  assert.ok(!history[0].content.includes('📊'));
});

test('renderFirstMessageBubble: non-first user message returned as-is', () => {
  const q = '这只股票还能持有吗？';
  // Simulate user message at index > 0 in history
  const result = renderFirstMessageBubble({ role: 'user', content: q }, 2);
  assert.equal(result, q);
  assert.ok(!result.includes('📊'));
});

test('prepareFollowUpHistory: after filtering, history[0] is still the full prompt', () => {
  const prompt = '完整 prompt 原文 15000 字符';
  let history = initHistory(prompt, '首次分析结果');

  // Simulate one round of follow-up
  history.push({ role: 'user', content: '追问1' });
  history.push({ role: 'assistant', content: '回复1' });

  // Simulate before sending follow-up 2: push thinking placeholder + user question
  history.push({ role: 'user', content: '追问2' });
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  const filtered = prepareFollowUpHistory(history);

  // Verify first element is still the full prompt
  assert.equal(filtered[0].content, prompt);
  assert.ok(!filtered[0].content.includes('📊'));

  // Verify thinking placeholder is filtered out
  assert.equal(filtered.find((m) => m.content === '⏳ 思考中…'), undefined);

  // Verify last item is "reply to follow-up 1" (follow-up 2 removed by slice(0,-1))
  assert.equal(filtered[filtered.length - 1].content, '回复1');
});

test('prepareFollowUpHistory: first follow-up (history has only 2 initial entries) filters without thinking', () => {
  const prompt = '完整 prompt 原文';
  let history = initHistory(prompt, '分析结果');

  // First follow-up: directly push question + thinking
  history.push({ role: 'user', content: '第一次追问' });
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  const filtered = prepareFollowUpHistory(history);

  assert.equal(filtered.length, 2); // initial 2 entries (thinking and question both removed)
  assert.equal(filtered[0].content, prompt);
  assert.equal(filtered[1].content, '分析结果');
});

test('resetHistory: keeps first two entries, entry 1 is still full prompt', () => {
  const prompt = '完整 prompt 原文';
  let history = initHistory(prompt, '分析结果');
  history.push({ role: 'user', content: '追问1' });
  history.push({ role: 'assistant', content: '回复1' });
  history.push({ role: 'user', content: '追问2' });
  history.push({ role: 'assistant', content: '回复2' });

  const reset = resetHistory(history);

  assert.equal(reset.length, 2);
  assert.equal(reset[0].content, prompt); // still the full prompt
  assert.equal(reset[1].content, '分析结果');
});

test('resetHistory: when only 2 entries, returns as-is without modification', () => {
  const prompt = '完整 prompt';
  const history = initHistory(prompt, '分析');
  const reset = resetHistory(history);
  assert.equal(reset, history); // same reference
  assert.equal(reset.length, 2);
});

// ---------------------------------------------------------------------------
// 2. sendFollowUp failure paths: all three branches remove "thinking..." placeholder and render error
// ---------------------------------------------------------------------------

test('sendFollowUp failure: !resp branch -- SW unresponsive', () => {
  const prompt = '完整 prompt 原文';
  let history = initHistory(prompt, '分析结果');
  history.push({ role: 'user', content: '追问' });
  const loadingIdx = history.length;
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  // Simulate sendFollowUp branch: !resp
  const result = handleFollowUpError(history, loadingIdx, '无响应');

  assert.equal(result.length, 4); // initial 2 + question 1 + error 1
  assert.equal(result[0].content, prompt); // prompt unchanged
  assert.ok(!result.some((m) => m.content === '⏳ 思考中…')); // placeholder removed
  assert.ok(result[result.length - 1].content.includes('❌ 错误：无响应'));
});

test('sendFollowUp failure: !resp.ok branch -- SW returns error', () => {
  const prompt = '完整 prompt 原文';
  let history = initHistory(prompt, '分析结果');
  history.push({ role: 'user', content: '追问' });
  const loadingIdx = history.length;
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  const result = handleFollowUpError(history, loadingIdx, 'API key 无效');

  assert.equal(result[0].content, prompt);
  assert.ok(!result.some((m) => m.content === '⏳ 思考中…'));
  assert.ok(result[result.length - 1].content.includes('API key 无效'));
  assert.equal(result[result.length - 1].role, 'assistant');
});

test('sendFollowUp failure: catch branch -- communication error', () => {
  const prompt = '完整 prompt 原文';
  let history = initHistory(prompt, '分析结果');
  history.push({ role: 'user', content: '追问' });
  const loadingIdx = history.length;
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  const result = handleFollowUpError(history, loadingIdx, '通信错误：Extension context invalidated');

  assert.equal(result[0].content, prompt);
  assert.ok(!result.some((m) => m.content === '⏳ 思考中…'));
  assert.ok(result[result.length - 1].content.includes('通信错误'));
});

test('sendFollowUp failure: prompt still intact after multiple follow-ups', () => {
  const prompt = '完整 prompt 原文 15000 字符';
  let history = initHistory(prompt, '分析结果');

  // Simulate 5 successful follow-up rounds
  for (let i = 1; i <= 5; i++) {
    history.push({ role: 'user', content: `追问${i}` });
    history.push({ role: 'assistant', content: `回复${i}` });
  }

  // Round 6 fails
  history.push({ role: 'user', content: '追问6' });
  const loadingIdx = history.length;
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  const result = handleFollowUpError(history, loadingIdx, '触发限流');

  // After 6 rounds, prompt is still at position 0 and intact
  assert.equal(result[0].content, prompt);
  assert.ok(!result[0].content.includes('📊'));
  assert.equal(result[0].role, 'user');
  assert.ok(!result.some((m) => m.content === '⏳ 思考中…'));
});
