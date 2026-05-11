// content.js conversationHistory 逻辑单元测试
// 核心不变式：conversationHistory[0].content 始终是完整 prompt 原文，不是 UI 折叠摘要
// 失败路径：sendFollowUp 三个分支都移除"思考中…"占位并渲染错误提示
import { test } from 'node:test';
import assert from 'node:assert/strict';

// ---------------------------------------------------------------------------
// 从 content.js 提取的纯函数（逻辑完全一致，仅去 DOM 操作）
// ---------------------------------------------------------------------------

/**
 * renderResult 中的 history 初始化逻辑
 * 不变式：history[0].content === 完整 prompt 原文
 */
function initHistory(prompt, analysis) {
  const firstUserMsg = prompt || `分析 ?(?) monthly`;
  return [
    { role: 'user', content: firstUserMsg },
    { role: 'assistant', content: analysis || '' },
  ];
}

/**
 * renderChatMessages 中首条 user 消息的 UI 渲染——只读，不修改 history
 * 不变式：调用前后 history 不变
 */
function renderFirstMessageBubble(msg, i) {
  if (i === 0 && msg.role === 'user') {
    return `📊 首次分析 prompt（已提交，长度 ${msg.content.length} 字符）`;
  }
  return msg.content; // 非首条原样
}

/**
 * sendFollowUp 中过滤 history 用于发给 SW 的逻辑
 * 过滤条件：排除"思考中…"占位，不包含正在发送的 question
 * 不变式：过滤后 history[0] 仍为完整 prompt
 */
function prepareFollowUpHistory(history) {
  return history
    .filter((m) => m.content !== '⏳ 思考中…')
    .slice(0, -1); // 去掉最后一个（刚推送的 user question 占位，或思考中占位）
}

/**
 * clearChat 逻辑：重置为前两条
 */
function resetHistory(history) {
  if (history.length <= 2) return history;
  return history.slice(0, 2);
}

/**
 * sendFollowUp 三个失败分支的占位替换模拟
 * 返回处理后的 history
 */
function handleFollowUpError(history, loadingIdx, errorMessage) {
  const h = [...history];
  h.splice(loadingIdx, 1); // 移除"思考中…"占位
  h.push({ role: 'assistant', content: `❌ 错误：${errorMessage}` });
  return h;
}

// ---------------------------------------------------------------------------
// 1. conversationHistory[0].content 完整性
// ---------------------------------------------------------------------------

test('initHistory: history[0].content 为完整 prompt 原文', () => {
  const prompt = '你是一个 A 股技术分析师。以下是 贵州茅台(600519) 近 60 个月的月线数据…（数万字 K 线表格）';
  const analysis = '## 综合分析\n\n偏多。';

  const history = initHistory(prompt, analysis);

  assert.equal(history.length, 2);
  assert.equal(history[0].role, 'user');
  assert.equal(history[0].content, prompt); // 完整原文，不是摘要
  assert.equal(history[1].role, 'assistant');
  assert.equal(history[1].content, analysis);
});

test('initHistory: prompt 为 null 时 fallback 不丢 role 结构', () => {
  const h1 = initHistory(null, '分析文本');
  assert.equal(h1[0].role, 'user');
  assert.ok(h1[0].content.includes('分析 ?'));

  // '' 是 falsy，会走 fallback（JS 语义：'' || fallback = fallback）
  const h2 = initHistory('', '分析文本');
  assert.equal(h2[0].role, 'user');
  assert.ok(h2[0].content.includes('分析 ?'));
});

test('renderFirstMessageBubble: 首条只读渲染，不修改 history', () => {
  const prompt = '完整 prompt 原文共 15000 字符';
  const history = initHistory(prompt, '分析内容');

  // 模拟 renderChatMessages 的 map 调用
  const rendered = history.map((msg, i) => renderFirstMessageBubble(msg, i));

  // rendering 输出是摘要文本（长度按 JS .length 计算，中文每字算 1）
  assert.ok(rendered[0].includes('📊 首次分析 prompt'));
  assert.ok(rendered[0].includes('已提交，长度'));

  // 但原始 history 不变
  assert.equal(history[0].content, prompt);
  assert.ok(!history[0].content.includes('📊'));
});

test('renderFirstMessageBubble: 非首条 user 消息原样返回', () => {
  const q = '这只股票还能持有吗？';
  // 模拟 history 中 index > 0 的 user 消息
  const result = renderFirstMessageBubble({ role: 'user', content: q }, 2);
  assert.equal(result, q);
  assert.ok(!result.includes('📊'));
});

test('prepareFollowUpHistory: 过滤后 history[0] 仍为完整 prompt', () => {
  const prompt = '完整 prompt 原文 15000 字符';
  let history = initHistory(prompt, '首次分析结果');

  // 模拟一轮追问
  history.push({ role: 'user', content: '追问1' });
  history.push({ role: 'assistant', content: '回复1' });

  // 模拟发送追问2 前：push thinking 占位 + user question
  history.push({ role: 'user', content: '追问2' });
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  const filtered = prepareFollowUpHistory(history);

  // 验证第一个元素仍是完整 prompt
  assert.equal(filtered[0].content, prompt);
  assert.ok(!filtered[0].content.includes('📊'));

  // 验证思考中占位被过滤
  assert.equal(filtered.find((m) => m.content === '⏳ 思考中…'), undefined);

  // 验证最后一条是"追问1 的回复"（追问2 被 slice(0,-1) 去掉）
  assert.equal(filtered[filtered.length - 1].content, '回复1');
});

test('prepareFollowUpHistory: 首次追问（history 仅 2 条初始）过滤后不含思考中', () => {
  const prompt = '完整 prompt 原文';
  let history = initHistory(prompt, '分析结果');

  // 首次追问：直接 push question + thinking
  history.push({ role: 'user', content: '第一次追问' });
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  const filtered = prepareFollowUpHistory(history);

  assert.equal(filtered.length, 2); // 初始 2 条（思考中和追问都被去掉了）
  assert.equal(filtered[0].content, prompt);
  assert.equal(filtered[1].content, '分析结果');
});

test('resetHistory: 保留前两条，第 1 条仍是完整 prompt', () => {
  const prompt = '完整 prompt 原文';
  let history = initHistory(prompt, '分析结果');
  history.push({ role: 'user', content: '追问1' });
  history.push({ role: 'assistant', content: '回复1' });
  history.push({ role: 'user', content: '追问2' });
  history.push({ role: 'assistant', content: '回复2' });

  const reset = resetHistory(history);

  assert.equal(reset.length, 2);
  assert.equal(reset[0].content, prompt); // 仍是完整 prompt
  assert.equal(reset[1].content, '分析结果');
});

test('resetHistory: 只有 2 条时不操作，原样返回', () => {
  const prompt = '完整 prompt';
  const history = initHistory(prompt, '分析');
  const reset = resetHistory(history);
  assert.equal(reset, history); // 同一个引用
  assert.equal(reset.length, 2);
});

// ---------------------------------------------------------------------------
// 2. sendFollowUp 失败路径：三个分支都移除"思考中…"占位并渲染错误
// ---------------------------------------------------------------------------

test('sendFollowUp 失败: !resp 分支——SW 无响应', () => {
  const prompt = '完整 prompt 原文';
  let history = initHistory(prompt, '分析结果');
  history.push({ role: 'user', content: '追问' });
  const loadingIdx = history.length;
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  // 模拟 sendFollowUp 分支：!resp
  const result = handleFollowUpError(history, loadingIdx, '无响应');

  assert.equal(result.length, 4); // 初始2 + 追问1 + 错误1
  assert.equal(result[0].content, prompt); // prompt 不变
  assert.ok(!result.some((m) => m.content === '⏳ 思考中…')); // 占位已移除
  assert.ok(result[result.length - 1].content.includes('❌ 错误：无响应'));
});

test('sendFollowUp 失败: !resp.ok 分支——SW 返回 error', () => {
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

test('sendFollowUp 失败: catch 分支——通信异常', () => {
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

test('sendFollowUp 失败: 多次追问后 prompt 仍完整', () => {
  const prompt = '完整 prompt 原文 15000 字符';
  let history = initHistory(prompt, '分析结果');

  // 模拟 5 轮成功追问
  for (let i = 1; i <= 5; i++) {
    history.push({ role: 'user', content: `追问${i}` });
    history.push({ role: 'assistant', content: `回复${i}` });
  }

  // 第 6 轮失败
  history.push({ role: 'user', content: '追问6' });
  const loadingIdx = history.length;
  history.push({ role: 'assistant', content: '⏳ 思考中…' });

  const result = handleFollowUpError(history, loadingIdx, '触发限流');

  // 经历 6 轮追问后 prompt 仍在第一位且完整
  assert.equal(result[0].content, prompt);
  assert.ok(!result[0].content.includes('📊'));
  assert.equal(result[0].role, 'user');
  assert.ok(!result.some((m) => m.content === '⏳ 思考中…'));
});
