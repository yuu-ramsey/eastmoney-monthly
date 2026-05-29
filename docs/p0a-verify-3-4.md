# P0a 验证 #3 #4：复用判定 + SW 死亡 UI 安全

> 分支: `p0a-verify-3-4` | 日期: 2026-05-29 | 基于 p0a-fix-fingerprint

---

## #3: 失败 Agent 不被复用

### 问题

`mergeCheckpointError` 写入的持久化错误是否会导致续跑时跳过该 Agent？（应
该重新调用 LLM，不应该跳过。）

### 证据

`lib/agents/runner.js:138`:

```js
const hasPartial = (role) => checkpoint && checkpoint.partials && checkpoint.partials[role];
```

仅检查 `checkpoint.partials[role]`，不看 `errors`。

`lib/agents/runner.js:101-113` (`mergeCheckpointError`):

```js
await chrome.storage.local.set({
  [key]: {
    v: CHECKPOINT_VERSION,
    ts: Date.now(),
    fp: fingerprint,
    partials: { ...prev.partials },          // ← 不写 role
    errors: { ...prev.errors, [role]: errorMsg },  // ← 只写 errors
  },
});
```

错误写入 `errors[role]`，不写 `partials[role]`。

`lib/agents/runner.js:150-153`:

```js
const promises = agentDefs.map(({ agent, role }) => {
  if (hasPartial(role)) {
    return Promise.resolve(checkpoint.partials[role]);  // 只有 partials 存在才复用
  }
  return agent.run(ctx, opts).then(/* ... */);
});
```

### 结论

**通过。** 失败 Agent 不会被复用——`hasPartial(role)` 对 error-only 的 role
返回 falsy，Agent 会重新调 `agent.run()`。

### 新增测试

`test/agents/runner.test.js`:
- 预置 checkpoint: `partials: { bull }` + `errors: { bear: 'LLM timeout' }`
- 断言: bull 复用（text='bull checkpoint 缓存'），bear 重新调用（fetch = 3
  次，含 bear+predictor+judge），bear.text ≠ 'bear checkpoint 缓存'

---

## #4: SW 终止 → UI 可重试失败态（非无限 spinner）

### 问题

SW 在辩论中途被 Chrome 终止后，content.js 的消息通道是否一定会落到"显示错
误、用户可以重试"的终止态，还是可能卡在无限 loading spinner？

### 证据

`content.js:383-422` (`sendAndShow` 函数):

```js
const resp = await chrome.runtime.sendMessage({
  type: 'ANALYZE', url: location.href, force, pageEvents,
});
if (!resp) {
  setBody('<div class="error">service worker 无响应，可能扩展未加载</div>');
  return;
}
if (!resp.ok) {
  setBody(`<div class="error">${escapeHtml(resp.error || '未知错误')}</div>`);
  return;
}
// ... 正常渲染 ...
} catch (err) {
  setBody(`<div class="error">通信错误: ${escapeHtml(err.message || String(err))}</div>`);
} finally {
  hideThinkingStream();
  busy = false;
  fabEl.disabled = false;
  reanalyzeEl.disabled = false;
}
```

三条错误路径：

| 路径 | 触发条件 | 结果 |
|------|---------|------|
| `!resp` | SW 无响应（被终止/未加载） | 显示"service worker 无响应" |
| `!resp.ok` | SW 返回 error | 显示 resp.error 文本 |
| `catch` | 通信异常（通道断开等） | 显示"通信错误: ..." |

`finally` 块在三种路径都会执行：隐藏 loading 动画，重置 busy 标志，恢复
按钮可用状态。用户看到错误提示后可以关闭面板重试。

不存在"消息发出后 SW 死亡、content 侧无限等待"的路径——
`chrome.runtime.sendMessage` 在 SW 无响应时返回 `undefined`（触发
`!resp` 分支），不抛出异常也不挂起。

### 结论

**通过。** SW 死亡 → `!resp` → 显示错误文本 → `finally` 清理状态。不会
出现无限 spinner。

---

## 未改动

| 模块 | 状态 |
|------|------|
| Agent prompt (bull/bear/predictor/judge) | 未碰 |
| LLM provider | 未碰 |
| score-fusion | 未碰 |
| 结构化输出解析 | 未碰 |
| content.js 错误处理 | 只读验证，未改 |
| runDebate 返回结构 | 未变 |

## 测试

```
252 tests | 0 fail | 0 skip
```

新增 1 个 #3 验证测试（pred 错误 → 不被复用），总计 7 个 checkpoint 专项测试。

## 人工审核清单

- [ ] 交易时段：Stop SW（`chrome://serviceworker-internals` → Stop）
- [ ] 点重试分析按钮
- [ ] 确认 UI 显示"service worker 无响应"（非无限 spinner）
- [ ] 再次 Stop → 重试 → console 确认复用了已完成 Agent
- [ ] `node --test test/agents/runner.test.js` → 11/11 pass
