# P0a 辩论 checkpoint 续跑（含指纹修复）

> 分支: `p0a-debate-checkpoint` → `p0a-fix-fingerprint` | 日期: 2026-05-29 | 基于 p0-audit 审计结论实施

---

## 问题

p0-audit 确认：辩论中间态纯存内存变量 `partials`，无断点续跑。Service Worker
在辩论中途被 Chrome 终止后，Bull/Bear/Predictor 三个 Agent 的 LLM 调用结果
全部丢失，用户重试需从头跑，重复消耗 token（单 Agent 约 0.01-0.05 USD）。

## 方案

在 `runDebate` 外围加"持久化 + 续跑"，不改动任何 Agent prompt / LLM provider /
score-fusion / 结构化输出解析器。

### checkpoint 数据流

```
runDebate(ctx, opts)
  │
  ├─ 1. loadCheckpoint(debate-wip:...)  ← chrome.storage.local
  │     ├─ fp 匹配 → 复用已完成 partial
  │     └─ fp 不匹配 → 丢弃，全部重跑
  │
  ├─ 2. Bull/Bear/Predictor 并发（串行写链避免竞态）
  │     ├─ 已复用: Promise.resolve(chk.partials[role])
  │     └─ 新调: agent.run() → fulfilled → mergeChain 落盘
  │
  ├─ 3. Judge 综合（successCount ≥ 2）
  │
  └─ 4. await mergeChain（确保落盘完成）
       → return { partials, errors, judge, ... }
       → background.js 写正式缓存 → clearDebateCheckpoint()
```

### checkpoint key

```
debate-wip:<market>.<code>:<period>:<bucket>:<style>:<decision>
```

与最终缓存 key `analysis:...` 共用同一身份标识，只换命名空间。

### 输入指纹

```
djb2(code|period|barCount|firstDate|lastDate|lastClose)  ← 仅已收盘 bar
```

**指纹修复 (p0a-fix-fingerprint)**: 初版指纹含当期未收盘 K 线的 close，
交易时段盘中每次 tick 都改变指纹，导致 checkpoint 在盘中永远无法命中。
修复后加入 `isBarClosed()` 过滤：

- monthly: 当期月不参与指纹
- weekly: 当期 ISO 周不参与指纹
- daily: 今日不参与指纹

当期 bar close 跳动不再影响指纹。若已收盘 bar 的 close 变了（复权基准
切换、数据源修正），指纹自动失效丢弃重跑。

### 超时清理

checkpoint 超过 1 小时 (`CHECKPOINT_STALE_MS = 60*60*1000`) 自动丢弃，
避免 storage 堆积。所有 catch 块均 `console.warn`，不再静默吞错。

### 串行写链

三个 Agent 并发调 LLM，但写 checkpoint 用串行写链 `mergeChain`：

```js
let mergeChain = Promise.resolve();
// Agent A 完成 → mergeChain = mergeChain.then(() => save(A))
// Agent B 完成 → mergeChain = mergeChain.then(() => save(B))
// ...
// await mergeChain → 确保全部落盘
```

避免并发读-改-写竞态（两个 Agent 同时读到空 checkpoint，后写覆盖先写）。

## 改动文件

### `lib/agents/runner.js` (+90 行)

| 新增 | 说明 |
|------|------|
| `CHECKPOINT_VERSION = 1` | schema 版本，将来改结构时可递增 |
| `CHECKPOINT_STALE_MS` | 1 小时超时，过期 checkpoint 自动丢弃 |
| `getIsoWeek(dateStr)` | ISO 周计算，与 background.js 一致 |
| `isBarClosed(dateStr, period)` | 判断 bar 是否属于当期（monthly/weekly/daily），用于指纹过滤 |
| `buildFingerprint(ctx)` | djb2 hash of `code\|period\|barCount\|firstDate\|lastDate\|lastClose`（仅已收盘 bar） |
| `loadCheckpoint(key, fp)` | 读 storage，验指纹+超时，不匹配则删除 |
| `mergeCheckpointPartial(key, role, value, fp)` | 合并写入单个 Agent 成功结果 |
| `mergeCheckpointError(key, role, errMsg, fp)` | 合并写入单个 Agent 失败信息 |
| `clearDebateCheckpoint(key)` | 删除 checkpoint（导出函数，background.js 调用） |

`runDebate` 修改：
- 入口：加载 checkpoint，验指纹
- Agent 循环：有 checkpoint 则 `Promise.resolve` 复用，无则调 `agent.run()`
- 每个 Agent 完成时 append 到 `mergeChain`（不阻塞兄弟 Agent）
- Judge 之后 `await mergeChain` 确保落盘完成

**未碰**: `sumCost`、Judge 逻辑、返回结构——逐字段与改前一致。

### `background.js` (+4 行)

```
:14   import { runDebate, clearDebateCheckpoint }  // 新增 clearDebateCheckpoint
:274  const checkpointKey = `debate-wip:...`       // 构造 key
:280  checkpointKey,                                 // 传入 opts
:507  await clearDebateCheckpoint(...)               // 正式缓存写入后清理
```

### `test/agents/runner.test.js` (+170 行)

新增 mock `chrome.storage.local`（Map 实现）+ 6 个 checkpoint 专项测试：

| 测试 | 场景 | 断言 |
|------|------|------|
| 无 checkpoint → 三个全调 | storage 为空 | fetch 4 次（Bull+Bear+Pred+Judge），checkpoint 已落盘 |
| bull+bear 已缓存 → 只调 predictor | 预设 checkpoint 含 bull+bear | fetch 2 次（Pred+Judge），bull.text='bull checkpoint 缓存' |
| 指纹不匹配 → 丢弃重跑 | 预设 checkpoint fp 错误 | fetch 4 次，旧值未出现在结果中 |
| ≥2 成功规则不破 | checkpoint 仅 bull，其余 500 | successCount=1，Judge 跳过 |
| 当期 bar close 变动 → 指纹不变 | closed bar 不变，加当期 bar 模拟盘中 | fetch 2 次，bull+bear 复用 checkpoint |
| closed bar close 变动 → 丢弃 | 修改 closed bar close 模拟复权切换 | fetch 4 次，全量重跑 |

### `PROGRESS.md` (+20 行)

"必须遵守的规则"下新增"辩论 checkpoint 续跑"节。

## 未改动（硬约束验证）

| 模块 | 验证方式 |
|------|---------|
| Agent prompt (bull/bear/predictor/judge) | `git diff master -- lib/agents/{bull,bear,predictor,judge}.js` → 无改动 |
| LLM provider (anthropic/deepseek) | `git diff master -- lib/llm/` → 无改动 |
| score-fusion | `git diff master -- lib/score-fusion.js` → 无改动 |
| 结构化输出解析 | `git diff master -- lib/parse-structured-output.js` → 无改动 |
| runDebate 返回结构 | 源码 diff 可见 `return { partials, errors, judge, totalCost, totalDurationMs }` 未变 |

## P0 边界

- **做了**: 用户重试时自动从 checkpoint 续跑，复用已完成的 Agent，不重复花钱
- **没做**: alarm 看门狗自动恢复（P1）、Judge 结果缓存、持久化 keepalive 定时器
- **静默降级**: storage 不可用时 catch 吞错，辩论正常进行（退回无 checkpoint 行为）

## 测试

```
251 tests | 0 fail | 0 skip
```

新增 6 个 checkpoint 测试（含 2 个指纹专项），现有测试零回归。

## 人工审核清单

- [ ] `git diff master...p0a-fix-fingerprint` 确认未碰 prompt/LLM/评分/解析
- [ ] 比较 `runDebate` 返回结构与改前逐字段一致
- [ ] 跑 `node --test test/agents/runner.test.js` → 10/10 pass
- [ ] 真实环境：交易时段 Stop SW → 重试分析 → console 确认 checkpoint 命中、复用 Agent
- [ ] 指纹专项：盘中当期 bar close 跳动不影响指纹 → checkpoint 可命中
- [ ] 指纹专项：修改 closed bar close → checkpoint 被丢弃重跑
