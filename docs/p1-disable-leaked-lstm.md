# P1 关闭泄漏的 LSTM 运行时注入

> 分支: `p1-disable-leaked-lstm` | 日期: 2026-05-29 | 基于 `docs/p1-lstm-leak-check.md` 审计结论

---

## 问题

审计确认 `daily_lstm7.pt` 训练数据无时间切分（`scripts/daily_to_monthly_aggr.py`），2024-2025 全在训练集内。所有 eval 时点 + 运行时均受前视泄漏影响。待 walk-forward 重训 + 基线验证前，不应向 prompt 注入任何 LSTM 信号。

## 方案

加 `ENABLE_LSTM_SIGNAL = false`（默认 OFF），跳过整个 native message 调用，利用已有的 null 降级路径。

### 改动

**`background.js:324-367`** — 用开关包裹 LSTM 获取逻辑：

```js
const ENABLE_LSTM_SIGNAL = false;
let lstmSignalData = null;
if (ENABLE_LSTM_SIGNAL) {
  // ...原有 native message + build lstmSignalData...
}
```

OFF 时 `lstmSignalData` 保持 `null`，下游路径：

| 步骤 | 行为 |
|------|------|
| `build-prompt.js:279` | 默认参数 `lstmSignalData = null` |
| `build-prompt.js:344` | `buildLstmSignalBlock(null)` |
| `prompt-templates.js:98` | `!signalData` → `return ''` |
| 模板约束 #14 | "若上方存在 LSTM 段" → 无 block 则跳过 |

与 native host 不可用完全相同的降级——审计已验证安全（`docs/p0a-verify-3-4.md` #4）。

### 未改动

| 模块 | 状态 |
|------|------|
| prompt 结构 | 未碰 |
| LLM provider | 未碰 |
| 评分/解析 | 未碰 |
| 模型文件/训练脚本 | 保留不动，待重训 |

## 测试

```
11 tests | 0 fail (prompt-templates.test.js)
```

## 重新启用条件

1. 用严格 walk-forward split（train ≤ 2021-12, val 2022-2023, test 2024+）重训 daily LSTM
2. 新模型在 test 集 IC 基线验证通过（优于无 LSTM baseline）
3. 改 `ENABLE_LSTM_SIGNAL = true`

## 人工审核

- [ ] 跑一次分析 → prompt 中无"LSTM 量化预测信号"块
- [ ] 分析正常输出结果，无报错
