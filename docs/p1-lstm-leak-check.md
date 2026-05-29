# P1 LSTM 训练/评估时间窗泄漏检查

> 分支: p0a-verify-3-4 (只读审计) | 日期: 2026-05-29
>
> **结论先行：`daily_lstm7.pt` 训练数据无时间切分，2024-2025 全在训练集内。
> 所有 eval 时点 (2024-05, 2024-11, 2025-05, 2025-11) 均被泄漏。
> mc-dropout eval 结果不可用，且运行时注入的 LSTM 信号同样受影响。**

---

## 1. 证据链

### 1a. 模型训练：无时间切分

**`scripts/daily_to_monthly_aggr.py:25-91`** — 这是训练 `daily_lstm7.pt` 的脚本:

```python
# Line 31-35: 选前 150 只 data-rich 股票
for c in stocks:
    g = d_df[d_df['code'] == c]
    if len(g) >= LOOKBACK + 504: train_codes.append(c)
    if len(train_codes) >= 150: break

# Line 37-62: 对每只股票的【全部历史】生成训练序列
for c in train_codes:
    g = d_df[d_df['code'] == c].sort_values('date').reset_index(drop=True)
    ...
    for i in range(LOOKBACK-1, n-252):  # n = 股票的全部历史长度
        ...
        all_seqs.append(feats[i-LOOKBACK+1:i+1]); all_y3.append(y3d)

# Line 64: 取前 250K 条作为训练集
Xtr = np.array(all_seqs[:250000], dtype=np.float32)
```

**没有 train/val/test 时间切分。** 第 64 行只取了前 250K 条序列（按股票顺序+时间顺序），未按日期筛选。包含全部 2010-2025 数据。

**`lib/lstm/dataset.py:17-19,308-310`** — 月线训练数据明确有测试集覆盖 eval 时点:

```python
TRAIN_END = '2021-12'
VAL_END = '2023-12'
TEST_END = '2026-05'

train_mask = (seq_dates >= '2015-01') & (seq_dates <= '2021-12')
val_mask = (seq_dates >= '2022-01') & (seq_dates <= '2023-12')
test_mask = (seq_dates >= '2024-01') & (seq_dates <= '2026-05')  # ← 覆盖所有 eval 时点
```

**但月线模型 (LSTMBaseline) 未被 mc_dropout_predict.py 使用**，且 v4 eval 中也未注入。

### 1b. MC Dropout 推理：全量数据

**`cli/mc_dropout_predict.py:128`**:

```python
WHERE code IN (...) AND date >= '2010-01-01'
```

加载所有股票的完整日线历史 (2010-01 至今)，第 157 行为每个交易日生成序列:

```python
for i in range(LOOKBACK - 1, n):  # n = 全部交易日
    seqs.append(feats[i - LOOKBACK + 1:i + 1])
```

输出每个日期的预测到 `mc_dropout_signals.parquet`。

### 1c. 运行时注入

**`cli/eval-mc-dropout.js:135-137,208`**:

```js
const mcCache = await loadMcDropoutCache();
...
const prompt = await buildPromptByTemplate({
    ...,
    lstmSignalData,   // ← LSTM 信号注入 prompt
});
```

**`cli/mc_export_json.py:22`** — mc_dropout JSON 导出:

```python
latest = df.sort_values('date').groupby('code').last()
```

每个股票取最新一条信号 → 存入 `.eastmoney-ai/storage/mc_dropout/<code>.json` → Chrome 扩展通过 native-host 读取。

---

## 2. 受影响时点

| Eval 时点 | v4-signals eval | MC-dropout eval | 运行时间入 |
|-----------|----------------|-----------------|-----------|
| 2024-05 | 不受影响 (无 LSTM) | **泄漏** | **泄漏** |
| 2024-11 | 不受影响 (无 LSTM) | **泄漏** | **泄漏** |
| 2025-05 | 不受影响 (无 LSTM) | **泄漏** | **泄漏** |
| 2025-11 | 不受影响 (无 LSTM) | **泄漏** | **泄漏** |

**v4-signals eval (0.187) 不受 LSTM 泄漏影响** — 已确认 prompt 中不含 LSTM 信号。

**mc-dropout eval (mc-dropout-*.jsonl) 全部受泄漏影响** — 模型训练用了 2024+ 数据，且推理在 2024+ 时点生成了信号。

**运行时系统受影响** — 用户在实际使用时，LSTM 信号注入到 prompt 中，该信号来自一个在 2024+ 数据上训练的模型。

---

## 3. 泄漏严重程度

| 项目 | 严重度 | 说明 |
|------|--------|------|
| v4 eval (0.187) | 无影响 | 未使用 LSTM |
| MC dropout eval | **高** | 结果不可用，信号包含未来信息 |
| 运行时 LSTM 信号 | **高** | 用户看到的 LSTM 信心度/不确定性来自泄漏模型 |
| Frozen baseline (0.1966) | 无影响 | runA-no-sector 未使用 LSTM |

---

## 4. 修复建议（不实施，仅记录）

若要修复:
1. 用严格 walk-forward split (e.g., train ≤2021-12, val 2022-2023, test 2024+) 重新训练 daily_lstm7
2. mc_dropout_predict.py 只对 train/val 日期做推理，test 日期严格留出
3. 重新生成 mc_dropout_signals.parquet 和 mc_dropout/*.json

---

## 未改动

| 模块 | 状态 |
|------|------|
| 任何 .py 训练脚本 | 只读 |
| 任何 .js eval 脚本 | 只读 |
| mc_dropout_signals.parquet | 未重新生成 |
