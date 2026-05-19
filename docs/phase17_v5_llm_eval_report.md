# Phase 17 v5 轨道 3: LLM Eval 报告

## 状态: 待执行

### 准备就绪

- `lib/prompt-templates.js`: HARD_CONSTRAINTS #14（LSTM 信号约束）
- `lib/build-prompt.js`: `buildLstmSignalBlock` + `lstmSignalData` 参数注入
- `monthly_lstm_signals_v2.parquet`: 43K 月度信号就绪
- `cli/eval_lstm_signal.py`: eval 框架就绪
- Frozen dataset: `data/frozen-eval-dataset-v1.json`（40 stocks, 640 samples）

### 对照设计

| | Run A | Run B |
|---|---|---|
| LSTM 信号 | 无 | 有（#14 约束） |
| Baseline score | 0.1966 | TBD |
| LLM 调用 | 0 | 640 |
| 成本 | ¥0 | ~¥14 |

### Kill Switch

- Δ score > +0.02 → ENABLE_LSTM_SIGNAL=true
- Δ ∈ [-0.01, +0.02] → marginal, 关闭
- Δ < -0.01 → LLM 用 LSTM 信号困惑, 关闭

### 风险

- LSTM 信号 IC=-0.05 很可能不足以让 LLM 做出更好的判断
- Phase 11/12 经验: 弱信号注入 LLM 反而不如不注入
- 成本 ¥14，但可能产生零或负收益

### 建议

在 LSTM 信号 IC 提升到 >0.10 之前，推迟 LLM eval。¥14 预算留给更高 ROI 的实验。
