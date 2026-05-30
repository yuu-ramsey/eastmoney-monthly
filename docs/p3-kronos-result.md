# P3 Kronos 评估 — 报错记录

> 分支: `p3-signal-gating` | 日期: 2026-05-30

## 结论: Kronos 被模块内部 bug 阻塞，无法在 v2 池完成评估。

---

## 报错链（5 已修 + 1 未修）

| # | 报错 | 状态 |
|---|------|------|
| 1 | `No module named 'einops'` | pip install ✓ |
| 2 | `No module named 'huggingface_hub'` | pip install ✓ |
| 3 | `NameError: name 'safetensors' is not defined` | pip install ✓ |
| 4 | `missing required arguments: 'x_timestamp', 'y_timestamp'` | 补传 ✓ |
| 5 | `历史数据不足: 需要 400 条，实际 96 条` | context_len=min(256,n) ✓ |
| **6** | **`'Kronos' object has no attribute 'encode'`** | **未修** |

**#6 根因**: KronosPredictor 内部将 self.model (Kronos transformer) 当 tokenizer 调用 .encode()。修此 bug 需改动 `kronos/predictor.py` 内部逻辑，超出"不改 kronos 模型逻辑"范围。

## LSTM

1583 preds, +0.91% CI[-3.6,+5.3], 不显著。
