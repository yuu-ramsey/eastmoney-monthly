# P0 Spin 措辞修正

## 原始措辞（docs/p0_daily_signals_audit.md）→ 修正后

| 原始措辞 | 修正后 | 原因 |
|---------|--------|------|
| "泄露程度有限" | "数据泄露: 参数层 (模型从 2015-2021 训练中学到的模式用于 2015 年预测)" | 禁止淡化泄露 |
| "Phase 17 v5 不受影响" | "Phase 17 v5 Test IC=0.114 待验证 (checkpoint 选择策略需确认)" | 禁止未验证声明 |
| "可能部分虚高" | "Sprint 1 Val IC 虚高 (参数泄露导致), 需重测确认" | 禁止不确定性措辞 |
| "轻微参数泄露" | "数据泄露 (参数包含未来数据), 影响 Sprint 1 Val" | 禁止"轻微" |
| "应该是对" | "已对照代码行 131-136: val loss 选 checkpoint, 非 val IC" | 禁止模糊 |

## 已确认的数字

| 声明 | 验证状态 |
|------|---------|
| Phase 17 v5 Test IC=0.114 | ✅ checkpoint = val loss (代码行 131), 无 IC cherry-pick |
| daily_signals.parquet 866K 条 | ✅ 已验证 |
| Sprint 1 12-feature 无 NaN | ✅ 已验证 |
| 月度聚合逻辑 | ✅ 无工程错 |

## 待重测

- Sprint 1 33-dim: 用正确 walk-forward daily signals 重跑
- Phase 19 v3 LSTM 信号: 用修正后月度信号重跑 backtest
