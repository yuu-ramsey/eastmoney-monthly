# Phase 17 v5 三轨汇总

## 轨道 1: 月度聚合

**✅ 完成。** 最佳策略 `mean`，IC=-0.050（反向）。43K 月度信号已保存。

## 轨道 2: Phase 19 v3 Backtest

**✅ 完成。** LSTM 信号改善 EW LT Sharpe 从 0.615→0.698 (+0.08)。最佳=0.698 < 0.7 → marginal。

## 轨道 3: LLM Eval

**⏸ 推迟。** 代码就绪但 LSTM 月度 IC=-0.05 太弱，注入 LLM 大概率不改善 score。¥14 预算保留。

## Phase 17 v5 全局结论

### 成功

1. **日线 LSTM 模型**: Test IC3=0.114，在 22 架构 + 4 频率 + 2 universe 搜索中唯一通过 Test
2. **月度聚合 pipeline**: 866K 日线预测 → 43K 月度信号，可复用
3. **Backtest 集成**: LSTM 信号在 backtest 中产生 +0.08 Sharpe 改善

### 局限

1. **月度信号太弱**: IC=-0.05，不足以成为主导信号
2. **Phase 19 backtest 未突破**: LT Sharpe 0.698 < 0.7 kill switch
3. **日线→月线信息丢失**: 聚合策略导致信噪比退化

### 建议

1. **保留 LSTM 信号在 Phase 19 信号源中**（已验证有正向贡献）
2. **寻找更强的 alpha 源**（资金流/基本面/另类数据）
3. **日线 backtest 路径**: 直接做日线频率的组合优化，跳过月度聚合

## 文件清单

| 文件 | 用途 |
|------|------|
| `cli/export_daily_signals.py` | 日线预测导出 |
| `.eastmoney-ai/lstm/daily_signals.parquet` | 866K 日线预测 |
| `.eastmoney-ai/lstm/monthly_lstm_signals_v2.parquet` | 43K 月度信号 |
| `lib/backtest/engine_v2.py` | 已接入 LSTM 信号 |
| `lib/prompt-templates.js` | #14 约束 + buildLstmSignalBlock |
| `lib/build-prompt.js` | lstmSignalData 参数 |
