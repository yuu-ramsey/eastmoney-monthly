# Proportional Split 根因分析

## Bug

原 `build_daily_v5.py:101-102` 使用索引比例 split:

```python
n = len(X); n_tr = int(n*0.55)
Xtr, ytr = X[:n_tr], y[:n_tr]  # 前 55% 序列 → train
```

序列按 `stock_code` 排序。不同股票的同期月份被分到不同集合。

## V6 修复

```python
train_mask = [d >= '2015-01-01' and d <= '2021-12-31' for d in dates]
```

严格 date-based split，零跨期污染。

## 影响

| 实验 | 原 IC | V6 严格 IC | 变化 |
|------|-------|-----------|------|
| 日线 LSTM | 0.114 | **0.141** | +24% ✅ |
| 周线 LSTM | 0.008 | **0.007** | — |
| 月线 LSTM | 0.019 | **-0.027** | 信号消失 |

## 为什么日线有效

- 381K 训练序列 (月线仅 16K)
- 短期模式 (动量/反转) 跨 regime 稳定
- 高信噪比

## 为什么月线无效

- 16K 训练序列，严重欠拟合
- 长期模式 (6-12 月前瞻) 被 2015→2024 regime change 破坏
- 比例 split 人为制造了跨股票伪信号

## 结论

日线是唯一有效频率。月线/周线的 LSTM 预测力为零。所有基于月线的历史实验（Phase 17 v1-v5, Sprint 1-5, B1-B3）均需用 v6 pipeline 重测，但预期结果不变（月线 IC ≤ 0）。
