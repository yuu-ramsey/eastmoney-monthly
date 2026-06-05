# P3 LSTM 文献调研 — 修复方案

> 分支: `p3-lstm-fix-v2` | 日期: 2026-05-30 | **硬停等审核**

---

## 诊断：为什么回归失败了

**月线**: 211K seqs, 10维特征, LSTM-2@64, MSE loss → val pred std=0, IC=NaN

根因: 回归直接预测 6m raw return。A 股月线 6m return std≈40%，可预测部分 <5%。Loss 最优解 = 输出均值（常数）→ spearmanr 除零 → IC=NaN。这是文献中已知的"predicting the mean"失败模式。

## 文献来源

- **López de Prado (2018)** *Advances in Financial Machine Learning*, Ch.3 — Triple-Barrier labeling: 三个 barrier（止盈/止损/时间），先碰到的决定标签 +1/-1/0
- **Kang & Kim (2025)** [arXiv 2504.02249](https://arxiv.org/html/2504.02249v1) — LSTM on raw OHLCV + Triple-Barrier 在韩国股市匹配 XGBoost。optimal: 100d window, hidden=8, 29d horizon, 9% barriers
- **Wang et al. (2026)** [Financial Innovation](https://link.springer.com/article/10.1186/s40854-026-00929-6) — Transformer/GRU > LSTM, classification > regression for A-share
- **Liu et al. (2025)** [ICDEBA 2024](https://www.atlantis-press.com/proceedings/icdeba-24/126008539) — deep learning overfitting challenges in stock prediction

## 方案：三处改动

| 组件 | 当前（失败） | 改为 |
|------|-----------|------|
| **标签** | 6m raw return MSE 回归 | Triple-Barrier 3-class 分类（涨/跌/震荡） |
| **Loss** | MSE | CrossEntropy |
| **模型** | LSTM-2@64 | GRU-1@32 baseline → GRU-2@64 |
| **评估** | IC（NaN, 无效） | spread（bullish/bearish WinsMean 差）+ CI |

### Triple-Barrier 标签实现

```
上轨 = 2 × 60d 波动率（止盈，涨 > 上轨 → +1）
下轨 = 1 × 60d 波动率（止损，跌 > 下轨 → -1）
时间轨 = 6 个月（都没碰 → 0）
波动率 = cutoff 前 60 日月线 high-low range 的 ema std
```

### GRU baseline 确认任务可学性（审核后细化）

日线 500 stocks × 10dim features，Triple-Barrier 3-class labels，CrossEntropy loss。

**σ 定义（审核要求）**: 各股 cutoff 前 12 个月月收益的 std，point-in-time，不用全样本/未来数据。

**三分类验收（审核要求，不用 33%）**:
- 报：各类占比（涨/跌/震荡 %）+ 混淆矩阵 + balanced accuracy + macro-F1
- baseline: "always predict majority class" 的 balanced acc / macro-F1
- **判读**: balanced acc > always-majority → 任务可学；≈ always-majority → 模型只学会输出最大类（同回归版"预测均值"），诚实停。

### 两级门控

| 级 | 指标 | 通过条件 | 不过则 |
|----|------|---------|--------|
| 1 可学性 | balanced acc vs always-majority | > always-majority | **停**, 任务不可学 |
| 2 选股 | v2 池 test spread CI | CI 不含 0 | 如实报, 不停 |

## 实现步骤

1. 修缩进 bug + 1-epoch 烟雾测试确认管线无 NaN
2. 正式 train GRU-1@32 on 日线 500 stocks
3. 混淆矩阵 + balanced acc 验收
4. 若过 1 级 → v2 池 spread + CI 门控

## 硬停点

1. ~~文献方案审核~~ ✓ 已审核（2026-05-30）
2. 烟雾测试通过 → 正式 train
3. 1 级门控通过（balanced acc > always-majority）→ 进 2 级
4. 2 级门控通过（test CI 不含 0）→ LSTM 最终过
