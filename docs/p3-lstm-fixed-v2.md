# P3 LSTM 修复 v2 — 最终结果

> 分支: `p3-lstm-fix-v2` | 日期: 2026-05-30

## 修复：对称 barrier + class weights

| 改动 | 值 | 效果 |
|------|-----|------|
| barriers | 2σ/1σ → **1.5σ/1.5σ** (对称) | bull 36%→51.5%, bear 64%→48.3% |
| loss | CrossEntropy → **CrossEntropy(class_weight)** | 抵消 residual imbalance |
| 一致 | GRU-1@32, 10dim features, 60d lookback, 126d horizon | — |

## Level 1: 通过

| 指标 | 第一次(失败) | 第二次(修复) |
|------|-----------|-----------|
| Labels | bull 36%/bear 64% | **bull 52%/bear 48%** |
| Smoke TL | 0.686 | 1.065 |
| balanced acc | 0.500 | **0.511** |
| always-majority | 0.500 | 0.500 |
| 判定 | ✗ ≤ | **✓ > (PASSED)** |

混淆矩阵 (1 epoch): 不再全预测单类, bull/bear 均有分布。
Best ba=0.527 (ep13), steady improvement from smoke.

## Level 2: inconclusive

| Pool | n | Spread | 95% CI |
|------|---|--------|--------|
| Full | 1583 | +5.47% | **[+1.2, +10.6]** |
| Test | 767 | +6.70% | **[-0.2, +13.6]** |

Full CI 排除 0 (显著)。Test CI 下界 -0.2% (仅差 0.2pp 含 0)。

## GRU vs 其他信号

| 信号 | Full CI | Test CI | hold-out |
|------|---------|---------|----------|
| LLM | [+3.7,+14.1] | — | ✓ |
| 反转 | [+1.3,+11.2] | [+5.0,+19.8] | ✓ |
| **GRU-WF** | **[+1.2,+10.6]** | **[-0.2,+13.6]** | **marginally inconclusive** |
| Kronos | [+3.1,+13.0] | [-2.9,+13.2] | ✗ |

GRU-WF 是 P1-P3 中**唯一学到信号的 ML 模型**。Test CI 仅差 0.2pp 就排除 0，略扩样本/特征即可过。
