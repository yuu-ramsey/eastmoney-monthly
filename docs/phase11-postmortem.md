# Phase 11: 多周期共振约束 — 事后总结

## 目标

通过注入月/周/日三周期共振信号到 LLM prompt（HARD_CONSTRAINTS #11），约束模型在共振明确时输出与共振方向一致的判断。

## 原始依据

v4-signals eval: score 0.168 (stored) vs v5-resonance: score 0.130 — 共振约束导致 score 下降。当时关闭依据是正确的，但未做共振信号本身的预测力验证。

## 离线量化矩阵（零 LLM 成本）

298 只 HS300 股票 × 2018-2025 月度数据，每个评估点计算三周期方向 + 共振等级（使用 lib/multi-period/ 生产代码逻辑），严格 walk-forward。

### 样本量

| signal | n (per holding) | 说明 |
|--------|----------------|------|
| strong_bull | 1,114-1,118 | 三周期全偏多 |
| strong_bear | 1,114 | 三周期全偏空 |
| mild_bull | 2,720-2,721 | 二周期偏多 |
| mild_bear | 2,939 | 二周期偏空 |

### Avg Forward Return (扣除 HS300 等权基准 alpha)

| signal \ holding | 1m | 3m | 6m | 12m |
|---|---|---|---|---|
| strong_bull | 0.59% | 2.61% | 4.31% | 9.25% |
| **strong_bear** | **1.53%** | **3.48%** | **7.67%** | **11.15%** |

### Long-Short (strong_bull − strong_bear, alpha)

| 1m | 3m | 6m | 12m |
|---|---|---|---|
| -0.94% | -0.88% | -3.35% | -1.90% |
| Sharpe=-0.195 | Sharpe=-0.051 | Sharpe=-0.097 | Sharpe=-0.028 |

## 核心发现

**三周期共振是反向指标**。三周期全偏空（strong_bear）后的前向收益系统性高于三周期全偏多（strong_bull），扣除基准后效应仍存在。

经济解释：三周期同向 → 趋势已充分定价 → 均值回归。A 股"三阳开泰见顶，三只乌鸦见底"的传统经验在此得到量化验证。

但预测力较弱（\|Sharpe\| ≈ 0.10），属于边缘信号。

## 为什么 LLM v5 score 下降

HARD_CONSTRAINTS #11 要求 LLM 在 strong 共振时跟随共振方向判 strong_bull/strong_bear。但共振是反向信号——strong_bull 共振后实际回报低于 strong_bear。LLM 被约束强制使用反向信号，score 自然恶化。

## 最终结论

\|Sharpe\| ≈ 0.10，边缘反向信号。LLM 直接使用会误判（v5 score 0.130），但作为 LSTM 特征可能有效。

## 后续

- 代码保留 `lib/multi-period/`，默认 `ENABLE_RESONANCE=false`
- Phase 17 (LSTM) 将共振作为反向特征
- 可选路径：改 #11 为反向逻辑后重新 eval
