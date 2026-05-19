# Phase 19 v3 Backtest 报告

## 配置

LSTM v2 月度信号（IC=-0.05, mean 聚合, 反向）替代 placeholder=0，接入 engine_v2.py。
4 配置: EW / EW+Timing / ICIR / ICIR+Timing。
In-sample 2015-2023，Live Test 2024-2025。

## 结果

| 配置 | IS Sharpe | IS MaxDD | IS AnnRet | **LT Sharpe** | LT MaxDD | LT AnnRet |
|------|-----------|----------|-----------|-------------|----------|-----------|
| v1 EW | 0.353 | -64.4% | 13.9% | **0.698** | -16.9% | 19.3% |
| v2 EW+Timing | 0.360 | -64.4% | 14.2% | 0.595 | -6.1% | 5.7% |
| v3 ICIR | 0.565 | -64.5% | 44.8% | 0.578 | -18.5% | 17.0% |
| v4 ICIR+Timing | 0.570 | -64.5% | 45.1% | 0.255 | -7.4% | 2.5% |
| HS300 EW | 0.588 | — | 24.0% | — | — | — |

## vs Phase 19 v2

| 配置 | v2 LT Sharpe | v3 LT Sharpe | Δ |
|------|-------------|-------------|-----|
| v1 EW | 0.615 | **0.698** | **+0.083** |
| v2 EW+Timing | 0.530 | 0.595 | +0.065 |
| v3 ICIR | 0.600 | 0.578 | -0.022 |
| v4 ICIR+Timing | 0.307 | 0.255 | -0.052 |

## 分析

1. LSTM 信号改善 EW 和 EW+Timing 的 Live Sharpe（+0.08），但 ICIR 变差
2. IS Sharpe 全部下降（0.517→0.353 for EW）——LSTM 反向信号在历史期引入噪声
3. MaxDD 无改善（-64% In-sample）
4. 最佳 LT Sharpe=0.698 < 0.7 → marginal

## Kill Switch

**LT Sharpe=0.698 < 0.7 → marginal。** LSTM 信号有正向贡献但不足以突破阈值。
建议: 保留 LSTM 信号在信号源中，但继续寻找更强的 alpha 源。
