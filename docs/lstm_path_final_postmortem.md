# LSTM Path Final Inventory

## Experiment Panorama

| # | Experiment | Frequency | Universe | Features | Arch | Val IC3 | **Test IC3** | 
|---|------|------|----------|------|------|---------|-------------|
| 1 | v1 baseline | Monthly | HS300 | 21 | LSTM-1 | 0.095 | 0.025 |
| 2 | v2 arch | Monthly | HS300 | 21 | 22 arch | 0.180 | 0.019 |
| 3 | v3 data | Monthly | HS300 | 40 | LSTM-7 | 0.127 | 0.009 |
| 4 | v4 weekly | Weekly | HS300 | 21 | LSTM-7 | 0.114 | 0.008 |
| 5 | v4 csi500 | Monthly | CSI500 | 21 | LSTM-7 | 0.172 | -0.080 |
| 6 | **v5 daily** | **Daily** | HS300 | 21 | LSTM-7 | 0.093 | **0.114** |
| 7 | Sprint 1 | Monthly | HS300 | 33 | LSTM-7 | 0.155 | -0.019 |
| 8 | Sprint 3 | Monthly | HS300 | 21 | MASTER | 0.118 | -0.018 |

## The Only Breakthrough

**v5 Daily LSTM: Test IC3=0.114.** 866K daily predictions, cross-sectionally effective.

Monthly return prediction (1m/3m/6m forward returns) all failed — Test IC never exceeded 0.03 (except v1's 0.025).

## Root Cause Analysis

1. **Regime change**: Patterns from 2015-2021 training do not transfer to 2024-2026. A-share market structure has fundamentally changed over the past decade (registration system, foreign access, rising quant fund share)
2. **Monthly signal degradation**: In a 60-month lookback, the most recent 6-12 months of information dominates; early historical information contributes almost nothing
3. **Price-volume alpha already absorbed**: 21-dim technical indicators (MA/MACD/RSI/KDJ etc.) are public information, already priced in by the market
4. **Small-cap contrarian signal**: CSI500 Test IC=-0.08 — small-cap price-volume signals are directionally opposite to large-caps; equal-weight aggregation cancels out the signal

## Conclusion

**Monthly frequency + large-cap universe + price-volume features = LSTM cannot produce out-of-sample predictive power.**

This is not an architecture problem (tried 22 architectures), not a feature problem (tried 21/33/40 dimensions), not a frequency problem (tried monthly/weekly/daily), not a universe problem (tried HS300/CSI500). It is the **information ceiling of the data itself**.

## Assets

- Daily LSTM IC=0.114: Usable quantitative signal, should be retained in Phase 19 backtest signal sources
- 866K daily predictions (daily_signals.parquet): Reusable data asset
- 43K monthly aggregated signals (monthly_lstm_signals_v2.parquet): Backtest-ready
- Frozen eval dataset v1: Standardized evaluation framework

## Pivot

**Pivot to Phase 23: Chrome extension productization.** Stop chasing prediction accuracy; focus on user value delivery.
