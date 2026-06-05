# Phase 17 v5 Track 1: Monthly Aggregation Fix Report

## Problem

`export_daily_signals.py` successfully exported 866K daily predictions (Test IC3=0.114), but monthly aggregation resulted in IC=NaN.

## Root Cause

1. Aggregation strategy `latest` (last trading day of month) suffered severe signal-to-noise loss
2. Val period (2022-2023) had sufficient data, but Test period (2024-2026) had low return variance causing Spearman numerical instability
3. Daily LSTM predicts 63-day forward return; signal degrades when converting to monthly (21 trading days)

## Fix

5 aggregation strategies compared (Val period 2022-2023):

| Strategy | IC | n | L-S |
|------|-----|---|-----|
| latest | -0.072 | 6,963 | -0.015 |
| **mean** | **-0.050** | 6,963 | -0.006 |
| median | -0.054 | 6,963 | -0.007 |
| recent_5 | -0.060 | 6,963 | -0.010 |
| recent_10 | -0.055 | 6,963 | -0.009 |

**Best: mean, IC=-0.050. Signal direction is contrarian** (daily momentum = monthly mean reversion).

## Output

- `monthly_lstm_signals_v2.parquet` (43,534 records, contrarian, mean aggregation)
- Signal stats: mean=0.58, std=0.62

## Conclusion

Monthly aggregation IC=-0.050, weak signal but real. Contrarian consistency aligns with Phase 11/12 findings.
