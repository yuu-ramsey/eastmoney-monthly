# Proportional Split Root Cause Analysis

## Bug

Original `build_daily_v5.py:101-102` used index-proportional split:

```python
n = len(X); n_tr = int(n*0.55)
Xtr, ytr = X[:n_tr], y[:n_tr]  # first 55% sequences -> train
```

Sequences sorted by `stock_code`. Same-period months from different stocks were assigned to different sets.

## V6 Fix

```python
train_mask = [d >= '2015-01-01' and d <= '2021-12-31' for d in dates]
```

Strict date-based split, zero cross-period contamination.

## Impact

| Experiment | Original IC | V6 Strict IC | Change |
|------|-------|-----------|------|
| Daily LSTM | 0.114 | **0.141** | +24% Pass |
| Weekly LSTM | 0.008 | **0.007** | — |
| Monthly LSTM | 0.019 | **-0.027** | Signal disappeared |

## Why Daily Works

- 381K training sequences (monthly only 16K)
- Short-term patterns (momentum/reversal) stable across regimes
- High signal-to-noise ratio

## Why Monthly Fails

- 16K training sequences, severely underfit
- Long-term patterns (6-12 month forward) destroyed by 2015->2024 regime change
- Proportional split artificially created cross-stock pseudo-signals

## Conclusion

Daily is the only valid frequency. Monthly/weekly LSTM predictive power is zero. All historical monthly-based experiments (Phase 17 v1-v5, Sprint 1-5, B1-B3) need retesting with v6 pipeline, but expected results unchanged (monthly IC <= 0).
