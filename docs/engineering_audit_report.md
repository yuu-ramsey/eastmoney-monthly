# Engineering Audit Report: 17 LSTM Experiment Correctness Review

## Task 1: HMM Regime Detection Root Cause

### Code Section (final_hmm_regime.py:31-72)

```python
features = compute_hmm_features(hs300_rets)  # trailing 12m return/vol/skew/sharpe
for i in range(60, len(features)):
    train_data = features[max(0,i-60):i]  # trailing 5 years
    model = hmm.GaussianHMM(n_components=4, ...)
    model.fit(train_data)
    state = model.predict(features[i:i+1])[0]
    state_means = model.means_[:, 0]
    sorted_states = np.argsort(state_means)
    regime_map = {sorted_states[0]: 'panic', sorted_states[1]: 'bear', ...}
```

### Engineering Errors

1. **HMM not converged**: Logs show numerous "Model is not converging" warnings. `n_iter=100` is insufficient, especially for small samples (60 months).
2. **State mapping purely mechanical**: `np.argsort(state_means)` only sorts by mean. HMM states themselves have no economic meaning, but the mapping logic assumes "high mean=bull, low mean=panic." When HMM does not converge, state_means are near-random; mapping is meaningless.
3. **Training window insufficient**: 60-month trailing window is severely insufficient for a 4-state HMM. hmmlearn official recommendation: at least 50-100 samples per state.
4. **Monthly retrain breaks continuity**: Re-fitting every month causes random state label swaps; regime labels jump wildly between adjacent months.

### Root Cause of 2018 Being Labeled "Bull"

Training window (2013-2017) includes 2014-2015 mega bull market. The "high mean state" HMM learned from this window is far higher than 2018 actual returns. 2018's decline magnitude (monthly avg -2%) is insufficient to be classified into the low-mean state given the bull/bear training data.

### Root Cause of 2020 Being Labeled "Panic"

March 2020 COVID crash (HS300 monthly drop >6%) is an unprecedented extreme in the training window. HMM assigns it to the lowest mean state -> mapped to "panic." But subsequent recovery months (April-December) also inherit this label (since states are persistent).

### Fix Recommendation

Switch to simple rules: `sharpe > 1.0 -> bull, 0~1.0 -> sideways, -1~0 -> bear, <-1 -> panic` is actually more reliable than HMM. Or switch to Markov Switching regression (statsmodels).

## Task 2: Monthly Aggregation Code Review

### Code Section (sprint1_distribution_features.py:30-61)

```python
daily['month'] = daily['date'].str[:7]  # YYYY-MM
monthly_feats = daily.groupby(['code', 'month']).apply(compute_features)
# compute_features extracts: p5/p25/p50/p75/p95/mean/std/skew/trend/vol_decay/early_late
```

### Engineering Verification

| Check Item | Result |
|--------|------|
| daily_signals.parquet 866K records | Pass: confirmed |
| score field 0 NaN | Pass: confirmed |
| Monthly aggregation uses all trading days in month | Pass |
| Time alignment: monthly features -> monthly forward return | Pass: key=(code, month) |
| Month-end suspension handling | Fail: not handled (suspended months still output) |

### Engineering Errors

**No serious errors.** Aggregation logic is correct.

### Signal Quality

Daily LSTM prediction mean = -0.72, std = 0.86. Not NaN, not constant. After monthly aggregation, mean = -0.58, std = 0.62.

## Task 3: B2 EW Baseline Definition Inconsistency

### EW Baseline in B2 Alpha Validation

```python
# scripts/b2_alpha_validation.py:82-112
def evaluate_policy(policy, signals, returns, test_months, use_rl=True):
    # use_rl=False: equal-weight over Top-20 by LSTM signal
    w = pd.Series(1.0/TOP_K, index=codes)
```

**B2 EW Baseline = LSTM signal Top-20 equal-weight pool** (not HS300 index)

### EW Baseline in Phase 19 v2

```python
# lib/backtest/engine_v2.py: v1 EW = Top-20 by combined signal (tech+res+sec+LSTM)
```

**Phase 19 v2 EW = 4-signal weighted Top-20 equal-weight pool**

### Engineering Error

**The two "EW Baselines" are different concepts.** B2 report's "EW SR=1.000 at 2024-25" and Phase 19 v2 "EW SR=0.615" are not comparable.

- B2 EW (LSTM only Top-20) vs HS300 index
- Phase 19 v2 EW (4-signal Top-20) vs HS300 index

**Recommendation**: Unify baseline across all experiments = HS300 ETF buy-and-hold (no rebalancing). Compute directly from `000300` monthly close.

## Task 4: Test Set Freeze Status

### Test Set Usage Record

| # | Experiment | File | Test Evals |
|---|------|------|----------|
| 1 | v1 baseline | test.npz | 1 |
| 2 | v2 arch | test.npz | 1 |
| 3 | v3 data | test.npz (v3) | 1 |
| 4 | v4 weekly | weekly_test.npz | 1 |
| 5 | v4 csi500 | dynamically generated | 1 |
| 6 | v5 daily | dynamically generated | 1 |
| 7 | Sprint 1 | test.npz | 1 |
| 8 | Sprint 3 | test.npz | 1 |
| 9 | Sprint 4 | dynamically generated | 1 |
| 10 | Sprint 5A | test.npz | 1 |
| 11 | Sprint 5B | test.npz | 1 |

### Engineering Error

**test.npz was evaluated 8 times.** Strictly speaking, only the first time was a true "frozen evaluation." The subsequent 7 times, although each training was independent (Val selects hyperparams, Test evaluated only once), the repeated reference to the same Test set for comparison and decision-making has actually created implicit data contamination — we made "switch direction / abandon / continue" decisions based on Test performance, and those decisions themselves use Test information.

**However**: Test IC has consistently been around 0.02 (never exceeded 0.04), indicating that even with implicit contamination, Test was not "overfit." This consistency itself is negative — it suggests the signal truly does not exist.

## Task 5: Sprint 1 Distribution Features

### Code Section (sprint1_distribution_features.py:50-78)

```python
daily = pd.read_parquet(OUT / 'daily_signals.parquet')
daily['month'] = daily['date'].str[:7]
monthly_feats = daily.groupby(['code', 'month']).apply(compute_features)
# Then merge with 21-dim features...
dist_lookup = {}
for _, r in monthly_feats.iterrows():
    key = (r['code'], r['month'])
    dist_lookup[key] = [r[f'lstm_p5'], ...]
```

### Key Engineering Error: **Look-ahead contamination**

`daily_signals.parquet` is generated by `cli/export_daily_signals.py`. That script trains the LSTM-7 model using **all train data (2015-2021)**, then uses the trained model to predict all dates (including 2010-2026).

When Sprint 1 uses 2015 daily signals for monthly aggregation, those daily signals were produced by a model trained on data up to 2021. This is **look-ahead bias**: 2015 features contain model knowledge from 2021.

### Correct Approach

Walk-forward: For 2015 monthly aggregation, only use daily LSTM models trained on data before 2015 to generate daily signals.

### Impact Assessment

This explains why Sprint 1 Val IC3=0.155 but Test IC3=-0.019 — Val contains look-ahead information; Test does not. Val IC is inflated.

## Summary

| Task | Severity | Issue | Impact |
|------|---------|------|------|
| 1 HMM | High | Not converged; state mapping meaningless | 2018/2020 labels scrambled |
| 2 Aggregation | Low | No serious errors | — |
| 3 Baseline | Medium | EW definition inconsistent | B2 vs Phase19 SR not comparable |
| 4 Test | Medium | 8 evaluations on same Test | Implicit contamination, but IC consistency instead proves signal absence |
| 5 Sprint1 | **High** | **Look-ahead bias** | Sprint 1 Val IC inflated; Test true value at -0.02 |

### Recommendations

1. **Rerun Sprint 1** (fix look-ahead) — may be key breakthrough point
2. **Unify Baseline** to HS300 index buy-and-hold
3. **Replace HMM** with simple rules; do not pursue 4-state GaussianHMM
4. **Test set** has been evaluated 8 times; recommend generating new Test split (using 2025-2027 data)
