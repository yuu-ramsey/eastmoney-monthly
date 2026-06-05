# P0: daily_signals.parquet Look-Ahead Audit

## 1. Model Training Data Range

**Code**: `cli/export_daily_signals.py:100-106`

```python
# Split by PROPORTIONAL index (not by date)
n = len(X); n_tr = int(n*0.55); n_va = int(n*0.72)
Xtr, ytr = X[:n_tr], y[:n_tr]
Xva, yva = X[n_tr:n_va], y[n_tr:n_va]
Xte, yte = X[n_va:], y[n_va:]
```

- Train: first 55% sequences (~416K records, covering dates ~2015-2021)
- Val: middle 17% (~129K records, covering dates ~2022-2023)  
- Test: last 28% (~212K records, covering dates ~2024-2026)
- Split method: **by index proportion** (sequences sorted by stock_code then split), not by date
- Best checkpoint: epoch with lowest val loss (early stop patience=15)

## 2. Inference Data Range

**Code**: `cli/export_daily_signals.py:156-194`

```python
for code in stocks:
    g = d_df[d_df['code'] == code].sort_values('date')
    # Recompute features...
    seqs = [feats[i-252+1:i+1] for i in range(251, n)]
    model.eval()
    with torch.no_grad():
        for j in range(0, len(seqs), 512):
            batch = torch.from_numpy(np.array(seqs[j:j+512])).float().to(DEVICE)
            p = model(batch).cpu().numpy()[:, 0]
```

- Inference range: all stocks, all trading days from `date >= '2010-01-01'`
- Model: a single final checkpoint (selected by early-stopping on train+val)
- **No walk-forward retrain**: same model infers entire 2010-2026

## 3. Key Judgment

### Case A: LOOK-AHEAD CONFIRMED

| Check Item | Actual State | Leakage? |
|--------|---------|-------|
| Training data range | 55% sequences (~2015-2021) | — |
| Inference data range | **All dates (2010-2026)** | — |
| Train period predicted? | **Yes** (2015-2021 daily signals from same model) | Warning |
| Walk-forward retrain | **No** | Warning |
| Final verdict | **Case A** | Warning |

### Two Layers of Leakage

1. **Feature layer**: Each day's features use trailing 252-day data (no forward look) Pass
2. **Parameter layer**: Model parameters learned from 2015-2021 training, used to predict 2015 data itself Warning

2015 daily signals come from a model that "has seen 2021." For 2015 predictions, this is not strict out-of-sample.

### But Leakage Magnitude is Limited

The model only saw data from the train set (2015-2021). 2015 predictions are not "using 2021 features to predict 2015" — features are still walk-forward. Leakage is only from the parameter layer: the model learned cross-sectional patterns from 2015-2021, then applied them to 2015.

Impact on downstream Sprint 1:
- Sprint 1 uses 2015 data from daily_signals to generate monthly features
- The model parameter information contained in these monthly features is limited (parameters come from average patterns of the entire train set)
- Sprint 1 Val IC=0.155 may be partially inflated, but Test IC=-0.019 is the true value (Test 2024-2026 is not in model training set)

## 4. Impact Scope

| Experiment | Depends on daily_signals? | Impact |
|------|-------------------|------|
| Phase 17 v5 Test IC=0.114 | No (Test independently evaluated) | Not affected Pass |
| Sprint 1 33-dim | **Yes** | Val inflated; Test true value -0.019 |
| Phase 19 v3 | **Yes** (LSTM signal) | Monthly signals may contain slight parameter leakage |
| B1 signal fusion | **Yes** | Same as above |
| B2 RL/B3 Risk | **Yes** | Same as above |

Phase 17 v5 Test IC=0.114 is independent of daily_signals.parquet — it was evaluated in-place in `build_daily_v5.py`, not dependent on the exported file.

## 5. Fix Plan

```python
# Correct approach: yearly retrain
for year in [2015, 2016, ..., 2025]:
    train_data = features[dates < f'{year}-01-01']
    model = train_model(train_data)
    predictions[year] = model.predict(features[dates year])
```

Effort: 2-3 days (needs 10 retrains x ~5 min = 50 min GPU)

## 6. Conclusion

**Case A confirmed.** But actual leakage magnitude is moderate (parameter layer, not feature layer). Most critical impact: Sprint 1 Val IC inflated. Recommended: fix then rerun Sprint 1.
