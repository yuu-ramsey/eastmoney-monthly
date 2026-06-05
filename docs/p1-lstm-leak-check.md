# P1 LSTM Training/Eval Time Window Leakage Check

> Branch: p0a-verify-3-4 (read-only audit) | Date: 2026-05-29
>
> **Conclusion first: `daily_lstm7.pt` training data has no time split; 2024-2025 is entirely in the training set.
> All eval timepoints (2024-05, 2024-11, 2025-05, 2025-11) are leaked.
> mc-dropout eval results are unusable, and runtime-injected LSTM signal is equally affected.**

---

## 1. Evidence Chain

### 1a. Model Training: No Time Split

**`scripts/daily_to_monthly_aggr.py:25-91`** — This is the script that trains `daily_lstm7.pt`:

```python
# Line 31-35: Select first 150 data-rich stocks
for c in stocks:
    g = d_df[d_df['code'] == c]
    if len(g) >= LOOKBACK + 504: train_codes.append(c)
    if len(train_codes) >= 150: break

# Line 37-62: Generate training sequences for each stock's [full history]
for c in train_codes:
    g = d_df[d_df['code'] == c].sort_values('date').reset_index(drop=True)
    ...
    for i in range(LOOKBACK-1, n-252):  # n = stock's full history length
        ...
        all_seqs.append(feats[i-LOOKBACK+1:i+1]); all_y3.append(y3d)

# Line 64: Take first 250K sequences as training set
Xtr = np.array(all_seqs[:250000], dtype=np.float32)
```

**No train/val/test time split.** Line 64 only takes the first 250K sequences (by stock order + time order), not filtered by date. Contains all 2010-2025 data.

**`lib/lstm/dataset.py:17-19,308-310`** — Monthly training data explicitly has test set covering eval timepoints:

```python
TRAIN_END = '2021-12'
VAL_END = '2023-12'
TEST_END = '2026-05'

train_mask = (seq_dates >= '2015-01') & (seq_dates <= '2021-12')
val_mask = (seq_dates >= '2022-01') & (seq_dates <= '2023-12')
test_mask = (seq_dates >= '2024-01') & (seq_dates <= '2026-05')  # <- covers all eval timepoints
```

**But the monthly model (LSTMBaseline) is not used by mc_dropout_predict.py**, and v4 eval also does not inject it.

### 1b. MC Dropout Inference: Full Data

**`cli/mc_dropout_predict.py:128`**:

```python
WHERE code IN (...) AND date >= '2010-01-01'
```

Loads full daily history for all stocks (2010-01 to present); line 157 generates sequences for every trading day:

```python
for i in range(LOOKBACK - 1, n):  # n = all trading days
    seqs.append(feats[i - LOOKBACK + 1:i + 1])
```

Outputs predictions for every date to `mc_dropout_signals.parquet`.

### 1c. Runtime Injection

**`cli/eval-mc-dropout.js:135-137,208`**:

```js
const mcCache = await loadMcDropoutCache();
...
const prompt = await buildPromptByTemplate({
    ...,
    lstmSignalData,   // <- LSTM signal injected into prompt
});
```

**`cli/mc_export_json.py:22`** — mc_dropout JSON export:

```python
latest = df.sort_values('date').groupby('code').last()
```

Each stock takes its latest signal -> stored to `.eastmoney-ai/storage/mc_dropout/<code>.json` -> Chrome extension reads via native-host.

---

## 2. Affected Timepoints

| Eval Timepoint | v4-signals eval | MC-dropout eval | Runtime inject |
|-----------|----------------|-----------------|-----------|
| 2024-05 | Not affected (no LSTM) | **Leaked** | **Leaked** |
| 2024-11 | Not affected (no LSTM) | **Leaked** | **Leaked** |
| 2025-05 | Not affected (no LSTM) | **Leaked** | **Leaked** |
| 2025-11 | Not affected (no LSTM) | **Leaked** | **Leaked** |

**v4-signals eval (0.187) not affected by LSTM leakage** — confirmed prompt contains no LSTM signal.

**mc-dropout eval (mc-dropout-*.jsonl) all affected by leakage** — model training used 2024+ data, and inference generated signals at 2024+ timepoints.

**Runtime system affected** — when users actually use it, LSTM signal is injected into prompt, and that signal comes from a model trained on 2024+ data.

---

## 3. Leakage Severity

| Item | Severity | Description |
|------|--------|------|
| v4 eval (0.187) | No impact | Did not use LSTM |
| MC dropout eval | **High** | Results unusable; signals contain future information |
| Runtime LSTM signal | **High** | LSTM confidence/uncertainty users see is from a leaked model |
| Frozen baseline (0.1966) | No impact | runA-no-sector did not use LSTM |

---

## 4. Fix Recommendations (not implemented, recorded only)

To fix:
1. Retrain daily_lstm7 with strict walk-forward split (e.g., train <=2021-12, val 2022-2023, test 2024+)
2. mc_dropout_predict.py only infers on train/val dates; test dates strictly held out
3. Regenerate mc_dropout_signals.parquet and mc_dropout/*.json

---

## Untouched

| Module | Status |
|------|------|
| Any .py training scripts | Read-only |
| Any .js eval scripts | Read-only |
| mc_dropout_signals.parquet | Not regenerated |
