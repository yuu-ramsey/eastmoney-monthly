# P1 Disable Leaked LSTM Runtime Injection

> Branch: `p1-disable-leaked-lstm` | Date: 2026-05-29 | Based on `docs/p1-lstm-leak-check.md` audit conclusion

---

## Problem

Audit confirmed `daily_lstm7.pt` training data has no time split (`scripts/daily_to_monthly_aggr.py`); 2024-2025 is entirely in training set. All eval timepoints + runtime are affected by look-ahead leakage. Pending walk-forward retrain + baseline verification, no LSTM signal should be injected into prompt.

## Solution

Add `ENABLE_LSTM_SIGNAL = false` (default OFF), skip the entire native message call, leveraging the existing null degradation path.

### Changes

**`background.js:324-367`** — Wrap LSTM acquisition logic with flag:

```js
const ENABLE_LSTM_SIGNAL = false;
let lstmSignalData = null;
if (ENABLE_LSTM_SIGNAL) {
  // ... original native message + build lstmSignalData ...
}
```

When OFF, `lstmSignalData` stays `null`; downstream path:

| Step | Behavior |
|------|------|
| `build-prompt.js:279` | Default parameter `lstmSignalData = null` |
| `build-prompt.js:344` | `buildLstmSignalBlock(null)` |
| `prompt-templates.js:98` | `!signalData` -> `return ''` |
| Template constraint #14 | "If LSTM section exists above" -> no block, skip |

Exactly the same degradation as when native host is unavailable — audit verified safe (`docs/p0a-verify-3-4.md` #4).

### Untouched

| Module | Status |
|------|------|
| prompt structure | Untouched |
| LLM provider | Untouched |
| Scoring/parsing | Untouched |
| Model files/training scripts | Preserved unchanged, pending retrain |

## Tests

```
11 tests | 0 fail (prompt-templates.test.js)
```

## Re-enable Conditions

1. Retrain daily LSTM with strict walk-forward split (train <= 2021-12, val 2022-2023, test 2024+)
2. New model passes test set IC baseline verification (better than no-LSTM baseline)
3. Change `ENABLE_LSTM_SIGNAL = true`

## Manual Review

- [ ] Run an analysis -> prompt has no "LSTM Quantitative Prediction Signal" block
- [ ] Analysis outputs results normally, no errors
