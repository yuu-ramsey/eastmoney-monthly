# P3 Kronos Evaluation — Error Record

> Branch: `p3-signal-gating` | Date: 2026-05-30

## Conclusion: Kronos blocked by internal module bug; cannot complete evaluation on v2 pool.

---

## Error Chain (5 fixed + 1 unfixed)

| # | Error | Status |
|---|------|------|
| 1 | `No module named 'einops'` | pip install Pass |
| 2 | `No module named 'huggingface_hub'` | pip install Pass |
| 3 | `NameError: name 'safetensors' is not defined` | pip install Pass |
| 4 | `missing required arguments: 'x_timestamp', 'y_timestamp'` | Passed additional args Pass |
| 5 | `Insufficient historical data: need 400, got 96` | context_len=min(256,n) Pass |
| **6** | **`'Kronos' object has no attribute 'encode'`** | **Unfixed** |

**#6 Root cause**: KronosPredictor internally calls self.model (Kronos transformer) as if it were a tokenizer via .encode(). Fixing this bug requires modifying `kronos/predictor.py` internal logic, which exceeds the "do not modify kronos model logic" scope.

## LSTM

1583 preds, +0.91% CI[-3.6,+5.3], not significant.
