# Phase 17 v5 Track 3: LLM Eval Report

## Status: Pending Execution

### Ready

- `lib/prompt-templates.js`: HARD_CONSTRAINTS #14 (LSTM signal constraint)
- `lib/build-prompt.js`: `buildLstmSignalBlock` + `lstmSignalData` parameter injection
- `monthly_lstm_signals_v2.parquet`: 43K monthly signals ready
- `cli/eval_lstm_signal.py`: eval framework ready
- Frozen dataset: `data/frozen-eval-dataset-v1.json` (40 stocks, 640 samples)

### Control Design

| | Run A | Run B |
|---|---|---|
| LSTM signal | None | Present (#14 constraint) |
| Baseline score | 0.1966 | TBD |
| LLM calls | 0 | 640 |
| Cost | 0 CNY | ~14 CNY |

### Kill Switch

- Delta score > +0.02 -> ENABLE_LSTM_SIGNAL=true
- Delta in [-0.01, +0.02] -> marginal, disabled
- Delta < -0.01 -> LLM confused by LSTM signal, disabled

### Risks

- LSTM signal IC=-0.05 likely insufficient for LLM to make better judgments
- Phase 11/12 experience: weak signal injection into LLM is worse than no injection
- Cost 14 CNY but may produce zero or negative return

### Recommendation

Defer LLM eval until LSTM signal IC improves to >0.10. Reserve 14 CNY budget for higher-ROI experiments.
