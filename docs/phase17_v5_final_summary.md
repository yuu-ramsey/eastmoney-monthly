# Phase 17 v5 Three-Track Summary

## Track 1: Monthly Aggregation

**Complete.** Best strategy `mean`, IC=-0.050 (contrarian). 43K monthly signals saved.

## Track 2: Phase 19 v3 Backtest

**Complete.** LSTM signal improved EW LT Sharpe from 0.615->0.698 (+0.08). Best=0.698 < 0.7 -> marginal.

## Track 3: LLM Eval

**Deferred.** Code ready but LSTM monthly IC=-0.05 too weak; injection into LLM highly unlikely to improve score. 14 CNY budget reserved.

## Phase 17 v5 Global Conclusions

### Successes

1. **Daily LSTM model**: Test IC3=0.114, only architecture that passed Test out of 22 architectures + 4 frequencies + 2 universes searched
2. **Monthly aggregation pipeline**: 866K daily predictions -> 43K monthly signals, reusable
3. **Backtest integration**: LSTM signal produced +0.08 Sharpe improvement in backtest

### Limitations

1. **Monthly signal too weak**: IC=-0.05, insufficient to be dominant signal
2. **Phase 19 backtest did not break through**: LT Sharpe 0.698 < 0.7 kill switch
3. **Daily->monthly information loss**: Aggregation strategy causes signal-to-noise degradation

### Recommendations

1. **Retain LSTM signal in Phase 19 signal sources** (verified positive contribution)
2. **Search for stronger alpha sources** (fund flows/fundamentals/alternative data)
3. **Daily backtest path**: Directly optimize portfolio at daily frequency, skip monthly aggregation

## File Checklist

| File | Purpose |
|------|------|
| `cli/export_daily_signals.py` | Daily prediction export |
| `.eastmoney-ai/lstm/daily_signals.parquet` | 866K daily predictions |
| `.eastmoney-ai/lstm/monthly_lstm_signals_v2.parquet` | 43K monthly signals |
| `lib/backtest/engine_v2.py` | Connected to LSTM signal |
| `lib/prompt-templates.js` | #14 constraint + buildLstmSignalBlock |
| `lib/build-prompt.js` | lstmSignalData parameter |
