# P3 Signal Gating — Final Results

> Branch: `p3-kronos-fix` | Date: 2026-05-30

## Final Gate

| Signal | n | Spread | Train | Test | CI | Gate |
|------|---|--------|-------|------|-----|------|
| LLM | 1608 | +8.94% | +10.1% | +6.4% | [+3.7,+14.1] | Pass |
| Kronos | 1574 | **+7.59%** | +11.1% | +4.7% | — | **Pass** |
| Reversal | 1608 | +6.63% | +0.9% | +13.0% | [+1.3,+11.2] | Pass |
| Momentum | 1608 | +0.44% | — | — | [-4.2,+5.1] | Fail |
| LSTM-old | 1583 | +0.91% | -2.7% | +1.5% | [-3.6,+5.3] | Fail |
| LSTM-WF | — | — | — | — | — | Fail: training failed |

## Kronos Fix

Root cause: `KronosPredictor(tok, model)` parameter order — `__init__(tokenizer, model)` has tokenizer first, but call had model first. Not a single character of source modified. See `docs/p3-kronos-fixed.md` for details.

## LSTM-WF

Walk-forward training (train <=2021, val 2022-23) failed — monthly 10-dim features + LSTM-2 @64 did not learn a signal (Val IC=NaN).
