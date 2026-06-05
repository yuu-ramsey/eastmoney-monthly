# P3 Kronos Fix — Final Results

> Branch: `p3-kronos-fix` | Date: 2026-05-30 | GPU: RTX 5070

## Root Cause

`KronosPredictor` constructor parameter order was reversed. `__init__(tokenizer, model)` but called with `(model, tok)`, so `self.tokenizer` pointed to model and `.encode()` raised `'Kronos' object has no attribute 'encode'`.

## Fix

```python
# Wrong: pred = KronosPredictor(model, tok, device=DEVICE)
# Correct:
pred = KronosPredictor(tok, model, device=DEVICE)
```

Not a single character of kronos source code was modified.

## Results

| Signal | n | Spread | Train | Test |
|------|---|--------|-------|------|
| LLM | 1608 | +8.94% CI[3.7,14.1] | +10.1% | +6.4% |
| **Kronos** | 1574 | **+7.59%** | +11.1% | +4.7% |
| Reversal | 1608 | +6.63% CI[1.3,11.2] | +0.9% | +13.0% |
| LSTM-old | 1583 | +0.91% | -2.7% | +1.5% |

## Judgment

**Passes gating.** Kronos +7.59% is at the same level as LLM/Reversal. Train>>Test (11% vs 5%) suggests regime dependency.
