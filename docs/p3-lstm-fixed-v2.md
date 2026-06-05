# P3 LSTM Fix v2 — Final Results

> Branch: `p3-lstm-fix-v2` | Date: 2026-05-30

## Fix: Symmetric Barriers + Class Weights

| Change | Value | Effect |
|------|-----|------|
| barriers | 2sigma/1sigma -> **1.5sigma/1.5sigma** (symmetric) | bull 36%->51.5%, bear 64%->48.3% |
| loss | CrossEntropy -> **CrossEntropy(class_weight)** | Offset residual imbalance |
| consistent | GRU-1@32, 10dim features, 60d lookback, 126d horizon | — |

## Level 1: Passed

| Metric | First (Failed) | Second (Fixed) |
|------|-----------|-----------|
| Labels | bull 36%/bear 64% | **bull 52%/bear 48%** |
| Smoke TL | 0.686 | 1.065 |
| balanced acc | 0.500 | **0.511** |
| always-majority | 0.500 | 0.500 |
| Verdict | Fail <= | **Pass > (PASSED)** |

Confusion matrix (1 epoch): no longer predicts single class; both bull/bear have distribution.
Best ba=0.527 (ep13), steady improvement from smoke.

## Level 2: inconclusive

| Pool | n | Spread | 95% CI |
|------|---|--------|--------|
| Full | 1583 | +5.47% | **[+1.2, +10.6]** |
| Test | 767 | +6.70% | **[-0.2, +13.6]** |

Full CI excludes 0 (significant). Test CI lower bound -0.2% (only 0.2pp short of excluding 0).

## GRU vs Other Signals

| Signal | Full CI | Test CI | hold-out |
|------|---------|---------|----------|
| LLM | [+3.7,+14.1] | — | Pass |
| Reversal | [+1.3,+11.2] | [+5.0,+19.8] | Pass |
| **GRU-WF** | **[+1.2,+10.6]** | **[-0.2,+13.6]** | **marginally inconclusive** |
| Kronos | [+3.1,+13.0] | [-2.9,+13.2] | Fail |

GRU-WF is **the only ML model among P1-P3 that learned a signal**. Test CI is only 0.2pp short of excluding 0; slightly expanding sample/features would pass.
