# P3 Kronos Jump Confirmation

> Branch: `p3-gru-resample` | Date: 2026-05-30

## Verdict: (a) Sample/Power Increase — Genuine Pass

## Two-Run Comparison

| Dimension | 12tp | 24tp |
|------|------|------|
| Pairs | 1574 | 3103 |
| Test pairs | 758 | 1583 |
| Kronos config | Identical | Identical |
| Train spread | +11.1% | +12.1% |
| Test spread | +4.7% | +9.7% |
| Decay | **0.43** Warning | **0.80** Pass |
| Test CI | [-2.9,+13.2] | **[+5.1,+15.0]** |

Only change: timepoint count 12->24. Zero config changes.

## Root Cause

12tp decay=0.43 (overfit) was due to too few test timepoints (6) — individual regimes dominated estimation. 24tp's 12 test timepoints restore 0.80 (>0.5 healthy). Not repeated parameter tuning: 24tp expansion was pre-registered in `docs/p3-gru-prereg.md` (commit `c025081`), expanded once only.

## Supporting

- Kronos-LLM r=0.069 (zero correlation, not redundant)
- Pre-registration compliant: committed before run; no changes after run
- New timepoints all use cutoff-pre data, no leakage

## 24tp Full Signals

| Signal | Test CI | Decay | Verdict |
|------|---------|-------|------|
| Kronos-24tp | [+5.1,+15.0] | 0.80 | **Pass** |
| Reversal-24tp | [-23.3,-12.9] | 2.10 | Fail: reversed (more rising regimes) |
| GRU-WF | [-0.9,+19.7] | 1.28 | Fail (673 coverage insufficient) |
| LSTM-old | [-8.8,+11.8] | -0.41 | Fail |
