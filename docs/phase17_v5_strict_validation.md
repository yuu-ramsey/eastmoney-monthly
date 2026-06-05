# Phase 17 v5 Strict Val IC Validation

## Checkpoint Selection Strategy Verification

**Code**: `scripts/build_daily_v5.py:112-133`

```python
best_vl, no_imp, best_ic3 = float('inf'), 0, -1.0
# ...
if vl < best_vl: best_vl, no_imp = vl, 0    # line 131: save on VAL LOSS
best_ic3 = max(best_ic3, ic3)                  # line 130: track IC for REPORT ONLY
if no_imp >= 15 and ep > 20: break             # line 133: early stop on LOSS
```

**Conclusion**: checkpoint selection = **val loss minimum** (line 131). `best_ic3` only used for report tracking (line 130), not for model selection.

## Test Evaluation Timing

- Model early-stopped at epoch 21 (val loss not improved for 15 consecutive epochs)
- Test evaluation uses model weights at early stop (no checkpoint reload)
- Test IC evaluation code at lines 137-148: independent of training loop

## Test IC=0.114 Verification

| Check Item | Status |
|--------|------|
| Checkpoint = val loss (not val IC) | Pass |
| Test evaluated only once (build_daily_v5.py) | Pass |
| daily_signals.parquet export = second inference (contains parameter leakage) | Confirmed |
| Original Test IC number | **0.114 credible** |

## Other Experiment Checks

| Experiment | Val IC Source | IC cherry-pick? | Credible? |
|------|-----------|----------------|-------|
| v1 baseline | val loss | No | Pass |
| v2 arch | val loss (IC tracking only) | No | Pass |
| Sprint 1 | val loss | No | Parameter leakage (daily_signals) |
| Sprint 3 MASTER | val loss | No | Pass |
| Sprint 5A ListNet | val loss | No | Pass |
| Sprint 5B Triple-Barrier | val accuracy | No | Pass |
| CSI1000 | val loss | No | Proportional split leak |

## Conclusion

- Phase 17 v5 Test IC=0.114 **credible** (checkpoint=val loss)
- Sprint 1 Val IC **needs retest** (daily_signals parameter leakage)
- CSI1000 Val IC **needs retest** (proportional split instead of date split)
- Other experiment checkpoint selection strategies all correct
