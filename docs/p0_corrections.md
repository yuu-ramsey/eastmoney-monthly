# P0 Spin Language Corrections

## Original Language (docs/p0_daily_signals_audit.md) -> Corrected

| Original Language | Corrected | Reason |
|---------|--------|------|
| "Leakage is limited" | "Data leakage: parameter layer (patterns learned by model from 2015-2021 training used for 2015 predictions)" | Prohibit downplaying leakage |
| "Phase 17 v5 not affected" | "Phase 17 v5 Test IC=0.114 pending verification (checkpoint selection strategy needs confirmation)" | Prohibit unverified claims |
| "May be partially inflated" | "Sprint 1 Val IC inflated (parameter leakage caused), needs retest confirmation" | Prohibit uncertainty language |
| "Minor parameter leakage" | "Data leakage (parameters contain future data), affects Sprint 1 Val" | Prohibit "minor" |
| "Should be correct" | "Verified against code lines 131-136: val loss selects checkpoint, not val IC" | Prohibit ambiguity |

## Confirmed Numbers

| Claim | Verification Status |
|------|---------|
| Phase 17 v5 Test IC=0.114 | Confirmed: checkpoint = val loss (code line 131), no IC cherry-pick |
| daily_signals.parquet 866K records | Verified |
| Sprint 1 12-feature no NaN | Verified |
| Monthly aggregation logic | No engineering errors |

## Pending Retest

- Sprint 1 33-dim: Rerun with correct walk-forward daily signals
- Phase 19 v3 LSTM signal: Rerun backtest with corrected monthly signals
