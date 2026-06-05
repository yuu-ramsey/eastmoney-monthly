# P2 Baostock Delisted Coverage Live Test

> Branch: `p2-verify-and-data-probe` | Date: 2026-05-29

## Conclusion: Baostock freely preserves delisted stock historical monthly data. Survivorship bias can be fixed for free.

---

## Live Test

`pip install baostock` (no registration/token needed), `bs.login()`.

| Code | Monthly Bars | Time Range | Last Price |
|------|------|---------|------|
| sz.000033 (Xindutong) | 30 | 2015-01 ~ 2017-06 | 1.20 |
| sz.000511 (*ST Xitan) | 42 | 2015-01 ~ 2018-06 | 0.79 |
| sz.002070 (*ST Zhonghe) | 54 | 2015-01 ~ 2019-06 | 0.50 |
| sz.000018 (ShenchengA Tui) | 60 | 2015-01 ~ 2019-12 | 0.24 |
| sh.600401 (*ST Hairun) | 54 | 2015-01 ~ 2019-06 | 0.12 |
| sh.600432 (*ST Jien) | 42 | 2015-01 ~ 2018-06 | 1.02 |
| sz.000001 (Ping An Bank) | Pass | Reference | — |

6/6 delisted stocks all have complete history. No data after delisting month; last price is actual closing price before delisting (not 0).

## Impact

- Delisted stock last price has actual value (not close=0), directly usable for alpha calculation
- Can pull 2018-2025 full delisted stock set and merge into SQLite; low-position pool recompute
- Free + no registration needed -> can serve as standard pipeline step
