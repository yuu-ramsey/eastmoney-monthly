# KoC Universe — Quality Report

**Generated**: 2026-06-08 16:57
**Input**: data/pead.sqlite (akshare YTD-based SUE, 2010-2024)
**Output**: data/universe.sqlite
**Mode**: DRY-RUN (see PIT status section for what is and isn't point-in-time)

## Coverage

| Metric | Liquid Pool | Full Sample |
|--------|-------------|-------------|
| Total observations | 168,508 | 168,508 |
| Observations in pool | 116,682 | 128,391 |
| Distinct stocks in pool | 4,510 | 4,748 |

## Filter Funnel (Liquid Pool)

Values: Pass=1, Unknown/Pending=-1, Fail=0

| Filter | Pass | Unknown/-1 | Fail |
|--------|------|------------|------|
| >= 8 quarters (trusted=1) | 168,508 | 0 | 0 |
| EPS >= 0 (no loss quarter) | 134,518 | 0 | 33,990 |
| Not ST/*ST [PROXY: current] | 158,349 | 0 | 10,159 |
| Market cap >= 5亿 [PROXY: SZ; SH=unknown] | 98,746 | 69,752 | 10 |
| Avg turnover >= 5% [DATA_PENDING] | 0 | 168,508 | 0 |
| Listed >= 24 months [PIT-correct] | 154,733 | 0 | 13,775 |

## Turnover Distribution

Sample data unavailable.

## Pool Size by Period

| Period | Liquid | Full |
|--------|--------|------|
| 2012Q1 | 1,207 | 1,513 |
| 2012Q2 | 1,350 | 1,575 |
| 2012Q3 | 1,349 | 1,568 |
| 2012Q4 | 1,413 | 1,527 |
| 2013Q1 | 1,522 | 1,718 |
| 2013Q2 | 1,612 | 1,732 |
| 2013Q3 | 1,605 | 1,742 |
| 2013Q4 | 1,624 | 1,692 |
| ... | ... | ... |
| 2024Q1 | 3,357 | 3,674 |
| 2024Q2 | 3,486 | 3,734 |
| 2024Q3 | 3,370 | 3,614 |
| 2024Q4 | 2,936 | 3,082 |

## PIT Status Summary

| Filter | PIT-correct? | Notes |
|--------|:-------------|-------|
| >= 8 quarters | ✅ Yes | `trusted=1` in pead.sqlite uses only historical EPS through pub_date |
| EPS >= 0 | ✅ Yes | `eps_single` anchored to pub_date; no future data |
| Not ST | ⚠️ Proxy | Current exchange name list; stocks that recovered (or became) ST since are misclassified |
| Market cap >= 5亿 | ⚠️ Proxy (SZ) | SZ: current float shares x current price; SH (69,532 obs) = unknown = pass |
| Avg turnover >= 5% | ❌ Pending | Requires daily price history per stock at each pub_date |
| Listed >= 24 months | ✅ Yes | Exchange listing date is static; 531 pre-IPO obs found and correctly excluded |

## Known Limitations (Dry-Run Mode)

1. **Turnover filter not applied** — filter logic is implemented but uses DATA_PENDING (-1 = all pass).
   Liquid pool observation count will decrease once this filter is active.
2. **Market cap for SH stocks** is unknown (69,532 observations, treated as pass).
   SH main board / STAR stocks tend to be large-cap, so over-inclusion is likely small.
3. **ST status** uses current exchange name list. Stocks that were ST historically (e.g., during
   2015 restructuring) but have since recovered will be incorrectly included.
4. **Market cap** uses current float shares x current price, not values at each pub_date.
   Stocks that have grown significantly since early periods are incorrectly included for those periods.

## Schema

```sql
universe_liquid(code, fiscal_year, fiscal_quarter, pub_date,
    pass_quarters, pass_pos_eps, pass_st, pass_mktcap, pass_turnover, pass_listing,
    in_pool)

universe_full(code, fiscal_year, fiscal_quarter, pub_date,
    pass_quarters, pass_pos_eps, pass_st, in_pool)

-- pass values: 1=pass, 0=fail, -1=unknown/DATA_PENDING (treated as pass in dry-run)
```

## Next Step (PIT Mode)

After pead-baostock.sqlite §10 is complete, update data sources:
- ST status: baostock historical stock name timeline
- Market cap: `total_share` from baostock x closing price at pub_date
  (requires separate `query_history_k_data_plus` fetch per stock per period)
- Turnover: rolling 20d average from daily close data at pub_date
