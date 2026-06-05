# P1 Low-Position Sample Pool — Gate 1 Report: Delisted Data Unavailable

> Branch: `p1-eval-lowpos-build` | Date: 2026-05-29
>
> **Gate 1: Not passed. SQLite does not contain monthly historical data for delisted stocks.**

---

## Evidence

### stocks Table

```sql
SELECT DISTINCT delisted FROM stocks -> [0]
SELECT * FROM stocks WHERE delisted != 0 -> 0 rows
```

- Only 300 rows, all `delisted=0`
- No delisting date or delisting status fields

### monthly_klines Table

```
4 known delisted stocks tested:
  000033 (Xindutong):  0 monthly bars
  000511 (*ST Xitan): 0 monthly bars
  002070 (*ST Zhonghe): 0 monthly bars
  000018 (ShenchengA Tui): 0 monthly bars
```

- 3265 unique codes, of which 2965 (91%) not in `stocks` table -> no status
- 1942 codes in `stock_industry_mapping` have no data in monthly/daily klines -> delisted stocks have no K-line history preserved

### Verdict

**Database only contains monthly data for currently listed 3265 stocks. Delisted stocks have no historical K-lines.**

---

## Impact

| Metric | Current (3265 survivors) | With Delisted |
|------|-------------------|--------|
| Unconditional alpha mean | Biased upward | True |
| Low-position pool tail quality | Beautified | Contains truly worst |
| Conclusion generalizability | Not usable for live trading | Usable |

---

## Options

### A: Accept bias, continue building pool (document known limitation)
- Cost: 0 CNY (this time), subsequent LLM 38 CNY
- Risk: eval inflated, conclusions not generalizable

### B: First supplement delisted data, then build pool
- Requires external data source (Wind/JoinQuant/Tushare)
- Cost: possibly paid + several days

### C: Switch UNIVERSE = "general"
- Delisting bias diluted in full cross-section
- But low-position edge may be weaker

---

**Not entered Gate 2, not built pool, not burned LLM. Awaiting decision.**
