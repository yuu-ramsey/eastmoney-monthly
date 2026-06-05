# P2 Sort Reference Spread Bug Investigation and Fix

> Branch: `p2-fix-sort-ref-bug` | Date: 2026-05-29

## Conclusion

**No sort signal contamination bug. B4 +29.5% true root cause: inline Z-score variance computation degenerated.**

`Math.pow(b - ref?.mean || 0, 2)` where `ref?.mean` for `{mean:{}, std:{}}` returns `undefined`, `(b-0)^2` = RMS, not variance. All Z-scores compressed near zero, coincidentally displaying the same spread. That script was not saved to file.

---

## Full Codebase Audit (70+ .sort() sites)

| File | Line | Pattern | Verdict |
|------|-----|------|------|
| `rulers.js` | 26/34/133 | `[...arr].sort()` | Pass |
| `eval-momentum-validate.js` | 52/114 | `[...pairs].sort()` | Pass |
| `eval-strongbull-vs-momentum.js` | 22/48/150 | `[...arr].sort()` | Pass |
| `sector/alpha.js` | 107 | `items.sort()` | Pass: new array |
| `sector/alpha.js` | 115 | `results.sort()` | Pass: created in function |
| Remaining ~65 sites | — | filter/spread then sort | Pass |

## Corrected Values

| Metric | B4 Wrong | Corrected |
|------|---------|------|
| Momentum spread | +29.5% | -13.1% |
| Momentum CI | All positive | Contains 0 |
