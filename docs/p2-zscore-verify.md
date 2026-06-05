# P2 Z-Score Variance Verification

> Branch: `p2-verify-and-data-probe` | Date: 2026-05-29

## Conclusion: All committed code is correct. No modifications needed.

---

## B4 Bug Root Cause

Inline script (not saved): `Math.pow(b - ref?.mean || 0, 2)`
- `ref?.mean` for `{mean:{}, std:{}}` -> `undefined`
- `undefined || 0` -> 0 -> `(b-0)^2` -> RMS, not variance
- Z-score compression -> signal degradation

## Full Codebase Audit

`Math.pow(.*-.*\?\.)` -> 0 hits. None remaining.

## Per-File Verification

| File | Line | Verdict |
|------|-----|------|
| `rulers.js` | 127 | Pass: caller pre-computes ref |
| `eval-momentum-validate.js` | 43 | Pass: `Math.pow(b - ref.mean[k], 2)` |
| `eval-strongbull-vs-momentum.js` | 124-126 | Pass: mean recomputed inline |
| `eval-power-analysis.js` | 21 | Pass: `Math.pow(a - alphaMean, 2)` |

Reversal -32.5%, momentum -13.1% both based on correct variance, credible.
