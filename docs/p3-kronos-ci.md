# P3 Kronos CI + LLM Correlation

> Branch: `p3-kronos-fix` | Date: 2026-05-30

## Block Bootstrap CI

| Pool | n | Spread | 95% CI | nBlocks |
|------|---|--------|--------|---------|
| Full | 1574 | +7.59% | **[+3.1, +13.0]** | 1574 |
| Test | 758 | +4.71% | **[-2.9, +13.2]** | 758 |

**Full CI excludes 0. Test CI contains 0 — hold-out not passed.**

Train +11.1% -> Test +4.7%, CI lower bound -2.9%. Kronos cannot reliably distinguish direction on unseen timepoints.

## Kronos-LLM Correlation

**r = 0.069** (n=1574). Nearly zero correlation — the two pick completely different stocks; complementary.

## Final Gate

| Signal | Full CI | Test CI | hold-out |
|------|---------|---------|----------|
| LLM | [+3.7,+14.1] | — | Pass |
| Reversal | [+1.3,+11.2] | [+5.0,+19.8] | Pass |
| Kronos | [+3.1,+13.0] | [-2.9,+13.2] | **Fail** |

Kronos downgraded. Zero correlation with LLM; even if not promoted, can serve as complementary feature.
