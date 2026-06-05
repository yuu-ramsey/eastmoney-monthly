# P2 Unbiased Pool Baseline v2 — Baostock Full Set

> Branch: `p2-rebuild-unbiased-pool` | Date: 2026-05-30 | Single-source: Baostock

---

## 1. Momentum (v1 vs v2)

| Pool | v1 spread | v1 CI | v2 spread | v2 CI |
|------|----------|-------|----------|-------|
| Full | -13.1% | [-44.7,+4.8] | **+0.4%** | **[-4.2,+5.1]** |
| Train | -24.1% | [-57.4,+5.5] | +0.4% | [-5.8,+8.1] |
| Test | -8.1% | [-47.3,+18.1] | -0.7% | [-8.2,+6.2] |

**Momentum goes to zero. CI tight (+/-5pp), completely excludes effects >5%.**

---

## 2. Reversal (v1 vs v2)

| Pool | v1 spread | v2 spread | v2 CI |
|------|----------|----------|-------|
| Full | -32.5% | **+6.6%** | **[+1.4,+11.3]** |
| Train | — | +0.9% | [-6.6,+6.9] |
| Test | — | **+13.0%** | **[+5.0,+19.8]** |

**Reversal flips positive and significant.** v1's -32.5% was a survivorship bias artifact.

---

## 3. Judgment

| Finding | Conclusion |
|------|------|
| Survivorship bias flipped reversal sign | -32.5% -> +6.6% |
| Momentum goes to zero | No predictive power on unbiased pool |
| Reversal is the only significant factor | Full CI[1.4,11.3], Test CI[5.0,19.8] |

C's LLM comparison should change: primary metric = (LLM spread - Reversal spread) difference CI.
