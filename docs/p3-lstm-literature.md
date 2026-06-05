# P3 LSTM Literature Review — Fix Plan

> Branch: `p3-lstm-fix-v2` | Date: 2026-05-30 | **Hard stop awaiting review**

---

## Diagnosis: Why Regression Failed

**Monthly**: 211K seqs, 10-dim features, LSTM-2@64, MSE loss -> val pred std=0, IC=NaN

Root cause: Regression directly predicts 6m raw return. A-share monthly 6m return std approx 40%, predictable portion <5%. The loss optimal solution = output mean (constant) -> spearmanr divides by zero -> IC=NaN. This is the well-known "predicting the mean" failure mode from literature.

## Literature Sources

- **Lopez de Prado (2018)** *Advances in Financial Machine Learning*, Ch.3 — Triple-Barrier labeling: three barriers (take-profit/stop-loss/time); first hit determines label +1/-1/0
- **Kang & Kim (2025)** [arXiv 2504.02249](https://arxiv.org/html/2504.02249v1) — LSTM on raw OHLCV + Triple-Barrier matches XGBoost on Korean market. optimal: 100d window, hidden=8, 29d horizon, 9% barriers
- **Wang et al. (2026)** [Financial Innovation](https://link.springer.com/article/10.1186/s40854-026-00929-6) — Transformer/GRU > LSTM, classification > regression for A-share
- **Liu et al. (2025)** [ICDEBA 2024](https://www.atlantis-press.com/proceedings/icdeba-24/126008539) — deep learning overfitting challenges in stock prediction

## Plan: Three Changes

| Component | Current (Failed) | Changed To |
|------|-----------|------|
| **Label** | 6m raw return MSE regression | Triple-Barrier 3-class classification (up/down/sideways) |
| **Loss** | MSE | CrossEntropy |
| **Model** | LSTM-2@64 | GRU-1@32 baseline -> GRU-2@64 |
| **Eval** | IC (NaN, invalid) | spread (bullish/bearish WinsMean difference) + CI |

### Triple-Barrier Label Implementation

```
Upper = 2 x 60d volatility (take-profit, up > upper -> +1)
Lower = 1 x 60d volatility (stop-loss, down > lower -> -1)
Time = 6 months (neither hit -> 0)
Volatility = cutoff-pre 60-day monthly high-low range ema std
```

### GRU Baseline to Confirm Task Learnability (details after review)

Daily 500 stocks x 10dim features, Triple-Barrier 3-class labels, CrossEntropy loss.

**sigma definition (review requirement)**: Each stock's pre-cutoff 12-month monthly return std, point-in-time, no full-sample/future data.

**3-class acceptance (review requirement, not 33%)**:
- Report: per-class proportions (up/down/sideways %) + confusion matrix + balanced accuracy + macro-F1
- baseline: "always predict majority class" balanced acc / macro-F1
- **Judgment**: balanced acc > always-majority -> task is learnable; approx always-majority -> model only learned to output majority class (same as regression "predict mean"), honest stop.

### Two-Level Gate

| Level | Metric | Pass Condition | If Not |
|----|------|---------|--------|
| 1 Learnability | balanced acc vs always-majority | > always-majority | **Stop**, task not learnable |
| 2 Stock selection | v2 pool test spread CI | CI excludes 0 | Report honestly, do not stop |

## Implementation Steps

1. Fix indentation bug + 1-epoch smoke test to confirm pipeline has no NaN
2. Formal train GRU-1@32 on daily 500 stocks
3. Confusion matrix + balanced acc acceptance
4. If passes Level 1 -> v2 pool spread + CI gate

## Hard Stop Points

1. ~~Literature plan review~~ Pass: reviewed (2026-05-30)
2. Smoke test passed -> formal train
3. Level 1 gate passed (balanced acc > always-majority) -> enter Level 2
4. Level 2 gate passed (test CI excludes 0) -> LSTM finally passes
