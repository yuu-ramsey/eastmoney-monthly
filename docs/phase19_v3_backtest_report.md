# Phase 19 v3 Backtest Report

## Config

LSTM v2 monthly signals (IC=-0.05, mean aggregation, contrarian) replace placeholder=0, connected to engine_v2.py.
4 configs: EW / EW+Timing / ICIR / ICIR+Timing.
In-sample 2015-2023, Live Test 2024-2025.

## Results

| Config | IS Sharpe | IS MaxDD | IS AnnRet | **LT Sharpe** | LT MaxDD | LT AnnRet |
|------|-----------|----------|-----------|-------------|----------|-----------|
| v1 EW | 0.353 | -64.4% | 13.9% | **0.698** | -16.9% | 19.3% |
| v2 EW+Timing | 0.360 | -64.4% | 14.2% | 0.595 | -6.1% | 5.7% |
| v3 ICIR | 0.565 | -64.5% | 44.8% | 0.578 | -18.5% | 17.0% |
| v4 ICIR+Timing | 0.570 | -64.5% | 45.1% | 0.255 | -7.4% | 2.5% |
| HS300 EW | 0.588 | — | 24.0% | — | — | — |

## vs Phase 19 v2

| Config | v2 LT Sharpe | v3 LT Sharpe | Delta |
|------|-------------|-------------|-----|
| v1 EW | 0.615 | **0.698** | **+0.083** |
| v2 EW+Timing | 0.530 | 0.595 | +0.065 |
| v3 ICIR | 0.600 | 0.578 | -0.022 |
| v4 ICIR+Timing | 0.307 | 0.255 | -0.052 |

## Analysis

1. LSTM signal improves EW and EW+Timing Live Sharpe (+0.08), but ICIR worsens
2. IS Sharpe all declined (0.517->0.353 for EW) — LSTM contrarian signal introduces noise in historical period
3. MaxDD no improvement (-64% In-sample)
4. Best LT Sharpe=0.698 < 0.7 -> marginal

## Kill Switch

**LT Sharpe=0.698 < 0.7 -> marginal.** LSTM signal has positive contribution but insufficient to break threshold.
Recommendation: Retain LSTM signal in signal sources but continue searching for stronger alpha sources.
