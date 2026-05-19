"""Phase 20: Portfolio optimization — Black-Litterman, Risk-Parity, Multi-Strategy"""
import numpy as np, pandas as pd

def black_litterman(signals, prices, tau=0.05, view_conf=0.5):
    """
    Black-Litterman: prior (market-cap or equal-weight) + views (signals).
    Simplified: use signal strength as view, combine with prior via confidence.
    Returns: (months × stocks) weight matrix
    """
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for month in signals.index:
        s = signals.loc[month].dropna()
        if len(s) < 10: continue
        # Prior: equal weight
        prior = pd.Series(1.0/len(s), index=s.index)
        # Views: signal-driven (stronger signal → higher weight deviation)
        view = s - s.mean()  # deviation from mean signal
        view = view / (view.abs().max() + 1e-8)  # normalize to [-1, 1]
        # BL blend
        bl = prior * (1 - view_conf) + (prior + view * 0.5/len(s)) * view_conf
        bl = bl.clip(lower=0)
        bl = bl / bl.sum() if bl.sum() > 0 else prior
        weights.loc[month, s.index] = bl
    return weights.fillna(0)

def risk_parity(signals, prices, lookback=12):
    """
    Risk parity: allocate inversely proportional to volatility.
    Returns: (months × stocks) weight matrix
    """
    returns = prices.pct_change()
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for i, month in enumerate(signals.index):
        if i < lookback: continue
        s = signals.loc[month].dropna()
        if len(s) < 10: continue
        # Volatility over trailing window
        window_returns = returns.loc[signals.index[i-lookback]:signals.index[i-1], s.index]
        vols = window_returns.std()
        inv_vol = 1.0 / (vols + 1e-8)
        w = inv_vol / inv_vol.sum()
        weights.loc[month, s.index] = w
    return weights.fillna(0)

def multi_strategy(signals, prices, mode='equal'):
    """
    Multi-strategy: combine mean-reversion (monthly), sector rotation (quarterly), trend following.
    Returns: (months × stocks) weight matrix
    """
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    # Strategy 1: Monthly mean-reversion = Top-20 by signal (done in backtest)
    # Strategy 2: Quarterly sector rotation = overweight top 3 sectors
    # Strategy 3: Trend following = long stocks above MA60

    for month in signals.index:
        s = signals.loc[month].dropna()
        if len(s) < 20: continue
        # Equal blend of strategies → just pick Top-20 (same as EW baseline for now)
        top = s.nlargest(20)
        w = pd.Series(1.0/20, index=top.index)
        weights.loc[month, w.index] = w
    return weights.fillna(0)
