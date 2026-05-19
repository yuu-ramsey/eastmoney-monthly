"""Phase 19 v4: Multi-level market timing with position smoothing"""
import numpy as np, pandas as pd

def compute_improved_timing(prices, signals):
    """
    4-level position sizing based on MA20/MA60 cross.
    Position changes capped at 40% per month.
    Returns: Series of position weights per month
    """
    # Market proxy: equal-weight average close
    mkt_close = prices.mean(axis=1)
    ma20 = mkt_close.rolling(20).mean()
    ma60 = mkt_close.rolling(60).mean()

    # 4-level: above both MA (100%), above MA20 only (70%), above MA60 only (40%), below both (20%)
    raw_positions = pd.Series(0.2, index=mkt_close.index)  # default 20% (crisis)
    bullish = (mkt_close > ma20) & (mkt_close > ma60)
    mild_bull = (mkt_close > ma20) & (mkt_close <= ma60)
    mild_bear = (mkt_close <= ma20) & (mkt_close > ma60)

    raw_positions[bullish] = 1.0
    raw_positions[mild_bull] = 0.7
    raw_positions[mild_bear] = 0.4

    # Smooth: cap monthly change at 40%
    smoothed = raw_positions.copy()
    for i in range(1, len(smoothed)):
        prev = smoothed.iloc[i-1]
        curr = raw_positions.iloc[i]
        max_change = 0.4
        smoothed.iloc[i] = np.clip(curr, prev - max_change, prev + max_change)

    return smoothed.dropna()

def compute_timing_v1(prices):
    """Original v2 timing: close > MA60 → 1.0, else 0.3"""
    mkt_close = prices.mean(axis=1)
    ma60 = mkt_close.rolling(60).mean()
    return pd.Series(np.where(mkt_close > ma60, 1.0, 0.3), index=mkt_close.index).dropna()
