"""Phase 19: Walk-Forward Backtest Engine — optimized (all data in-memory)"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT_DIR = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'backtest'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLIPPAGE = 0.003
COMMISSION = 0.00025
TOP_K = 20

# ============================================================
# 1. Load ALL data into memory
# ============================================================

def load_all_data():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("""
        SELECT code, date, open, high, low, close, volume
        FROM monthly_klines
        WHERE code IN (SELECT DISTINCT stock_code FROM stock_industry_mapping)
        AND date >= '2010-01'
        ORDER BY code, date
    """, conn)
    # Industry mapping
    ind_df = pd.read_sql_query(
        "SELECT stock_code, industry_code FROM stock_industry_mapping", conn
    )
    conn.close()
    stock_to_ind = dict(zip(ind_df['stock_code'], ind_df['industry_code']))
    return df, stock_to_ind

# ============================================================
# 2. Signal computation (vectorized per stock, walk-forward)
# ============================================================

def compute_signals_for_stock(code, group, stock_to_ind):
    """Compute 3 signals for one stock across all months. Returns DataFrame of scores."""
    group = group.sort_values('date').reset_index(drop=True)
    closes = group['close'].values.astype(float)
    highs = group['high'].values.astype(float)
    lows = group['low'].values.astype(float)
    volumes = group['volume'].values.astype(float)
    dates = group['date'].tolist()
    n = len(group)

    if n < 60:
        return None

    industry = stock_to_ind.get(code, None)

    results = []
    for i in range(60, n):
        as_of = dates[i]
        window_closes = closes[:i+1]
        window_highs = highs[:i+1]
        window_lows = lows[:i+1]
        window_volumes = volumes[:i+1]
        w = i + 1  # window size

        # --- Signal 1: Technical Indicators ---
        ma20 = np.mean(window_closes[-20:])
        ma60 = np.mean(window_closes[-min(60,w):]) if w >= 60 else ma20
        # MACD
        ema12 = pd.Series(window_closes).ewm(span=12).mean().values[-1]
        ema26 = pd.Series(window_closes).ewm(span=26).mean().values[-1]
        dif = ema12 - ema26
        dea_arr = pd.Series(window_closes).ewm(span=12).mean().values
        dea_arr2 = pd.Series(dea_arr).ewm(span=26).mean().values
        # Simplified: dif and signal
        macd_signal = 1 if dif > 0 else -1

        # RSI14
        deltas = np.diff(window_closes, prepend=window_closes[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = pd.Series(gains).ewm(alpha=1/14).mean().values[-1]
        avg_loss = pd.Series(losses).ewm(alpha=1/14).mean().values[-1]
        rsi = 100 - 100/(1 + avg_gain/max(avg_loss, 1e-8)) if avg_loss > 0 else 50

        s_tech = 0.0
        s_tech += np.clip(dif * 3, -0.3, 0.3)  # MACD
        s_tech += 0.2 if rsi < 30 else (-0.2 if rsi > 70 else 0)  # RSI
        s_tech += 0.15 if window_closes[-1] > ma20 else -0.15  # MA20 position
        s_tech += 0.15 if window_closes[-1] > ma60 else -0.15  # MA60 position

        # --- Signal 2: Resonance Reverse ---
        ma20_slope = np.polyfit(range(5), window_closes[-5:], 1)[0]
        macd_pos = dif > 0
        s_res = 0.0
        if window_closes[-1] > ma60 and ma20_slope > 0 and macd_pos:
            s_res = -0.5  # strong bull → bearish (reverse)
        elif window_closes[-1] < ma60 and ma20_slope < 0 and not macd_pos:
            s_res = 0.5   # strong bear → bullish (reverse)

        # --- Signal 3: Sector Alpha 24m reverse ---
        s_sec = 0.0
        if industry and i >= 24:
            lookback = min(24, i)
            stock_ret = (window_closes[-1] - window_closes[-lookback-1]) / max(window_closes[-lookback-1], 0.01)
            # Reverse: high alpha → mean reversion
            s_sec = np.clip(-stock_ret * 5, -0.5, 0.5)

        # Aggregate (equal weight)
        total = np.clip(0.25 * s_tech + 0.25 * s_res + 0.25 * s_sec + 0.25 * 0, -1.0, 1.0)
        results.append({'date': as_of, 'score': total,
                        's_tech': s_tech, 's_res': s_res, 's_sec': s_sec})

    if not results:
        return None
    return pd.DataFrame(results).set_index('date')

# ============================================================
# 3. Backtest
# ============================================================

def run_backtest(prices, signals, name='default'):
    common_dates = sorted(set(prices.index) & set(signals.index))
    print(f"\n{name}: {len(common_dates)} months")

    daily_rets = []
    equity = [1.0]
    turnovers = []

    for i, date in enumerate(common_dates[:-1]):
        next_date = common_dates[i + 1]
        sig = signals.loc[date].dropna()
        if len(sig) < TOP_K:
            continue

        top = sig.nlargest(TOP_K)
        w = pd.Series(1.0 / TOP_K, index=top.index)

        rets = []
        for c in w.index:
            p0 = prices.at[date, c] if c in prices.columns else np.nan
            p1 = prices.at[next_date, c] if c in prices.columns else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                rets.append((p1 - p0) / p0)
            else:
                rets.append(0.0)

        port_ret = np.dot(w.values, rets)
        if i > 0 and len(common_dates) > 1:
            prev_sig = signals.loc[common_dates[max(0,i-1)]].dropna()
            if len(prev_sig) >= TOP_K:
                prev_top = set(prev_sig.nlargest(TOP_K).index)
                to = len(set(top.index) - prev_top) / TOP_K
                port_ret -= to * (SLIPPAGE + COMMISSION * 2)
                turnovers.append(to)

        daily_rets.append(port_ret)
        equity.append(equity[-1] * (1 + port_ret))

    rets = pd.Series(daily_rets, index=common_dates[1:])
    eq = pd.Series(equity, index=common_dates[:len(equity)])

    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cummax = eq.expanding().max()
    dd = (eq - cummax) / cummax
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0
    win_rate = (rets > 0).mean()

    print(f"  Ann Ret: {ann_ret:.1%}  Vol: {ann_vol:.1%}  Sharpe: {sharpe:.3f}")
    print(f"  Max DD: {max_dd:.1%}  Calmar: {calmar:.3f}  Win: {win_rate:.1%}")

    return rets, eq, {'sharpe': sharpe, 'max_dd': max_dd, 'calmar': calmar,
                      'ann_ret': ann_ret, 'ann_vol': ann_vol, 'win_rate': win_rate}

# ============================================================
# 4. Main
# ============================================================

def main():
    print("Loading data...")
    df, stock_to_ind = load_all_data()
    stocks = sorted(df['code'].unique())
    print(f"Stocks: {len(stocks)}, rows: {len(df)}")

    # Compute signals per stock
    print("Computing signals...")
    signal_frames = {}
    for code in stocks:
        group = df[df['code'] == code]
        sigs = compute_signals_for_stock(code, group, stock_to_ind)
        if sigs is not None:
            signal_frames[code] = sigs['score']

    signals = pd.DataFrame(signal_frames).sort_index()
    # Forward fill missing months
    signals = signals.reindex(sorted(signals.index)).fillna(0.0)
    print(f"Signal matrix: {signals.shape}")

    # Price matrix
    prices = df.pivot(index='date', columns='code', values='close').sort_index()
    prices = prices.ffill()
    print(f"Price matrix: {prices.shape}")

    # In-sample
    print("\n" + "=" * 60)
    print("IN-SAMPLE: 2015-01 ~ 2023-12")
    print("=" * 60)
    is_ret, is_eq, is_m = run_backtest(
        prices.loc['2014-12':'2024-01'],
        signals.loc['2015-01':'2023-12'],
        'In-Sample')

    # Benchmark
    bench = prices.pct_change().mean(axis=1).dropna()
    bench_is = bench.loc['2015-01':'2023-12']
    bench_sr = bench_is.mean() * 12 / (bench_is.std() * np.sqrt(12))
    bench_ann = bench_is.mean() * 12
    print(f"\n  HS300 Equal-Weight: Sharpe={bench_sr:.3f} Ann Ret={bench_ann:.1%}")
    alpha = is_m['ann_ret'] - bench_ann
    print(f"  Alpha: {alpha:.1%}")

    # Live test
    print("\n" + "=" * 60)
    print("LIVE TEST: 2024-01 ~ 2025-12 (FROZEN)")
    print("=" * 60)
    lt_ret, lt_eq, lt_m = run_backtest(
        prices.loc['2023-12':],
        signals.loc['2024-01':'2025-12'],
        'Live Test')

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<15} {'In-Sample':>12} {'Live Test':>12} {'HS300(IS)':>12}")
    print(f"{'Sharpe':<15} {is_m['sharpe']:12.3f} {lt_m['sharpe']:12.3f} {bench_sr:12.3f}")
    print(f"{'Max DD':<15} {is_m['max_dd']:11.1%} {lt_m['max_dd']:11.1%}")
    print(f"{'Calmar':<15} {is_m['calmar']:12.3f} {lt_m['calmar']:12.3f}")
    print(f"{'Ann Ret':<15} {is_m['ann_ret']:11.1%} {lt_m['ann_ret']:11.1%} {bench_ann:11.1%}")

    # Save
    is_eq.to_csv(OUT_DIR / 'equity_is.csv')
    lt_eq.to_csv(OUT_DIR / 'equity_lt.csv')

if __name__ == '__main__':
    main()
