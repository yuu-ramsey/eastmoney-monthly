"""
Phase 17 LSTM: Data pipeline
Extract 21-dim features from SQLite, strict walk-forward z-score, output train/val/test npz
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

DB_PATH = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT_DIR = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'lstm'
LOOKBACK = 60
TRAIN_END = '2021-12'
VAL_END = '2023-12'
TEST_END = '2026-05'

# ---- Indicator calculation (pure Python, no external deps) ----

def ema(series, period):
    """EMA with Wilder's smoothing"""
    result = np.full(len(series), np.nan)
    if len(series) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        result[i] = series[i] * k + result[i - 1] * (1 - k)
    return result

def sma(series, period):
    result = np.full(len(series), np.nan)
    for i in range(period - 1, len(series)):
        result[i] = np.mean(series[i - period + 1:i + 1])
    return result

def rsi(closes, period=14):
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full(len(closes), np.nan)
    avg_loss = np.full(len(closes), np.nan)
    if len(closes) <= period:
        return np.full(len(closes), np.nan)
    avg_gain[period] = np.mean(gains[1:period + 1])
    avg_loss[period] = np.mean(losses[1:period + 1])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100.0 - 100.0 / (1.0 + rs)

def macd(closes, fast=12, slow=26, signal=9):
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    dif = ema_fast - ema_slow
    dea = ema(np.nan_to_num(dif, nan=0), signal)
    hist = (dif - dea) * 2
    return dif, dea, hist

def kdj(highs, lows, closes, n=9):
    k = np.full(len(closes), np.nan)
    d = np.full(len(closes), np.nan)
    j = np.full(len(closes), np.nan)
    for i in range(n - 1, len(closes)):
        hh = np.max(highs[i - n + 1:i + 1])
        ll = np.min(lows[i - n + 1:i + 1])
        rsv = (closes[i] - ll) / max(hh - ll, 0.01) * 100
        if i == n - 1:
            k[i] = 50.0
            d[i] = 50.0
        else:
            k[i] = k[i - 1] * 2 / 3 + rsv * 1 / 3
            d[i] = d[i - 1] * 2 / 3 + k[i] * 1 / 3
        j[i] = 3 * k[i] - 2 * d[i]
    return k, d, j

def bollinger(closes, period=20, n_std=2):
    mid = sma(closes, period)
    upper = np.full(len(closes), np.nan)
    lower = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        std = np.std(closes[i - period + 1:i + 1])
        upper[i] = mid[i] + n_std * std
        lower[i] = mid[i] - n_std * std
    return upper, mid, lower

def atr(highs, lows, closes, period=14):
    result = np.full(len(closes), np.nan)
    tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1)))
    tr[0] = highs[0] - lows[0]
    for i in range(period - 1, len(closes)):
        result[i] = np.mean(tr[i - period + 1:i + 1])
    return result

def obv(closes, volumes):
    result = np.full(len(closes), np.nan)
    result[0] = volumes[0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            result[i] = result[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            result[i] = result[i - 1] - volumes[i]
        else:
            result[i] = result[i - 1]
    return result

def cci(highs, lows, closes, period=20):
    tp = (highs + lows + closes) / 3
    result = np.full(len(closes), np.nan)
    sma_tp = sma(tp, period)
    for i in range(period - 1, len(closes)):
        mean_dev = np.mean(np.abs(tp[i - period + 1:i + 1] - sma_tp[i]))
        result[i] = (tp[i] - sma_tp[i]) / max(mean_dev * 0.015, 0.001)
    return result

def wr(highs, lows, closes, period=14):
    result = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        hh = np.max(highs[i - period + 1:i + 1])
        ll = np.min(lows[i - period + 1:i + 1])
        result[i] = (hh - closes[i]) / max(hh - ll, 0.01) * -100
    return result

def mfi(highs, lows, closes, volumes, period=14):
    tp = (highs + lows + closes) / 3
    mf = tp * volumes
    result = np.full(len(closes), np.nan)
    for i in range(period, len(closes)):
        pos = neg = 0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos += mf[j]
            else:
                neg += mf[j]
        mr = pos / max(neg, 0.001)
        result[i] = 100.0 - 100.0 / (1.0 + mr)
    return result

def stoch(highs, lows, closes, n=9):
    k = np.full(len(closes), np.nan)
    for i in range(n - 1, len(closes)):
        hh = np.max(highs[i - n + 1:i + 1])
        ll = np.min(lows[i - n + 1:i + 1])
        k[i] = (closes[i] - ll) / max(hh - ll, 0.01) * 100
    return k

# ---- Rolling z-score (strict walk-forward) ----

def rolling_zscore(series, lookback=60):
    """Calculate rolling z-score using only past data"""
    result = np.full(len(series), 0.0)
    for i in range(lookback, len(series)):
        window = series[max(0, i - lookback):i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-8:
            result[i] = (series[i] - mean) / std
    return result

# ---- Feature extraction ----

def extract_features(db_path, output_dir):
    conn = sqlite3.connect(str(db_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all HS300 stocks with monthly klines
    query = """
    SELECT m.code, m.date, m.open, m.high, m.low, m.close, m.volume,
           COALESCE(sim.industry_code, '000000') as industry_code
    FROM monthly_klines m
    LEFT JOIN stock_industry_mapping sim ON m.code = sim.stock_code
    WHERE m.code IN (SELECT DISTINCT stock_code FROM stock_industry_mapping)
      AND m.date >= '2010-01'
    ORDER BY m.code, m.date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df)} rows, {df['code'].nunique()} stocks")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")

    # Encode industry codes (31 classes -> 0-based index)
    industry_codes = sorted(df['industry_code'].unique())
    ind_map = {c: i for i, c in enumerate(industry_codes)}
    print(f"Industries: {len(industry_codes)}")
    df['industry_id'] = df['industry_code'].map(ind_map)

    # Build features per stock
    all_sequences = []
    all_targets = []
    all_seq_dates = []
    stock_count = 0
    skipped = 0

    for code, group in df.groupby('code'):
        group = group.sort_values('date').reset_index(drop=True)
        if len(group) < LOOKBACK + 24:  # need lookback + horizon months
            skipped += 1
            continue

        closes = group['close'].values.astype(np.float64)
        dates = group['date'].tolist()
        opens = group['open'].values.astype(np.float64)
        highs = group['high'].values.astype(np.float64)
        lows = group['low'].values.astype(np.float64)
        volumes = group['volume'].values.astype(np.float64)
        industry_id = group['industry_id'].iloc[0]

        # OHLCV: 5 features (z-score normalized)
        close_zs = rolling_zscore(closes, LOOKBACK)
        open_zs = rolling_zscore(opens, LOOKBACK)
        high_zs = rolling_zscore(highs, LOOKBACK)
        low_zs = rolling_zscore(lows, LOOKBACK)
        vol_zs = rolling_zscore(volumes, LOOKBACK)

        # Indicators
        dif, dea, hist = macd(closes)
        rsi14 = rsi(closes, 14)
        k, d, j = kdj(highs, lows, closes, 9)
        boll_u, boll_m, boll_l = bollinger(closes, 20)
        atr14 = atr(highs, lows, closes, 14)
        obv_line = obv(closes, volumes)
        cci20 = cci(highs, lows, closes, 20)
        wr14 = wr(highs, lows, closes, 14)
        mfi14 = mfi(highs, lows, closes, volumes, 14)
        sto9 = stoch(highs, lows, closes, 9)
        sma20 = sma(closes, 20)
        ema20 = ema(closes, 20)
        ma60 = sma(closes, 60)

        # Sector alpha (24m lookback) — placeholder, will be loaded separately

        # Build feature matrix: (time, 21)
        n = len(group)
        features = np.zeros((n, 21), dtype=np.float32)

        # 0-4: OHLCV z-score
        features[:, 0] = close_zs
        features[:, 1] = open_zs
        features[:, 2] = high_zs
        features[:, 3] = low_zs
        features[:, 4] = vol_zs

        # 5-19: Indicators
        features[:, 5] = np.nan_to_num(dif, nan=0)
        features[:, 6] = np.nan_to_num(dea, nan=0)
        features[:, 7] = np.nan_to_num(hist, nan=0)
        features[:, 8] = np.nan_to_num(rsi14, nan=50)
        features[:, 9] = np.nan_to_num(k, nan=50)
        features[:, 10] = np.nan_to_num(j, nan=50)
        features[:, 11] = np.nan_to_num(((closes - boll_l) / np.maximum(boll_u - boll_l, 0.01)), nan=0.5)  # Boll position
        features[:, 12] = np.nan_to_num(atr14 / closes, nan=0)  # ATR ratio
        features[:, 13] = np.nan_to_num(obv_line / np.maximum(volumes.cumsum(), 1), nan=0)  # OBV ratio
        features[:, 14] = np.nan_to_num(cci20, nan=0)
        features[:, 15] = np.nan_to_num(wr14, nan=-50)
        features[:, 16] = np.nan_to_num(mfi14, nan=50)
        features[:, 17] = np.nan_to_num(sto9, nan=50)
        features[:, 18] = np.nan_to_num((closes - sma20) / np.maximum(sma20, 0.01), nan=0)  # SMA20 dev
        features[:, 19] = np.nan_to_num((closes - ma60) / np.maximum(ma60, 0.01), nan=0)  # MA60 dev

        # 20: Industry embedding ID (normalized 0-1)
        features[:, 20] = industry_id / max(len(industry_codes) - 1, 1)

        # Forward fill NaN (first LOOKBACK rows may have NaN from rolling z-score)
        features = np.nan_to_num(features, nan=0.0)

        # Generate sequences: for each starting month >= LOOKBACK-1 with enough horizon
        for i in range(LOOKBACK - 1, n - 12):
            # Skip if current close <= 0 (bad data)
            if closes[i] <= 0.01:
                continue
            seq = features[i - LOOKBACK + 1:i + 1]  # (60, 21)

            # Forward returns (as percentage ratio, e.g. 0.05 = 5%)
            y3 = (closes[i + 3] - closes[i]) / closes[i]
            y6 = (closes[i + 6] - closes[i]) / closes[i]

            # Clip extreme returns (±200%)
            y3 = np.clip(y3, -2.0, 2.0)
            y6 = np.clip(y6, -2.0, 2.0)

            all_sequences.append(seq)
            all_targets.append([y3, y6])
            all_seq_dates.append(dates[i])  # cutoff date for this sequence

        stock_count += 1
        if stock_count % 50 == 0:
            print(f"  Processed {stock_count} stocks, {len(all_sequences)} sequences")

    print(f"\nStocks: {stock_count} processed, {skipped} skipped (too short)")
    print(f"Total sequences: {len(all_sequences)}")

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    seq_dates = np.array(all_seq_dates)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Save full dataset
    np.savez_compressed(out_dir / 'lstm_dataset_full.npz', X=X, y=y, dates=seq_dates)

    # ---- Strict walk-forward split by cutoff date ----
    train_mask = (seq_dates >= '2015-01') & (seq_dates <= '2021-12')
    val_mask = (seq_dates >= '2022-01') & (seq_dates <= '2023-12')
    test_mask = (seq_dates >= '2024-01') & (seq_dates <= '2026-05')

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nWalk-forward split by cutoff date:")
    print(f"  Train (2015-01~2021-12): {len(X_train)} seqs")
    print(f"  Val   (2022-01~2023-12): {len(X_val)} seqs")
    print(f"  Test  (2024-01~2026-05): {len(X_test)} seqs")
    print(f"  (dropped outside range): {len(X) - len(X_train) - len(X_val) - len(X_test)} seqs")

    # Filter NaN targets
    def filter_nan(X, y):
        mask = ~np.isnan(y).any(axis=1)
        return X[mask], y[mask]

    X_train, y_train = filter_nan(X_train, y_train)
    X_val, y_val = filter_nan(X_val, y_val)
    X_test, y_test = filter_nan(X_test, y_test)

    np.savez_compressed(out_dir / 'train.npz', X=X_train, y=y_train)
    np.savez_compressed(out_dir / 'val.npz', X=X_val, y=y_val)
    np.savez_compressed(out_dir / 'test.npz', X=X_test, y=y_test)

    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"y_val range: [{y_val.min():.4f}, {y_val.max():.4f}]")
    print(f"y_test range: [{y_test.min():.4f}, {y_test.max():.4f}]")

    # Sanity checks
    for name, (xx, yy) in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        nan_x = np.any(np.isnan(xx))
        nan_y = np.any(np.isnan(yy))
        inf_x = np.any(np.isinf(xx))
        print(f"  {name}: NaN_X={nan_x} NaN_Y={nan_y} Inf_X={inf_x} shape={xx.shape} y_range=[{yy.min():.4f}, {yy.max():.4f}]")
    assert not np.any(np.isnan(X_train)), "NaN in train X!"
    assert not np.any(np.isnan(y_train)), "NaN in train y!"

    print("\nData pipeline complete.")
    return out_dir

if __name__ == '__main__':
    extract_features(DB_PATH, OUT_DIR)
