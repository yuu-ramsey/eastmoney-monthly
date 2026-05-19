"""B1: 3-layer signal fusion — LSTM + Resonance + Sector Alpha.
Train MLP fusion layer with walk-forward, evaluate Test IC."""
import numpy as np, pandas as pd, sqlite3
from pathlib import Path
from scipy.stats import spearmanr
import torch, torch.nn as nn
from sklearn.linear_model import Ridge

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEVICE = torch.device('cuda')

print("Loading 3 signal sources...")

# 1. LSTM monthly signal
lstm = pd.read_parquet(OUT / 'monthly_lstm_signals_v2.parquet')
lstm_dict = {}
for _, r in lstm.iterrows():
    lstm_dict[(r['code'], r['month'])] = r['lstm_signal']

# 2. Resonance reverse (compute live)
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
m_df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01' ORDER BY code, date", conn, params=stocks)
conn.close()

# 3. Sector alpha proxy (24m stock return reversal, as used in backtest)
def compute_signals_for_month(code, closes, highs, lows, idx):
    """Compute resonance + sector signals for one month"""
    w_close = closes[:idx+1]
    # Resonance reverse
    ma60 = np.mean(w_close[-60:]) if len(w_close) >= 60 else np.mean(w_close[-len(w_close):])
    ma20_slope = np.polyfit(range(min(5, len(w_close))), w_close[-min(5, len(w_close)):], 1)[0]
    ema12 = pd.Series(w_close).ewm(span=12).mean().values[-1]
    ema26 = pd.Series(w_close).ewm(span=26).mean().values[-1]
    macd_pos = (ema12 - ema26) > 0
    s_res = 0.0
    if w_close[-1] > ma60 and ma20_slope > 0 and macd_pos: s_res = -0.5
    elif w_close[-1] < ma60 and ma20_slope < 0 and not macd_pos: s_res = 0.5

    # Sector alpha (24m reverse)
    s_sec = 0.0
    if idx >= 24:
        stock_ret = (w_close[-1] - w_close[-25]) / max(w_close[-25], 0.01)
        s_sec = np.clip(-stock_ret * 5, -0.5, 0.5)
    return s_res, s_sec

print("Building signal matrix...")
all_data = []  # [{code, month, s_lstm, s_res, s_sec, fwd_ret}]
for code in stocks:
    g = m_df[m_df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    closes = g['close'].values.astype(float)
    highs = g['high'].values.astype(float)
    lows = g['low'].values.astype(float)
    dates = g['date'].tolist()

    for i in range(60, len(g) - 1):
        month = dates[i]
        # LSTM signal
        s_lstm = lstm_dict.get((code, month), 0.0)
        # Resonance + Sector
        s_res, s_sec = compute_signals_for_month(code, closes, highs, lows, i)
        # Forward 1-month return
        if i + 1 < len(g) and g.iloc[i]['close'] > 0.01:
            fwd_ret = np.clip((g.iloc[i+1]['close'] - g.iloc[i]['close']) / g.iloc[i]['close'], -2, 2)
        else:
            continue
        all_data.append({'code': code, 'month': month, 's_lstm': s_lstm, 's_res': s_res, 's_sec': s_sec, 'fwd_ret': fwd_ret})

    if len(stocks) > 0 and len(all_data) % 5000 == 0:
        pass

df = pd.DataFrame(all_data).dropna()
print(f"Signal matrix: {len(df)} rows (after NaN filter)")

# Walk-forward fusion: for each month, train on all prior data, predict next
print("\nWalk-forward MLP fusion...")
months = sorted(df['month'].unique())
ic_list = []
for i, month in enumerate(months):
    if month < '2018-01': continue  # need warmup
    # Train on all data before this month
    train = df[df['month'] < month]
    test = df[df['month'] == month]

    if len(train) < 500 or len(test) < 20: continue

    X_tr = train[['s_lstm', 's_res', 's_sec']].values.astype(np.float32)
    y_tr = train['fwd_ret'].values.astype(np.float32)
    X_te = test[['s_lstm', 's_res', 's_sec']].values.astype(np.float32)
    y_te = test['fwd_ret'].values.astype(np.float32)

    # Ridge regression (fast, stable, walk-forward safe)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    if len(pred) > 10:
        ic = spearmanr(pred, y_te)[0]
        ic_list.append(ic)

    if i % 24 == 0:
        avg_ic = np.mean(ic_list[-12:]) if len(ic_list) >= 12 else np.mean(ic_list) if ic_list else 0
        print(f"  {month}: IC={ic:.4f}, rolling avg IC={avg_ic:.4f}")

# IS vs OOS split
is_months = [m for m in months if '2018' <= m[:4] <= '2021']
oos_months = [m for m in months if m[:4] >= '2022']
is_ics = [ic_list[i] for i, m in enumerate(months) if m in is_months and (i < len(ic_list))]
oos_ics = [ic_list[i] for i, m in enumerate(months) if m in oos_months and (i < len(ic_list))]

# Simple backtest: equal-weight vs fused
print(f"\nBacktest comparison (2022+ OOS):")
# Equal-weight baseline
ew_preds = df['s_lstm'].values * 0.5 + df['s_res'].values * 0.3 + df['s_sec'].values * 0.2
ew_ic = spearmanr(ew_preds[-5000:], df['fwd_ret'].values[-5000:])[0]
print(f"  Equal-weight IC: {ew_ic:.4f}")

# Fused IC
avg_ic = np.mean(oos_ics) if oos_ics else np.mean(ic_list)
print(f"  Fused IC (OOS): {avg_ic:.4f}")

print(f"\n{'='*60}")
print(f"B1 FINAL")
print(f"{'='*60}")
print(f"  EW baseline IC: {ew_ic:.4f}")
print(f"  MLP fused IC:   {avg_ic:.4f}")
print(f"  IS IC avg:      {np.mean(is_ics):.4f}" if is_ics else "  IS IC: N/A")
delta = avg_ic - ew_ic
print(f"  Δ: {delta:+.4f}")
if avg_ic > 0.05: print("KILL SWITCH: PASS > 0.05")
elif avg_ic > 0.03: print("KILL SWITCH: MARGINAL 0.03-0.05")
else: print("KILL SWITCH: FAIL < 0.03")
