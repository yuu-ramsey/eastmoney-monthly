"""B1-XGB: XGBoost stacking ensemble vs Ridge baseline.
Walk-forward 逐月训练，严格防未来信息泄露。
对比 Ridge IC=0.063 baseline vs XGBoost with same + extended features.
"""
import numpy as np, pandas as pd, sqlite3, warnings
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

# ======== 1. 加载 LSTM monthly signals ========
print("1/3 Loading data...")
lstm_path = OUT / 'monthly_lstm_signals_v2.parquet'
lstm_dict = {}
if lstm_path.exists():
    lstm = pd.read_parquet(lstm_path)
    for _, r in lstm.iterrows():
        lstm_dict[(r['code'], r['month'])] = r['lstm_signal']
print(f"  {len(lstm_dict)} LSTM signals")

# ======== 2. 加载月线 ========
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute(
    "SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
m_df = pd.read_sql_query(
    f"SELECT code, date, open, high, low, close, volume FROM monthly_klines "
    f"WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01' "
    f"ORDER BY code, date", conn, params=stocks)
conn.close()
print(f"  {len(stocks)} stocks, {len(m_df)} monthly rows")

# ======== 3. 构建特征矩阵 ========
print("2/3 Building features...")

def compute_quant_score(closes, highs, lows, volumes, idx):
    """Quant factor score at index"""
    n = idx + 1
    if n < 60: return 0.0

    # f1: trend strength (MA alignment + slope) — weight 0.35
    ma5 = np.mean(closes[max(0,n-5):n])
    ma20 = np.mean(closes[max(0,n-20):n])
    ma60 = np.mean(closes[max(0,n-60):n])
    slope = np.polyfit(range(min(5,n)), closes[-min(5,n):], 1)[0] if n >= 5 else 0
    align = 1 if closes[-1] > ma5 > ma20 > ma60 else (-1 if closes[-1] < ma5 < ma20 < ma60 else 0)
    f1 = 0.35 * np.tanh(slope * 10) + 0.35 * align

    # f2: price position (percentile in 60-bar) — weight 0.25
    r60 = closes[max(0,n-60):n]
    f2 = 0.25 * (np.clip((closes[-1] - np.min(r60)) / max(np.max(r60) - np.min(r60), 0.01), 0, 1) - 0.5)

    # f3: volatility percentile — weight 0.15
    amps = [(highs[i] - lows[i]) / max(closes[i], 0.01) for i in range(max(0,n-30), n)]
    pct = np.mean([1 if a < amps[-1] else 0 for a in amps]) if amps else 0.5
    f3 = 0.15 * (pct - 0.5)

    # f4: volume-price correlation — weight 0.25
    w = min(21, n)
    close_win = closes[n-w:n]
    vol_win = volumes[n-w:n]
    rets = np.diff(close_win) / np.maximum(close_win[:-1], 0.01)
    vols_paired = vol_win[1:]  # volume for each return period
    if len(rets) >= 10 and np.std(rets) > 0 and np.std(vols_paired) > 0:
        corr = np.corrcoef(rets, vols_paired)[0,1]
        f4 = 0.25 * np.clip(0 if np.isnan(corr) else corr, -1, 1)
    else:
        f4 = 0.0

    return np.clip((f1 + f2 + f3 + f4) * 100, -100, 100)

# Feature sets
FEATURES_3 = ['s_lstm', 's_res', 's_sec']          # Ridge baseline (3)
FEATURES_5 = ['s_lstm', 's_res', 's_sec', 'quant', 'lstm_x_quant']  # Extended (5)

all_rows = []
for code in stocks:
    g = m_df[m_df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    closes = g['close'].values.astype(float)
    highs = g['high'].values.astype(float)
    lows = g['low'].values.astype(float)
    vols = g['volume'].values.astype(float)
    dates = g['date'].tolist()

    for i in range(60, len(g) - 1):
        month = dates[i][:7]
        s_lstm = lstm_dict.get((code, month), 0.0)

        # Resonance
        ma60 = np.mean(closes[max(0,i-60):i+1])
        slope20 = np.polyfit(range(min(5,i+1)), closes[-min(5,i+1):], 1)[0]
        ema12 = pd.Series(closes[:i+1]).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(closes[:i+1]).ewm(span=26).mean().iloc[-1]
        macd_pos = (ema12 - ema26) > 0
        s_res = 0.0
        if closes[-1] > ma60 and slope20 > 0 and macd_pos: s_res = -0.5
        elif closes[-1] < ma60 and slope20 < 0 and not macd_pos: s_res = 0.5

        # Sector alpha (24m reversal)
        s_sec = 0.0
        if i >= 24:
            ret24 = (closes[-1] - closes[i-24]) / max(closes[i-24], 0.01)
            s_sec = np.clip(-ret24 * 5, -0.5, 0.5)

        # Quant
        quant = compute_quant_score(closes, highs, lows, vols, i)
        lstm_x_quant = s_lstm * quant / 100.0

        # Forward return
        if closes[i] > 0.01:
            fwd_ret = np.clip((closes[i+1] - closes[i]) / closes[i], -2, 2)
        else:
            continue

        all_rows.append({
            'code': code, 'month': month,
            's_lstm': s_lstm, 's_res': s_res, 's_sec': s_sec,
            'quant': quant, 'lstm_x_quant': lstm_x_quant,
            'fwd_ret': fwd_ret,
        })

df = pd.DataFrame(all_rows).dropna()
print(f"  {len(df)} rows")

# ======== Walk-Forward Comparison ========
print("\n3/3 Walk-Forward: Ridge(3) vs Ridge(5) vs XGB(3) vs XGB(5)")
print("="*70)

months = sorted(df['month'].unique())
results = {name: [] for name in ['Ridge3', 'Ridge5', 'XGB3', 'XGB5']}

for i, month in enumerate(months):
    if month < '2018-01': continue

    tr = df[df['month'] < month]
    te = df[df['month'] == month]
    if len(tr) < 500 or len(te) < 20: continue

    # Ridge 3
    X3_tr = tr[FEATURES_3].values.astype(np.float32)
    X3_te = te[FEATURES_3].values.astype(np.float32)
    y_tr = tr['fwd_ret'].values.astype(np.float32)
    y_te = te['fwd_ret'].values.astype(np.float32)

    ridge3 = Ridge(alpha=1.0).fit(X3_tr, y_tr)
    p = ridge3.predict(X3_te)
    if len(p) > 10:
        ic = spearmanr(p, y_te)[0]
        if not np.isnan(ic): results['Ridge3'].append(ic)

    # Ridge 5
    X5_tr = tr[FEATURES_5].values.astype(np.float32)
    X5_te = te[FEATURES_5].values.astype(np.float32)
    ridge5 = Ridge(alpha=1.0).fit(X5_tr, y_tr)
    p = ridge5.predict(X5_te)
    if len(p) > 10:
        ic = spearmanr(p, y_te)[0]
        if not np.isnan(ic): results['Ridge5'].append(ic)

    # XGB 3
    xgb3 = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0,
    ).fit(X3_tr, y_tr)
    p = xgb3.predict(X3_te)
    if len(p) > 10:
        ic = spearmanr(p, y_te)[0]
        if not np.isnan(ic): results['XGB3'].append(ic)

    # XGB 5
    xgb5 = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0,
    ).fit(X5_tr, y_tr)
    p = xgb5.predict(X5_te)
    if len(p) > 10:
        ic = spearmanr(p, y_te)[0]
        if not np.isnan(ic): results['XGB5'].append(ic)

    if i % 24 == 0:
        def r(name): return np.mean(results[name][-12:]) if len(results[name]) >= 12 else (np.mean(results[name]) if results[name] else 0)
        print(f"  {month}: R3={results['Ridge3'][-1]:.4f} R5={results['Ridge5'][-1]:.4f} X3={results['XGB3'][-1]:.4f} X5={results['XGB5'][-1]:.4f}")

# ======== Report ========
print(f"\n{'='*60}")
print("FINAL: Ridge vs XGBoost Walk-Forward IC")
print(f"{'='*60}")
for name in ['Ridge3', 'Ridge5', 'XGB3', 'XGB5']:
    if results[name]:
        m = np.mean(results[name])
        s = np.std(results[name])
        print(f"  {name:10s}: IC={m:.4f} ± {s:.4f} (n={len(results[name])})")

# Feature importance (XGB5 only)
if results['XGB5']:
    # Retrain on all data for final importance
    X_all = df[FEATURES_5].values.astype(np.float32)
    y_all = df['fwd_ret'].values.astype(np.float32)
    final_xgb = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0,
    ).fit(X_all, y_all)

    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.argsort(final_xgb.feature_importances_)[::-1]
    ax.barh([FEATURES_5[i] for i in idx][::-1],
            [final_xgb.feature_importances_[i] for i in idx][::-1])
    ax.set_xlabel('Importance')
    ax.set_title('XGBoost Feature Importance (5 features)')
    fig.tight_layout()
    fig.savefig(OUT / 'xgboost_feature_importance.png', dpi=150)
    print(f"\n  Feature importance saved: {OUT / 'xgboost_feature_importance.png'}")

print(f"\n{'='*60}")
