"""Monthly cross-sectional ranking with LightGBM + strict v6.
Target: within-month rank of forward 3m return (not absolute return).
Easier problem than regression. LightGBM faster than LSTM for tabular data."""
import numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

# Load monthly data
conn = sqlite3.connect(str(DB))
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
# Use only HS300+CSI1000 (813 optimal from previous test)
hs300_csi1000 = set()
for idx in ['000300', '000852']:
    import akshare as ak
    df = ak.index_stock_cons(idx)
    hs300_csi1000.update(df['品种代码'].tolist())
codes = [c for c in codes if c in hs300_csi1000]
print(f"Stocks (HS300+CSI1000): {len(codes)}")

params_str = ','.join('?' * len(codes))
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({params_str}) AND date >= '2010-01' ORDER BY code, date", conn, params=codes)
conn.close()
print(f"Rows: {len(df)}")

# Build tabular features (not sequences — flat features per stock-month)
print("Building features...")
rows = []
for code in codes:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    n = len(g)
    c = g['close'].values.astype(float); o = g['open'].values.astype(float)
    h = g['high'].values.astype(float); l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    dates = g['date'].tolist()

    for i in range(60, n - 6):
        if c[i] <= 0.01: continue
        # Forward 3m return (target for ranking)
        fwd_ret = (c[i+3] - c[i]) / c[i]
        if abs(fwd_ret) > 2: continue

        # Features: 1m/3m/6m/12m momentum, vol, MA deviations, volume change
        feats = {}
        feats['ret_1m'] = (c[i] - c[i-1]) / c[i-1] if i >= 1 else 0
        feats['ret_3m'] = (c[i] - c[i-3]) / c[i-3] if i >= 3 else 0
        feats['ret_6m'] = (c[i] - c[i-6]) / c[i-6] if i >= 6 else 0
        feats['ret_12m'] = (c[i] - c[i-12]) / c[i-12] if i >= 12 else 0
        feats['vol_3m'] = np.std(np.diff(c[i-3:i+1]) / c[i-3:i]) if i >= 3 else 0
        feats['vol_6m'] = np.std(np.diff(c[i-6:i+1]) / c[i-6:i]) if i >= 6 else 0
        feats['ma20_dev'] = (c[i] - np.mean(c[max(0,i-20):i+1])) / c[i]
        feats['ma60_dev'] = (c[i] - np.mean(c[max(0,i-60):i+1])) / c[i]
        feats['vol_chg'] = np.mean(v[max(0,i-3):i+1]) / max(np.mean(v[max(0,i-12):i+1]), 1) - 1 if i >= 12 else 0
        feats['high_low'] = (h[i] - l[i]) / c[i]
        feats['turnover_proxy'] = v[i] / max(np.mean(v[max(0,i-12):i+1]), 1) - 1 if i >= 12 else 0

        rows.append({'code': code, 'month': dates[i], 'fwd_ret': fwd_ret, **feats})

data = pd.DataFrame(rows).dropna()
print(f"Data: {len(data)} rows, {len(data.columns)-3} features")

# Cross-sectional ranking: within each month, rank stocks by fwd_ret
data['rank'] = data.groupby('month')['fwd_ret'].rank(pct=True)  # 0-1 percentile rank

# Strict date split
train = data[data['month'].between('2015-01', '2021-12')]
test = data[data['month'] >= '2024-01']
print(f"Train: {len(train)}, Test: {len(test)}")

feature_cols = [c for c in data.columns if c not in ['code', 'month', 'fwd_ret', 'rank']]
Xtr, ytr = train[feature_cols].values, train['rank'].values
Xte, yte = test[feature_cols].values, test['rank'].values

# LightGBM ranking (no val data, fixed params, no early stop)
print("Training LightGBM ranker...")
model = lgb.LGBMRegressor(
    objective='regression', metric='l1', num_leaves=31, learning_rate=0.05,
    n_estimators=100, min_child_samples=20, random_state=456, verbosity=-1
)

t0 = time.time()
model.fit(Xtr, ytr)
pred = model.predict(Xte)

# Evaluate: IC per month
test_months = test['month'].unique()
ics = []
for m in test_months:
    mask = test['month'] == m
    if mask.sum() < 10: continue
    ic = spearmanr(pred[mask], test.loc[mask, 'fwd_ret'].values)[0]
    ics.append(ic)

avg_ic = np.mean(ics)
print(f"Test avg monthly IC: {avg_ic:+.4f} ({len(ics)} months, {time.time()-t0:.0f}s)")

# Top-bottom spread
test_df = test.copy()
test_df['pred'] = pred
n = len(test_df); cut = int(n * 0.3)
test_df_sorted = test_df.sort_values('pred')
top = test_df_sorted.iloc[-cut:]['fwd_ret'].mean()
bot = test_df_sorted.iloc[:cut]['fwd_ret'].mean()
spread = top - bot
print(f"Top-Bottom spread: {spread:+.4f} (monthly, {cut} stocks each)")

# Daily baseline
print(f"\nDaily LSTM IC3:      +0.141 (381K seqs, LSTM)")
print(f"813-stock Monthly LSTM: +0.027 (46K seqs, LSTM)")
print(f"Monthly LightGBM Rank:  {avg_ic:+.4f} ({len(train)} rows, LightGBM)")
