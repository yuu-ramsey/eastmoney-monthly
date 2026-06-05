# perm_ridge.py - Ridge permutation test for IC=0.0603
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json, sys
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DB = '.eastmoney-ai/db/klines-v2.sqlite'
N_PERM = 500

print('[1/3] Loading data...', flush=True)
t0 = time.time()
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute(
    'SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84'
).fetchall()]
ind_map = {r[0]: r[1] for r in conn.execute(
    'SELECT stock_code, industry_code FROM stock_industry_mapping'
)}
params = ','.join('?' * len(codes))
df = pd.read_sql_query(
    f"SELECT code,date,open,high,low,close,volume,turnover_rate "
    f"FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' "
    f"ORDER BY code,date", conn, params=codes
)
conn.close()

rows = []
for code in codes:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 72:
        continue
    c = g['close'].values.astype(float)
    o = g['open'].values.astype(float)
    h = g['high'].values.astype(float)
    l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    dates = g['date'].tolist()
    industry = ind_map.get(code, 'unknown')

    ma5 = pd.Series(c).rolling(5).mean().values
    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    e12 = pd.Series(c).ewm(span=12).mean().values
    e26 = pd.Series(c).ewm(span=26).mean().values
    dif = e12 - e26
    dea = pd.Series(dif).ewm(span=9).mean().values
    macd_hist = (dif - dea) * 2
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100 - 100/(1 + avg_gain/np.maximum(avg_loss, 1e-8)), 50)
    bb_std = pd.Series(c).rolling(20).std().values
    bb_pos = np.nan_to_num((c - (ma20 - 2*bb_std)) / np.maximum(4*bb_std, 0.01), 0.5)
    trange = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
    atr14 = pd.Series(trange).rolling(14).mean().values

    for i in range(60, len(g) - 6):
        if c[i] <= 0.01:
            continue
        fwd = (c[i+3] - c[i]) / c[i]
        if abs(fwd) > 2:
            continue
        rows.append({
            'code': code, 'date': dates[i], 'fwd_ret': fwd,
            'r1': (c[i]-c[i-1])/max(abs(c[i-1]), 0.01) if i >= 1 else 0,
            'r3': (c[i]-c[i-3])/max(abs(c[i-3]), 0.01) if i >= 3 else 0,
            'r6': (c[i]-c[i-6])/max(abs(c[i-6]), 0.01) if i >= 6 else 0,
            'r12': (c[i]-c[i-12])/max(abs(c[i-12]), 0.01) if i >= 12 else 0,
            'ma5d': (c[i]-ma5[i])/max(abs(c[i]), 0.01),
            'ma20d': (c[i]-ma20[i])/max(abs(c[i]), 0.01),
            'ma60d': (c[i]-ma60[i])/max(abs(c[i]), 0.01),
            'macd_dif': dif[i], 'macd_dea': dea[i], 'macd_hist': macd_hist[i],
            'rsi14': rsi14[i], 'bb_pos': bb_pos[i],
            'vol_6': np.std(np.diff(c[max(0,i-6):i+1]) /
                           np.maximum(np.abs(c[max(0,i-5):i+1]), 0.01)) if i >= 6 else 0,
            'atr_ratio': atr14[i]/max(abs(c[i]), 0.01) if not np.isnan(atr14[i]) else 0,
            'hilo': (h[i]-l[i])/max(abs(c[i]), 0.01),
            'above_ma20': 1.0 if c[i] > ma20[i] else 0.0,
            'above_ma60': 1.0 if c[i] > ma60[i] else 0.0,
        })

data = pd.DataFrame(rows).dropna()
feat_cols = [c for c in data.columns if c not in ['code', 'date', 'fwd_ret']]
train = data[(data['date'] >= '2015-01') & (data['date'] <= '2021-12')]
test = data[data['date'] >= '2024-01']
print(f'Data: train={len(train):,} test={len(test):,} features={len(feat_cols)} ({time.time()-t0:.0f}s)', flush=True)

# Real IC
print('[2/3] Computing real Ridge IC...', flush=True)
sc = StandardScaler()
X_tr = sc.fit_transform(train[feat_cols].values.astype(float))
X_te = sc.transform(test[feat_cols].values.astype(float))
y_tr = train['fwd_ret'].values.astype(float)
y_te = test['fwd_ret'].values.astype(float)

m = Ridge(alpha=1.0)
m.fit(X_tr, y_tr)
pred = m.predict(X_te)
real_ics = []
for dt in test['date'].unique():
    mask = test['date'] == dt
    if mask.sum() >= 20:
        real_ics.append(spearmanr(pred[mask], test.loc[mask, 'fwd_ret'].values)[0])
real_ic = np.mean(real_ics)
print(f'Real Ridge IC = {real_ic:+.4f} (std={np.std(real_ics):.4f}, {len(real_ics)} months)', flush=True)

# Permutation test
print(f'[3/3] Cross-section permutation {N_PERM} iterations...', flush=True)
perm_ics = []
better = 0
t0 = time.time()
for n in range(N_PERM):
    test_p = test.copy()
    for dt in test_p['date'].unique():
        mask = test_p['date'] == dt
        test_p.loc[mask, 'fwd_ret'] = test_p.loc[mask, 'fwd_ret'].sample(frac=1, random_state=None).values
    X_te_p = sc.transform(test_p[feat_cols].values.astype(float))
    m = Ridge(alpha=1.0)
    m.fit(X_tr, y_tr)
    pred_p = m.predict(X_te_p)
    ics = []
    for dt in test_p['date'].unique():
        mask = test_p['date'] == dt
        if mask.sum() >= 20:
            ics.append(spearmanr(pred_p[mask], test_p.loc[mask, 'fwd_ret'].values)[0])
    ic = np.mean(ics) if ics else np.nan
    perm_ics.append(ic)
    if ic >= real_ic:
        better += 1
    if (n + 1) % 100 == 0:
        e = time.time() - t0
        eta = e / (n + 1) * (N_PERM - n - 1)
        print(f'  [{n+1}/{N_PERM}] perm_IC={ic:+.4f} better={better}/{n+1} p={better/(n+1):.4f} ({e:.0f}s ETA {eta:.0f}s)', flush=True)

p_val = better / N_PERM

print('')
print('=' * 60)
print('Ridge Permutation Test Results')
print(f'  Real IC:        {real_ic:+.4f}')
print(f'  P-value:        {p_val:.4f}')
print(f'  Iterations:     {N_PERM}')
print(f'  Perm IC mean:   {np.mean(perm_ics):+.4f} +/- {np.std(perm_ics):.4f}')
print(f'  Perm IC 95%:    {np.percentile(perm_ics, 95):+.4f}')
print(f'  Perm IC 99%:    {np.percentile(perm_ics, 99):+.4f}')
if p_val < 0.05:
    verdict = 'SIGNIFICANT - IC unlikely from noise'
elif p_val < 0.10:
    verdict = 'MARGINAL - borderline significance'
else:
    verdict = 'NOT SIGNIFICANT - IC may be noise'
print(f'  Verdict:        {verdict}')
print('=' * 60)

Path('.eastmoney-ai/eval').mkdir(parents=True, exist_ok=True)
result = {
    'real_ic': float(real_ic),
    'p_value': float(p_val),
    'n_perm': N_PERM,
    'perm_ics': [float(x) for x in perm_ics],
    'real_ics': [float(x) for x in real_ics],
    'perm_mean': float(np.mean(perm_ics)),
    'perm_std': float(np.std(perm_ics)),
    'perm_p95': float(np.percentile(perm_ics, 95)),
    'perm_p99': float(np.percentile(perm_ics, 99)),
    'verdict': verdict,
    'n_stocks': len(codes),
    'n_train': len(train),
    'n_test': len(test),
    'n_features': len(feat_cols),
}
with open('.eastmoney-ai/eval/perm_ridge.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved to .eastmoney-ai/eval/perm_ridge.json')
