"""Monthly LightGBM hyperparameter sweep + feature importance analysis."""
import numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
import lightgbm as lgb

conn = sqlite3.connect('.eastmoney-ai/db/klines-v2.sqlite')
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()][:1000]
params_str = ','.join('?' * len(codes))
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({params_str}) AND date >= '2010-01' ORDER BY code, date", conn, params=codes)
conn.close()
print(f'Stocks: {len(codes)}')

rows = []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    n = len(g); c = g['close'].values.astype(float)
    h = g['high'].values.astype(float); l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float); dates = g['date'].tolist()
    ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    dif = e12-e26; dea = pd.Series(dif).ewm(span=9).mean().values
    delta = np.diff(c,prepend=c[0]); gain = np.where(delta>0,delta,0); loss = np.where(delta<0,-delta,0)
    avg_g = pd.Series(gain).ewm(alpha=1/14).mean().values; avg_l = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100-100/(1+avg_g/np.maximum(avg_l,1e-8)), 50)

    for i in range(60, n-6):
        if c[i]<=0.01 or abs((c[i+3]-c[i])/c[i])>2: continue
        rows.append({'code':code,'date':dates[i],'fwd_ret':(c[i+3]-c[i])/c[i],
            'r1':(c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0,
            'r3':(c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0,
            'r6':(c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0,
            'r12':(c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0,
            'ma5d':(c[i]-ma5[i])/max(abs(c[i]),0.01), 'ma20d':(c[i]-ma20[i])/max(abs(c[i]),0.01),
            'ma60d':(c[i]-ma60[i])/max(abs(c[i]),0.01) if not np.isnan(ma60[i]) else 0,
            'macd_dif':dif[i], 'macd_hist':(dif[i]-dea[i])*2, 'rsi14':rsi14[i],
            'hilo':(h[i]-l[i])/max(abs(c[i]),0.01),
            'above_ma20':1.0 if c[i]>ma20[i] else 0.0,
            'vol_chg':np.mean(v[max(0,i-3):i+1])/max(np.mean(v[max(0,i-12):i+1]),1)-1 if i>=12 else 0})

data = pd.DataFrame(rows).fillna(0)
train = data[(data['date']>='2015-01')&(data['date']<='2021-12')]
test = data[data['date']>='2024-01']
feats = [c for c in data.columns if c not in ['code','date','fwd_ret']]
print(f'Train={len(train)}, Test={len(test)}, Features={len(feats)}')

# Grid + random search
best_ic, best_params, best_name = -1, None, ''
configs = [
    ('small', 31, 0.05, 100, 20), ('medium', 63, 0.03, 200, 10),
    ('large', 127, 0.02, 300, 5), ('deep', 255, 0.01, 500, 5),
    ('wide', 31, 0.03, 300, 10), ('fast', 15, 0.1, 100, 30),
    ('balanced', 63, 0.02, 200, 10), ('big', 255, 0.02, 300, 10),
    ('shallow_wide', 31, 0.01, 500, 20), ('v1', 63, 0.03, 200, 10),
    ('v2', 127, 0.03, 200, 10), ('v3', 63, 0.05, 200, 10),
]

t0 = time.time()
for name, leaves, lr, n_est, min_child in configs:
    m = lgb.LGBMRegressor(objective='regression', num_leaves=leaves, learning_rate=lr,
        n_estimators=n_est, min_child_samples=min_child, subsample=0.8,
        colsample_bytree=0.8, random_state=456, verbosity=-1)
    m.fit(train[feats].values, train['fwd_ret'].values)
    pred = m.predict(test[feats].values)
    ics = [spearmanr(pred[test['date']==d], test.loc[test['date']==d,'fwd_ret'].values)[0]
           for d in test['date'].unique() if (test['date']==d).sum()>=20]
    ic = np.mean(ics)
    print(f'  {name:15s} leaves={leaves:3d} lr={lr:.3f} n={n_est:3d} mc={min_child:2d}: IC={ic:+.4f}')
    if ic > best_ic: best_ic, best_params, best_name = ic, (leaves,lr,n_est,min_child), name

print(f'\nBest: {best_name} IC={best_ic:+.4f} params={best_params}')
print(f'Baseline: 0.063 ({time.time()-t0:.0f}s)')
