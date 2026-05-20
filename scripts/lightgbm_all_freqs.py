"""LightGBM on all frequencies: monthly (full 1158 stocks) + weekly (300 stocks).
Strict v6 date split. Flat tabular features."""
import numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb

DB = '.eastmoney-ai/db/klines-v2.sqlite'
conn = sqlite3.connect(DB)

results = {}
for freq, table, min_len, fwd_step in [('Monthly','monthly_klines',84,3), ('Weekly','weekly_klines',200,13)]:
    codes = [r[0] for r in conn.execute(f'SELECT code FROM {table} GROUP BY code HAVING COUNT(*)>=?',(min_len,)).fetchall()]
    print(f'\n{freq}: {len(codes)} stocks')
    params_str = ','.join('?' * len(codes))
    df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM {table} WHERE code IN ({params_str}) AND date >= '2010-01-01' ORDER BY code, date", conn, params=codes)

    rows = []
    for code in codes[:300] if freq == 'Weekly' else codes:  # limit weekly to 300 for speed
        g = df[df['code']==code].sort_values('date').reset_index(drop=True)
        if len(g) < min_len: continue
        n = len(g); c = g['close'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float); dates = g['date'].tolist()
        for i in range(60, n - fwd_step - 1):
            if c[i] <= 0.01: continue
            fwd_ret = (c[i+fwd_step] - c[i]) / c[i]
            if abs(fwd_ret) > 2: continue
            feats = {
                'ret_1': (c[i]-c[i-1])/max(abs(c[i-1]), 0.01) if i>=1 else 0,
                'ret_3': (c[i]-c[i-3])/max(abs(c[i-3]), 0.01) if i>=3 else 0,
                'ret_6': (c[i]-c[i-6])/max(abs(c[i-6]), 0.01) if i>=6 else 0,
                'ret_12': (c[i]-c[i-12])/max(abs(c[i-12]), 0.01) if i>=12 else 0,
                'vol_6': np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]), 0.01)) if i>=6 else 0,
                'ma20_d': (c[i]-np.mean(c[max(0,i-20):i+1]))/max(abs(c[i]), 0.01),
                'ma60_d': (c[i]-np.mean(c[max(0,i-60):i+1]))/max(abs(c[i]), 0.01),
                'vol_chg': np.mean(v[max(0,i-3):i+1])/max(np.mean(v[max(0,i-12):i+1]), 1)-1 if i>=12 else 0,
                'hilo': (h[i]-l[i])/max(abs(c[i]), 0.01),
                'to': v[i]/max(np.mean(v[max(0,i-12):i+1]), 1)-1 if i>=12 else 0,
            }
            rows.append({'code':code,'date':dates[i],'fwd_ret':fwd_ret,**feats})

    data = pd.DataFrame(rows).dropna()
    train = data[(data['date']>='2015-01') & (data['date']<='2021-12')]
    test = data[data['date']>='2024-01']
    feat_cols = [c for c in data.columns if c not in ['code','date','fwd_ret']]
    model = lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=31,learning_rate=0.05,n_estimators=100,min_child_samples=20,random_state=456,verbosity=-1)
    model.fit(train[feat_cols].values, train['fwd_ret'].values)
    pred = model.predict(test[feat_cols].values)
    ics = []
    for m in test['date'].unique():
        mask = test['date']==m
        if mask.sum()<10: continue
        ics.append(spearmanr(pred[mask], test.loc[mask,'fwd_ret'].values)[0])
    avg_ic = np.mean(ics)
    print(f'  Train={len(train)}, Test={len(test)}, avg IC={avg_ic:+.4f} ({len(ics)} periods)')
    results[freq] = avg_ic

conn.close()
print(f'\n=== FINAL ===')
for f, ic in results.items(): print(f'{f}: IC={ic:+.4f}')
print(f'Daily LSTM: IC=+0.141')
