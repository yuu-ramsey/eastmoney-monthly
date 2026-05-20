"""Alpha158-inspired: inline feature building per stock, no intermediate dicts.
50+ factors for 2000 stocks, LightGBM, strict v6 date split."""
import numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

conn = sqlite3.connect(str(DB))
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=60').fetchall()][:1500]
params_str = ','.join('?' * len(codes))
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({params_str}) AND date >= '2010-01' ORDER BY code, date", conn, params=codes)
conn.close()
print(f'Stocks: {len(codes)}, Rows: {len(df)}')

rows = []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    n = len(g); c = g['close'].values.astype(float); o = g['open'].values.astype(float)
    h = g['high'].values.astype(float); l = g['low'].values.astype(float); v = g['volume'].values.astype(float)
    dates = g['date'].tolist(); ch = hash(code) % 31 / 31.0
    daily_ret = np.diff(np.log(np.maximum(c, 0.01)), prepend=0)

    # Pre-compute all factors as arrays
    # Momentum
    mom1 = c / np.maximum(pd.Series(c).shift(1).fillna(c[0]).values, 0.01) - 1
    mom3 = c / np.maximum(pd.Series(c).shift(3).fillna(c[0]).values, 0.01) - 1
    mom6 = c / np.maximum(pd.Series(c).shift(6).fillna(c[0]).values, 0.01) - 1
    mom12 = c / np.maximum(pd.Series(c).shift(12).fillna(c[0]).values, 0.01) - 1
    # Norm momentum
    for arr, name in [(mom1,'m1z'),(mom3,'m3z'),(mom6,'m6z'),(mom12,'m12z')]:
        rm = pd.Series(arr).rolling(60,min_periods=12).mean().values
        rs = pd.Series(arr).rolling(60,min_periods=12).std().values
        locals()[name] = np.nan_to_num((arr - rm) / np.maximum(rs, 0.01), 0)

    # Vol
    vol3 = pd.Series(daily_ret).rolling(3,min_periods=3).std().fillna(0).values
    vol6 = pd.Series(daily_ret).rolling(6,min_periods=3).std().fillna(0).values
    vol12 = pd.Series(daily_ret).rolling(12,min_periods=3).std().fillna(0).values
    vol24 = pd.Series(daily_ret).rolling(24,min_periods=6).std().fillna(0).values

    # MA dev
    for p, name in [(5,'ma5'),(10,'ma10'),(20,'ma20'),(30,'ma30'),(60,'ma60')]:
        ma = pd.Series(c).rolling(p,min_periods=3).mean().values
        locals()[name] = (c - ma) / np.maximum(c, 0.01)

    # Volume/Turnover
    vma3 = v / np.maximum(pd.Series(v).rolling(3,min_periods=1).mean().values, 1) - 1
    vma6 = v / np.maximum(pd.Series(v).rolling(6,min_periods=1).mean().values, 1) - 1
    vma12 = v / np.maximum(pd.Series(v).rolling(12,min_periods=1).mean().values, 1) - 1

    # Price patterns
    hilo = (h - l) / np.maximum(c, 0.01)
    body_ratio = np.abs(c - o) / np.maximum(h - l, 0.01)

    # Correlation
    corr12 = pd.Series(c).rolling(12,min_periods=6).apply(lambda x: np.corrcoef(x, np.arange(len(x)))[0,1] if len(x)>1 else 0, raw=False).fillna(0).values
    corr24 = pd.Series(c).rolling(24,min_periods=6).apply(lambda x: np.corrcoef(x, np.arange(len(x)))[0,1] if len(x)>1 else 0, raw=False).fillna(0).values

    # RSI
    delta = np.diff(c, prepend=c[0]); gain = np.where(delta>0,delta,0); loss = np.where(delta<0,-delta,0)
    rsi6 = np.nan_to_num(100-100/(1+pd.Series(gain).ewm(span=6).mean().values/np.maximum(pd.Series(loss).ewm(span=6).mean().values,1e-8)), 50)
    rsi14 = np.nan_to_num(100-100/(1+pd.Series(gain).ewm(span=14).mean().values/np.maximum(pd.Series(loss).ewm(span=14).mean().values,1e-8)), 50)
    rsi24 = np.nan_to_num(100-100/(1+pd.Series(gain).ewm(span=24).mean().values/np.maximum(pd.Series(loss).ewm(span=24).mean().values,1e-8)), 50)

    # MACD
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    macd_dif = e12 - e26; macd_dea = pd.Series(macd_dif).ewm(span=9).mean().values
    macd_hist = (macd_dif - macd_dea) * 2

    # Build rows
    for i in range(60, n-6):
        if c[i] <= 0.01: continue
        fwd = (c[i+3] - c[i]) / c[i]
        if abs(fwd) > 2: continue
        rows.append({'code':code,'date':dates[i],'fwd_ret':fwd,
            'mom1':mom1[i],'mom3':mom3[i],'mom6':mom6[i],'mom12':mom12[i],
            'm1z':m1z[i],'m3z':m3z[i],'m6z':m6z[i],'m12z':m12z[i],
            'vol3':vol3[i],'vol6':vol6[i],'vol12':vol12[i],'vol24':vol24[i],
            'ma5':ma5[i],'ma10':ma10[i],'ma20':ma20[i],'ma30':ma30[i],'ma60':ma60[i],
            'vma3':vma3[i],'vma6':vma6[i],'vma12':vma12[i],
            'hilo':hilo[i],'body':body_ratio[i],
            'corr12':corr12[i],'corr24':corr24[i],
            'rsi6':rsi6[i],'rsi14':rsi14[i],'rsi24':rsi24[i],
            'dif':macd_dif[i],'dea':macd_dea[i],'hist':macd_hist[i],
            'hash':ch})

data = pd.DataFrame(rows).fillna(0)
train = data[(data['date']>='2015-01')&(data['date']<='2021-12')]
test = data[data['date']>='2024-01']
feats = [c for c in data.columns if c not in ['code','date','fwd_ret']]
print(f'Train={len(train)}, Test={len(test)}, Features={len(feats)}')

model = lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=200,min_child_samples=10,subsample=0.8,random_state=456,verbosity=-1)
model.fit(train[feats].values, train['fwd_ret'].values)
pred = model.predict(test[feats].values)
ics = [spearmanr(pred[test['date']==d], test.loc[test['date']==d,'fwd_ret'].values)[0] for d in test['date'].unique() if (test['date']==d).sum()>=20]
ic = np.mean(ics)
print(f'Alpha158-LightGBM IC={ic:+.4f} ({len(ics)} months)')
print(f'Our baseline (22 features): 0.063 | Daily LSTM: 0.141')
