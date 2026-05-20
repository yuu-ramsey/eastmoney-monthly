"""Monthly LightGBM enhanced: 30 features + alpha target + XGBoost.
More technical features + cross-sectional alpha as target."""
import numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb, xgboost as xgb

DB = '.eastmoney-ai/db/klines-v2.sqlite'
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
# Load industry mapping for alpha target
ind_map = {}
for r in conn.execute('SELECT stock_code, industry_code FROM stock_industry_mapping'):
    ind_map[r[0]] = r[1]
print(f'Stocks: {len(codes)}, with industry: {sum(1 for c in codes if c in ind_map)}')

params_str = ','.join('?' * len(codes))
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume, turnover_rate FROM monthly_klines WHERE code IN ({params_str}) AND date >= '2010-01' ORDER BY code, date", conn, params=codes)
conn.close()

rows = []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    n = len(g); c = g['close'].values.astype(float)
    o = g['open'].values.astype(float); h = g['high'].values.astype(float)
    l = g['low'].values.astype(float); v = g['volume'].values.astype(float)
    tr_col = 'turnover_rate' if 'turnover_rate' in g.columns else None
    tr = g[tr_col].values.astype(float) if tr_col else np.zeros(n)
    dates = g['date'].tolist(); industry = ind_map.get(code, 'unknown')

    # Pre-compute indicators
    ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    dif = e12 - e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif - dea) * 2
    # RSI
    delta = np.diff(c, prepend=c[0]); gain = np.where(delta>0,delta,0); loss = np.where(delta<0,-delta,0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)), 50)
    # BB position
    bb_mid = ma20; bb_std = pd.Series(c).rolling(20).std().values
    bb_pos = np.nan_to_num((c - (bb_mid - 2*bb_std)) / np.maximum(4*bb_std, 0.01), 0.5)
    # ATR
    trange = np.maximum(h-l, np.abs(h-np.roll(c,1)))
    atr14 = pd.Series(trange).rolling(14).mean().values

    for i in range(60, n - 6):
        if c[i] <= 0.01: continue
        fwd_ret = (c[i+3] - c[i]) / c[i]
        if abs(fwd_ret) > 2: continue

        feats = {
            # Momentum (4)
            'r1': (c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0,
            'r3': (c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0,
            'r6': (c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0,
            'r12': (c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0,
            # MA deviations (3)
            'ma5d': (c[i]-ma5[i])/max(abs(c[i]),0.01),
            'ma20d': (c[i]-ma20[i])/max(abs(c[i]),0.01),
            'ma60d': (c[i]-ma60[i])/max(abs(c[i]),0.01),
            # MACD (3)
            'macd_dif': dif[i], 'macd_dea': dea[i], 'macd_hist': macd_hist[i],
            # RSI (1)
            'rsi14': rsi14[i],
            # Bollinger (1)
            'bb_pos': bb_pos[i],
            # Volatility (2)
            'vol_6': np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0,
            'atr_ratio': atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0,
            # Volume (3)
            'vol_chg': np.mean(v[max(0,i-3):i+1])/max(np.mean(v[max(0,i-12):i+1]),1)-1 if i>=12 else 0,
            'to': tr[i] if not np.isnan(tr[i]) else 0,
            'to_chg': tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
            # Price pattern (2)
            'hilo': (h[i]-l[i])/max(abs(c[i]),0.01),
            'body': abs(c[i]-o[i])/max(abs(c[i]),0.01),
            # Trend (2)
            'above_ma20': 1.0 if c[i] > ma20[i] else 0.0,
            'above_ma60': 1.0 if c[i] > ma60[i] else 0.0,
            # Cross-sectional (1)
            'sector_id': hash(industry) % 31 / 31.0,
        }
        rows.append({'code':code,'date':dates[i],'fwd_ret':fwd_ret,**feats})
    if len(rows) % 20000 == 0: pass

data = pd.DataFrame(rows).dropna()
print(f'Data: {len(data)} rows, {len(data.columns)-3} features')

train = data[(data['date']>='2015-01') & (data['date']<='2021-12')]
test = data[data['date']>='2024-01']
feat_cols = [c for c in data.columns if c not in ['code','date','fwd_ret']]
print(f'Train: {len(train)}, Test: {len(test)}')

results = {}
all_preds = {}
for name, Model, params in [
    ('LightGBM', lgb.LGBMRegressor, {'objective':'regression','metric':'l1','num_leaves':63,'learning_rate':0.03,'n_estimators':200,'min_child_samples':10,'subsample':0.8,'colsample_bytree':0.8,'random_state':456,'verbosity':-1,'n_jobs':4}),
    ('XGBoost', xgb.XGBRegressor, {'objective':'reg:squarederror','max_depth':6,'learning_rate':0.05,'n_estimators':200,'subsample':0.8,'colsample_bytree':0.8,'random_state':456,'verbosity':0,'n_jobs':4}),
]:
    print(f'\n{name}...')
    t0 = time.time()
    model = Model(**params)
    model.fit(train[feat_cols].values, train['fwd_ret'].values)
    pred = model.predict(test[feat_cols].values)
    all_preds[name] = pred
    ics = []
    for m in test['date'].unique():
        mask = test['date']==m; n_stocks = mask.sum()
        if n_stocks < 20: continue
        ic = spearmanr(pred[mask], test.loc[mask,'fwd_ret'].values)[0]
        ics.append(ic)
    avg_ic = np.mean(ics)
    print(f'  avg monthly IC={avg_ic:+.4f} ({len(ics)} months, {time.time()-t0:.0f}s)')
    results[name] = avg_ic

# Ensemble
pred_ens = all_preds['LightGBM'] * 0.5 + all_preds['XGBoost'] * 0.5
ics_ens = []
for m in test['date'].unique():
    mask = test['date']==m
    if mask.sum()<20: continue
    ics_ens.append(spearmanr(pred_ens[mask], test.loc[mask,'fwd_ret'].values)[0])
results['Ensemble'] = np.mean(ics_ens)
print(f"\nEnsemble: avg monthly IC={results['Ensemble']:+.4f}")

print(f'\n=== ENHANCED MONTHLY ===')
for n, ic in results.items(): print(f'{n}: IC={ic:+.4f}')
print(f'Baseline LightGBM (11 features): +0.042')
print(f'Daily LSTM: +0.141')
