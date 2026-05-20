"""Monthly LightGBM with daily LSTM signal as bridge feature.
Hypothesis: daily LSTM (IC=0.141) aggregated to monthly → strongest monthly feature.
Target: maximize monthly IC beyond current 0.063 ceiling."""
import numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb, xgboost as xgb

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

# Load daily LSTM signals and aggregate to monthly features
print("Loading daily LSTM signals...")
daily = pd.read_parquet(OUT / 'daily_signals.parquet')
daily['month'] = daily['date'].str[:7]

# Monthly distribution features from daily signals
print("Computing monthly distribution features from daily LSTM...")
def dist_features(g):
    s = g['score'].values
    if len(s) < 5: return None
    return {
        'lstm_mean': np.mean(s), 'lstm_std': np.std(s),
        'lstm_p10': np.percentile(s, 10), 'lstm_p50': np.percentile(s, 50),
        'lstm_p90': np.percentile(s, 90), 'lstm_skew': pd.Series(s).skew() if len(s)>2 else 0,
        'lstm_last': s[-1], 'lstm_trend': np.polyfit(range(len(s)), s, 1)[0] if len(s)>=5 else 0,
    }

monthly_lstm = daily.groupby(['code','month']).apply(dist_features).dropna().reset_index()
lstm_feats = pd.DataFrame(monthly_lstm[0].tolist())
monthly_lstm = pd.concat([monthly_lstm[['code','month']], lstm_feats], axis=1)
print(f"Monthly LSTM features: {len(monthly_lstm)} rows, {len(lstm_feats.columns)} features")

# Build lookup
lstm_lookup = {}
for _, r in monthly_lstm.iterrows():
    key = (r['code'], r['month'])
    lstm_lookup[key] = [r[c] for c in lstm_feats.columns]

# Load monthly klines and build enhanced features
conn = sqlite3.connect(str(DB))
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
print(f'Monthly stocks: {len(codes)}')

params_str = ','.join('?' * len(codes))
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({params_str}) AND date >= '2010-01' ORDER BY code, date", conn, params=codes)
conn.close()

rows = []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    n = len(g); c = g['close'].values.astype(float)
    o = g['open'].values.astype(float); h = g['high'].values.astype(float)
    l = g['low'].values.astype(float); v = g['volume'].values.astype(float)
    dates = g['date'].tolist()
    ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    dif = e12-e26; dea = pd.Series(dif).ewm(span=9).mean().values
    delta = np.diff(c,prepend=c[0]); gain = np.where(delta>0,delta,0); loss = np.where(delta<0,-delta,0)
    avg_g = pd.Series(gain).ewm(alpha=1/14).mean().values; avg_l = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100-100/(1+avg_g/np.maximum(avg_l,1e-8)), 50)
    for i in range(60, n-6):
        if c[i]<=0.01 or abs((c[i+3]-c[i])/c[i])>2: continue
        key = (code, dates[i])
        lstm_f = lstm_lookup.get(key, [0]*8)
        feats = {
            'r1':(c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0,
            'r3':(c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0,
            'r6':(c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0,
            'r12':(c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0,
            'ma20d':(c[i]-ma20[i])/max(abs(c[i]),0.01),
            'ma60d':(c[i]-ma60[i])/max(abs(c[i]),0.01) if not np.isnan(ma60[i]) else 0,
            'macd_dif':dif[i], 'macd_hist':(dif[i]-dea[i])*2,
            'rsi14':rsi14[i], 'hilo':(h[i]-l[i])/max(abs(c[i]),0.01),
            'above_ma20':1.0 if c[i]>ma20[i] else 0.0,
            'lstm_m': lstm_f[0], 'lstm_s': lstm_f[1], 'lstm_p10': lstm_f[2],
            'lstm_p50': lstm_f[3], 'lstm_p90': lstm_f[4], 'lstm_sk': lstm_f[5],
            'lstm_last': lstm_f[6], 'lstm_tr': lstm_f[7],
        }
        rows.append({'code':code,'date':dates[i],'fwd_ret':(c[i+3]-c[i])/c[i],**feats})

data = pd.DataFrame(rows).fillna(0)
train = data[(data['date']>='2015-01') & (data['date']<='2021-12')]
test = data[data['date']>='2024-01']
feats = [c for c in data.columns if c not in ['code','date','fwd_ret']]
print(f'Train={len(train)}, Test={len(test)}, Features={len(feats)}')

results = {}
for name, Model, params in [
    ('LightGBM+Daily', lgb.LGBMRegressor, {'objective':'regression','num_leaves':63,'learning_rate':0.03,'n_estimators':200,'min_child_samples':10,'subsample':0.8,'random_state':456,'verbosity':-1}),
    ('XGBoost+Daily', xgb.XGBRegressor, {'max_depth':6,'learning_rate':0.05,'n_estimators':200,'subsample':0.8,'random_state':456,'verbosity':0}),
]:
    model = Model(**params)
    model.fit(train[feats].values, train['fwd_ret'].values)
    pred = model.predict(test[feats].values)
    ics = []
    for m in test['date'].unique():
        mask = test['date']==m
        if mask.sum()<20: continue
        ics.append(spearmanr(pred[mask], test.loc[mask,'fwd_ret'].values)[0])
    results[name] = np.mean(ics)
    print(f'{name}: IC={results[name]:+.4f} ({len(ics)} months)')

print(f'\n=== WITH DAILY BRIDGE ===')
for n, ic in results.items(): print(f'{n}: IC={ic:+.4f}')
print(f'Best monthly without daily: 0.063')
print(f'Daily LSTM standalone: 0.141')
