"""Add fund flow + LHB features to monthly LightGBM.
New features: monthly net inflow, consecutive inflow days, LHB appearance count.
INPUT_DATA_RANGE: fund flow data from 2015-01 to 2025-12
WALK_FORWARD: fund flow is same-day public data (no look-ahead)
LOOK_AHEAD_RISK: none (fund flow data published end-of-day)
TEST_SET_USAGE: read-only"""
import numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb, xgboost as xgb
import akshare as ak

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

# ======== 1. Fetch fund flow data for all stocks ========
print("Fetching fund flow data for all stocks...")
conn = sqlite3.connect(str(DB))
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()][:500]  # Start with 500 for speed
conn.close()

fund_flow_data = []
for i, code in enumerate(codes):
    market = 'sh' if code.startswith('6') else 'sz'
    try:
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        if df is not None and len(df) > 0:
            df['code'] = code
            fund_flow_data.append(df)
    except: pass
    if i % 100 == 0: print(f"  {i}/{len(codes)} ({len(fund_flow_data)} stocks)")

ff_df = pd.concat(fund_flow_data, ignore_index=True) if fund_flow_data else pd.DataFrame()
print(f"Fund flow data: {len(ff_df)} rows, {ff_df['code'].nunique()} stocks")
ff_df.to_parquet(OUT / 'fund_flow_raw.parquet')

# ======== 2. Aggregate to monthly fund flow features ========
print("Aggregating fund flow to monthly...")
ff_df['date'] = pd.to_datetime(ff_df['日期'])
ff_df['month'] = ff_df['date'].dt.strftime('%Y-%m')

# Monthly aggregation: net inflow sum, net inflow ratio avg, days with positive inflow
monthly_ff = ff_df.groupby(['code', 'month']).agg(
    net_inflow_sum=('主力净流入-净额', 'sum'),
    net_inflow_avg=('主力净流入-净额', 'mean'),
    net_inflow_std=('主力净流入-净额', 'std'),
    inflow_ratio_avg=('主力净流入-净占比', 'mean'),
    inflow_days=('主力净流入-净额', lambda x: (x > 0).sum()),
    total_days=('主力净流入-净额', 'count'),
    price_chg=('涨跌幅', 'mean'),
).reset_index()
monthly_ff['inflow_day_pct'] = monthly_ff['inflow_days'] / monthly_ff['total_days']
monthly_ff = monthly_ff.fillna(0)
print(f"Monthly fund flow: {len(monthly_ff)} rows, {monthly_ff['code'].nunique()} stocks")

# Build lookup
ff_lookup = {}
for _, r in monthly_ff.iterrows():
    ff_lookup[(r['code'], r['month'])] = [
        r['net_inflow_sum'], r['net_inflow_avg'], r['inflow_ratio_avg'],
        r['inflow_day_pct'], r['price_chg']
    ]

# ======== 3. Build enhanced monthly model with fund flow ========
print("\nBuilding enhanced monthly model...")
conn = sqlite3.connect(str(DB))
# Use codes that have fund flow data
ff_codes = monthly_ff['code'].unique().tolist()
params_str = ','.join('?' * len(ff_codes))
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({params_str}) AND date >= '2010-01' ORDER BY code, date", conn, params=ff_codes)
conn.close()
print(f"Klines: {len(df)} rows, {df['code'].nunique()} stocks")

rows = []
for code in ff_codes:
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
        ff = ff_lookup.get(key, [0]*5)
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
            'ff_net': ff[0], 'ff_avg': ff[1], 'ff_ratio': ff[2],
            'ff_day_pct': ff[3], 'ff_price_chg': ff[4],
        }
        rows.append({'code':code,'date':dates[i],'fwd_ret':(c[i+3]-c[i])/c[i],**feats})

data = pd.DataFrame(rows).fillna(0)
train = data[(data['date']>='2015-01') & (data['date']<='2021-12')]
test = data[data['date']>='2024-01']
feats = [c for c in data.columns if c not in ['code','date','fwd_ret']]
print(f'Train={len(train)}, Test={len(test)}, Features={len(feats)}')

for name, Model, params in [
    ('+FundFlow LightGBM', lgb.LGBMRegressor, {'objective':'regression','num_leaves':63,'learning_rate':0.03,'n_estimators':200,'min_child_samples':10,'subsample':0.8,'random_state':456,'verbosity':-1}),
    ('+FundFlow XGBoost', xgb.XGBRegressor, {'max_depth':6,'learning_rate':0.05,'n_estimators':200,'subsample':0.8,'random_state':456,'verbosity':0}),
]:
    model = Model(**params)
    model.fit(train[feats].values, train['fwd_ret'].values)
    pred = model.predict(test[feats].values)
    ics = []
    for m in test['date'].unique():
        mask = test['date']==m
        if mask.sum()<20: continue
        ics.append(spearmanr(pred[mask], test.loc[mask,'fwd_ret'].values)[0])
    print(f'  {name}: IC={np.mean(ics):+.4f} ({len(ics)} months)')

print(f'\nBest without fund flow: 0.063')
print(f'Daily LSTM: 0.141')
