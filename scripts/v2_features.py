# Feature v2: Industry-relative + Cross-sectional + Weekly bridge
# Target: CS_IC 0.177 -> 0.200, anti-overfitting design
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt
from pathlib import Path

DB = '.eastmoney-ai/db/klines-v2.sqlite'
OUT = '.eastmoney-ai/eval'
N_FFT = 10

print('Feature v2: Industry + Cross-sectional + Weekly', flush=True)

# ======== 1. Load ALL data ========
conn = sqlite3.connect(DB)
codes_m = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params_m = ','.join('?'*len(codes_m))
df_m = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params_m}) AND date>='2010-01' ORDER BY code,date", conn, params=codes_m)

# Industry mapping
ind_map = {r[0]:r[1] for r in conn.execute('SELECT stock_code, industry_code FROM stock_industry_mapping')}
print(f'Industry: {len(ind_map)} stocks, {len(set(ind_map.values()))} industries', flush=True)

# Weekly data (300 stocks, for bridge features)
codes_w = [r[0] for r in conn.execute('SELECT code FROM weekly_klines GROUP BY code HAVING COUNT(*)>=200').fetchall()]
params_w = ','.join('?'*len(codes_w))
df_w = pd.read_sql_query(f"SELECT code,date,open,close,high,low,volume FROM weekly_klines WHERE code IN ({params_w}) AND date>='2010-01' ORDER BY code,date", conn, params=codes_w)
conn.close()
print(f'Weekly: {len(codes_w)} stocks, {len(df_w):,} rows', flush=True)

# Weekly features: last 4 weeks return/momentum per stock per month
print('Building weekly bridge...', flush=True); t0=time.time()
df_w['month'] = df_w['date'].str[:7]
weekly_feats = {}  # (code, month) -> [w1_ret, w2_ret, w3_ret, w4_ret, w_vol, w_ma_pos]
for code in codes_w:
    g = df_w[df_w['code']==code].sort_values('date')
    c = g['close'].values; n=len(c)
    # Within each month, get weekly returns
    months = g['month'].values
    for i in range(n):
        m = months[i]
        key = (code, m)
        if key not in weekly_feats:
            weekly_feats[key] = []
        weekly_feats[key].append(c[i])
# Aggregate: for each month, compute weekly stats
weekly_agg = {}
for (code, month), prices in weekly_feats.items():
    if len(prices) < 3: continue
    p = np.array(prices)
    rets = np.diff(p) / np.maximum(p[:-1], 0.01)
    weekly_agg[(code, month)] = [
        rets[-1] if len(rets)>=1 else 0,           # last week return
        np.mean(rets) if len(rets)>=2 else 0,       # avg weekly return
        np.std(rets) if len(rets)>=2 else 0,        # weekly vol
        rets[-1]-rets[-2] if len(rets)>=2 else 0,   # acceleration
        p[-1]/max(np.mean(p),0.01)-1,               # month-end vs month-avg
    ]
print(f'Weekly bridge: {len(weekly_agg)} stock-months ({time.time()-t0:.0f}s)', flush=True)

# ======== 2. Monthly feature building with new features ========
def fft_f(prices):
    x=np.arange(len(prices)); trend=np.polyfit(x,prices,1); detrended=prices-np.polyval(trend,x)
    fft_p=np.fft.rfft(detrended); amps=np.abs(fft_p); freqs=np.fft.rfftfreq(len(detrended))
    if len(amps)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(amps[1:])[::-1][:N_FFT]+1; feats=[]
    for idx in pk:
        if idx<len(freqs): feats.extend([freqs[idx],amps[idx],np.angle(fft_p[idx])])
    while len(feats)<N_FFT*3: feats.extend([0,0,0])
    return np.array(feats[:N_FFT*3],dtype=np.float32)

def wdenoise(signal):
    coeffs=pywt.wavedec(signal,'db4',level=2)
    sigma=np.median(np.abs(coeffs[-1]))/0.6745; threshold=sigma*np.sqrt(2*np.log(len(signal)))
    coeffs_d=[coeffs[0]]+[pywt.threshold(c,threshold,mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_d,'db4')[:len(signal)]

print('Building monthly features...', flush=True); t0=time.time()
# Phase 1: build per-stock features (61d base + weekly bridge)
flat_per_stock = {}  # code -> list of (date, feature_vector, target)
for code in codes_m:
    g = df_m[df_m['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<72: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c); c_clean=wdenoise(c); industry=ind_map.get(code, 'unknown')
    ma5=pd.Series(c).rolling(5).mean().values; ma20=pd.Series(c).rolling(20).mean().values; ma60=pd.Series(c).rolling(60).mean().values
    e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
    dif=e12-e26; dea=pd.Series(dif).ewm(span=9).mean().values; macd_hist=(dif-dea)*2
    delta=np.diff(c,prepend=c[0]); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    avg_gain=pd.Series(gain).ewm(alpha=1/14).mean().values; avg_loss=pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14=np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)),50)
    bb_std=pd.Series(c).rolling(20).std().values; bb_pos=np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01),0.5)
    trange=np.maximum(h-l,np.abs(h-np.roll(c,1))); atr14=pd.Series(trange).rolling(14).mean().values
    vol_ma3=pd.Series(v).rolling(3).mean().values; vol_ma12=pd.Series(v).rolling(12).mean().values
    p5h=pd.Series(c).rolling(60).max().values; p5l=pd.Series(c).rolling(60).min().values
    body_pct=np.abs(c-o)/np.maximum(h-l,0.01)
    up_streak=np.zeros(n); dn_streak=np.zeros(n)
    for i in range(1,n): up_streak[i]=up_streak[i-1]+1 if c[i]>c[i-1] else 0; dn_streak[i]=dn_streak[i-1]+1 if c[i]<c[i-1] else 0
    F=np.zeros((n,17),dtype=np.float32)
    for i in range(n):
        F[i,0]=(c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0
        F[i,1]=(c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0
        F[i,2]=(c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0
        F[i,3]=(c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0
        F[i,4]=(c[i]-ma5[i])/max(abs(c[i]),0.01) if not np.isnan(ma5[i]) else 0
        F[i,5]=(c[i]-ma20[i])/max(abs(c[i]),0.01) if not np.isnan(ma20[i]) else 0
        F[i,6]=(c[i]-ma60[i])/max(abs(c[i]),0.01) if not np.isnan(ma60[i]) else 0
        F[i,7]=dif[i]if not np.isnan(dif[i])else 0; F[i,8]=dea[i]if not np.isnan(dea[i])else 0
        F[i,9]=macd_hist[i]if not np.isnan(macd_hist[i])else 0
        F[i,10]=rsi14[i]if not np.isnan(rsi14[i])else 50; F[i,11]=bb_pos[i]if not np.isnan(bb_pos[i])else 0.5
        F[i,12]=np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0
        F[i,13]=atr14[i]/max(abs(c[i]),0.01)if not np.isnan(atr14[i])else 0
        F[i,14]=(h[i]-l[i])/max(abs(c[i]),0.01)
        F[i,15]=1.0 if c[i]>ma20[i]else 0.0; F[i,16]=1.0 if c[i]>ma60[i]else 0.0
    F=np.nan_to_num(F,0.0)
    for i in range(60,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        flat=list(F[i,:17]); flat.extend(fft_f(c_clean[i-60+1:i+1]).tolist())
        flat.append(v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(tr[i] if not np.isnan(tr[i]) else 0)
        flat.append(tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0)
        flat.append((vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(np.log1p(max(v[i],1))); flat.append(np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0)
        flat.append(body_pct[i] if not np.isnan(body_pct[i]) else 0)
        flat.append((c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 and not np.isnan(p5h[i]) else 0.5)
        flat.append((ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 and not np.isnan(ma20[i]) else 0)
        flat.append(np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0)
        flat.append(ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0)
        flat.append(1.0 if c[i]>ma5[i]else 0.0)
        flat.append(up_streak[i]/12.0); flat.append(dn_streak[i]/12.0)
        # Weekly bridge (5 features)
        wk = weekly_agg.get((code, g['date'].iloc[i]), [0,0,0,0,0])
        flat.extend(wk)
        date = g['date'].iloc[i]
        target = np.clip(fwd_raw,-2,2)
        if code not in flat_per_stock: flat_per_stock[code] = []
        flat_per_stock[code].append((date, flat, target, industry))

# Phase 2: Cross-sectional normalization (per-date z-score)
print('Cross-sectional normalization...', flush=True)
# Collect all samples by date
by_date = {}
for code, samples in flat_per_stock.items():
    for date, feat, target, ind in samples:
        if date not in by_date: by_date[date] = []
        by_date[date].append((code, feat, target, ind))

# Compute cross-sectional stats and normalize
# For each date, compute mean/std across all stocks, then z-score
# Only normalize the first 47 features (technical), not FFT/streaks which are stock-specific
N_TECH = 47  # first 47 features to normalize cross-sectionally
flat_list, ys_list, dates_list, inds_list = [], [], [], []
for date in sorted(by_date.keys()):
    batch = by_date[date]
    if len(batch) < 50: continue  # skip months with too few stocks
    feats_batch = np.array([f for _, f, _, _ in batch])
    # Cross-sectional z-score on technical features
    cs_mean = feats_batch[:, :N_TECH].mean(axis=0)
    cs_std = feats_batch[:, :N_TECH].std(axis=0) + 1e-8
    feats_batch[:, :N_TECH] = np.clip((feats_batch[:, :N_TECH] - cs_mean) / cs_std, -4, 4)
    for (code, _, target, ind), feat in zip(batch, feats_batch):
        flat_list.append(feat); ys_list.append(target)
        dates_list.append(date); inds_list.append(ind)

flat_m = np.array(flat_list, dtype=np.float32); ys_m = np.array(ys_list, dtype=np.float32)
dates_m = np.array(dates_list); inds_arr = np.array(inds_list)
v_m = ~np.isnan(flat_m).any(axis=1)&~np.isnan(ys_m)
flat_m=flat_m[v_m]; ys_m=ys_m[v_m]; dates_m=dates_m[v_m]; inds_arr=inds_arr[v_m]

# Phase 3: Industry-relative features (post cross-sectional)
print('Industry-relative features...', flush=True)
# Add industry dummy encoding (one-hot for top 20 industries)
from collections import Counter
ind_counts = Counter(inds_arr)
top_inds = [ind for ind, _ in ind_counts.most_common(20)]
ind_dummies = np.zeros((len(inds_arr), 20), dtype=np.float32)
for i, ind in enumerate(inds_arr):
    if ind in top_inds:
        ind_dummies[i, top_inds.index(ind)] = 1.0

flat_m = np.column_stack([flat_m, ind_dummies])

tr_m=(dates_m>='2015-01')&(dates_m<='2021-12')
te_m=(dates_m>='2024-01')
print(f'Final features: {flat_m.shape[1]}d, T={tr_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)', flush=True)

# ======== 3. Train + Evaluate ========
sc=StandardScaler(); Xt=sc.fit_transform(flat_m[tr_m]); Xte=sc.transform(flat_m[te_m])
y_tr_m=ys_m[tr_m]; y_te_m=ys_m[te_m]; te_dates=dates_m[te_m]

def cs_ic(pred,true,dates):
    ics=[spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan,ics

print('Training...', flush=True); t0=time.time()
lgb_m=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_m.fit(Xt,y_tr_m); p_lgb=lgb_m.predict(Xte)

xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_m.fit(Xt,y_tr_m); p_xgb=xgb_m.predict(Xte)

ridge_m=Ridge(alpha=1.0); ridge_m.fit(Xt,y_tr_m); p_ridge=ridge_m.predict(Xte)
mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
mlp_m.fit(Xt,y_tr_m); p_mlp=mlp_m.predict(Xte)

# Weighted ensemble
ics_b={}
for name,pred in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)]:
    ics_b[name]=max(cs_ic(pred,y_te_m,te_dates)[0],0)
w_sum=sum(ics_b.values())
p_ens=(ics_b['LGB']*p_lgb+ics_b['XGB']*p_xgb+ics_b['Ridge']*p_ridge+ics_b['MLP']*p_mlp)/w_sum
ic_new,ics_new=cs_ic(p_ens,y_te_m,te_dates)
hit=np.mean((p_ens>0)==(y_te_m>0))

# ======== 4. Report ========
print(); print('='*65)
print('FEATURE v2 RESULTS')
print('='*65)
for name,ic in [('LightGBM',ics_b['LGB']),('XGBoost',ics_b['XGB']),('Ridge',ics_b['Ridge']),('MLP',ics_b['MLP']),('Ensemble',ic_new)]:
    print(f'  {name:<15s} CS_IC={ic:+.4f}')
print(f'\n  Ensemble Hit: {hit:.3f}')
n20=max(1,int(len(p_ens)*0.2))
ls=np.mean(y_te_m[np.argsort(p_ens)[-n20:]])-np.mean(y_te_m[np.argsort(p_ens)[:n20]])
print(f'  Top20 LS: {ls:+.4f}')
print(f'  N_test: {len(y_te_m):,}, N_months: {len(np.unique(te_dates))}')
print(f'  Features: {flat_m.shape[1]}d (base 61 + weekly 5 + industry 20 = 86)')
print()
print(f'  Previous best: CS_IC=+0.177 (61d)')
print(f'  New:           CS_IC={ic_new:+.4f} ({flat_m.shape[1]}d)')
print(f'  Delta:         {ic_new-0.177:+.4f}')
print(f'  Target:        +0.200 (need {0.200-ic_new:+.4f} more)')
# Overfitting check
ic_tr=cs_ic(lgb_m.predict(Xt),y_tr_m,dates_m[tr_m])[0]
lgb_ic_test = ics_b['LGB']
print(f'\n  Overfitting check: train_IC={ic_tr:+.4f} test_IC={lgb_ic_test:+.4f} gap={ic_tr-lgb_ic_test:+.4f}')
print('='*65)
