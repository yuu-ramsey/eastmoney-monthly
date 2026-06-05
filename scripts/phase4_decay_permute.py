# Phase 4: Signal Decay + Permutation Test (on Industry-Neutralized factors)
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
import lightgbm as lgb, xgboost as xgb, pywt

DB = '.eastmoney-ai/db/klines-v2.sqlite'; N_FFT = 10; N_PERM = 1000

def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates==m], true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    ics = np.array(ics); return np.mean(ics), np.mean(ics)/np.std(ics) if np.std(ics)>0 else 0, ics

def fft_f(p):
    x=np.arange(len(p));t=np.polyfit(x,p,1);d=p-np.polyval(t,x)
    fp=np.fft.rfft(d);a=np.abs(fp);fq=np.fft.rfftfreq(len(d))
    if len(a)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(a[1:])[::-1][:N_FFT]+1;fs=[]
    for i in pk:
        if i<len(fq): fs.extend([fq[i],a[i],np.angle(fp[i])])
    while len(fs)<N_FFT*3: fs.extend([0,0,0])
    return np.array(fs[:N_FFT*3],dtype=np.float32)

def wd(s):
    c=pywt.wavedec(s,'db4',level=2);sigma=np.median(np.abs(c[-1]))/0.6745;th=sigma*np.sqrt(2*np.log(len(s)))
    cd=[c[0]]+[pywt.threshold(cf,th,mode='soft') for cf in c[1:]]; return pywt.waverec(cd,'db4')[:len(s)]

def cross_sectional_neutralize(features, dates, neutralizer, ntype='categorical'):
    neutralized = features.copy()
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 50: continue
        if ntype == 'categorical':
            groups = neutralizer[mask]
            for g in np.unique(groups):
                gm = mask & (neutralizer == g)
                if gm.sum() >= 3: neutralized[gm] -= features[gm].mean(axis=0)
    return neutralized

# ======== Build data with T+1, T+2, T+3 targets ========
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
ind_map = {r[0]:r[1] for r in conn.execute('SELECT stock_code, industry_code FROM stock_industry_mapping')}
params = ','.join('?'*len(codes))
df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2005-01' ORDER BY code,date", conn, params=codes)
conn.close()
df['month'] = df['date'].str[:7]

print(f'Building with T+1,T+2,T+3 targets...', flush=True); t0=time.time()
flat_list, y1_list, y2_list, y3_list, dates_list, inds_list = [], [], [], [], [], []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c=g['close'].values.astype(float);o=g['open'].values.astype(float);h=g['high'].values.astype(float)
    l=g['low'].values.astype(float);v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c);cc=wd(c);industry=ind_map.get(code,'unknown')
    ma5=pd.Series(c).rolling(5).mean().values;ma20=pd.Series(c).rolling(20).mean().values;ma60=pd.Series(c).rolling(60).mean().values
    e12=pd.Series(c).ewm(span=12).mean().values;e26=pd.Series(c).ewm(span=26).mean().values
    dif=e12-e26;dea=pd.Series(dif).ewm(span=9).mean().values;macd_hist=(dif-dea)*2
    delta=np.diff(c,prepend=c[0]);gain=np.where(delta>0,delta,0);loss=np.where(delta<0,-delta,0)
    avg_gain=pd.Series(gain).ewm(alpha=1/14).mean().values;avg_loss=pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14=np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)),50)
    bb_std=pd.Series(c).rolling(20).std().values;bb_pos=np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01),0.5)
    trange=np.maximum(h-l,np.abs(h-np.roll(c,1)));atr14=pd.Series(trange).rolling(14).mean().values
    vol_ma3=pd.Series(v).rolling(3).mean().values;vol_ma12=pd.Series(v).rolling(12).mean().values
    p5h=pd.Series(c).rolling(60).max().values;p5l=pd.Series(c).rolling(60).min().values
    body_pct=np.abs(c-o)/np.maximum(h-l,0.01)
    up_streak=np.zeros(n);dn_streak=np.zeros(n)
    for i in range(1,n): up_streak[i]=up_streak[i-1]+1 if c[i]>c[i-1] else 0;dn_streak[i]=dn_streak[i-1]+1 if c[i]<c[i-1] else 0
    for i in range(60,n-6):
        if c[i]<=0.01:continue
        fwd1=np.clip((c[i+1]-c[i])/c[i],-2,2)
        fwd2=np.clip((c[i+2]-c[i])/c[i],-2,2) if i+2<n else 0
        fwd3=np.clip((c[i+3]-c[i])/c[i],-2,2) if i+3<n else 0
        if abs((c[i+3]-c[i])/c[i])>2:continue
        flat=[(c[i]-c[i-j])/max(abs(c[i-j]),0.01) if i>=j else 0 for j in [1,3,6,12]]
        for ma in [ma5,ma20,ma60]: flat.append((c[i]-ma[i])/max(abs(c[i]),0.01) if not np.isnan(ma[i]) else 0)
        flat.extend([dif[i]if not np.isnan(dif[i])else 0,dea[i]if not np.isnan(dea[i])else 0,macd_hist[i]if not np.isnan(macd_hist[i])else 0])
        flat.append(rsi14[i]if not np.isnan(rsi14[i])else 50);flat.append(bb_pos[i]if not np.isnan(bb_pos[i])else 0.5)
        flat.append(np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0)
        flat.append(atr14[i]/max(abs(c[i]),0.01)if not np.isnan(atr14[i])else 0);flat.append((h[i]-l[i])/max(abs(c[i]),0.01))
        flat.append(1.0 if c[i]>ma20[i]else 0.0);flat.append(1.0 if c[i]>ma60[i]else 0.0)
        flat.extend(fft_f(cc[i-60+1:i+1]).tolist())
        flat.extend([v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0,tr[i] if not np.isnan(tr[i]) else 0,
            tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
            (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0,np.log1p(max(v[i],1)),np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0])
        flat.extend([body_pct[i] if not np.isnan(body_pct[i]) else 0,(c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
            (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0,np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0,
            ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0,1.0 if c[i]>ma5[i]else 0.0,up_streak[i]/12.0,dn_streak[i]/12.0])
        flat_list.append(flat);y1_list.append(fwd1);y2_list.append(fwd2);y3_list.append(fwd3)
        dates_list.append(g['month'].iloc[i]);inds_list.append(industry)

flat=np.array(flat_list,dtype=np.float32)
y1=np.array(y1_list,dtype=np.float32);y2=np.array(y2_list,dtype=np.float32);y3=np.array(y3_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(y1)&~np.isnan(y2)&~np.isnan(y3)
flat=flat[v];y1=y1[v];y2=y2[v];y3=y3[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
inds_arr=np.array([inds_list[i] for i in range(len(v)) if v[i]])
print(f'{len(flat):,} samples ({time.time()-t0:.0f}s)', flush=True)

# Industry neutralize
print('Industry neutralizing...', flush=True); t0=time.time()
flat_ind = cross_sectional_neutralize(flat.copy(), dates_arr, inds_arr, 'categorical')
print(f'Done ({time.time()-t0:.0f}s)', flush=True)

# Split
tr_m=(dates_arr>='2010-01')&(dates_arr<='2014-12');te_m=(dates_arr>='2015-01')
sc=StandardScaler();Xt=sc.fit_transform(flat_ind[tr_m]);Xte=sc.transform(flat_ind[te_m])
te_dates=dates_arr[te_m]

# Train model (on T+3 target)
lgb_m=lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_m.fit(Xt,y3[tr_m]);p_lgb=lgb_m.predict(Xte)
xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_m.fit(Xt,y3[tr_m]);p_xgb=xgb_m.predict(Xte)
ridge_m=Ridge(alpha=1.0);ridge_m.fit(Xt,y3[tr_m]);p_ridge=ridge_m.predict(Xte)

ics_m={}
for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge)]:
    ic_tmp=np.mean([spearmanr(p[te_dates==m],y3[te_m][te_dates==m])[0] for m in np.unique(te_dates) if (te_dates==m).sum()>=20])
    ics_m[n]=max(ic_tmp,0)
w_sum=sum(ics_m.values())
p_ens=sum(ics_m[n]*p for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge)])/w_sum

# ======== 4.1 Signal Decay ========
print(); print('='*65)
print('4.1 SIGNAL DECAY: Industry-Neutral Factors')
print('='*65)
print(f'  {"Horizon":<12s} {"IC":>8s} {"ICIR":>8s} {"IC>0":>8s} {"vs T+3":>10s}')
base_ic, base_ir, _ = cs_ic(p_ens, y3[te_m], te_dates)
for horizon, y_target, label in [(1, y1[te_m], 'T+1 (1m)'), (2, y2[te_m], 'T+2 (2m)'), (3, y3[te_m], 'T+3 (3m)')]:
    ic, icir, ics_arr = cs_ic(p_ens, y_target, te_dates)
    pct = ic/base_ic*100 if base_ic != 0 else 0
    print(f'  {label:<12s} {ic:+8.4f} {icir:+8.3f} {np.mean(ics_arr>0):7.0%} {pct:9.0f}%')

# Decay slope
ic1,_,_ = cs_ic(p_ens, y1[te_m], te_dates)
ic2,_,_ = cs_ic(p_ens, y2[te_m], te_dates)
ic3,_,_ = cs_ic(p_ens, y3[te_m], te_dates)
print(f'\n  Decay: T+1={ic1:+.4f} → T+2={ic2:+.4f} → T+3={ic3:+.4f}')
print(f'  T+2 / T+1 = {ic2/max(ic1,0.001)*100:.0f}%')
if ic2/max(ic1,0.001) >= 0.7:
    print(f'  VERDICT: Monthly rebalancing is REASONABLE (T+2 retains >=70% of T+1)')
else:
    print(f'  VERDICT: Signal decays fast. Consider higher-frequency rebalancing.')

# ======== 4.2 Permutation Test ========
print(f'\n{"="*65}')
print(f'4.2 PERMUTATION TEST ({N_PERM} iterations)')
print(f'{"="*65}')
print(f'  Real IC (T+3, Industry Neut): {base_ic:+.4f}')

# Permute: shuffle y within each month (break X→y, preserve cross-sectional distribution)
perm_ics = []
better_count = 0
t0_perm = time.time()
y_te_target = y3[te_m]
unique_dates = np.unique(te_dates)

for i in range(N_PERM):
    # Shuffle targets within each month independently
    y_permuted = y_te_target.copy()
    for m in unique_dates:
        mask = te_dates == m
        if mask.sum() >= 20:
            y_permuted[mask] = np.random.permutation(y_permuted[mask])

    # Compute IC: same predictions, shuffled targets
    ic_p, _, _ = cs_ic(p_ens, y_permuted, te_dates)
    perm_ics.append(ic_p)
    if ic_p >= base_ic: better_count += 1

    if (i+1) % 200 == 0:
        elapsed = time.time() - t0_perm
        eta = elapsed/(i+1)*(N_PERM-i-1)
        print(f'  [{i+1}/{N_PERM}] perm_IC={ic_p:+.4f} better={better_count}/{i+1} '
              f'p={better_count/(i+1):.4f} ({elapsed:.0f}s ETA {eta:.0f}s)', flush=True)

perm_ics = np.array(perm_ics)
p_value = better_count / N_PERM
perm_mean = np.mean(perm_ics)
perm_std = np.std(perm_ics)
perm_p95 = np.percentile(perm_ics, 95)
perm_p99 = np.percentile(perm_ics, 99)

print(f'\n  Permutation Results:')
print(f'  Real IC:        {base_ic:+.4f}')
print(f'  Perm IC mean:   {perm_mean:+.4f} ± {perm_std:.4f}')
print(f'  Perm IC 95%:    {perm_p95:+.4f}')
print(f'  Perm IC 99%:    {perm_p99:+.4f}')
print(f'  P-value:        {p_value:.4f} ({better_count}/{N_PERM})')
print(f'  Significance:   {"*** p<0.01" if p_value<0.01 else "** p<0.05" if p_value<0.05 else "* p<0.10" if p_value<0.10 else "NOT SIGNIFICANT"}')
print('='*65)
