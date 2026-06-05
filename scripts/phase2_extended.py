# Phase 2 Extended: Max backtest window, multi-regime IC audit
# Train: 2010-2014, Test: 2015-01 to latest (~130 months)
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt

DB = '.eastmoney-ai/db/klines-v2.sqlite'; N_FFT = 10

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

conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params = ','.join('?'*len(codes))
df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2005-01' ORDER BY code,date", conn, params=codes)
conn.close()
df['month'] = df['date'].str[:7]

print(f'Building features ({len(codes)} stocks, data from {df["month"].min()})...', flush=True); t0=time.time()
flat_list, ys_list, dates_list = [], [], []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c=g['close'].values.astype(float);o=g['open'].values.astype(float);h=g['high'].values.astype(float)
    l=g['low'].values.astype(float);v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c);cc=wd(c)
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
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
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
        flat_list.append(flat);ys_list.append(np.clip(fwd_raw,-2,2));dates_list.append(g['month'].iloc[i])

flat=np.array(flat_list,dtype=np.float32);ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(ys);flat=flat[v];ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
print(f'{len(flat):,} samples, {flat.shape[1]}d ({time.time()-t0:.0f}s)', flush=True)

# ======== Extended split ========
tr_m = (dates_arr >= '2010-01') & (dates_arr <= '2014-12')
te_m = (dates_arr >= '2015-01')
print(f'Train: {tr_m.sum():,} (2010-2014), Test: {te_m.sum():,} (2015+)', flush=True)

sc=StandardScaler();Xt=sc.fit_transform(flat[tr_m]);Xte=sc.transform(flat[te_m])
y_tr=ys[tr_m];y_te=ys[te_m];te_dates=dates_arr[te_m]
print(f'Test months: {len(np.unique(te_dates))}', flush=True)

lgb_m=lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_m.fit(Xt,y_tr);p_lgb=lgb_m.predict(Xte)
xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_m.fit(Xt,y_tr);p_xgb=xgb_m.predict(Xte)
ridge_m=Ridge(alpha=1.0);ridge_m.fit(Xt,y_tr);p_ridge=ridge_m.predict(Xte)
mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
mlp_m.fit(Xt,y_tr);p_mlp=mlp_m.predict(Xte)

ics_m={}
for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)]:
    ic_tmp = np.mean([spearmanr(p[te_dates==m],y_te[te_dates==m])[0] for m in np.unique(te_dates) if (te_dates==m).sum()>=20])
    ics_m[n]=max(ic_tmp,0)
w_sum=sum(ics_m.values())
p_ens=sum(ics_m[n]*p for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)])/w_sum

# ======== Compute per-month IC ========
test_months_arr = np.array(sorted(set(te_dates)))
monthly_ics = {}; monthly_n = {}
for m in test_months_arr:
    mask = te_dates == m
    if mask.sum() >= 20:
        monthly_ics[m] = spearmanr(p_ens[mask], y_te[mask])[0]
        monthly_n[m] = mask.sum()

ics_arr = np.array(list(monthly_ics.values()))
months_arr = np.array(list(monthly_ics.keys()))
n_arr = np.array(list(monthly_n.values()))
n_periods = len(ics_arr)
full_ic = np.mean(ics_arr)
full_icir = full_ic/np.std(ics_arr) if np.std(ics_arr)>0 else 0

print(); print('='*65)
print(f'EXTENDED PHASE 2: {n_periods} test months (2015-01 ~ {months_arr[-1]})')
print('='*65)

# 2.1
print(f'\n[2.1] IC Time Series:')
print(f'  Mean={full_ic:+.4f}  Median={np.median(ics_arr):+.4f}  Std={np.std(ics_arr):.4f}')
print(f'  Min={np.min(ics_arr):+.4f} ({months_arr[np.argmin(ics_arr)]})')
print(f'  Max={np.max(ics_arr):+.4f} ({months_arr[np.argmax(ics_arr)]})')
print(f'  IC>0: {np.mean(ics_arr>0):.0%} ({np.sum(ics_arr>0)}/{n_periods})')
print(f'  IC<-0.05: {np.sum(ics_arr<-0.05)} periods')

# 2.2 — Split-half
split = n_periods // 2
first, second = ics_arr[:split], ics_arr[split:]
ic1, icir1 = np.mean(first), np.mean(first)/np.std(first) if np.std(first)>0 else 0
ic2, icir2 = np.mean(second), np.mean(second)/np.std(second) if np.std(second)>0 else 0
print(f'\n[2.2] Split-Half:')
print(f'  H1 ({months_arr[0]}~{months_arr[split-1]}): IC={ic1:+.4f} ICIR={icir1:+.2f} IC>0={np.mean(first>0):.0%}')
print(f'  H2 ({months_arr[split]}~{months_arr[-1]}): IC={ic2:+.4f} ICIR={icir2:+.2f} IC>0={np.mean(second>0):.0%}')
print(f'  Delta: {(ic2-ic1):+.4f}')

# 2.3 — By regime (not just year)
regimes = [
    ('2015-2016', '2015-01', '2016-12', 'Bull crash + Circuit breaker'),
    ('2017-2018', '2017-01', '2018-12', 'Blue chip + Trade war bear'),
    ('2019-2020', '2019-01', '2020-12', 'Recovery + Core asset start'),
    ('2021-2022', '2021-01', '2022-12', 'Core asset peak + Long bear start'),
    ('2023-2024', '2023-01', '2024-12', 'Bear continuation + Recovery'),
    ('2025',     '2025-01', '2025-12', 'Current'),
]
print(f'\n[2.3] By Market Regime:')
print(f'  {"Period":<12s} {"Months":>6s} {"IC":>8s} {"ICIR":>8s} {"IC>0":>8s} {"ICstd":>8s} {"Context":<30s}')
print(f'  {"-"*85}')
for label, start, end, context in regimes:
    mask = (months_arr >= start) & (months_arr <= end)
    if mask.sum() < 2: continue
    ri = ics_arr[mask]
    ric = np.mean(ri)
    rir = ric/np.std(ri) if np.std(ri)>0 else 0
    print(f'  {label:<12s} {mask.sum():>6d} {ric:+8.4f} {rir:+8.2f} {np.mean(ri>0):7.0%} {np.std(ri):8.4f} {context:<30s}')

# 2.4 — Trim
trim_n = max(1, int(n_periods * 0.10))
extreme_idx = np.argsort(np.abs(ics_arr))[-trim_n:]
trimmed_ics = np.delete(ics_arr, extreme_idx)
trim_ic = np.mean(trimmed_ics)
trim_icir = trim_ic/np.std(trimmed_ics) if np.std(trimmed_ics)>0 else 0
print(f'\n[2.4] Trim Top 10% ({trim_n}/{n_periods}):')
print(f'  Trimmed IC={trim_ic:+.4f} ICIR={trim_icir:+.2f} vs Full IC={full_ic:+.4f} ICIR={full_icir:+.2f}')
print(f'  {"ROBUST" if abs(trim_icir-full_icir)<0.5 else "OUTLIER-DRIVEN"}')

# Final verdict
can_proceed = full_ic > 0.03 and full_icir > 0.5 and np.mean(ics_arr > 0) > 0.6
print(); print('='*65)
print(f'FINAL VERDICT:')
print(f'  IC={full_ic:+.4f}  ICIR={full_icir:+.2f}  IC>0={np.mean(ics_arr>0):.0%}')
print(f'  Can proceed to Phase 3: {"YES" if can_proceed else "NO"}')
print('='*65)
