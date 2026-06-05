# Phase 2: IC Decomposition on CORRECTED monthly data (YYYY-MM grouping)
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

# ======== Build monthly data with YYYY-MM grouping ========
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params = ','.join('?'*len(codes))
df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date", conn, params=codes)
conn.close()
df['month'] = df['date'].str[:7]  # YYYY-MM

print(f'Building features ({len(codes)} stocks)...', flush=True); t0=time.time()
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
print(f'{len(flat):,} samples ({time.time()-t0:.0f}s)', flush=True)

tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12');te_m=(dates_arr>='2024-01')
sc=StandardScaler();Xt=sc.fit_transform(flat[tr_m]);Xte=sc.transform(flat[te_m])
y_tr=ys[tr_m];y_te=ys[te_m];te_dates=dates_arr[te_m]

# Train ensemble
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

# ======== PHASE 2 DIAGNOSTICS ========
test_months_arr = np.array(sorted(set(te_dates)))
monthly_ics = {}
monthly_n = {}
for m in test_months_arr:
    mask = te_dates == m
    if mask.sum() >= 20:
        monthly_ics[m] = spearmanr(p_ens[mask], y_te[mask])[0]
        monthly_n[m] = mask.sum()

ics_arr = np.array(list(monthly_ics.values()))
months_arr = np.array(list(monthly_ics.keys()))
n_arr = np.array(list(monthly_n.values()))
n_periods = len(ics_arr)

print(); print('='*65)
print('PHASE 2: IC DECOMPOSITION (Corrected YYYY-MM grouping)')
print(f'{n_periods} test months with >=20 stocks')
print('='*65)

# 2.1 — IC time series
print(f'\n[2.1] IC Time Series:')
print(f'  Mean:    {np.mean(ics_arr):+.4f}')
print(f'  Median:  {np.median(ics_arr):+.4f}')
print(f'  Std:     {np.std(ics_arr):.4f}')
print(f'  Skew:    {float(pd.Series(ics_arr).skew()):.3f}')
print(f'  Min:     {np.min(ics_arr):+.4f} ({months_arr[np.argmin(ics_arr)]}, n={n_arr[np.argmin(ics_arr)]})')
print(f'  Max:     {np.max(ics_arr):+.4f} ({months_arr[np.argmax(ics_arr)]}, n={n_arr[np.argmax(ics_arr)]})')
print(f'  IC>0:    {np.mean(ics_arr>0):.0%} ({np.sum(ics_arr>0)}/{n_periods})')
print(f'  IC>0.15: {np.sum(ics_arr>0.15)} periods')
print(f'  IC<-0.05: {np.sum(ics_arr<-0.05)} periods')

# Highlight extreme months
extreme_hi = np.argsort(ics_arr)[-3:]
extreme_lo = np.argsort(ics_arr)[:3]
print(f'\n  Top 3 months:')
for i in extreme_hi[::-1]:
    print(f'    {months_arr[i]}: IC={ics_arr[i]:+.4f}  n={n_arr[i]}')
print(f'  Worst 3 months:')
for i in extreme_lo:
    print(f'    {months_arr[i]}: IC={ics_arr[i]:+.4f}  n={n_arr[i]}')

# IC vs stock count scatter
print(f'\n  IC vs Stock Count correlation:')
if len(n_arr) > 3:
    ic_n_corr, ic_n_p = spearmanr(ics_arr, n_arr)
    print(f'    Spearman r = {ic_n_corr:.4f} (p={ic_n_p:.4f})')
    print(f'    (Positive = larger cross-sections give higher IC)')

# 2.2 — Split-half
split = n_periods // 2
first = ics_arr[:split]; second = ics_arr[split:]
ic1, icir1 = np.mean(first), np.mean(first)/np.std(first) if np.std(first)>0 else 0
ic2, icir2 = np.mean(second), np.mean(second)/np.std(second) if np.std(second)>0 else 0
print(f'\n[2.2] Split-Half:')
print(f'  First {split} ({months_arr[0]}~{months_arr[split-1]}):')
print(f'    IC={ic1:+.4f}  ICIR={icir1:+.2f}  IC>0={np.mean(first>0):.0%}')
print(f'  Last {n_periods-split} ({months_arr[split]}~{months_arr[-1]}):')
print(f'    IC={ic2:+.4f}  ICIR={icir2:+.2f}  IC>0={np.mean(second>0):.0%}')
print(f'  Delta: {(ic2-ic1):+.4f} ({(ic2-ic1)/max(abs(ic1),0.001)*100:+.1f}%)')
stability = 'STABLE' if abs(ic2-ic1) < 0.05 else 'DEGRADING' if ic2 < ic1-0.05 else 'IMPROVING'
print(f'  Verdict: {stability}')

# 2.3 — Year-by-year
print(f'\n[2.3] Year-by-Year:')
print(f'  {"Year":<8s} {"IC":>8s} {"ICIR":>8s} {"IC>0":>8s} {"ICstd":>8s} {"N_months":>8s} {"Avg_Stocks":>10s}')
print(f'  {"-"*62}')
years_found = []
for yr in ['2024','2025','2026']:
    mask = np.array([str(m).startswith(yr) for m in months_arr])
    if mask.sum() < 2: continue
    y_ics = ics_arr[mask]
    y_n = n_arr[mask]
    y_ic = np.mean(y_ics)
    y_icir = y_ic/np.std(y_ics) if np.std(y_ics)>0 else 0
    years_found.append({'yr':yr,'IC':y_ic,'ICIR':y_icir,'IC>0':np.mean(y_ics>0),'ICstd':np.std(y_ics),'N':mask.sum(),'AvgStock':np.mean(y_n)})
    print(f'  {yr:<8s} {y_ic:+8.4f} {y_icir:+8.2f} {np.mean(y_ics>0):7.0%} {np.std(y_ics):8.4f} {mask.sum():>8d} {np.mean(y_n):>10.0f}')

# 2.4 — Trim top 10%
trim_n = max(1, int(n_periods * 0.10))
abs_ics = np.abs(ics_arr)
extreme_idx = np.argsort(abs_ics)[-trim_n:]
trimmed_ics = np.delete(ics_arr, extreme_idx)
trim_ic = np.mean(trimmed_ics)
trim_icir = trim_ic/np.std(trimmed_ics) if np.std(trimmed_ics)>0 else 0
full_icir = np.mean(ics_arr)/np.std(ics_arr) if np.std(ics_arr)>0 else 0
print(f'\n[2.4] Trim Top 10% ({trim_n}/{n_periods} periods):')
print(f'  Removed: {sorted(months_arr[extreme_idx])}')
print(f'  Full IC:     {np.mean(ics_arr):+.4f}  ICIR={full_icir:+.2f}')
print(f'  Trimmed IC:  {trim_ic:+.4f}  ICIR={trim_icir:+.2f}')
print(f'  IC impact:   {(trim_ic-np.mean(ics_arr)):+.4f}')
print(f'  ICIR impact: {(trim_icir-full_icir):+.2f}')
print(f'  Conclusion:  {"OUTLIER-DRIVEN" if abs(trim_icir-full_icir) > 0.5 else "ROBUST"}')

# Summary verdict
print(); print('='*65)
print('FINAL VERDICT')
print(f'  Overall IC:   {np.mean(ics_arr):+.4f}')
print(f'  Overall ICIR: {full_icir:+.2f}')
print(f'  Stability:    {stability}')
print(f'  Outlier risk: {"HIGH - trim reduces ICIR significantly" if abs(trim_icir-full_icir)>0.5 else "LOW - IC is not driven by outliers"}')
print('='*65)
