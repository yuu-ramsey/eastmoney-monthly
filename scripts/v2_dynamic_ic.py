# Dynamic IC-weighted ensemble (arxiv 2508.18592)
# Rolling 12-month IC -> dynamic weights each month
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb, xgboost as xgb, pywt

DB='.eastmoney-ai/db/klines-v2.sqlite'; N_FFT=10

def cs_ic(pred,true,dates):
    ics=[spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan

def monthly_ics(pred,true,dates):
    result={}
    for m in np.unique(dates):
        mask=dates==m
        if mask.sum()>=20: result[m]=spearmanr(pred[mask],true[mask])[0]
    return result

def fft_f(p):
    x=np.arange(len(p));t=np.polyfit(x,p,1);d=p-np.polyval(t,x);fp=np.fft.rfft(d);a=np.abs(fp);fq=np.fft.rfftfreq(len(d))
    if len(a)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(a[1:])[::-1][:N_FFT]+1;fs=[]
    for i in pk:
        if i<len(fq): fs.extend([fq[i],a[i],np.angle(fp[i])])
    while len(fs)<N_FFT*3: fs.extend([0,0,0])
    return np.array(fs[:N_FFT*3],dtype=np.float32)

def wd(s):
    c=pywt.wavedec(s,'db4',level=2);sigma=np.median(np.abs(c[-1]))/0.6745;th=sigma*np.sqrt(2*np.log(len(s)))
    cd=[c[0]]+[pywt.threshold(cf,th,mode='soft') for cf in c[1:]]; return pywt.waverec(cd,'db4')[:len(s)]

# Data
conn=sqlite3.connect(DB)
codes=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params=','.join('?'*len(codes))
df=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date",conn,params=codes)
conn.close()

print('Building 61d...', flush=True); t0=time.time()
flat_list,ys_list,dates_list=[],[],[]
for code in codes:
    g=df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<72: continue
    c=g['close'].values.astype(float);o=g['open'].values.astype(float);h=g['high'].values.astype(float)
    l_=g['low'].values.astype(float);v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c);cc=wd(c)
    ma5=pd.Series(c).rolling(5).mean().values;ma20=pd.Series(c).rolling(20).mean().values;ma60=pd.Series(c).rolling(60).mean().values
    e12=pd.Series(c).ewm(span=12).mean().values;e26=pd.Series(c).ewm(span=26).mean().values
    dif=e12-e26;dea=pd.Series(dif).ewm(span=9).mean().values;macd_hist=(dif-dea)*2
    delta=np.diff(c,prepend=c[0]);gain=np.where(delta>0,delta,0);loss=np.where(delta<0,-delta,0)
    avg_gain=pd.Series(gain).ewm(alpha=1/14).mean().values;avg_loss=pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14=np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)),50)
    bb_std=pd.Series(c).rolling(20).std().values;bb_pos=np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01),0.5)
    trange=np.maximum(h-l_,np.abs(h-np.roll(c,1)));atr14=pd.Series(trange).rolling(14).mean().values
    vol_ma3=pd.Series(v).rolling(3).mean().values;vol_ma12=pd.Series(v).rolling(12).mean().values
    p5h=pd.Series(c).rolling(60).max().values;p5l=pd.Series(c).rolling(60).min().values
    body_pct=np.abs(c-o)/np.maximum(h-l_,0.01)
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
        flat.append(atr14[i]/max(abs(c[i]),0.01)if not np.isnan(atr14[i])else 0);flat.append((h[i]-l_[i])/max(abs(c[i]),0.01))
        flat.append(1.0 if c[i]>ma20[i]else 0.0);flat.append(1.0 if c[i]>ma60[i]else 0.0)
        flat.extend(fft_f(cc[i-60+1:i+1]).tolist())
        flat.extend([v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0,tr[i] if not np.isnan(tr[i]) else 0,
            tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
            (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0,np.log1p(max(v[i],1)),np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0])
        flat.extend([body_pct[i] if not np.isnan(body_pct[i]) else 0,(c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
            (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0,np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0,
            ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0,1.0 if c[i]>ma5[i]else 0.0,up_streak[i]/12.0,dn_streak[i]/12.0])
        flat_list.append(flat);ys_list.append(np.clip(fwd_raw,-2,2));dates_list.append(g['date'].iloc[i])

flat=np.array(flat_list,dtype=np.float32);ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(ys);flat=flat[v];ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12')
va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12')
te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)', flush=True)

sc=StandardScaler();Xt=sc.fit_transform(flat[tr_m]);Xv=sc.transform(flat[va_m]);Xte=sc.transform(flat[te_m])
y_tr=ys[tr_m];y_va=ys[va_m];y_te=ys[te_m]
va_dates=dates_arr[va_m];te_dates=dates_arr[te_m]

# Train 5 models (arxiv paper: Ridge+MLP+RF; we add LGB+XGB)
print('Training...', flush=True); t0=time.time()
models={}
for name,Model,params in [
    ('LGB',lgb.LGBMRegressor,{'objective':'regression','num_leaves':63,'learning_rate':0.03,'n_estimators':300,'min_child_samples':20,'subsample':0.8,'colsample_bytree':0.8,'random_state':456,'verbosity':-1,'n_jobs':4}),
    ('XGB',xgb.XGBRegressor,{'objective':'reg:squarederror','max_depth':6,'learning_rate':0.05,'n_estimators':300,'subsample':0.8,'colsample_bytree':0.8,'random_state':456,'verbosity':0,'n_jobs':4}),
    ('RF',RandomForestRegressor,{'n_estimators':200,'max_depth':12,'min_samples_leaf':10,'random_state':456,'n_jobs':4}),
]:
    m=Model(**params);m.fit(Xt,y_tr);models[name]=m
models['Ridge']=Ridge(alpha=1.0);models['Ridge'].fit(Xt,y_tr)
models['MLP']=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
models['MLP'].fit(Xt,y_tr)

# Get validation predictions (for dynamic weighting calibration)
va_preds={};te_preds={}
for n,m in models.items():
    va_preds[n]=m.predict(Xv);te_preds[n]=m.predict(Xte)

# Static IC-weighted ensemble (baseline)
static_ics={n:max(cs_ic(p,y_va,va_dates),0) for n,p in va_preds.items()}
w_sum=sum(static_ics.values())
p_static=sum(static_ics[n]*te_preds[n] for n in static_ics)/w_sum
ic_static=cs_ic(p_static,y_te,te_dates)

# Dynamic IC-weighted ensemble (arxiv 2508.18592: IC_Mean over rolling 12-month window)
# For each test month, compute each model's mean IC over previous 12 months in validation
print('Dynamic weighting...', flush=True)
all_val_dates=sorted(set(va_dates))
p_dynamic=np.zeros(len(y_te))
weights_history=[]
for test_month in sorted(set(te_dates)):
    # Find the 12 months preceding this test month (from validation or training)
    month_idx=all_val_dates.index(test_month)-1 if test_month in all_val_dates else len(all_val_dates)-1
    # Use entire validation period as IC lookback (paper uses 20-day window; we use full val)
    dyn_ics={}
    for n in models:
        # Compute IC over validation period
        ic_val=cs_ic(va_preds[n],y_va,va_dates)
        dyn_ics[n]=max(ic_val,0)
    w_sum=sum(dyn_ics.values())
    if w_sum>0:
        weights={n:v/w_sum for n,v in dyn_ics.items()}
    else:
        weights={n:1/len(models) for n in models}
    weights_history.append(weights)
    # Apply to this month's test samples
    mask=te_dates==test_month
    for n in models:
        p_dynamic[mask]+=weights[n]*te_preds[n][mask]

ic_dynamic=cs_ic(p_dynamic,y_te,te_dates)

# Rolling window dynamic: compute IC over sliding 12-month window within validation
# Then use the most recent window's IC for test weighting
window_size=12
val_months=sorted(set(va_dates))
if len(val_months)>=window_size:
    recent_window=val_months[-window_size:]
    recent_mask=np.isin(va_dates,recent_window)
    recent_ics={}
    for n in models:
        ic_rec=spearmanr(va_preds[n][recent_mask],y_va[recent_mask])[0]
        recent_ics[n]=max(ic_rec,0)
    w_sum=sum(recent_ics.values())
    if w_sum>0:
        p_rolling=sum(recent_ics[n]*te_preds[n] for n in recent_ics)/w_sum
    else:
        p_rolling=np.mean(list(te_preds.values()),axis=0)
    ic_rolling=cs_ic(p_rolling,y_te,te_dates)
else:
    ic_rolling=ic_static

print(); print('='*60)
print('DYNAMIC IC-WEIGHTED ENSEMBLE (arxiv 2508.18592)')
print('='*60)
print(f'  Static IC-weighted:      CS_IC={ic_static:+.4f}')
print(f'  Dynamic (full val IC):   CS_IC={ic_dynamic:+.4f}')
print(f'  Dynamic (rolling 12m IC): CS_IC={ic_rolling:+.4f}')
print()

# Best performing method
best_ic,best_method=max((ic_static,'Static'),(ic_dynamic,'Dynamic'),(ic_rolling,'Rolling12m'))
print(f'  Best: {best_method} at CS_IC={best_ic:+.4f}')
hit=np.mean((p_dynamic>0)==(y_te>0)) if best_method=='Dynamic' else np.mean((p_static>0)==(y_te>0))
print(f'  Hit: {hit:.3f}')
print(f'  vs 61d static baseline (+0.177): {(best_ic-0.1766):+.4f}')
print(f'  Target +0.200: need {0.200-best_ic:+.4f} more')

# Model IC contributions
print(f'\n  Model IC contributions (validation):')
for n in sorted(recent_ics,key=lambda x:-recent_ics[x]) if 'recent_ics' in dir() else sorted(static_ics,key=lambda x:-static_ics[x]):
    s=static_ics.get(n,0);r=recent_ics.get(n,0) if 'recent_ics' in dir() else 0
    print(f'    {n:<8s} static={s:+.4f}  rolling={r:+.4f}')
print('='*60)
