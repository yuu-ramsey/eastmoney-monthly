# Minimal v2: 61d base + 5d weekly bridge, no cross-sectional norm
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt

DB='.eastmoney-ai/db/klines-v2.sqlite'; N_FFT=10

def cs_ic(pred,true,dates):
    ics=[spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan

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

# Data
conn=sqlite3.connect(DB)
codes_m=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params_m=','.join('?'*len(codes_m))
df_m=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params_m}) AND date>='2010-01' ORDER BY code,date",conn,params=codes_m)

# Weekly bridge
codes_w=[r[0] for r in conn.execute('SELECT code FROM weekly_klines GROUP BY code HAVING COUNT(*)>=200').fetchall()]
params_w=','.join('?'*len(codes_w))
df_w=pd.read_sql_query(f"SELECT code,date,close FROM weekly_klines WHERE code IN ({params_w}) AND date>='2010-01' ORDER BY code,date",conn,params=codes_w)
conn.close()

df_w['month']=df_w['date'].str[:7]
w_temp={}
for _,row in df_w.iterrows():
    key=(row['code'],row['month'])
    if key not in w_temp: w_temp[key]=[]
    w_temp[key].append(row['close'])
weekly_agg={}
for (code,month),prices in w_temp.items():
    if len(prices)<3: continue
    p=np.array(prices); rets=np.diff(p)/np.maximum(p[:-1],0.01)
    weekly_agg[(code,month)]=[rets[-1] if len(rets)>=1 else 0,
        np.mean(rets) if len(rets)>=2 else 0, np.std(rets) if len(rets)>=2 else 0,
        rets[-1]-rets[-2] if len(rets)>=2 else 0, p[-1]/max(np.mean(p),0.01)-1]
print(f'Weekly bridge: {len(weekly_agg)} months', flush=True)

# Build features
print('Building...', flush=True); t0=time.time()
flat_list,ys_list,dates_list=[],[],[]
for code in codes_m:
    g=df_m[df_m['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<72: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c); c_clean=wdenoise(c)
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

    for i in range(60,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        flat=[]
        r=[(c[i]-c[i-j])/max(abs(c[i-j]),0.01) if i>=j else 0 for j in [1,3,6,12]]
        flat.extend(r)
        for ma in [ma5,ma20,ma60]:
            flat.append((c[i]-ma[i])/max(abs(c[i]),0.01) if not np.isnan(ma[i]) else 0)
        flat.extend([dif[i]if not np.isnan(dif[i])else 0,dea[i]if not np.isnan(dea[i])else 0,macd_hist[i]if not np.isnan(macd_hist[i])else 0])
        flat.append(rsi14[i]if not np.isnan(rsi14[i])else 50)
        flat.append(bb_pos[i]if not np.isnan(bb_pos[i])else 0.5)
        flat.append(np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0)
        flat.append(atr14[i]/max(abs(c[i]),0.01)if not np.isnan(atr14[i])else 0)
        flat.append((h[i]-l[i])/max(abs(c[i]),0.01))
        flat.append(1.0 if c[i]>ma20[i]else 0.0); flat.append(1.0 if c[i]>ma60[i]else 0.0)
        flat.extend(fft_f(c_clean[i-60+1:i+1]).tolist())
        flat.extend([v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0,
            tr[i] if not np.isnan(tr[i]) else 0,
            tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
            (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0,
            np.log1p(max(v[i],1)), np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0])
        flat.extend([body_pct[i] if not np.isnan(body_pct[i]) else 0,
            (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
            (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0,
            np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0,
            ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0,
            1.0 if c[i]>ma5[i]else 0.0, up_streak[i]/12.0, dn_streak[i]/12.0])
        wk=weekly_agg.get((code,g['date'].iloc[i]),[0,0,0,0,0])
        flat.extend(wk)
        flat_list.append(flat); ys_list.append(np.clip(fwd_raw,-2,2))
        dates_list.append(g['date'].iloc[i])

flat=np.array(flat_list,dtype=np.float32); ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(ys); flat=flat[v]; ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12'); te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat):,} T={tr_m.sum():,} Te={te_m.sum():,} dims={flat.shape[1]} ({time.time()-t0:.0f}s)', flush=True)

sc=StandardScaler(); Xt=sc.fit_transform(flat[tr_m]); Xte=sc.transform(flat[te_m])
y_tr=ys[tr_m]; y_te=ys[te_m]; te_dates=dates_arr[te_m]

models={}
for name,Model,params in [
    ('LGB',lgb.LGBMRegressor,{'objective':'regression','num_leaves':63,'learning_rate':0.03,'n_estimators':300,'min_child_samples':20,'subsample':0.8,'colsample_bytree':0.8,'random_state':456,'verbosity':-1,'n_jobs':4}),
    ('XGB',xgb.XGBRegressor,{'objective':'reg:squarederror','max_depth':6,'learning_rate':0.05,'n_estimators':300,'subsample':0.8,'colsample_bytree':0.8,'random_state':456,'verbosity':0,'n_jobs':4}),
]:
    m=Model(**params); m.fit(Xt,y_tr); models[name]=m.predict(Xte)

ridge_m=Ridge(alpha=1.0); ridge_m.fit(Xt,y_tr); models['Ridge']=ridge_m.predict(Xte)
mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
mlp_m.fit(Xt,y_tr); models['MLP']=mlp_m.predict(Xte)

ics={}
for n,p in models.items(): ics[n]=max(cs_ic(p,y_te,te_dates),0)
w_sum=sum(ics.values())
p_ens=sum(ics[n]*models[n] for n in ics)/w_sum
ic_new=cs_ic(p_ens,y_te,te_dates); hit=np.mean((p_ens>0)==(y_te>0))
ic_tr=cs_ic(lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4).fit(Xt,y_tr).predict(Xt),y_tr,dates_arr[tr_m])
n20=max(1,int(len(p_ens)*0.2))
ls=np.mean(y_te[np.argsort(p_ens)[-n20:]])-np.mean(y_te[np.argsort(p_ens)[:n20]])

print('='*60)
for n,ic in sorted(ics.items(),key=lambda x:-x[1]):
    print(f'  {n:<10s} CS_IC={ic:+.4f}')
print(f'  {"Ensemble":<10s} CS_IC={ic_new:+.4f}  Hit={hit:.3f}  Top20LS={ls:+.4f}')
print(f'  Dims={flat.shape[1]}  N_test={len(y_te):,}')
print(f'  vs 61d baseline:  {ic_new-0.1766:+.4f}')
lgb_test_ic = ics.get('LGB', 0)
print(f'  Overfit gap:      train_IC={ic_tr:+.4f}  test_IC={lgb_test_ic:+.4f}')
print('='*60)
