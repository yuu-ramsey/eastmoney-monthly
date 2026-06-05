# Final comprehensive accuracy evaluation of best model
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt

DB = '.eastmoney-ai/db/klines-v2.sqlite'
N_FFT = 10
OUT = '.eastmoney-ai/eval'

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

def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan, ics

# ======== Load + Build Features (66d best set) ========
conn=sqlite3.connect(DB)
codes=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params=','.join('?'*len(codes))
df=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date",conn,params=codes)
conn.close()

print('Building 66d features...', flush=True); t0=time.time()
flat_list, ys_list, dates_list = [], [], []
for code in codes:
    g=df[df['code']==code].sort_values('date').reset_index(drop=True)
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
        flat.append(ma5[i]/max(ma20[i],0.01)-1 if i>=20 and np.isfinite(ma5[i]) else 0)
        flat.append(1.0 if c[i]>ma5[i]else 0.0)
        flat.append(up_streak[i]/12.0); flat.append(dn_streak[i]/12.0)
        flat_list.append(flat); ys_list.append(np.clip(fwd_raw,-2,2))
        dates_list.append(g['date'].iloc[i])

flat=np.array(flat_list,dtype=np.float32); ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(ys); flat=flat[v]; ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12'); va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12'); te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} dims={flat.shape[1]} ({time.time()-t0:.0f}s)', flush=True)

sc=StandardScaler(); Xt=sc.fit_transform(flat[tr_m]); Xv=sc.transform(flat[va_m]); Xte=sc.transform(flat[te_m])
y_tr=ys[tr_m]; y_va=ys[va_m]; y_te=ys[te_m]; te_dates=dates_arr[te_m]

# ======== Train All Models ========
print('\nTraining models...', flush=True)
models = {}

# LightGBM
t0=time.time()
lgb_m=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,
    n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_m.fit(Xt,y_tr); p_lgb=lgb_m.predict(Xte)
ic_lgb, ics_lgb = cs_ic(p_lgb, y_te, te_dates)
models['LightGBM'] = {'pred':p_lgb, 'IC':ic_lgb, 'ics':ics_lgb, 'time':time.time()-t0}
print(f'  LightGBM:          CS_IC={ic_lgb:+.4f} ({time.time()-t0:.0f}s)', flush=True)

# XGBoost
t0=time.time()
xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,
    n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_m.fit(Xt,y_tr); p_xgb=xgb_m.predict(Xte)
ic_xgb, ics_xgb = cs_ic(p_xgb, y_te, te_dates)
models['XGBoost'] = {'pred':p_xgb, 'IC':ic_xgb, 'ics':ics_xgb, 'time':time.time()-t0}
print(f'  XGBoost:           CS_IC={ic_xgb:+.4f} ({time.time()-t0:.0f}s)', flush=True)

# Ridge
t0=time.time()
ridge_m=Ridge(alpha=1.0,random_state=456); ridge_m.fit(Xt,y_tr); p_ridge=ridge_m.predict(Xte)
ic_ridge, ics_ridge = cs_ic(p_ridge, y_te, te_dates)
models['Ridge'] = {'pred':p_ridge, 'IC':ic_ridge, 'ics':ics_ridge, 'time':time.time()-t0}
print(f'  Ridge:             CS_IC={ic_ridge:+.4f} ({time.time()-t0:.0f}s)', flush=True)

# MLP
t0=time.time()
mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,
    max_iter=300,early_stopping=True,random_state=456)
mlp_m.fit(Xt,y_tr); p_mlp=mlp_m.predict(Xte)
ic_mlp, ics_mlp = cs_ic(p_mlp, y_te, te_dates)
models['MLP'] = {'pred':p_mlp, 'IC':ic_mlp, 'ics':ics_mlp, 'time':time.time()-t0}
print(f'  MLP(64,32):        CS_IC={ic_mlp:+.4f} ({time.time()-t0:.0f}s)', flush=True)

# Ensemble (IC-weighted)
weights = {k: max(m['IC'], 0) for k, m in models.items()}
w_sum = sum(weights.values())
p_ens = sum(max(m['IC'],0)*m['pred'] for m in models.values()) / w_sum
ic_ens, ics_ens = cs_ic(p_ens, y_te, te_dates)

# Simple average
p_avg = np.mean([m['pred'] for m in models.values()], axis=0)
ic_avg, ics_avg = cs_ic(p_avg, y_te, te_dates)

# ======== Comprehensive Metrics ========
print(); print('='*70)
print('FINAL MODEL ACCURACY EVALUATION')
print(f'Data: {len(codes)} stocks, {len(flat):,} samples, {flat.shape[1]} features')
print(f'Train: {tr_m.sum():,} (2015-2021) | Val: {va_m.sum():,} (2022-2023) | Test: {te_m.sum():,} (2024+)')
print('='*70)

# 1. CS_IC Summary
print(f'\n{"Model":<20s} {"CS_IC":>8s} {"ICIR":>8s} {"IC>0%":>8s} {"IC std":>8s}')
print('-'*56)
for name, m in [('LightGBM',models['LightGBM']),('XGBoost',models['XGBoost']),
                 ('Ridge',models['Ridge']),('MLP(64,32)',models['MLP']),
                 ('Avg Ensemble',{'IC':ic_avg,'ics':ics_avg}),
                 ('Weighted Ensemble',{'IC':ic_ens,'ics':ics_ens})]:
    icir = m['IC']/np.std(m['ics']) if np.std(m['ics'])>0 else np.nan
    pos_pct = np.mean(np.array(m['ics'])>0)
    print(f'{name:<20s} {m["IC"]:+8.4f} {icir:+8.4f} {pos_pct:7.1%} {np.std(m["ics"]):+8.4f}')

# 2. Direction Accuracy
print(f'\n{"Direction Accuracy":-^56}')
for name, pred in [('LightGBM',p_lgb),('XGBoost',p_xgb),('Ridge',p_ridge),
                    ('MLP',p_mlp),('Avg Ensemble',p_avg),('Weighted Ensemble',p_ens)]:
    hit = np.mean((pred>0)==(y_te>0))
    up_prec = np.sum((pred>0)&(y_te>0))/max(np.sum(pred>0),1)
    dn_prec = np.sum((pred<0)&(y_te<0))/max(np.sum(pred<0),1)
    print(f'  {name:<20s} Hit={hit:.3f}  UPprec={up_prec:.3f}  DNprec={dn_prec:.3f}')

# 3. Top-K Performance
print(f'\n{"Top-K Long-Short":-^56}')
for name, pred in [('LightGBM',p_lgb),('XGBoost',p_xgb),('Avg Ensemble',p_avg),('Weighted Ensemble',p_ens)]:
    for k in [5,10,20,30]:
        nk=max(1,int(len(pred)*k//100))
        t=np.argsort(pred)[-nk:]; b=np.argsort(pred)[:nk]
        ls=np.mean(y_te[t])-np.mean(y_te[b])
        print(f'  {name:<20s} Top{k:2d}%: LS={ls:+.4f}  Long={np.mean(y_te[t]):+.4f}  Short={np.mean(y_te[b]):+.4f}')

# 4. Year-by-Year
print(f'\n{"Yearly Breakdown":-^56}')
print(f'  {"Year":<8s} {"LGB IC":>8s} {"XGB IC":>8s} {"Ens IC":>8s} {"Ens Hit":>8s} {"N":>8s}')
for yr in ['2024','2025','2026']:
    m=np.array([str(d).startswith(yr) for d in te_dates])
    if m.sum()<100: continue
    ic_l=spearmanr(p_lgb[m],y_te[m])[0]; ic_x=spearmanr(p_xgb[m],y_te[m])[0]
    ic_e=spearmanr(p_ens[m],y_te[m])[0]; hit=np.mean((p_ens[m]>0)==(y_te[m]>0))
    print(f'  {yr:<8s} {ic_l:+8.4f} {ic_x:+8.4f} {ic_e:+8.4f} {hit:8.3f} {m.sum():>8,}')

# 5. Monthly IC Distribution
print(f'\n{"Monthly IC Distribution":-^56}')
for name, ics in [('LightGBM',ics_lgb),('XGBoost',ics_xgb),('Ensemble',ics_ens)]:
    ics_a=np.array(ics)
    print(f'  {name:<15s} Mean={np.mean(ics_a):+.4f} Med={np.median(ics_a):+.4f} '
          f'Std={np.std(ics_a):.4f} Pos={np.sum(ics_a>0)}/{len(ics_a)} '
          f'Best={np.max(ics_a):+.4f} Worst={np.min(ics_a):+.4f}')

# 6. Model Correlation
print(f'\n{"Model Correlation Matrix":-^56}')
all_preds = {'LGB':p_lgb, 'XGB':p_xgb, 'Ridge':p_ridge, 'MLP':p_mlp, 'Ens':p_ens}
names = list(all_preds.keys())
print(f'  {"":>8s}', end='')
for n in names: print(f'{n:>8s}', end='')
print()
for n1 in names:
    print(f'  {n1:>8s}', end='')
    for n2 in names:
        r = spearmanr(all_preds[n1], all_preds[n2])[0]
        print(f'{r:+8.4f}', end='')
    print()

# 7. Error Metrics
print(f'\n{"Error Metrics (Ensemble)":-^56}')
mae=np.mean(np.abs(p_ens-y_te)); rmse=np.sqrt(np.mean((p_ens-y_te)**2))
r2 = 1 - np.sum((y_te-p_ens)**2)/np.sum((y_te-np.mean(y_te))**2)
print(f'  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}')

# 8. Feature Count & Data Summary
print(f'\n{"Summary":-^56}')
print(f'  Stocks:               {len(codes)}')
print(f'  Features:             {flat.shape[1]}')
print(f'  Training samples:     {tr_m.sum():,} (2015-2021)')
print(f'  Test samples:         {te_m.sum():,} (2024+)')
print(f'  Test months:          {len(np.unique(te_dates))}')
print(f'  Best CS_IC:           {ic_ens:+.4f} (Weighted Ensemble)')
print(f'  Best Hit Rate:        {np.mean((p_ens>0)==(y_te>0)):.3f}')
print(f'  Top20 Long-Short:     {np.mean(y_te[np.argsort(p_ens)[-int(len(p_ens)*0.2):]])-np.mean(y_te[np.argsort(p_ens)[:int(len(p_ens)*0.2)]]):+.4f}')
print('='*70)

# Save
import json
out = {
    'config': {'stocks':len(codes), 'features':int(flat.shape[1]), 'train':int(tr_m.sum()), 'test':int(te_m.sum())},
    'models': {k:{'CS_IC':float(v['IC']), 'n_months':len(v['ics']), 'ics':[float(x) for x in v['ics']]} for k,v in models.items()},
    'ensemble': {'CS_IC':float(ic_ens), 'Hit':float(np.mean((p_ens>0)==(y_te>0))), 'MAE':float(mae), 'RMSE':float(rmse), 'R2':float(r2)},
    'yearly': {}
}
for yr in ['2024','2025','2026']:
    m=np.array([str(d).startswith(yr) for d in te_dates])
    if m.sum()>=100:
        out['yearly'][yr] = {'IC_LGB':float(spearmanr(p_lgb[m],y_te[m])[0]), 'IC_ENS':float(spearmanr(p_ens[m],y_te[m])[0]), 'Hit':float(np.mean((p_ens[m]>0)==(y_te[m]>0))), 'N':int(m.sum())}
import os; os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/final_eval.json','w') as f: json.dump(out, f, indent=2)
print(f'Saved to {OUT}/final_eval.json')
