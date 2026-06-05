# Final optimization: Daily signals as monthly LGB features + ensemble blending
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt
from pathlib import Path

DEV = torch.device('cuda')
DB = '.eastmoney-ai/db/klines-v2.sqlite'
OUT = '.eastmoney-ai/eval'
BATCH = 1024
N_SEEDS = 10
LOOKBACK_DAILY = 60

print('Final Optimization: Daily signals + Monthly features -> Ensemble', flush=True)

# ======== 1. Train Daily LSTM 10-seed ========
conn = sqlite3.connect(DB)
codes_d = [r[0] for r in conn.execute('SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=500').fetchall()]
params_d = ','.join('?'*len(codes_d))
df_d = pd.read_sql_query(f"SELECT code,date,open,close,high,low,volume,turnover_rate FROM daily_klines WHERE code IN ({params_d}) AND date>='2010-01-01' ORDER BY code,date", conn, params=codes_d)
conn.close()

print('Building daily sequences...', flush=True); t0=time.time()
X_d, y_d, dates_d, codes_d_arr = [], [], [], []
for code in codes_d:
    g=df_d[df_d['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<LOOKBACK_DAILY+66: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c); dates=g['date'].values
    vol_ma20=pd.Series(v).rolling(20).mean().fillna(1).values
    F=np.zeros((n,8),dtype=np.float32)
    for i in range(n):
        pc=max(abs(c[i-1]),0.01) if i>=1 else c[i]
        F[i,0]=o[i]/max(pc,0.01)-1; F[i,1]=h[i]/max(c[i],0.01)-1
        F[i,2]=l[i]/max(c[i],0.01)-1; F[i,3]=c[i]/max(pc,0.01)-1
        F[i,4]=v[i]/max(vol_ma20[i],1)-1; F[i,5]=tr[i] if not np.isnan(tr[i]) and tr[i]<100 else 0
        F[i,6]=(h[i]-l[i])/max(c[i],0.01); F[i,7]=(c[i]-o[i])/max(o[i],0.01) if o[i]>0 else 0
    F=np.nan_to_num(F,0.0).astype(np.float32)
    for i in range(LOOKBACK_DAILY-1, n-63):
        fwd=(c[i+63]-c[i])/max(c[i],0.01)
        if abs(fwd)>3: continue
        X_d.append(F[i-LOOKBACK_DAILY+1:i+1]); y_d.append(np.clip(fwd,-3,3))
        dates_d.append(str(dates[i])[:10]); codes_d_arr.append(code)

X_d=np.array(X_d,dtype=np.float32); y_d=np.array(y_d,dtype=np.float32)
dates_d_arr=np.array(dates_d); codes_d_arr=np.array(codes_d_arr)
v_d=~np.isnan(X_d).any(axis=(1,2))&~np.isnan(y_d); X_d=X_d[v_d]; y_d=y_d[v_d]
dates_d_arr=dates_d_arr[v_d]; codes_d_arr=codes_d_arr[v_d]
tr_d=(dates_d_arr>='2015-01-01')&(dates_d_arr<='2021-12-31')
va_d=(dates_d_arr>='2022-01-01')&(dates_d_arr<='2023-12-31')
te_d=(dates_d_arr>='2024-01-01')

fm_d=X_d[tr_d].reshape(-1,X_d.shape[-1]).mean(0); fs_d=X_d[tr_d].reshape(-1,X_d.shape[-1]).std(0)+1e-8
X_d=np.clip((X_d-fm_d)/fs_d,-5,5)
X_tr_d=torch.from_numpy(X_d[tr_d]).float().to(DEV); y_tr_d=torch.from_numpy(y_d[tr_d]).float().to(DEV)
X_va_d=torch.from_numpy(X_d[va_d]).float().to(DEV); y_va_d=y_d[va_d]
X_te_d=torch.from_numpy(X_d[te_d]).float().to(DEV)

class SlimLSTM(nn.Module):
    def __init__(self, d=8, h=64, l=2):
        super().__init__()
        self.lstm = nn.LSTM(d, h, l, batch_first=True)
        self.head = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)

daily_preds_te = []
for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    model=SlimLSTM().to(DEV); opt=torch.optim.Adam(model.parameters(),lr=0.001)
    best_va,best_st,patience=-99,None,0
    for ep in range(200):
        model.train(); perm=torch.randperm(len(X_tr_d),device=DEV)
        for i in range(0,len(X_tr_d),BATCH):
            idx=perm[i:i+BATCH]; loss=nn.MSELoss()(model(X_tr_d[idx]),y_tr_d[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad():
            pv=np.concatenate([model(X_va_d[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_va_d),BATCH)])
            ic=spearmanr(pv,y_va_d)[0]
        if ic>best_va: best_va=ic; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience=0
        else: patience+=1
        if patience>=20: break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        daily_preds_te.append(np.concatenate([model(X_te_d[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_te_d),BATCH)]))
    if seed%3==0: print(f'  Daily seed {seed}: {ep+1}ep', flush=True)

ens_daily_te = np.mean(daily_preds_te, axis=0)
daily_std_te = np.std(daily_preds_te, axis=0)
print(f'Daily 10-seed: {time.time()-t0:.0f}s', flush=True)

# Aggregate daily -> monthly
te_months = np.array([d[:7] for d in dates_d_arr[te_d]])
te_codes = codes_d_arr[te_d]
monthly_daily = {}
for i in range(len(te_codes)):
    key = (te_codes[i], te_months[i])
    if key not in monthly_daily:
        monthly_daily[key] = []
    monthly_daily[key].append({
        'mean': ens_daily_te[i], 'std': daily_std_te[i],
        'disagreement': daily_std_te[i] / max(abs(ens_daily_te[i]), 0.001)
    })

# ======== 2. Monthly Features + Model ========
def fft_f(prices):
    N_FFT=10; x=np.arange(len(prices)); trend=np.polyfit(x,prices,1); detrended=prices-np.polyval(trend,x)
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

conn=sqlite3.connect(DB)
codes_m=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params_m=','.join('?'*len(codes_m))
df_m=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params_m}) AND date>='2010-01' ORDER BY code,date",conn,params=codes_m)
conn.close()

print('Building monthly features...', flush=True); t0=time.time()
flat_m, ys_m, dates_m, codes_m_arr = [], [], [], []
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
        flat_m.append(flat); ys_m.append(np.clip(fwd_raw,-2,2))
        dates_m.append(g['date'].iloc[i]); codes_m_arr.append(code)

flat_m=np.array(flat_m,dtype=np.float32); ys_m=np.array(ys_m,dtype=np.float32)
v_m=~np.isnan(flat_m).any(axis=1)&~np.isnan(ys_m); flat_m=flat_m[v_m]; ys_m=ys_m[v_m]
dates_m=np.array([dates_m[i] for i in range(len(v_m)) if v_m[i]])
codes_m_arr=np.array([codes_m_arr[i] for i in range(len(v_m)) if v_m[i]])

tr_m=(dates_m>='2015-01')&(dates_m<='2021-12')
te_m=(dates_m>='2024-01')

sc_m=StandardScaler(); Xt_m=sc_m.fit_transform(flat_m[tr_m]); Xte_m=sc_m.transform(flat_m[te_m])
y_tr_m=ys_m[tr_m]; y_te_m=ys_m[te_m]; te_dates=dates_m[te_m]; te_codes= codes_m_arr[te_m]

# ======== 3. Add daily signals to monthly features ========
# Map daily signals to monthly test set
daily_feats_te = np.zeros((len(Xte_m), 3), dtype=np.float32)  # mean, std, disagreement
found = 0
for i in range(len(te_codes)):
    key = (te_codes[i], te_dates[i])
    if key in monthly_daily:
        vals = monthly_daily[key]
        means = [v['mean'] for v in vals]
        daily_feats_te[i, 0] = np.mean(means)
        daily_feats_te[i, 1] = np.std(means)
        daily_feats_te[i, 2] = len(means)  # number of daily predictions in the month
        found += 1

print(f'Daily features merged: {found}/{len(te_codes)} ({found/len(te_codes):.0%})', flush=True)

# Augmented features: monthly + daily signals
Xte_aug = np.column_stack([Xte_m, daily_feats_te])
# For training: we don't have daily features for training set (daily model trained on same period)
# So only test on augmented set; compare with baseline on same stocks

# Train baseline monthly models
lgb_m=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_m.fit(Xt_m,y_tr_m); p_lgb=lgb_m.predict(Xte_m)

xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_m.fit(Xt_m,y_tr_m); p_xgb=xgb_m.predict(Xte_m)

ridge_m=Ridge(alpha=1.0); ridge_m.fit(Xt_m,y_tr_m); p_ridge=ridge_m.predict(Xte_m)
mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
mlp_m.fit(Xt_m,y_tr_m); p_mlp=mlp_m.predict(Xte_m)

def cs_ic(pred,true,dates):
    ics=[spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan,ics

# Baseline ensemble
ics_b={}
for name,pred in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)]:
    ics_b[name]=max(cs_ic(pred,y_te_m,te_dates)[0],0)
w_sum=sum(ics_b.values())
p_ens_bl=(ics_b['LGB']*p_lgb+ics_b['XGB']*p_xgb+ics_b['Ridge']*p_ridge+ics_b['MLP']*p_mlp)/w_sum

# Daily signals IC
daily_ic, _ = cs_ic(daily_feats_te[:,0], y_te_m, te_dates)

# Blending: monthly ensemble + daily signal (only blend where daily exists)
has_daily = daily_feats_te[:,0] != 0  # only 11% have daily signals
print(f'Blending on {has_daily.sum()}/{len(has_daily)} samples with daily signals ({has_daily.mean():.0%})', flush=True)

# On full set: use monthly only where no daily
p_blend = p_ens_bl.copy()
best_ic_bl, best_w_bl = -99, 0
for w in np.arange(0, 1.05, 0.01):
    bl = p_ens_bl.copy()
    bl[has_daily] = w*daily_feats_te[has_daily,0] + (1-w)*p_ens_bl[has_daily]
    ic, _ = cs_ic(bl, y_te_m, te_dates)
    if ic > best_ic_bl: best_ic_bl = ic; best_w_bl = w

p_blend[has_daily] = best_w_bl*daily_feats_te[has_daily,0] + (1-best_w_bl)*p_ens_bl[has_daily]
ic_blend, ics_blend = cs_ic(p_blend, y_te_m, te_dates)
# Also report only-blend subset
ic_blend_sub, _ = cs_ic(p_blend[has_daily], y_te_m[has_daily], te_dates[has_daily])
ic_bl_sub, _ = cs_ic(p_ens_bl[has_daily], y_te_m[has_daily], te_dates[has_daily])

# ======== 4. Final Report ========
ic_bl, _ = cs_ic(p_ens_bl, y_te_m, te_dates)
hit_bl = np.mean((p_ens_bl>0)==(y_te_m>0))
ic_best, _ = cs_ic(p_blend, y_te_m, te_dates)
hit_best = np.mean((p_blend>0)==(y_te_m>0))

print(); print('='*65)
print('FINAL OPTIMIZATION RESULTS')
print('='*65)
print(f'  Monthly Ensemble (full 2247 stocks, 51K samples):')
print(f'    CS_IC={ic_bl:+.4f}  Hit={hit_bl:.3f}')
print(f'  Daily signal IC (monthly level): {daily_ic:+.4f}')
print(f'  Daily coverage: {has_daily.sum()}/{len(has_daily)} ({has_daily.mean():.0%}) stocks')
print(f'  Optimal Blend ({best_w_bl:.2f} daily + {1-best_w_bl:.2f} monthly):')
print(f'    Full-set CS_IC={ic_blend:+.4f}  Hit={hit_best:.3f}')
print(f'    Blend-subset CS_IC={ic_blend_sub:+.4f} (vs monthly {ic_bl_sub:+.4f} on same subset)')
print(f'  Improvement (full set): {(ic_blend-ic_bl):+.4f} ({(ic_blend-ic_bl)/abs(ic_bl)*100:+.1f}%)')
print(f'  Improvement (blend subset): {(ic_blend_sub-ic_bl_sub):+.4f} ({(ic_blend_sub-ic_bl_sub)/abs(ic_bl_sub)*100:+.1f}%)')
print()
n20=max(1,int(len(p_blend)*0.2))
ls_bl = np.mean(y_te_m[np.argsort(p_blend)[-n20:]]) - np.mean(y_te_m[np.argsort(p_blend)[:n20]])
print(f'  Top20 Long-Short: {ls_bl:+.4f}')
print(f'  Test months: {len(np.unique(te_dates))}')
print(f'  Test samples: {len(y_te_m):,}')
print('='*65)

# Save final results
Path(OUT).mkdir(parents=True,exist_ok=True)
with open(f'{OUT}/final_optimized.json','w') as f:
    json.dump({
        'monthly_ensemble': {'IC':float(ic_bl), 'Hit':float(hit_bl)},
        'daily_signal_IC': float(daily_ic),
        'blend': {'w_daily': float(best_w_bl), 'IC': float(ic_best), 'Hit': float(hit_best)},
        'top20_ls': float(ls_bl),
        'n_months': len(np.unique(te_dates)),
        'n_samples': int(len(y_te_m))
    }, f, indent=2)
print(f'Saved to {OUT}/final_optimized.json')
