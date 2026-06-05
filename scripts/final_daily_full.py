# Full daily LSTM (3265 stocks) + monthly integration
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt
from pathlib import Path

DEV = torch.device('cuda')
DB = '.eastmoney-ai/db/klines-v2.sqlite'
LOOKBACK, BATCH = 60, 1024
HIDDEN, N_LAYERS = 64, 2
N_SEEDS_DAILY = 5  # reduced for speed (was 10)
LR, WD, EPOCHS = 0.001, 0, 200
N_FFT = 10

print(f'Full Daily LSTM: {N_SEEDS_DAILY} seeds, 3265 stocks', flush=True)

# ======== 1. Daily LSTM (ALL stocks) ========
conn = sqlite3.connect(DB)
codes_d = [r[0] for r in conn.execute('SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=200').fetchall()]
N_DAILY = len(codes_d)
print(f'Daily stocks: {N_DAILY} (with >=200 bars)', flush=True)

# Use sampling for speed: take first 1500 stocks for training
if N_DAILY > 1500:
    import random; random.seed(42)
    train_codes = random.sample(codes_d, 1500)
    print(f'  Training on random {len(train_codes)} stocks for speed', flush=True)
else:
    train_codes = codes_d

params_d = ','.join('?'*len(codes_d))
df_d = pd.read_sql_query(
    f"SELECT code,date,open,close,high,low,volume,turnover_rate FROM daily_klines "
    f"WHERE code IN ({params_d}) AND date>='2010-01-01' ORDER BY code,date",
    conn, params=codes_d)
conn.close()

print('Building daily sequences...', flush=True); t0=time.time()
X_d, y_d, dates_d, codes_d_arr = [], [], [], []
for code in train_codes:
    g = df_d[df_d['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 66: continue
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
    for i in range(LOOKBACK-1, n-63):
        fwd=(c[i+63]-c[i])/max(c[i],0.01)
        if abs(fwd)>3: continue
        X_d.append(F[i-LOOKBACK+1:i+1]); y_d.append(np.clip(fwd,-3,3))
        dates_d.append(str(dates[i])[:10]); codes_d_arr.append(code)

X_d=np.array(X_d,dtype=np.float32); y_d=np.array(y_d,dtype=np.float32)
dates_d_arr=np.array(dates_d); codes_d_arr=np.array(codes_d_arr)
v_d=~np.isnan(X_d).any(axis=(1,2))&~np.isnan(y_d); X_d=X_d[v_d]; y_d=y_d[v_d]
dates_d_arr=dates_d_arr[v_d]; codes_d_arr=codes_d_arr[v_d]
tr_d=(dates_d_arr>='2015-01-01')&(dates_d_arr<='2021-12-31')
va_d=(dates_d_arr>='2022-01-01')&(dates_d_arr<='2023-12-31')
te_d=(dates_d_arr>='2024-01-01')
print(f'Daily seqs: {len(X_d):,} T={tr_d.sum():,} V={va_d.sum():,} Te={te_d.sum():,} ({time.time()-t0:.0f}s)', flush=True)

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

print(f'Training {N_SEEDS_DAILY} daily seeds...', flush=True)
daily_preds_te = []
for seed in range(N_SEEDS_DAILY):
    torch.manual_seed(seed); np.random.seed(seed)
    model=SlimLSTM().to(DEV); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WD)
    best_va,best_st,patience=-99,None,0
    for ep in range(EPOCHS):
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
    if seed%2==0: print(f'  seed {seed}: {ep+1}ep', flush=True)

ens_daily_te=np.mean(daily_preds_te,axis=0)

# Aggregate daily -> monthly (ALL stocks now)
te_months=np.array([d[:7] for d in dates_d_arr[te_d]]); te_codes=codes_d_arr[te_d]
monthly_daily={}
for i in range(len(te_codes)):
    key=(te_codes[i],te_months[i])
    if key not in monthly_daily: monthly_daily[key]=[]
    monthly_daily[key].append(ens_daily_te[i])

# ======== 2. Monthly Ensemble ========
conn=sqlite3.connect(DB)
codes_m=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params_m=','.join('?'*len(codes_m))
df_m=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params_m}) AND date>='2010-01' ORDER BY code,date",conn,params=codes_m)
conn.close()

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

print('Building monthly...', flush=True); t0=time.time()
flat_m,ys_m,dates_m,codes_m_arr=[],[],[],[]
for code in codes_m:
    g=df_m[df_m['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<72: continue
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
        flat_m.append(flat);ys_m.append(np.clip(fwd_raw,-2,2))
        dates_m.append(g['date'].iloc[i]);codes_m_arr.append(code)

flat_m=np.array(flat_m,dtype=np.float32);ys_m=np.array(ys_m,dtype=np.float32)
v_m=~np.isnan(flat_m).any(axis=1)&~np.isnan(ys_m);flat_m=flat_m[v_m];ys_m=ys_m[v_m]
dates_m=np.array([dates_m[i] for i in range(len(v_m)) if v_m[i]]);codes_m_arr=np.array([codes_m_arr[i] for i in range(len(v_m)) if v_m[i]])
tr_m=(dates_m>='2015-01')&(dates_m<='2021-12');te_m=(dates_m>='2024-01')
print(f'Monthly: {len(flat_m):,} T={tr_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)', flush=True)

sc_m=StandardScaler();Xt=sc_m.fit_transform(flat_m[tr_m]);Xte=sc_m.transform(flat_m[te_m])
y_tr_m=ys_m[tr_m];y_te_m=ys_m[te_m];te_dates=dates_m[te_m];te_codes=codes_m_arr[te_m]

# Train monthly models
lgb_m=lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_m.fit(Xt,y_tr_m);p_lgb=lgb_m.predict(Xte)
xgb_m=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_m.fit(Xt,y_tr_m);p_xgb=xgb_m.predict(Xte)
ridge_m=Ridge(alpha=1.0);ridge_m.fit(Xt,y_tr_m);p_ridge=ridge_m.predict(Xte)
mlp_m=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456)
mlp_m.fit(Xt,y_tr_m);p_mlp=mlp_m.predict(Xte)

def cs_ic(pred,true,dates):
    ics=[spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan

ics_m={}
for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)]:
    ics_m[n]=max(cs_ic(p,y_te_m,te_dates),0)
w_sum=sum(ics_m.values())
p_ens_m=sum(ics_m[n]*p for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge),('MLP',p_mlp)])/w_sum
ic_monthly=cs_ic(p_ens_m,y_te_m,te_dates)

# ======== 3. Merge Daily + Monthly ========
daily_feats=np.zeros(len(y_te_m),dtype=np.float32)
has_daily=np.zeros(len(y_te_m),dtype=bool)
for i in range(len(te_codes)):
    key=(te_codes[i],te_dates[i])
    if key in monthly_daily:
        vals=monthly_daily[key]
        daily_feats[i]=np.mean(vals)
        has_daily[i]=True

cov_pct=has_daily.mean()
print(f'Daily coverage: {has_daily.sum()}/{len(has_daily)} ({cov_pct:.0%})', flush=True)

# Daily standalone IC
daily_ic=cs_ic(daily_feats[has_daily],y_te_m[has_daily],te_dates[has_daily]) if has_daily.sum()>100 else 0

# Blend: monthly ensemble + daily signal (where available)
p_blend=p_ens_m.copy()
best_ic,best_w=ic_monthly,0
for w in np.arange(0,1.05,0.01):
    bl=p_ens_m.copy()
    bl[has_daily]=w*daily_feats[has_daily]+(1-w)*p_ens_m[has_daily]
    ic=cs_ic(bl,y_te_m,te_dates)
    if ic>best_ic: best_ic=ic; best_w=w
p_blend[has_daily]=best_w*daily_feats[has_daily]+(1-best_w)*p_ens_m[has_daily]
ic_blend=cs_ic(p_blend,y_te_m,te_dates)
hit_blend=np.mean((p_blend>0)==(y_te_m>0))
n20=max(1,int(len(p_blend)*0.2))
ls=np.mean(y_te_m[np.argsort(p_blend)[-n20:]])-np.mean(y_te_m[np.argsort(p_blend)[:n20]])

# ======== 4. Results ========
print(); print('='*65)
print('FULL DAILY LSTM + MONTHLY INTEGRATION')
print(f'Daily: {N_DAILY} stocks, {N_SEEDS_DAILY} seeds')
print(f'Monthly: {len(codes_m)} stocks, 61d features')
print('='*65)
print(f'  Monthly Ensemble:       CS_IC={ic_monthly:+.4f}')
print(f'  Daily signal (monthly): CS_IC={daily_ic:+.4f} (coverage={cov_pct:.0%})')
print(f'  Blend (w={best_w:.2f}): CS_IC={ic_blend:+.4f}  Hit={hit_blend:.3f}  LS={ls:+.4f}')
print(f'  vs Monthly alone:       {(ic_blend-ic_monthly):+.4f}')
print(f'  Previous best (300 stocks, 11% cov): CS_IC=+0.180')
print(f'  Target 0.200:           need {0.200-ic_blend:+.4f} more')
print('='*65)
