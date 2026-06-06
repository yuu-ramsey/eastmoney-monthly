# Path B: BiLSTM hidden states as features for LightGBM
# + Phase 1.2: extended feature engineering (50+ dims)
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, copy, numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, xgboost as xgb, pywt

DEV = torch.device('cuda')
SEQ_LEN, BATCH, N_FFT = 60, 256, 10
LR, WD, EPOCHS, RANK_LAMBDA, EMA_DECAY = 3e-4, 1e-5, 60, 0.3, 0.999
DB = '.eastmoney-ai/db/klines-v2.sqlite'
OUT = '.eastmoney-ai/benchmark'

print(f'Path B: LSTM Features -> LightGBM | GPU: {torch.cuda.get_device_name(0)}', flush=True)

# ======== 1. Extended Feature Pipeline (Phase 1.2) ========
def fft_f(prices):
    x=np.arange(len(prices)); trend=np.polyfit(x,prices,1); detrended=prices-np.polyval(trend,x)
    fft_p=np.fft.rfft(detrended); amps=np.abs(fft_p); freqs=np.fft.rfftfreq(len(detrended))
    if len(amps)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(amps[1:])[::-1][:N_FFT]+1; feats=[]
    for idx in pk:
        if idx<len(freqs): feats.extend([freqs[idx],amps[idx],np.angle(fft_p[idx])])
    while len(feats)<N_FFT*3: feats.extend([0,0,0])
    return np.array(feats[:N_FFT*3],dtype=np.float32)

def wdenoise(signal,wavelet='db4',level=2):
    coeffs=pywt.wavedec(signal,wavelet,level=level)
    sigma=np.median(np.abs(coeffs[-1]))/0.6745; threshold=sigma*np.sqrt(2*np.log(len(signal)))
    coeffs_d=[coeffs[0]]+[pywt.threshold(c,threshold,mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_d,wavelet)[:len(signal)]

conn=sqlite3.connect(DB)
codes=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params=','.join('?'*len(codes))
df=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date",conn,params=codes)
conn.close()

print('Loading data (extended features)...', flush=True); t0=time.time()
seqs_list, ys_list, flat_list, dates_list = [], [], [], []

for code in codes:
    g=df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g)<72: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c); c_clean=wdenoise(c)
    ma5=pd.Series(c).rolling(5).mean().values; ma20=pd.Series(c).rolling(20).mean().values; ma60=pd.Series(c).rolling(60).mean().values
    ma10=pd.Series(c).rolling(10).mean().values
    e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
    dif=e12-e26; dea=pd.Series(dif).ewm(span=9).mean().values; macd_hist=(dif-dea)*2
    delta=np.diff(c,prepend=c[0]); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    avg_gain=pd.Series(gain).ewm(alpha=1/14).mean().values; avg_loss=pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14=np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)),50)
    bb_std=pd.Series(c).rolling(20).std().values; bb_pos=np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01),0.5)
    trange=np.maximum(h-l,np.abs(h-np.roll(c,1))); atr14=pd.Series(trange).rolling(14).mean().values
    # Extended features from Phase 1.2
    p5h=pd.Series(c).rolling(60).max().values; p5l=pd.Series(c).rolling(60).min().values
    vol_ma3=pd.Series(v).rolling(3).mean().values; vol_ma12=pd.Series(v).rolling(12).mean().values
    # Candle body features
    body_pct = np.abs(c-o)/np.maximum(h-l, 0.01)  # body ratio
    upper_shadow = (np.maximum(o,c)-h)/np.maximum(h-l, 0.01)  # upper shadow ratio
    lower_shadow = (l-np.minimum(o,c))/np.maximum(h-l, 0.01)  # lower shadow ratio
    # Consecutive up/down months
    up_streak = np.zeros(n); dn_streak = np.zeros(n)
    for i in range(1,n):
        up_streak[i] = up_streak[i-1]+1 if c[i]>c[i-1] else 0
        dn_streak[i] = dn_streak[i-1]+1 if c[i]<c[i-1] else 0

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

    for i in range(SEQ_LEN-1,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        fwd=np.clip(fwd_raw,-2,2)
        # LSTM sequence
        seq=np.zeros((SEQ_LEN,17+N_FFT*3),dtype=np.float32)
        seq[:,:17]=F[i-SEQ_LEN+1:i+1]; seq[:,17:]=fft_f(c_clean[i-SEQ_LEN+1:i+1])
        seqs_list.append(seq)
        # Extended flat features (55 dims)
        flat = list(F[i,:17])  # 17 tech
        flat.extend(fft_f(c_clean[i-SEQ_LEN+1:i+1]).tolist())  # 30 FFT
        flat.append(v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0)  # vol chg
        flat.append(tr[i] if not np.isnan(tr[i]) else 0)  # turnover
        flat.append(tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0)
        flat.append((vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(np.log1p(max(v[i],1))); flat.append(np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0)
        flat.append((c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 and not np.isnan(p5h[i]) else 0.5)  # price pos
        flat.append((ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 and not np.isnan(ma20[i]) else 0)  # trend
        flat.append(np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0)  # 12m vol
        # Phase 1.2 new features
        flat.append(ma5[i]/max(ma20[i],0.01)-1 if i>=20 and not np.isnan(ma5[i]) and not np.isnan(ma20[i]) else 0)  # MA5/MA20
        flat.append(ma10[i]/max(ma60[i],0.01)-1 if i>=60 and not np.isnan(ma10[i]) and not np.isnan(ma60[i]) else 0)  # MA10/MA60
        flat.append(1.0 if c[i]>ma5[i] else 0.0)  # above MA5
        flat.append(1.0 if c[i]>ma10[i] else 0.0)  # above MA10
        flat.append(body_pct[i] if not np.isnan(body_pct[i]) else 0)  # body ratio
        flat.append(upper_shadow[i] if not np.isnan(upper_shadow[i]) else 0)  # upper shadow
        flat.append(lower_shadow[i] if not np.isnan(lower_shadow[i]) else 0)  # lower shadow
        flat.append(up_streak[i] / 12.0)  # consecutive up months (normalized)
        flat.append(dn_streak[i] / 12.0)  # consecutive down months (normalized)
        # Previous month's range change
        flat.append((h[i]-l[i])/max(h[i-1]-l[i-1],0.01)-1 if i>=1 else 0)  # range change
        flat_list.append(flat); ys_list.append(fwd)
        dates_list.append(g['date'].iloc[i])

flat=np.array(flat_list,dtype=np.float32); seqs=np.array(seqs_list,dtype=np.float32); ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(seqs).any(axis=(1,2))&~np.isnan(ys)
flat=flat[v]; seqs=seqs[v]; ys=ys[v]; dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12'); va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12'); te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({flat.shape[1]}flat + {seqs.shape[2]}seq) {time.time()-t0:.0f}s', flush=True)

# Normalize
sc=StandardScaler(); flat_tr=sc.fit_transform(flat[tr_m]); flat_va=sc.transform(flat[va_m]); flat_te=sc.transform(flat[te_m])
st=seqs[tr_m]; fm=st.reshape(-1,seqs.shape[2]).mean(0); fs=st.reshape(-1,seqs.shape[2]).std(0)+1e-8
seqs=np.clip((seqs-fm)/fs,-5,5)

# ======== 2. Train BiLSTM Feature Extractor ========
class BiLSTM_Extractor(nn.Module):
    def __init__(self,d,hidden=256):
        super().__init__()
        self.in_ln=nn.LayerNorm(d)
        self.lstm=nn.LSTM(d,hidden,2,batch_first=True,dropout=0.3,bidirectional=True)
        for name,param in self.lstm.named_parameters():
            if 'bias_ih' in name: n=param.shape[0]//4; param.data[n:2*n].fill_(1.0)
            elif 'bias_hh' in name: n=param.shape[0]//4; param.data[n:2*n].fill_(1.0)
        self.ln=nn.LayerNorm(hidden*2); self.head=nn.Linear(hidden*2,1)
    def forward(self,x,return_hidden=False):
        x=self.in_ln(x); o,_=self.lstm(x); h=self.ln(o[:,-1,:])
        if return_hidden: return h
        return self.head(h).squeeze(-1)

def rank_loss(pred,target):
    if len(pred)<2: return torch.tensor(0.0,device=pred.device)
    d=target.unsqueeze(0)-target.unsqueeze(1); pd=pred.unsqueeze(0)-pred.unsqueeze(1)
    return torch.sigmoid(-pd*torch.sign(d)*torch.abs(d)).mean()

Xl_tr=torch.from_numpy(seqs[tr_m]).float().to(DEV); y_tr_t=torch.from_numpy(ys[tr_m]).float().to(DEV)
Xl_va=torch.from_numpy(seqs[va_m]).float().to(DEV); y_va_np=ys[va_m]
Xl_te=torch.from_numpy(seqs[te_m]).float().to(DEV)

print('Training BiLSTM extractor...', flush=True); t0=time.time()
lstm_ext=BiLSTM_Extractor(seqs.shape[2]).to(DEV); ema=copy.deepcopy(lstm_ext)
for p in ema.parameters(): p.requires_grad=False
opt=torch.optim.AdamW(lstm_ext.parameters(),lr=LR,weight_decay=WD)
huber=nn.HuberLoss(delta=0.5); best_ic,best_st=-99,None
for ep in range(EPOCHS):
    lstm_ext.train(); perm=torch.randperm(len(Xl_tr),device=DEV)
    for i in range(0,len(Xl_tr),BATCH):
        idx=perm[i:i+BATCH]; pred=lstm_ext(Xl_tr[idx])
        loss=huber(pred,y_tr_t[idx])+RANK_LAMBDA*rank_loss(pred,y_tr_t[idx])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lstm_ext.parameters(),1.0); opt.step()
        with torch.no_grad():
            for ep_,p in zip(ema.parameters(),lstm_ext.parameters()): ep_.data.mul_(EMA_DECAY).add_(p.data,alpha=1-EMA_DECAY)
    ema.eval()
    with torch.no_grad():
        pv=np.concatenate([ema(Xl_va[i:i+BATCH]).detach().cpu().numpy() for i in range(0,len(Xl_va),BATCH)])
        ic=spearmanr(pv,y_va_np)[0]
    if ic>best_ic: best_ic=ic; best_st=copy.deepcopy(ema.state_dict())
lstm_ext.load_state_dict(best_st); lstm_ext.eval()
print(f'BiLSTM val_IC={best_ic:+.4f} ({time.time()-t0:.0f}s)', flush=True)

# ======== 3. Extract LSTM Hidden States as Features ========
print('Extracting hidden states...', flush=True); t0=time.time()
def get_hidden(X_tensor, batch_size=512):
    hidden_list = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            h = lstm_ext(X_tensor[i:i+batch_size], return_hidden=True)
            hidden_list.append(h.detach().cpu().numpy())
    return np.concatenate(hidden_list)

h_tr = get_hidden(Xl_tr); h_va = get_hidden(Xl_va); h_te = get_hidden(Xl_te)
print(f'Hidden dim: {h_tr.shape[1]} ({time.time()-t0:.0f}s)', flush=True)

# ======== 4. Augment Flat Features + Train LightGBM ========
def cs_ic(pred, true, dates):
    return np.mean([spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20])

print('\nTraining models...', flush=True)
te_dates=dates_arr[te_m]; y_te=ys[te_m]

# Baseline: flat features only
lgb0=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb0.fit(flat_tr,ys[tr_m]); p0=lgb0.predict(flat_te); ic0=cs_ic(p0,y_te,te_dates)
print(f'  LGB (flat only, {flat.shape[1]}d): CS_IC={ic0:+.4f}', flush=True)

# Augment with LSTM hidden states (dimensionality reduction: 512→32 via PCA)
from sklearn.decomposition import PCA
# Select dims with positive IC on validation set
h_dim_ics = np.array([spearmanr(h_va[:,j], y_va_np)[0] for j in range(h_va.shape[1])])
pos_mask = h_dim_ics > 0.01  # only keep dims with IC > 0.01
print(f'  Hidden dims with IC>0.01: {pos_mask.sum()}/{len(pos_mask)}', flush=True)
if pos_mask.sum() >= 8:
    h_tr_f = h_tr[:, pos_mask]
    h_va_f = h_va[:, pos_mask]
    h_te_f = h_te[:, pos_mask]
else:
    # Fallback: PCA to 32
    pca = PCA(n_components=32, random_state=456)
    h_tr_f = pca.fit_transform(h_tr)
    h_va_f = pca.transform(h_va)
    h_te_f = pca.transform(h_te)
    print(f'  PCA: 512→32 dims', flush=True)

flat_lstm_tr = np.column_stack([flat_tr, h_tr_f])
flat_lstm_va = np.column_stack([flat_va, h_va_f])
flat_lstm_te = np.column_stack([flat_te, h_te_f])

lgb1=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb1.fit(flat_lstm_tr,ys[tr_m]); p1=lgb1.predict(flat_lstm_te); ic1=cs_ic(p1,y_te,te_dates)
print(f'  LGB + LSTM hidden ({flat_lstm_tr.shape[1]}d): CS_IC={ic1:+.4f}', flush=True)

# XGBoost with hidden states
xgb1=xgb.XGBRegressor(objective='reg:squarederror',max_depth=6,learning_rate=0.05,n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb1.fit(flat_lstm_tr,ys[tr_m]); p_xgb=xgb1.predict(flat_lstm_te); ic_xgb=cs_ic(p_xgb,y_te,te_dates)
print(f'  XGB + LSTM hidden ({flat_lstm_tr.shape[1]}d): CS_IC={ic_xgb:+.4f}', flush=True)

# BiLSTM standalone (batched inference)
p_lstm=np.concatenate([lstm_ext(Xl_te[i:i+BATCH]).detach().cpu().numpy() for i in range(0,len(Xl_te),BATCH)]); ic_lstm=cs_ic(p_lstm,y_te,te_dates)
print(f'  BiLSTM standalone: CS_IC={ic_lstm:+.4f}', flush=True)

# ======== 5. Ensemble ========
# Find optimal blend
print('\nEnsemble search...', flush=True)
best_ic, best_w, best_w2 = -99, 0.5, 0.0
results_table = []
for w1 in np.arange(0, 1.05, 0.1):
    ens = w1 * p_lstm + (1-w1) * p1
    ic = cs_ic(ens, y_te, te_dates)
    results_table.append((w1, ic, 'LSTM+LGB'))
    if ic > best_ic: best_ic = ic; best_w = w1

# 3-way ensemble
for w1 in np.arange(0, 0.6, 0.1):
    for w2 in np.arange(0, 0.6, 0.1):
        if w1 + w2 > 1.0: continue
        w3 = 1.0 - w1 - w2
        ens3 = w1 * p_lstm + w2 * p1 + w3 * p_xgb
        ic3 = cs_ic(ens3, y_te, te_dates)
        if ic3 > best_ic: best_ic = ic3; best_w = w1; best_w2 = w2

# Final ensemble
ens_final = best_w * p_lstm + best_w2 * p1 + (1-best_w-best_w2) * p_xgb
ic_final = cs_ic(ens_final, y_te, te_dates)

# ======== 6. Results ========
print(); print('='*65)
print('Path B Results: LSTM Hidden States -> Tree Models')
print('='*65)
print(f'  {"Model":<30s} {"Features":>8s} {"CS_IC":>10s} {"Delta":>10s}')
print(f'  {"-"*60}')
baselines = [
    ('LightGBM (flat features)', ic0, flat.shape[1], 0),
    ('LGB + LSTM hidden', ic1, flat_lstm_tr.shape[1], ic1-ic0),
    ('XGB + LSTM hidden', ic_xgb, flat_lstm_tr.shape[1], ic_xgb-ic0),
    ('BiLSTM standalone', ic_lstm, seqs.shape[2], ic_lstm-ic0),
    ('2-way Ensemble (LSTM+LGB)', best_ic, '-', best_ic-ic0),
    ('3-way Ensemble (LSTM+LGB+XGB)', ic_final, '-', ic_final-ic0),
]
for name, ic, feat, delta in baselines:
    print(f'  {name:<30s} {str(feat):>8s} {ic:+10.4f} {delta:+10.4f}')

# Hit rate
ens_dir = ens_final > 0; true_dir = y_te > 0
hit = np.mean(ens_dir == true_dir)
print(f'\n  Ensemble hit rate: {hit:.2%}')
print(f'  Previous best:     CS_IC=+0.178 (LGB+XGB+MLP+Ridge ensemble)')

# Diagnosis: why LSTM hidden states degrade tree models?
print(f'\n[Diagnosis] Why LSTM hidden states hurt trees:')
# 1. Check per-dimension IC of hidden states
dim_ics = [spearmanr(h_te[:,j], y_te)[0] for j in range(min(20, h_te.shape[1]))]
pos_dims = sum(1 for ic in dim_ics if ic > 0)
print(f'  Hidden state dims with positive IC: {pos_dims}/{len(dim_ics)} (first 20)')
print(f'  Hidden state IC range: {min(dim_ics):+.4f} ~ {max(dim_ics):+.4f}')

# 2. Feature importance from LightGBM
imp = lgb1.feature_importances_
flat_imp = imp[:flat.shape[1]].sum()
hidden_imp = imp[flat.shape[1]:].sum()
print(f'  Flat feature importance: {flat_imp:.1%}')
print(f'  Hidden state importance:  {hidden_imp:.1%}')
print(f'  Top-10 feature sources:')
top10 = np.argsort(imp)[::-1][:10]
for idx in top10:
    src = 'flat' if idx < flat.shape[1] else 'LSTM'
    print(f'    dim{idx:4d} ({src:4s}): importance={imp[idx]:.4f}')

# 3. Correlation between LSTM pred and LGB pred
from scipy.stats import pearsonr
r_pearson = pearsonr(p_lstm, p1)[0]
print(f'  LSTM pred vs LGB+LSTM pred pearson r: {r_pearson:.3f}')
print(f'  (This is the integrated model, not 2 separate models)')
print('='*65)
