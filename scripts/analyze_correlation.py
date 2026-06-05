# Analyze BiLSTM vs LightGBM correlation and ensemble benefit
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, copy, numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, pywt

DEV = torch.device('cuda')
SEQ_LEN, BATCH, N_FFT = 60, 256, 10
LR, WD, EPOCHS, RANK_LAMBDA, EMA_DECAY = 3e-4, 1e-5, 60, 0.3, 0.999
DB = '.eastmoney-ai/db/klines-v2.sqlite'

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

def rank_loss(pred,target):
    if len(pred)<2: return torch.tensor(0.0,device=pred.device)
    d=target.unsqueeze(0)-target.unsqueeze(1); pd=pred.unsqueeze(0)-pred.unsqueeze(1)
    return torch.sigmoid(-pd*torch.sign(d)*torch.abs(d)).mean()

class BiLSTM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.in_ln = nn.LayerNorm(d)
        self.lstm = nn.LSTM(d, 256, 2, batch_first=True, dropout=0.3, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if 'bias_ih' in name:
                n = param.shape[0] // 4
                param.data[n:2*n].fill_(1.0)
            elif 'bias_hh' in name:
                n = param.shape[0] // 4
                param.data[n:2*n].fill_(1.0)
        self.ln = nn.LayerNorm(512)
        self.head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        x = self.in_ln(x)
        o, _ = self.lstm(x)
        return self.head(self.ln(o[:, -1, :])).squeeze(-1)

# ---- Data ----
conn=sqlite3.connect(DB)
codes=[r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params=','.join('?'*len(codes))
df=pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date",conn,params=codes)
conn.close()

print('Loading data...', flush=True); t0=time.time()
seqs_list,ys_list,flat_list,dates_list=[],[],[],[]
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
    vol_ma3=pd.Series(v).rolling(3).mean().values; vol_ma12=pd.Series(v).rolling(12).mean().values
    p5h=pd.Series(c).rolling(60).max().values; p5l=pd.Series(c).rolling(60).min().values
    for i in range(SEQ_LEN-1,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        fwd=np.clip(fwd_raw,-2,2)
        seq=np.zeros((SEQ_LEN,17+N_FFT*3),dtype=np.float32)
        seq[:,:17]=F[i-SEQ_LEN+1:i+1]; seq[:,17:]=fft_f(c_clean[i-SEQ_LEN+1:i+1])
        seqs_list.append(seq)
        flat=list(F[i,:17]); flat.extend(fft_f(c_clean[i-SEQ_LEN+1:i+1]).tolist())
        flat.append(v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(tr[i] if not np.isnan(tr[i]) else 0)
        flat.append(tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0)
        flat.append((vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(np.log1p(max(v[i],1))); flat.append(np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0)
        flat.append((c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 and not np.isnan(p5h[i]) else 0.5)
        flat.append((ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 and not np.isnan(ma20[i]) else 0)
        flat.append(np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0)
        flat_list.append(flat); ys_list.append(fwd); dates_list.append(g['date'].iloc[i])

flat=np.array(flat_list,dtype=np.float32); seqs=np.array(seqs_list,dtype=np.float32); ys=np.array(ys_list,dtype=np.float32)
v=~np.isnan(flat).any(axis=1)&~np.isnan(seqs).any(axis=(1,2))&~np.isnan(ys)
flat=flat[v]; seqs=seqs[v]; ys=ys[v]; dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12'); va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12'); te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)', flush=True)

# Normalize
sc=StandardScaler(); flat_tr=sc.fit_transform(flat[tr_m]); flat_te=sc.transform(flat[te_m])
st=seqs[tr_m]; fm=st.reshape(-1,seqs.shape[2]).mean(0); fs=st.reshape(-1,seqs.shape[2]).std(0)+1e-8
seqs=np.clip((seqs-fm)/fs,-5,5)
Xl_tr=torch.from_numpy(seqs[tr_m]).float().to(DEV); y_tr_t=torch.from_numpy(ys[tr_m]).float().to(DEV)
Xl_te=torch.from_numpy(seqs[te_m]).float().to(DEV)
y_te_np=ys[te_m]; te_dates=dates_arr[te_m]

# ---- Train BiLSTM ----
print('Training BiLSTM...', flush=True); t0=time.time()
bilstm=BiLSTM(seqs.shape[2]).to(DEV); ema=copy.deepcopy(bilstm)
for p in ema.parameters(): p.requires_grad=False
opt=torch.optim.AdamW(bilstm.parameters(),lr=LR,weight_decay=WD)
huber=nn.HuberLoss(delta=0.5); best_ic,best_st=-99,None
for ep in range(EPOCHS):
    bilstm.train(); perm=torch.randperm(len(Xl_tr),device=DEV)
    for i in range(0,len(Xl_tr),BATCH):
        idx=perm[i:i+BATCH]; pred=bilstm(Xl_tr[idx])
        loss=huber(pred,y_tr_t[idx])+RANK_LAMBDA*rank_loss(pred,y_tr_t[idx])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(bilstm.parameters(),1.0); opt.step()
        with torch.no_grad():
            for ep_,p in zip(ema.parameters(),bilstm.parameters()): ep_.data.mul_(EMA_DECAY).add_(p.data,alpha=1-EMA_DECAY)
    bilstm.eval()
    with torch.no_grad():
        ic=spearmanr(ema(Xl_tr[:5000]).detach().cpu().numpy(),ys[tr_m][:5000])[0]
    if ic>best_ic: best_ic=ic; best_st=copy.deepcopy(ema.state_dict())
bilstm.load_state_dict(best_st); bilstm.eval()
p_bilstm=np.concatenate([bilstm(Xl_te[i:i+BATCH]).detach().cpu().numpy() for i in range(0,len(Xl_te),BATCH)])

def cs_ic(pred,true):
    return np.mean([spearmanr(pred[te_dates==m],true[te_dates==m])[0] for m in np.unique(te_dates) if (te_dates==m).sum()>=20])

print(f'BiLSTM CS_IC={cs_ic(p_bilstm,y_te_np):+.4f} ({time.time()-t0:.0f}s)', flush=True)

# ---- Train LightGBM ----
print('Training LightGBM...', flush=True); t0=time.time()
lgb_reg=lgb.LGBMRegressor(objective='regression',metric='l1',num_leaves=63,learning_rate=0.03,
    n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4)
lgb_reg.fit(flat_tr,ys[tr_m]); p_lgb=lgb_reg.predict(flat_te)
print(f'LightGBM CS_IC={cs_ic(p_lgb,y_te_np):+.4f} ({time.time()-t0:.0f}s)', flush=True)

# ---- Analysis ----
SEP = '=' * 60
print(); print(SEP)
print('Model Correlation & Ensemble Analysis')
print(SEP)

# 1. Rank correlation
rank_corr, rank_p = spearmanr(p_bilstm, p_lgb)
pearson_c, pearson_p = pearsonr(p_bilstm, p_lgb)
print(f'BiLSTM vs LightGBM rank correlation: {rank_corr:.4f} (p={rank_p:.2e})')
print(f'BiLSTM vs LightGBM pearson correlation: {pearson_c:.4f}')

# 2. Ensemble weights
print('\nEnsemble IC by weight:')
best_ens_ic, best_w = -99, 0
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ens = w * p_bilstm + (1-w) * p_lgb
    ic = cs_ic(ens, y_te_np)
    tag = ' <-- BEST' if ic > best_ens_ic else ''
    if ic > best_ens_ic: best_ens_ic = ic; best_w = w
    print(f'  w={w:.1f} BiLSTM + {(1-w):.1f} LGB: CS_IC={ic:+.4f}{tag}')

# 3. Residual analysis
residual = p_bilstm - p_lgb
res_ic = spearmanr(residual, y_te_np)[0]
print(f'\nResidual (BiLSTM - LGB) standalone IC: {res_ic:+.4f}')
print(f'(Positive = BiLSTM captures unique signal not in LGB)')

# 4. Monthly correlation stability
monthly_corrs = []
for m in np.unique(te_dates):
    mask = te_dates == m
    if mask.sum() >= 20:
        monthly_corrs.append(spearmanr(p_bilstm[mask], p_lgb[mask])[0])
monthly_corrs = np.array(monthly_corrs)
print(f'\nMonthly rank correlation stability:')
print(f'  Mean={np.mean(monthly_corrs):.3f}  Std={np.std(monthly_corrs):.3f}')
print(f'  Min={np.min(monthly_corrs):.3f}  Max={np.max(monthly_corrs):.3f}')

# 5. Verdict
print(); print(SEP)
if rank_corr < 0.6:
    v = 'HIGH ENSEMBLE BENEFIT — models capture different signals'
elif rank_corr < 0.8:
    v = 'MODERATE ENSEMBLE BENEFIT — some overlap but worth combining'
else:
    v = 'LOW ENSEMBLE BENEFIT — models are redundant'
print(f'Rank correlation: {rank_corr:.4f} -> {v}')
print(f'Best ensemble CS_IC: {best_ens_ic:+.4f} (w={best_w:.1f} BiLSTM)')
print(f'vs standalone: BiLSTM={cs_ic(p_bilstm,y_te_np):+.4f} LightGBM={cs_ic(p_lgb,y_te_np):+.4f}')
print(SEP)
