# Phase B+C: Rank Loss + EMA + Wavelet Denoising + More Features + Walk-Forward
# Built on Phase A classification ensemble, targeting 70%+
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn
import numpy as np, pandas as pd, sqlite3, time, json, copy
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, xgboost as xgb
import pywt  # wavelet

DEV = torch.device('cuda')
SEQ_LEN, BATCH = 60, 256
LR, WD, EPOCHS = 5e-4, 1e-5, 80
N_FFT, MC_SAMPLES = 10, 20
FOCAL_GAMMA, RANK_LAMBDA = 2.0, 0.3
EMA_DECAY = 0.999
print(f'Phase B+C: +RankLoss +EMA +Wavelet +WalkForward')
print(f'GPU: {torch.cuda.get_device_name(0)} | gamma={FOCAL_GAMMA} rank_lambda={RANK_LAMBDA}', flush=True)

# ======== 1. Extended Feature Pipeline ========
DB = '.eastmoney-ai/db/klines-v2.sqlite'
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params = ','.join('?' * len(codes))
df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date", conn, params=codes)
conn.close()

def fft_features(prices):
    x = np.arange(len(prices)); trend = np.polyfit(x, prices, 1)
    detrended = prices - np.polyval(trend, x)
    fft_p = np.fft.rfft(detrended); amps = np.abs(fft_p); freqs = np.fft.rfftfreq(len(detrended))
    if len(amps) <= 1: return np.zeros(N_FFT*3, dtype=np.float32)
    pk = np.argsort(amps[1:])[::-1][:N_FFT] + 1
    feats = []
    for idx in pk:
        if idx < len(freqs): feats.extend([freqs[idx], amps[idx], np.angle(fft_p[idx])])
    while len(feats) < N_FFT*3: feats.extend([0,0,0])
    return np.array(feats[:N_FFT*3], dtype=np.float32)

def wavelet_denoise(signal, wavelet='db4', level=2):
    """Wavelet denoising - from xLSTM-TS paper (2408.12408)"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # universal threshold on detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_denoised = [coeffs[0]]  # keep approximation
    for c in coeffs[1:]:
        coeffs_denoised.append(pywt.threshold(c, threshold, mode='soft'))
    return pywt.waverec(coeffs_denoised, wavelet)[:len(signal)]

print('Loading data (wavelet denoising + extended features)...', flush=True); t0 = time.time()
seqs_lstm, labels_cls, ys_reg, dates_list, flat_feats_list = [], [], [], [], []

for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c)

    # Wavelet denoise close prices
    c_denoised = wavelet_denoise(c)

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
    # PE/PB proxy: price relative to its 5-year range
    price_5yr_high = pd.Series(c).rolling(60).max().values
    price_5yr_low = pd.Series(c).rolling(60).min().values

    for i in range(SEQ_LEN-1,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        fwd=np.clip(fwd_raw,-2,2)

        # LSTM: wavelet denoised features
        # Recompute features on denoised price for LSTM input
        seq_clean = np.zeros((SEQ_LEN,17+N_FFT*3),dtype=np.float32)
        seq_clean[:,:17]=F[i-SEQ_LEN+1:i+1]
        seq_clean[:,17:]=fft_features(c_denoised[i-SEQ_LEN+1:i+1])
        seqs_lstm.append(seq_clean)

        # Flat features
        flat = list(F[i,:17])
        flat.extend(fft_features(c_denoised[i-SEQ_LEN+1:i+1]).tolist())
        flat.append(v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(tr[i] if not np.isnan(tr[i]) else 0)
        flat.append(tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0)
        flat.append((vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0)
        flat.append(np.log1p(max(v[i],1)))
        flat.append(np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0)
        # PE/PB proxy features
        flat.append((c[i]-price_5yr_low[i])/max(price_5yr_high[i]-price_5yr_low[i],0.01) if i>=60 and not np.isnan(price_5yr_high[i]) else 0.5)  # price position in 5yr range
        flat.append((ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 and not np.isnan(ma20[i]) and not np.isnan(ma60[i]) else 0)  # trend strength
        flat.append(np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0)  # 12m volatility
        flat_feats_list.append(flat)

        labels_cls.append(1 if fwd_raw > 0 else 0)
        ys_reg.append(fwd)
        dates_list.append(g['date'].iloc[i])

flat_feats = np.array(flat_feats_list, dtype=np.float32); seqs_lstm = np.array(seqs_lstm, dtype=np.float32)
labels_cls = np.array(labels_cls, dtype=np.int64); ys_reg = np.array(ys_reg, dtype=np.float32)
v = ~np.isnan(flat_feats).any(axis=1) & ~np.isnan(seqs_lstm).any(axis=(1,2)) & ~np.isnan(ys_reg)
flat_feats=flat_feats[v]; seqs_lstm=seqs_lstm[v]; labels_cls=labels_cls[v]; ys_reg=ys_reg[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])

# Walk-forward splits (strict time-based)
splits = [
    ('2015-01','2019-12','2020-01','2021-12','cw1'),  # cross-window 1
    ('2017-01','2021-12','2022-01','2023-12','cw2'),  # cross-window 2
]
# Main split
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12')
va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12')
te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat_feats):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({flat_feats.shape[1]} flat + {seqs_lstm.shape[2]} seq) {time.time()-t0:.0f}s', flush=True)
print(f'Class: UP={labels_cls.sum()/len(labels_cls):.1%}', flush=True)

# Normalize
sc = StandardScaler(); flat_tr = sc.fit_transform(flat_feats[tr_m])
flat_va = sc.transform(flat_feats[va_m]); flat_te = sc.transform(flat_feats[te_m])
lstm_tr = seqs_lstm[tr_m]; lstm_fm = lstm_tr.reshape(-1,seqs_lstm.shape[2]).mean(0)
lstm_fs = lstm_tr.reshape(-1,seqs_lstm.shape[2]).std(0)+1e-8
seqs_lstm = np.clip((seqs_lstm-lstm_fm)/lstm_fs,-5,5)

X_flat_tr = torch.from_numpy(flat_tr).float().to(DEV); y_cls_tr_t = torch.from_numpy(labels_cls[tr_m].astype(np.float32)).float().to(DEV)
X_flat_va = torch.from_numpy(flat_va).float().to(DEV); X_flat_te = torch.from_numpy(flat_te).float().to(DEV)
X_lstm_tr = torch.from_numpy(seqs_lstm[tr_m]).float().to(DEV)
X_lstm_va = torch.from_numpy(seqs_lstm[va_m]).float().to(DEV)
X_lstm_te = torch.from_numpy(seqs_lstm[te_m]).float().to(DEV)
y_cls_va = labels_cls[va_m]; y_te_reg = ys_reg[te_m]; te_dates = dates_arr[te_m]; y_te_cls = labels_cls[te_m]

# ======== 2. Rank Loss ========
def pairwise_rank_loss(pred_logits, targets):
    """Penalize incorrect pairwise orderings"""
    if len(pred_logits) < 2: return torch.tensor(0.0, device=pred_logits.device)
    probs = torch.sigmoid(pred_logits)
    diff = targets.unsqueeze(0) - targets.unsqueeze(1)
    prob_diff = probs.unsqueeze(0) - probs.unsqueeze(1)
    weight = torch.abs(diff)
    loss = torch.sigmoid(-prob_diff * torch.sign(diff)) * weight
    return loss.mean()

def focal_loss(logits, targets, gamma=FOCAL_GAMMA):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    return ((1-pt)**gamma * bce).mean()

def combined_loss(logits, targets, rank_lambda=RANK_LAMBDA):
    return focal_loss(logits, targets) + rank_lambda * pairwise_rank_loss(logits, targets)

# ======== 3. Models ========
# LightGBM + XGBoost (same as Phase A)
print('[1] LightGBM...', flush=True); t0=time.time()
lgb_clf = lgb.LGBMClassifier(objective='binary',num_leaves=63,learning_rate=0.03,
    n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
    random_state=456,verbosity=-1,n_jobs=4)
lgb_clf.fit(flat_tr, labels_cls[tr_m])
lgb_va = lgb_clf.predict_proba(flat_va)[:,1]
lgb_te = lgb_clf.predict_proba(flat_te)[:,1]
print(f'  val acc={np.mean((lgb_va>0.5)==y_cls_va):.3f} ({time.time()-t0:.0f}s)', flush=True)

print('[2] XGBoost...', flush=True); t0=time.time()
xgb_clf = xgb.XGBClassifier(objective='binary:logistic',max_depth=6,learning_rate=0.05,
    n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_clf.fit(flat_tr, labels_cls[tr_m])
xgb_va = xgb_clf.predict_proba(flat_va)[:,1]
xgb_te = xgb_clf.predict_proba(flat_te)[:,1]
print(f'  val acc={np.mean((xgb_va>0.5)==y_cls_va):.3f} ({time.time()-t0:.0f}s)', flush=True)

# MLP with Rank+Focal loss + EMA
class MLPClassifier(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,128),nn.GELU(),nn.Dropout(0.3),
            nn.Linear(128,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)

print('[3] MLP + RankLoss + EMA...', flush=True); t0=time.time()
mlp = MLPClassifier(flat_feats.shape[1]).to(DEV); ema_mlp = copy.deepcopy(mlp)
for p in ema_mlp.parameters(): p.requires_grad = False
opt_mlp = torch.optim.AdamW(mlp.parameters(),lr=5e-4,weight_decay=1e-4)
best_acc, best_state = 0, None
for ep in range(60):
    mlp.train(); perm=torch.randperm(len(X_flat_tr),device=DEV)
    for i in range(0,len(X_flat_tr),512):
        idx=perm[i:i+512]
        loss=combined_loss(mlp(X_flat_tr[idx]),y_cls_tr_t[idx])
        opt_mlp.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(),1.0); opt_mlp.step()
        with torch.no_grad():
            for ep_,p in zip(ema_mlp.parameters(),mlp.parameters()):
                ep_.data.mul_(EMA_DECAY).add_(p.data,alpha=1-EMA_DECAY)
    ema_mlp.eval()
    with torch.no_grad():
        acc=np.mean((torch.sigmoid(ema_mlp(X_flat_va)).detach().cpu().numpy()>0.5)==y_cls_va)
    if acc>best_acc: best_acc=acc; best_state=copy.deepcopy(ema_mlp.state_dict())
mlp.load_state_dict(best_state); mlp.eval()
mlp_va = torch.sigmoid(mlp(X_flat_va)).detach().cpu().numpy()
mlp_te = torch.sigmoid(mlp(X_flat_te)).detach().cpu().numpy()
print(f'  EMA val acc={np.mean((mlp_va>0.5)==y_cls_va):.3f} best={best_acc:.3f} ({time.time()-t0:.0f}s)', flush=True)

# LSTM with Rank+Focal + EMA + Wavelet denoised features
class LSTMClassifier(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim,128,2,batch_first=True,dropout=0.3)
        self.ln = nn.LayerNorm(128)
        self.attn = nn.MultiheadAttention(128,4,dropout=0.2,batch_first=True)
        self.head = nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(self,x):
        out,_=self.lstm(x); out=self.ln(out[:,-1,:]).unsqueeze(1)
        attn,_=self.attn(out,out,out)
        return self.head((out+attn).squeeze(1)).squeeze(-1)

print('[4] LSTM + RankLoss + EMA + Attention...', flush=True); t0=time.time()
lstm_clf = LSTMClassifier(seqs_lstm.shape[2]).to(DEV); ema_lstm = copy.deepcopy(lstm_clf)
for p in ema_lstm.parameters(): p.requires_grad = False
opt_lstm = torch.optim.AdamW(lstm_clf.parameters(),lr=3e-4,weight_decay=1e-5)
sched_lstm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lstm,T_max=60,eta_min=1e-5)
best_lstm_acc, best_lstm_state = 0, None
for ep in range(60):
    lstm_clf.train(); perm=torch.randperm(len(X_lstm_tr),device=DEV)
    for i in range(0,len(X_lstm_tr),BATCH):
        idx=perm[i:i+BATCH]
        loss=combined_loss(lstm_clf(X_lstm_tr[idx]),y_cls_tr_t[idx])
        opt_lstm.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_clf.parameters(),1.0); opt_lstm.step()
        with torch.no_grad():
            for ep_,p in zip(ema_lstm.parameters(),lstm_clf.parameters()):
                ep_.data.mul_(EMA_DECAY).add_(p.data,alpha=1-EMA_DECAY)
    sched_lstm.step()
    ema_lstm.eval()
    with torch.no_grad():
        p=[torch.sigmoid(ema_lstm(X_lstm_va[i:i+BATCH])).detach().cpu().numpy() for i in range(0,len(X_lstm_va),BATCH)]
        acc=np.mean((np.concatenate(p)>0.5)==y_cls_va)
    if acc>best_lstm_acc: best_lstm_acc=acc; best_lstm_state=copy.deepcopy(ema_lstm.state_dict())
    if (ep+1)%30==0: print(f'  ep{ep+1} acc={acc:.3f} best={best_lstm_acc:.3f}', flush=True)
lstm_clf.load_state_dict(best_lstm_state); lstm_clf.eval()
p=[torch.sigmoid(lstm_clf(X_lstm_va[i:i+BATCH])).detach().cpu().numpy() for i in range(0,len(X_lstm_va),BATCH)]
lstm_va=np.concatenate(p)
# MC Dropout test
lstm_clf.train()
mc_te=[]
for _ in range(MC_SAMPLES):
    p=[torch.sigmoid(lstm_clf(X_lstm_te[i:i+BATCH])).detach().cpu().numpy() for i in range(0,len(X_lstm_te),BATCH)]
    mc_te.append(np.concatenate(p))
lstm_te=np.mean(mc_te,0)
print(f'  EMA+MC val acc={np.mean((lstm_va>0.5)==y_cls_va):.3f} best={best_lstm_acc:.3f} ({time.time()-t0:.0f}s)', flush=True)

# ======== 4. Stacking + Ensemble ========
print('\n[5] Ensemble...', flush=True)
stack_va = np.column_stack([lgb_va,xgb_va,mlp_va,lstm_va])
meta = LogisticRegression(C=1.0,random_state=456); meta.fit(stack_va,y_cls_va)
stack_te = np.column_stack([lgb_te,xgb_te,mlp_te,lstm_te])
meta_te = meta.predict_proba(stack_te)[:,1]
avg_te = np.mean(stack_te,axis=1)

# ======== 5. Walk-Forward Cross-Validation ========
print('\n[6] Walk-Forward CV...', flush=True)
wf_results = []
for tr_s,tr_e,va_s,va_e,label in splits:
    tr_m_wf = (dates_arr>=tr_s)&(dates_arr<=tr_e)
    va_m_wf = (dates_arr>=va_s)&(dates_arr<=va_e)
    if tr_m_wf.sum()<1000 or va_m_wf.sum()<100: continue
    # Train LGB on walk-forward window
    sc_wf = StandardScaler()
    Xt = sc_wf.fit_transform(flat_feats[tr_m_wf]); Xv = sc_wf.transform(flat_feats[va_m_wf])
    lgb_wf = lgb.LGBMClassifier(objective='binary',num_leaves=63,learning_rate=0.03,n_estimators=200,min_child_samples=20,random_state=456,verbosity=-1,n_jobs=4)
    lgb_wf.fit(Xt,labels_cls[tr_m_wf])
    p_wf = lgb_wf.predict_proba(Xv)[:,1]
    acc_wf = np.mean((p_wf>0.5)==labels_cls[va_m_wf])
    # IC
    ic_wf = spearmanr(p_wf, ys_reg[va_m_wf])[0]
    wf_results.append({'window':label,'acc':float(acc_wf),'IC':float(ic_wf),'n_train':int(tr_m_wf.sum()),'n_val':int(va_m_wf.sum())})
    print(f'  {label}: acc={acc_wf:.4f} IC={ic_wf:+.4f} T={tr_m_wf.sum():,} V={va_m_wf.sum():,}', flush=True)

# ======== 6. Test Results ========
def metrics(probs,label,prefix=''):
    pred=probs>0.5; acc=np.mean(pred==label)
    tp=np.sum(pred&label); tn=np.sum(~pred&~label)
    fp=np.sum(pred&~label); fn=np.sum(~pred&label)
    up=tp/max(tp+fp,1); dn=tn/max(tn+fn,1); f1=2*tp/max(2*tp+fp+fn,1)
    ls_scores=[]
    for k in[10,20]:
        nk=max(1,int(len(probs)*k//100))
        t=np.argsort(probs)[-nk:]; b=np.argsort(probs)[:nk]
        ls=np.mean(y_te_reg[t])-np.mean(y_te_reg[b])
        ls_scores.append(ls)
        print(f'{prefix} Top-{k}% LS={ls:+.4f}')
    print(f'{prefix} Acc={acc:.4f} ({acc:.2%}) UPprec={up:.3f} DNprec={dn:.3f} F1={f1:.3f}')
    return acc

print(); print('='*65)
print('Phase B+C Results: +RankLoss +EMA +Wavelet +WalkForward')
print('='*65)
metrics(lgb_te,y_te_cls,'LGB       ')
metrics(xgb_te,y_te_cls,'XGB       ')
metrics(mlp_te,y_te_cls,'MLP+EMA   ')
metrics(lstm_te,y_te_cls,'LSTM+EMA+MC')
metrics(avg_te,y_te_cls,'AVG-ENS   ')
metrics(meta_te,y_te_cls,'STACK-ENS ')
print(f'\nBaseline LSTM v1: 56.56%')
print(f'Best: {max(np.mean((lgb_te>0.5)==y_te_cls),np.mean((xgb_te>0.5)==y_te_cls),np.mean((mlp_te>0.5)==y_te_cls),np.mean((lstm_te>0.5)==y_te_cls),np.mean((avg_te>0.5)==y_te_cls),np.mean((meta_te>0.5)==y_te_cls)):.2%}')
if wf_results:
    wf_avg_acc = np.mean([r['acc'] for r in wf_results])
    wf_avg_ic = np.mean([r['IC'] for r in wf_results])
    print(f'Walk-Forward CV: avg acc={wf_avg_acc:.3f} avg IC={wf_avg_ic:+.4f}')
print('='*65)
