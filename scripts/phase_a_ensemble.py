# Phase A: Classification Ensemble (Focal Loss + 4 models + FFT + MC Dropout)
# Target: >70% direction accuracy on A-share monthly data
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, sqlite3, time, json, copy
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, xgboost as xgb

DEV = torch.device('cuda')
SEQ_LEN, BATCH = 60, 256
LR, WD, EPOCHS = 5e-4, 1e-5, 80
N_FFT, MC_SAMPLES = 10, 20
FOCAL_GAMMA = 2.0  # focal loss gamma
print(f'Phase A: Classification Ensemble')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Config: FFT={N_FFT} MC={MC_SAMPLES} FocalGamma={FOCAL_GAMMA}')

# ======== 1. Extended Data Pipeline ========
DB = '.eastmoney-ai/db/klines-v2.sqlite'
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute(
    'SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params = ','.join('?' * len(codes))
df = pd.read_sql_query(
    f"SELECT code,date,open,high,low,close,volume,turnover_rate "
    f"FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' "
    f"ORDER BY code,date", conn, params=codes)
conn.close()

def fft_features(prices):
    x = np.arange(len(prices)); trend = np.polyfit(x, prices, 1)
    detrended = prices - np.polyval(trend, x)
    fft = np.fft.rfft(detrended); amps = np.abs(fft); freqs = np.fft.rfftfreq(len(detrended))
    if len(amps) <= 1: return np.zeros(N_FFT*3, dtype=np.float32)
    pk = np.argsort(amps[1:])[::-1][:N_FFT] + 1
    feats = []
    for idx in pk:
        if idx < len(freqs): feats.extend([freqs[idx], amps[idx], np.angle(fft[idx])])
    while len(feats) < N_FFT*3: feats.extend([0,0,0])
    return np.array(feats[:N_FFT*3], dtype=np.float32)

print('Loading data...', flush=True); t0 = time.time()
seqs_lstm, labels_cls, ys_reg, dates_list = [], [], [], []
# Flat features for tree/MLP models
flat_feats_list = []

for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c)

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

    # Volume/turnover extended features
    vol_ma3 = pd.Series(v).rolling(3).mean().values
    vol_ma12 = pd.Series(v).rolling(12).mean().values

    for i in range(SEQ_LEN-1,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        fwd=np.clip(fwd_raw,-2,2)

        # LSTM sequence with FFT
        seq_d = np.zeros((SEQ_LEN,17+N_FFT*3),dtype=np.float32)
        seq_d[:,:17]=F[i-SEQ_LEN+1:i+1]
        seq_d[:,17:]=fft_features(c[i-SEQ_LEN+1:i+1])
        seqs_lstm.append(seq_d)

        # Flat features (for tree models): 17 tech + 30 FFT + 6 extended
        flat = list(F[i, :17])  # 17 tech indicators
        flat.extend(fft_features(c[i-SEQ_LEN+1:i+1]).tolist())  # 30 FFT features
        # Extended features
        flat.append(v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0)  # vol change
        flat.append(tr[i] if not np.isnan(tr[i]) else 0)  # turnover rate
        flat.append(tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0)  # to change
        flat.append((vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0)  # vol ratio
        flat.append(np.log1p(max(v[i],1)))  # log volume
        flat.append(np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0)  # log turnover
        flat_feats_list.append(flat)

        # Label: binary up/down for classification
        labels_cls.append(1 if fwd_raw > 0 else 0)
        ys_reg.append(fwd)
        dates_list.append(g['date'].iloc[i])

flat_feats = np.array(flat_feats_list, dtype=np.float32)
seqs_lstm = np.array(seqs_lstm, dtype=np.float32)
labels_cls = np.array(labels_cls, dtype=np.int64)
ys_reg = np.array(ys_reg, dtype=np.float32)

# Filter NaN
v = ~np.isnan(flat_feats).any(axis=1) & ~np.isnan(seqs_lstm).any(axis=(1,2)) & ~np.isnan(ys_reg)
flat_feats=flat_feats[v]; seqs_lstm=seqs_lstm[v]; labels_cls=labels_cls[v]; ys_reg=ys_reg[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])

tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12')
va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12')
te_m=(dates_arr>='2024-01')
print(f'Data: {len(flat_feats):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({flat_feats.shape[1]} flat + {seqs_lstm.shape[2]} seq features)', flush=True)
print(f'Class balance: UP={labels_cls.sum()/len(labels_cls):.1%} DOWN={1-labels_cls.sum()/len(labels_cls):.1%}', flush=True)
print(f'Train+Val time: {time.time()-t0:.0f}s', flush=True)

# Normalize flat features
sc = StandardScaler(); flat_tr = sc.fit_transform(flat_feats[tr_m])
flat_va = sc.transform(flat_feats[va_m]); flat_te = sc.transform(flat_feats[te_m])

# Normalize LSTM features
lstm_tr = seqs_lstm[tr_m]; lstm_fm = lstm_tr.reshape(-1,seqs_lstm.shape[2]).mean(0)
lstm_fs = lstm_tr.reshape(-1,seqs_lstm.shape[2]).std(0)+1e-8
seqs_lstm = np.clip((seqs_lstm-lstm_fm)/lstm_fs,-5,5)

# ======== 2. Define 4 Base Models ========

# Model 1&2: LightGBM + XGBoost (tree-based)
print('\n[1] Training LightGBM...', flush=True); t0=time.time()
lgb_clf = lgb.LGBMClassifier(objective='binary',num_leaves=63,learning_rate=0.03,
    n_estimators=300,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
    random_state=456,verbosity=-1,n_jobs=4)
lgb_clf.fit(flat_tr, labels_cls[tr_m])
lgb_p = lgb_clf.predict_proba(flat_va)[:,1]
lgb_acc = np.mean((lgb_p>0.5)==labels_cls[va_m])
print(f'  LightGBM val acc={lgb_acc:.3f} ({time.time()-t0:.0f}s)', flush=True)

print('[2] Training XGBoost...', flush=True); t0=time.time()
xgb_clf = xgb.XGBClassifier(objective='binary:logistic',max_depth=6,learning_rate=0.05,
    n_estimators=300,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=0,n_jobs=4)
xgb_clf.fit(flat_tr, labels_cls[tr_m])
xgb_p = xgb_clf.predict_proba(flat_va)[:,1]
xgb_acc = np.mean((xgb_p>0.5)==labels_cls[va_m])
print(f'  XGBoost val acc={xgb_acc:.3f} ({time.time()-t0:.0f}s)', flush=True)

# Model 3: MLP with Focal Loss
print('[3] Training MLP...', flush=True); t0=time.time()
class MLPClassifier(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128),nn.GELU(),nn.Dropout(0.3),
            nn.Linear(128,64),nn.GELU(),nn.Dropout(0.3),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x).squeeze(-1)

mlp = MLPClassifier(flat_feats.shape[1]).to(DEV)
opt_mlp = torch.optim.AdamW(mlp.parameters(),lr=5e-4,weight_decay=1e-4)
X_flat_tr = torch.from_numpy(flat_tr).float().to(DEV)
y_cls_tr = torch.from_numpy(labels_cls[tr_m].astype(np.float32)).float().to(DEV)
X_flat_va = torch.from_numpy(flat_va).float().to(DEV)
y_cls_va = labels_cls[va_m]

def focal_loss(logits, targets, gamma=FOCAL_GAMMA):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    return ((1-pt)**gamma * bce).mean()

best_mlp_acc, best_mlp = 0, None
for ep in range(60):
    mlp.train()
    perm=torch.randperm(len(X_flat_tr),device=DEV)
    for i in range(0,len(X_flat_tr),512):
        idx=perm[i:i+512]
        loss=focal_loss(mlp(X_flat_tr[idx]),y_cls_tr[idx])
        opt_mlp.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(),1.0)
        opt_mlp.step()
    mlp.eval()
    with torch.no_grad():
        p=torch.sigmoid(mlp(X_flat_va)).detach().cpu().numpy()
        acc=np.mean((p>0.5)==y_cls_va)
    if acc>best_mlp_acc: best_mlp_acc=acc; best_mlp=copy.deepcopy(mlp.state_dict())
mlp.load_state_dict(best_mlp)
mlp.eval()
with torch.no_grad():
    mlp_p = torch.sigmoid(mlp(X_flat_va)).detach().cpu().numpy()
print(f'  MLP val acc={best_mlp_acc:.3f} ({time.time()-t0:.0f}s)', flush=True)

# Model 4: LSTM with Classification + Focal Loss
print('[4] Training LSTM Classifier...', flush=True); t0=time.time()
X_lstm_tr = torch.from_numpy(seqs_lstm[tr_m]).float().to(DEV)
X_lstm_va = torch.from_numpy(seqs_lstm[va_m]).float().to(DEV)

class LSTMClassifier(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim,128,2,batch_first=True,dropout=0.3)
        self.ln = nn.LayerNorm(128)
        self.head = nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(0.3),nn.Linear(64,1))
    def forward(self,x):
        out,_=self.lstm(x)
        return self.head(self.ln(out[:,-1,:])).squeeze(-1)

lstm_clf = LSTMClassifier(seqs_lstm.shape[2]).to(DEV)
opt_lstm = torch.optim.AdamW(lstm_clf.parameters(),lr=3e-4,weight_decay=1e-5)
sched_lstm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lstm,T_max=60,eta_min=1e-5)
best_lstm_acc, best_lstm_state = 0, None

for ep in range(60):
    lstm_clf.train()
    perm=torch.randperm(len(X_lstm_tr),device=DEV)
    total_loss=0
    for i in range(0,len(X_lstm_tr),BATCH):
        idx=perm[i:i+BATCH]
        loss=focal_loss(lstm_clf(X_lstm_tr[idx]),y_cls_tr[idx])
        opt_lstm.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_clf.parameters(),1.0)
        opt_lstm.step(); total_loss+=loss.item()
    sched_lstm.step()
    lstm_clf.eval()
    with torch.no_grad():
        p_list = [torch.sigmoid(lstm_clf(X_lstm_va[i:i+BATCH])).detach().cpu().numpy() for i in range(0,len(X_lstm_va),BATCH)]
        p = np.concatenate(p_list)
        acc=np.mean((p>0.5)==y_cls_va)
    if acc>best_lstm_acc: best_lstm_acc=acc; best_lstm_state=copy.deepcopy(lstm_clf.state_dict())
lstm_clf.load_state_dict(best_lstm_state)
p_list = [torch.sigmoid(lstm_clf(X_lstm_va[i:i+BATCH])).detach().cpu().numpy() for i in range(0,len(X_lstm_va),BATCH)]
lstm_p = np.concatenate(p_list)
print(f'  LSTM val acc={best_lstm_acc:.3f} ({time.time()-t0:.0f}s)', flush=True)

# ======== 3. Stacking Meta-Learner ========
print('\n[5] Stacking ensemble...', flush=True)
# Train base models on full train, predict on validation
# Build stacking features from validation predictions
lgb_va = lgb_clf.predict_proba(flat_va)[:,1]
xgb_va = xgb_clf.predict_proba(flat_va)[:,1]
mlp_va = torch.sigmoid(mlp(X_flat_va)).detach().cpu().numpy()
p_list = [torch.sigmoid(lstm_clf(X_lstm_va[i:i+BATCH])).detach().cpu().numpy() for i in range(0,len(X_lstm_va),BATCH)]
lstm_va = np.concatenate(p_list)

stack_X_va = np.column_stack([lgb_va, xgb_va, mlp_va, lstm_va])
stack_y_va = labels_cls[va_m]

meta = LogisticRegression(C=1.0, random_state=456)
meta.fit(stack_X_va, stack_y_va)
stack_p_va = meta.predict_proba(stack_X_va)[:,1]
stack_acc_va = np.mean((stack_p_va>0.5)==stack_y_va)
# Simple average ensemble
avg_p_va = np.mean(stack_X_va, axis=1)
avg_acc_va = np.mean((avg_p_va>0.5)==stack_y_va)
print(f'  Individual: LGB={lgb_acc:.3f} XGB={xgb_acc:.3f} MLP={best_mlp_acc:.3f} LSTM={best_lstm_acc:.3f}')
print(f'  Average ensemble: {avg_acc_va:.3f}')
print(f'  Stacked ensemble: {stack_acc_va:.3f}')

# ======== 4. Test Evaluation ========
print('\n[6] Test evaluation...', flush=True)
# Base model predictions on test
lgb_te = lgb_clf.predict_proba(flat_te)[:,1]
xgb_te = xgb_clf.predict_proba(flat_te)[:,1]
X_flat_te = torch.from_numpy(flat_te).float().to(DEV)
mlp_te = torch.sigmoid(mlp(X_flat_te)).detach().cpu().numpy()
X_lstm_te = torch.from_numpy(seqs_lstm[te_m]).float().to(DEV)
# MC Dropout for LSTM
lstm_clf.train()
mc_preds = []
for _ in range(MC_SAMPLES):
    p_batch = [torch.sigmoid(lstm_clf(X_lstm_te[i:i+BATCH])).detach().cpu().numpy() for i in range(0,len(X_lstm_te),BATCH)]
mc_preds.append(np.concatenate(p_batch))
lstm_te_mc = np.mean(mc_preds, axis=0)

stack_X_te = np.column_stack([lgb_te, xgb_te, mlp_te, lstm_te_mc])
stack_p_te = meta.predict_proba(stack_X_te)[:,1]
avg_p_te = np.mean(stack_X_te, axis=1)

y_te_true = labels_cls[te_m]
y_te_reg = ys_reg[te_m]
te_dates = dates_arr[te_m]

def metrics(probs, label, prefix=''):
    pred = probs > 0.5
    acc = np.mean(pred == label)
    tp=np.sum(pred&label); fp=np.sum(pred&~label)
    tn=np.sum(~pred&~label); fn=np.sum(~pred&label)
    up_p = tp/max(tp+fp,1); dn_p = tn/max(tn+fn,1)
    f1 = 2*tp/max(2*tp+fp+fn,1)
    print(f'{prefix} Acc={acc:.4f} ({acc:.2%}) UPprec={up_p:.3f} DNprec={dn_p:.3f} F1={f1:.3f}')
    # Top-K
    for k in [10,20]:
        nk = max(1,int(len(probs)*k//100))
        t = np.argsort(probs)[-nk:]; b = np.argsort(probs)[:nk]
        ls = np.mean(y_te_reg[t]) - np.mean(y_te_reg[b])
        print(f'{prefix} Top-{k}% LS={ls:+.4f}')
    return acc

print(); print('='*65)
print('Phase A: Classification Ensemble Results')
print('='*65)
a1 = metrics(lgb_te, y_te_true, 'LGB    ')
a2 = metrics(xgb_te, y_te_true, 'XGB    ')
a3 = metrics(mlp_te, y_te_true, 'MLP    ')
a4 = metrics(lstm_te_mc, y_te_true, 'LSTM-MC')
a5 = metrics(avg_p_te, y_te_true, 'AVG-ENS')
a6 = metrics(stack_p_te, y_te_true, 'STACK  ')
print(f'\nBaseline LSTM v1: 56.56%')
print(f'Best ensemble:     {max(a5,a6):.2%}')
print(f'Improvement:       {max(a5,a6)-0.5656:+.2%}')
print('='*65)
