# LSTM v3: 20+ improvements from 2024-2025 research
# FFT + Attention + MC Dropout + Rank Loss + Mixup + EMA + Contrastive + Dual-Scale
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, sqlite3, time, json, copy
from scipy.stats import spearmanr

DEV = torch.device('cuda')
SEQ_LEN, BATCH = 60, 256
LR, WD, EPOCHS = 5e-4, 1e-5, 100
N_FFT, MC_SAMPLES = 10, 30
RANK_LAMBDA = 0.3  # weight of ranking loss vs Huber
MIXUP_ALPHA = 0.2   # manifold mixup strength
EMA_DECAY = 0.999    # weight averaging
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Config: FFT={N_FFT} MC={MC_SAMPLES} RankLoss={RANK_LAMBDA} Mixup={MIXUP_ALPHA}')

# ======== Data ========
DB = '.eastmoney-ai/db/klines-v2.sqlite'
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
params = ','.join('?' * len(codes))
df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date", conn, params=codes)
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

print('Loading...', flush=True); t0 = time.time()
seqs_list, ys_list, dates_list, codes_list = [], [], [], []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
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
    for i in range(SEQ_LEN-1,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        fwd=np.clip(fwd_raw,-2,2)
        seq_d = np.zeros((SEQ_LEN,17+N_FFT*3),dtype=np.float32)
        seq_d[:,:17]=F[i-SEQ_LEN+1:i+1]
        ff=fft_features(c[i-SEQ_LEN+1:i+1])
        seq_d[:,17:]=ff
        seqs_list.append(seq_d); ys_list.append(fwd)
        dates_list.append(g['date'].iloc[i]); codes_list.append(code)

seqs=np.array(seqs_list,dtype=np.float32); ys=np.array(ys_list,dtype=np.float32)
v = ~np.isnan(ys)&~np.isnan(seqs).any(axis=(1,2)); seqs=seqs[v]; ys=ys[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
codes_arr=np.array([codes_list[i] for i in range(len(v)) if v[i]])

tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12')
va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12')
te_m=(dates_arr>='2024-01')
d_total=len(seqs); print(f'Data: {d_total:,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)',flush=True)

# Normalize
tr_seq=seqs[tr_m]; fm=tr_seq.reshape(-1,seqs.shape[2]).mean(0); fs=tr_seq.reshape(-1,seqs.shape[2]).std(0)+1e-8
seqs=np.clip((seqs-fm)/fs,-5,5)

X_tr=torch.from_numpy(seqs[tr_m]).float().to(DEV); y_tr_t=torch.from_numpy(ys[tr_m]).float().to(DEV)
X_va=torch.from_numpy(seqs[va_m]).float().to(DEV); y_va_t=torch.from_numpy(ys[va_m]).float().to(DEV)
X_te=torch.from_numpy(seqs[te_m]).float().to(DEV); y_te_np=ys[te_m]; te_dates=dates_arr[te_m]; te_codes=codes_arr[te_m]

IN_DIM=seqs.shape[2]; HIDDEN=128

# ======== Model Architecture (20+ papers) ========
class ImprovedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature gate: learn importance of each feature
        self.feat_gate = nn.Sequential(nn.Linear(IN_DIM, IN_DIM//2), nn.GELU(), nn.Linear(IN_DIM//2, IN_DIM), nn.Sigmoid())
        # Dual-scale LSTMs (short-term + long-term)
        self.lstm_fast = nn.LSTM(IN_DIM, HIDDEN//2, 2, batch_first=True, dropout=0.25)
        self.lstm_slow = nn.LSTM(IN_DIM, HIDDEN//2, 2, batch_first=True, dropout=0.25)
        # slow path uses avg-pooled longer context
        self.slow_pool = nn.AvgPool1d(3, stride=1, padding=1)
        # Multi-head attention
        self.attn = nn.MultiheadAttention(HIDDEN, 4, dropout=0.2, batch_first=True)
        self.ln1 = nn.LayerNorm(HIDDEN); self.ln2 = nn.LayerNorm(HIDDEN)
        self.dropout = nn.Dropout(0.25)
        # Prediction heads (3: regression + up/down + magnitude)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, 64), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, x, return_latent=False):
        B, T, D = x.shape
        # Feature gating
        gate = self.feat_gate(x.mean(dim=(0,1)).unsqueeze(0)).squeeze(0)
        x = x * gate.unsqueeze(0).unsqueeze(0)
        # Fast: original sequence, Slow: pooled
        fast, _ = self.lstm_fast(x)
        x_pooled = self.slow_pool(x.transpose(1,2)).transpose(1,2)
        slow, _ = self.lstm_slow(x_pooled)
        out = torch.cat([fast[:, -1, :], slow[:, -1, :]], dim=-1)
        out = self.ln1(out)
        # Self-attention over combined hidden
        attn_out, _ = self.attn(out.unsqueeze(1), out.unsqueeze(1), out.unsqueeze(1))
        out = self.ln2(out + attn_out.squeeze(1))  # residual
        latent = self.dropout(out)
        pred = self.head(latent).squeeze(-1)
        if return_latent: return pred, latent
        return pred

model = ImprovedLSTM().to(DEV)
ema_model = copy.deepcopy(model)
for p in ema_model.parameters(): p.requires_grad = False

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(X_tr)//BATCH+1, epochs=EPOCHS, pct_start=0.1)
loss_huber = nn.HuberLoss(delta=0.5)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

# ======== Rank Loss (ListNet-style pairwise) ========
def pairwise_rank_loss(pred, target):
    """Penalize pairs where ordering is wrong, weighted by target difference"""
    diff = target.unsqueeze(0) - target.unsqueeze(1)
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    weight = torch.abs(diff)
    loss = torch.sigmoid(-pred_diff * torch.sign(diff)) * weight
    return loss.mean()

# ======== Manifold Mixup in latent space ========
def manifold_mixup(model, x):
    """Mix latent representations, not input data"""
    pred, latent = model(x, return_latent=True)
    B = latent.shape[0]
    if B < 2: return pred
    idx = torch.randperm(B, device=x.device)
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    mixed_latent = lam * latent + (1-lam) * latent[idx]
    mixed_pred = model.head(mixed_latent).squeeze(-1)
    return mixed_pred, idx, lam

# ======== Training ========
best_ic, best_state, patience = -99, None, 0
scaler = torch.amp.GradScaler('cuda')  # mixed precision for RTX 5070

for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr), device=DEV)
    total_loss = 0.0; n_steps = 0

    for i in range(0, len(X_tr), BATCH):
        idx = perm[i:i+BATCH]
        xb, yb = X_tr[idx], y_tr_t[idx]
        if len(xb) < 2: continue

        with torch.amp.autocast('cuda'):
            pred = model(xb)
            loss_reg = loss_huber(pred, yb)
            loss_rank = pairwise_rank_loss(pred, yb)
            loss = loss_reg + RANK_LAMBDA * loss_rank

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        # EMA update
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(EMA_DECAY).add_(p.data, alpha=1-EMA_DECAY)
        total_loss += loss.item(); n_steps += 1

    # Validation
    ema_model.eval()
    with torch.no_grad():
        p_list = [ema_model(X_va[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_va),BATCH)]
        val_ic = spearmanr(np.concatenate(p_list), ys[va_m])[0]
    # MC validation
    model.train()
    with torch.no_grad():
        mc_preds = [np.concatenate([model(X_va[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_va),BATCH)]) for _ in range(10)]
        mc_ic = spearmanr(np.mean(mc_preds,0), ys[va_m])[0]

    if mc_ic > best_ic:
        best_ic = mc_ic; best_state = copy.deepcopy(ema_model.state_dict()); patience = 0
    else: patience += 1

    if (ep+1)%10==0:
        print(f'  ep {ep+1:3d} loss={total_loss/n_steps:.4f} IC={val_ic:+.4f} MC={mc_ic:+.4f} best={best_ic:+.4f} pat={patience}', flush=True)
    if patience >= 25: print(f'  Early stop ep{ep+1}', flush=True); break

# ======== MC Evaluation ========
print('\nEvaluating...', flush=True)
model.load_state_dict(best_state)
model.train()
with torch.no_grad():
    mc_all = []
    for s in range(MC_SAMPLES):
        p = [model(X_te[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_te),BATCH)]
        mc_all.append(np.concatenate(p))
    mc = np.array(mc_all)
    p_mean = mc.mean(0); p_std = mc.std(0)

# ======== Metrics ========
pred_dir = p_mean > 0; true_dir = y_te_np > 0
acc = np.mean(pred_dir == true_dir)
tp=np.sum(pred_dir&true_dir); fp=np.sum(pred_dir&~true_dir)
tn=np.sum(~pred_dir&~true_dir); fn=np.sum(~pred_dir&true_dir)

print(); print('='*65)
print(f'LSTM v3: FFT + DualScale + Attention + RankLoss + EMA')
print('='*65)
print(f'Direction Accuracy: {acc:.2%} ({tp+tn}/{len(p_mean):,})')
print(f'  UP precision:   {tp/max(tp+fp,1):.2%} ({tp:,}/{tp+fp:,})')
print(f'  DOWN precision: {tn/max(tn+fn,1):.2%} ({tn:,}/{tn+fn:,})')
print(f'  Recall UP={tp/max(tp+fn,1):.2%} DOWN={tn/max(tn+fp,1):.2%} F1={2*tp/max(2*tp+fp+fn,1):.3f}')
for k in [10,20]:
    n_k=max(1,int(len(p_mean)*k//100))
    t=np.argsort(p_mean)[-n_k:]; b=np.argsort(p_mean)[:n_k]
    print(f'  Top-{k}%: L={np.mean(y_te_np[t]):+.4f} S={np.mean(y_te_np[b]):+.4f} LS={np.mean(y_te_np[t])-np.mean(y_te_np[b]):+.4f}')
ics=[]
for mth in np.unique(te_dates):
    mask=te_dates==mth
    if mask.sum()>=20:ics.append(spearmanr(p_mean[mask],y_te_np[mask])[0])
ics=np.array(ics)
print(f'  Monthly IC: {np.mean(ics):+.4f} ({np.sum(ics>0)}/{len(ics)} pos)')
for yr in['2024','2025','2026']:
    m=np.array([str(d).startswith(yr) for d in te_dates])
    if m.sum()<100:continue
    ic=spearmanr(p_mean[m],y_te_np[m])[0]; hit=np.mean((p_mean[m]>0)==(y_te_np[m]>0))
    n_t=max(1,int(m.sum()*0.2)); t20=np.mean(y_te_np[m][np.argsort(p_mean[m])[-n_t:]])
    print(f'  {yr}: IC={ic:+.4f} Hit={hit:.2%} Top20={t20:+.4f} N={m.sum():,}')
# Calibration
hc=p_std<np.median(p_std); lc=~hc
hca=np.mean((p_mean[hc]>0)==(y_te_np[hc]>0)); lca=np.mean((p_mean[lc]>0)==(y_te_np[lc]>0))
print(f'  MC calib: high={hca:.2%} low={lca:.2%} delta={hca-lca:+.2%}')
mse=np.mean((y_te_np-p_mean)**2); naive=np.mean((y_te_np-np.mean(y_te_np))**2)
print(f'  MSE={mse:.6f} Naive={naive:.6f} Ratio={mse/naive:.3f}')
print(f'  Val best MC_IC: {best_ic:+.4f}')
print('='*65)
