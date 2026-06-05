# LSTM v4: Roadmap implementation
# Phase 1.1: Cross-stock training (already 2247 stocks, 217K seqs)
# Phase 4.1: 3-class labels (UP>+2%, FLAT[-2%,+2%], DOWN<-2%)
# Phase 3.1: Temporal Attention BiLSTM
# Phase 5: MC Dropout confidence filtering
# Target: 68-70% Hit on high-confidence subset, coverage >= 30%
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F, copy
import numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import pywt

DEV = torch.device('cuda')
SEQ_LEN, BATCH = 60, 256
LR, WD, EPOCHS = 5e-4, 1e-5, 80
N_FFT, MC_SAMPLES = 10, 30
EMA_DECAY = 0.999
DB = '.eastmoney-ai/db/klines-v2.sqlite'
THRESH_UP = 0.02    # up: monthly return > +2%
THRESH_DN = -0.02   # 跌：月收益率 < -2%

print(f'LSTM v4 Roadmap | GPU: {torch.cuda.get_device_name(0)}', flush=True)
print(f'3-class: UP>{THRESH_UP:+.0%} FLAT=[{THRESH_DN:+.0%},{THRESH_UP:+.0%}] DOWN<{THRESH_DN:+.0%}', flush=True)

# ======== Data ========
def fft_f(prices):
    x=np.arange(len(prices)); trend=np.polyfit(x,prices,1); detrended=prices-np.polyval(trend,x)
    fft_p=np.fft.rfft(detrended); amps=np.abs(fft_p); freqs=np.fft.rfftfreq(len(detrended))
    if len(amps)<=1: return np.zeros(N_FFT*3,dtype=np.float32)
    pk=np.argsort(amps[1:])[::-1][:N_FFT]+1; feats=[]
    for idx in pk:
        if idx<len(freqs): feats.extend([freqs[idx], amps[idx], np.angle(fft_p[idx])])
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

print('Loading data...', flush=True); t0=time.time()
seqs_list, labels_3cls, ys_reg, dates_list = [], [], [], []
cnt_up, cnt_flat, cnt_dn = 0, 0, 0
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
    for i in range(SEQ_LEN-1,n-6):
        if c[i]<=0.01:continue
        fwd_raw=(c[i+3]-c[i])/c[i]
        if abs(fwd_raw)>2:continue
        fwd=np.clip(fwd_raw,-2,2)
        seq=np.zeros((SEQ_LEN,17+N_FFT*3),dtype=np.float32)
        seq[:,:17]=F[i-SEQ_LEN+1:i+1]; seq[:,17:]=fft_f(c_clean[i-SEQ_LEN+1:i+1])
        seqs_list.append(seq); ys_reg.append(fwd)
        # 3-class label
        if fwd_raw > THRESH_UP:
            labels_3cls.append(2); cnt_up += 1
        elif fwd_raw < THRESH_DN:
            labels_3cls.append(0); cnt_dn += 1
        else:
            labels_3cls.append(1); cnt_flat += 1
        dates_list.append(g['date'].iloc[i])

seqs=np.array(seqs_list,dtype=np.float32); ys=np.array(ys_reg,dtype=np.float32); lbls=np.array(labels_3cls,dtype=np.int64)
v=~np.isnan(seqs).any(axis=(1,2))&~np.isnan(ys); seqs=seqs[v]; ys=ys[v]; lbls=lbls[v]
dates_arr=np.array([dates_list[i] for i in range(len(v)) if v[i]])
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12'); va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12'); te_m=(dates_arr>='2024-01')
print(f'Data: {len(seqs):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)', flush=True)
print(f'3-class: UP={cnt_up:,}({cnt_up/(cnt_up+cnt_flat+cnt_dn):.1%}) FLAT={cnt_flat:,}({cnt_flat/(cnt_up+cnt_flat+cnt_dn):.1%}) DOWN={cnt_dn:,}({cnt_dn/(cnt_up+cnt_flat+cnt_dn):.1%})', flush=True)

st=seqs[tr_m]; fm=st.reshape(-1,seqs.shape[2]).mean(0); fs=st.reshape(-1,seqs.shape[2]).std(0)+1e-8
seqs=np.clip((seqs-fm)/fs,-5,5)

X_tr=torch.from_numpy(seqs[tr_m]).float().to(DEV); y_tr_t=torch.from_numpy(ys[tr_m]).float().to(DEV); l_tr=torch.from_numpy(lbls[tr_m]).long().to(DEV)
X_va=torch.from_numpy(seqs[va_m]).float().to(DEV); y_va_np=ys[va_m]; l_va=lbls[va_m]
X_te=torch.from_numpy(seqs[te_m]).float().to(DEV); y_te_np=ys[te_m]; l_te=lbls[te_m]; te_dates=dates_arr[te_m]
l_va_gpu = torch.from_numpy(l_va).long().to(DEV)

# ======== Model: Attention BiLSTM + 3-class head ========
class TemporalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Linear(hidden//2, 1))
    def forward(self, lstm_out):
        w = self.attn(lstm_out).squeeze(-1)  # (B, T)
        w = torch.nn.functional.softmax(w, dim=1)
        return (w.unsqueeze(-1) * lstm_out).sum(dim=1), w  # (B,H), (B,T)

class AttnBiLSTM(nn.Module):
    def __init__(self, in_dim, hidden=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.in_ln = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0, bidirectional=True)
        for name, param in self.lstm.named_parameters():
            if 'bias_ih' in name: n=param.shape[0]//4; param.data[n:2*n].fill_(1.0)
            elif 'bias_hh' in name: n=param.shape[0]//4; param.data[n:2*n].fill_(1.0)
        self.attn = TemporalAttention(hidden * 2)
        self.ln = nn.LayerNorm(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.reg_head = nn.Sequential(nn.Linear(hidden*2, hidden//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1))
    def forward(self, x):
        x = self.in_ln(x)
        o, _ = self.lstm(x)
        context, attn_w = self.attn(o)
        context = self.ln(context)
        c = self.dropout(context)
        return self.cls_head(c), self.reg_head(c).squeeze(-1), attn_w

def rank_loss(pred, target):
    if len(pred)<2: return torch.tensor(0.0, device=pred.device)
    d=target.unsqueeze(0)-target.unsqueeze(1); pd=pred.unsqueeze(0)-pred.unsqueeze(1)
    return torch.sigmoid(-pd*torch.sign(d)*torch.abs(d)).mean()

def focal_loss_3cls(logits, targets, gamma=2.0):
    """Focal loss for 3-class, only penalizing UP/DOWN (ignore FLAT)"""
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    focal = ((1-pt)**gamma * ce)
    # Zero out FLAT class loss
    mask = (targets != 1).float()  # class 1 = FLAT, ignore
    return (focal * mask).sum() / max(mask.sum(), 1)

# ======== Training ========
print('Training AttnBiLSTM...', flush=True); t0=time.time()
model = AttnBiLSTM(seqs.shape[2]).to(DEV); ema = copy.deepcopy(model)
for p in ema.parameters(): p.requires_grad = False
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
huber = nn.HuberLoss(delta=0.5)

def eval_hit(model, X, lbls, ys, batch_size=512):
    """Evaluate hit rate ignoring FLAT class"""
    model.eval()
    with torch.no_grad():
        cls_list, reg_list = [], []
        for i in range(0, len(X), batch_size):
            c, r, _ = model(X[i:i+batch_size])
            cls_list.append(c.detach().cpu()); reg_list.append(r.detach().cpu())
        cls_p = torch.cat(cls_list); reg_p = torch.cat(reg_list)
    pred_cls = cls_p.argmax(dim=1).numpy()
    # Only count UP/DOWN predictions
    valid_mask = lbls != 1  # ignore FLAT
    if valid_mask.sum() == 0: return 0.5, 0
    hit = (pred_cls[valid_mask] == lbls[valid_mask]).mean()
    ic = spearmanr(reg_p.numpy(), ys)[0]
    return float(hit), float(ic)

best_hit, best_state, patience = 0, None, 0
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr), device=DEV)
    total_loss = 0.0
    for i in range(0, len(X_tr), BATCH):
        idx = perm[i:i+BATCH]
        xb, yb, lb = X_tr[idx], y_tr_t[idx], l_tr[idx]
        cls_p, reg_p, _ = model(xb)
        loss_cls = focal_loss_3cls(cls_p, lb, gamma=2.0)
        loss_reg = huber(reg_p, yb)
        loss_rank = rank_loss(reg_p, yb)
        loss = loss_cls + loss_reg + 0.3 * loss_rank
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for ep_, p in zip(ema.parameters(), model.parameters()):
                ep_.data.mul_(EMA_DECAY).add_(p.data, alpha=1-EMA_DECAY)
        total_loss += loss.item()
    hit_va, ic_va = eval_hit(ema, X_va, l_va, y_va_np)
    if hit_va > best_hit:
        best_hit = hit_va; best_state = copy.deepcopy(ema.state_dict()); patience = 0
    else:
        patience += 1
    if (ep+1) % 15 == 0:
        print(f'  ep{ep+1:3d} loss={total_loss:.2f} val_hit={hit_va:.4f} val_IC={ic_va:+.4f} best={best_hit:.4f}', flush=True)
    if patience >= 20:
        print(f'  Early stop ep{ep+1}', flush=True); break

# ======== Evaluation ========
print('\nEvaluating...', flush=True)
model.load_state_dict(best_state)

# MC Dropout for confidence
model.train()  # dropout ON
mc_preds_cls, mc_preds_reg = [], []
for s in range(MC_SAMPLES):
    cls_list, reg_list = [], []
    with torch.no_grad():
        for i in range(0, len(X_te), BATCH):
            c, r, _ = model(X_te[i:i+BATCH])
            cls_list.append(c.detach().cpu()); reg_list.append(r.detach().cpu())
    mc_preds_cls.append(torch.nn.functional.softmax(torch.cat(cls_list), dim=1).numpy())
    mc_preds_reg.append(torch.cat(reg_list).numpy())
mc_cls = np.array(mc_preds_cls)  # (MC, N, 3)
mc_reg = np.array(mc_preds_reg)  # (MC, N)

cls_mean = mc_cls.mean(axis=0); cls_std = mc_cls.std(axis=0)
reg_mean = mc_reg.mean(axis=0); reg_std = mc_reg.std(axis=0)
pred_cls = cls_mean.argmax(axis=1)

# ======== Metrics ========
def report(label, pred, prob, std, tag):
    valid = label != 1  # ignore FLAT
    n_total = len(label); n_valid = valid.sum()
    hit_all = (pred == label).mean()
    hit_sig = (pred[valid] == label[valid]).mean() if n_valid > 0 else 0
    print(f'\n[{tag}]')
    print(f'  All-sample Hit: {hit_all:.4f} ({hit_all:.2%})')
    print(f'  Signal-only Hit (UP+DOWN): {hit_sig:.4f} ({hit_sig:.2%})  coverage={n_valid/n_total:.1%}')

    # Confidence filtering
    max_prob = prob.max(axis=1)
    results = []
    sep = '-' * 55
    print(f'  {"Conf":>8s}  {"Hit":>8s}  {"Coverage":>10s}  {"UPprec":>8s}  {"DNprec":>8s}  {"N_pred":>8s}')
    print(f'  {sep}')
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        conf_mask = max_prob > thr
        if conf_mask.sum() == 0: continue
        cp = pred[conf_mask]; cl = label[conf_mask]
        # Only count UP/DOWN
        sc = (cl != 1)
        if sc.sum() == 0: continue
        hit = (cp[sc] == cl[sc]).mean()
        cov = sc.sum() / n_total
        up_p = ((cp==2)&(cl==2)).sum() / max((cp==2).sum(), 1)
        dn_p = ((cp==0)&(cl==0)).sum() / max((cp==0).sum(), 1)
        print(f'  {thr:8.2f}  {hit:8.4f}  {cov:10.1%}  {up_p:8.2%}  {dn_p:8.2%}  {sc.sum():>8,}')
        results.append({'threshold': thr, 'hit': float(hit), 'coverage': float(cov), 'up_prec': float(up_p), 'dn_prec': float(dn_p)})

    # CS_IC
    cs_ics = [spearmanr(reg_mean[te_dates==m], y_te_np[te_dates==m])[0] for m in np.unique(te_dates) if (te_dates==m).sum()>=20]
    cs_ic = np.mean(cs_ics) if cs_ics else np.nan
    print(f'  CS_IC: {cs_ic:+.4f} ({len(cs_ics)} months)')
    return hit_sig, cs_ic, results

hit3, ic3, conf_results = report(l_te, pred_cls, cls_mean, cls_std, '3-Class AttnBiLSTM')

# Summary
print(); print('='*60)
print('Roadmap v4 Summary')
print(f'  Signal-only Hit (UP+DOWN): {hit3:.2%}')
print(f'  CS_IC: {ic3:+.4f}')
print(f'  Best high-conf Hit: {max(r["hit"] for r in conf_results):.2%}' if conf_results else '  No conf results')
for r in conf_results:
    if r['hit'] >= 0.68 and r['coverage'] >= 0.30:
        print(f'  TARGET MET: Hit={r["hit"]:.2%} Coverage={r["coverage"]:.1%} @ conf={r["threshold"]:.2f}')
print('='*60)
