# LSTM v2: Fourier features + Multi-head Attention + MC Dropout + Residual
# Target: 70% direction accuracy on test set
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr
from pathlib import Path

DEV = torch.device('cuda')
SEQ_LEN, BATCH = 60, 256
LR, WD, EPOCHS = 3e-4, 1e-5, 80
N_FFT = 10  # top K dominant frequencies to extract
MC_SAMPLES = 20  # Monte Carlo forward passes at test time
print(f'Device: {DEV} ({torch.cuda.get_device_name(0)})')
print(f'FFT features: {N_FFT}, MC samples: {MC_SAMPLES}')

# ======== 1. Data Loading ========
DB = '.eastmoney-ai/db/klines-v2.sqlite'
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute(
    'SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84'
).fetchall()]
params = ','.join('?' * len(codes))
df = pd.read_sql_query(
    f"SELECT code,date,open,high,low,close,volume,turnover_rate "
    f"FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' "
    f"ORDER BY code,date", conn, params=codes)
conn.close()
print(f'Loading {len(codes)} stocks...', flush=True)

t0 = time.time()
seqs_list, ys_list, dates_list = [], [], []

def extract_fft_features(price_window):
    """Extract top N dominant frequencies from a price window using FFT"""
    # detrend
    x = np.arange(len(price_window))
    trend = np.polyfit(x, price_window, 1)
    detrended = price_window - np.polyval(trend, x)
    # FFT
    fft = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(len(detrended))
    amplitudes = np.abs(fft)
    # top K peaks by amplitude (skip DC component at index 0)
    if len(amplitudes) <= 1:
        return np.zeros(N_FFT * 3)
    peak_idx = np.argsort(amplitudes[1:])[::-1][:N_FFT] + 1
    features = []
    for idx in peak_idx:
        if idx < len(freqs):
            features.extend([
                freqs[idx],                    # frequency
                amplitudes[idx],               # amplitude
                np.angle(fft[idx]),            # phase
            ])
    # pad if fewer peaks than N_FFT
    while len(features) < N_FFT * 3:
        features.extend([0, 0, 0])
    return np.array(features[:N_FFT*3], dtype=np.float32)

for code in codes:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c = g['close'].values.astype(float); o = g['open'].values.astype(float)
    h = g['high'].values.astype(float); l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n = len(c)

    # same 17 technical features
    ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values; ma60 = pd.Series(c).rolling(60).mean().values
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    dif = e12 - e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif - dea) * 2
    delta = np.diff(c, prepend=c[0]); gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values; avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100 - 100/(1 + avg_gain/np.maximum(avg_loss, 1e-8)), 50)
    bb_std = pd.Series(c).rolling(20).std().values; bb_pos = np.nan_to_num((c - (ma20 - 2*bb_std)) / np.maximum(4*bb_std, 0.01), 0.5)
    trange = np.maximum(h - l, np.abs(h - np.roll(c, 1))); atr14 = pd.Series(trange).rolling(14).mean().values

    F = np.zeros((n, 17), dtype=np.float32)
    for i in range(n):
        F[i, 0] = (c[i]-c[i-1])/max(abs(c[i-1]), 0.01) if i >= 1 else 0
        F[i, 1] = (c[i]-c[i-3])/max(abs(c[i-3]), 0.01) if i >= 3 else 0
        F[i, 2] = (c[i]-c[i-6])/max(abs(c[i-6]), 0.01) if i >= 6 else 0
        F[i, 3] = (c[i]-c[i-12])/max(abs(c[i-12]), 0.01) if i >= 12 else 0
        F[i, 4] = (c[i]-ma5[i])/max(abs(c[i]), 0.01) if not np.isnan(ma5[i]) else 0
        F[i, 5] = (c[i]-ma20[i])/max(abs(c[i]), 0.01) if not np.isnan(ma20[i]) else 0
        F[i, 6] = (c[i]-ma60[i])/max(abs(c[i]), 0.01) if not np.isnan(ma60[i]) else 0
        F[i, 7] = dif[i] if not np.isnan(dif[i]) else 0; F[i, 8] = dea[i] if not np.isnan(dea[i]) else 0
        F[i, 9] = macd_hist[i] if not np.isnan(macd_hist[i]) else 0
        F[i, 10] = rsi14[i] if not np.isnan(rsi14[i]) else 50; F[i, 11] = bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5
        F[i, 12] = np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]), 0.01)) if i >= 6 else 0
        F[i, 13] = atr14[i]/max(abs(c[i]), 0.01) if not np.isnan(atr14[i]) else 0
        F[i, 14] = (h[i]-l[i])/max(abs(c[i]), 0.01)
        F[i, 15] = 1.0 if c[i] > ma20[i] else 0.0; F[i, 16] = 1.0 if c[i] > ma60[i] else 0.0
    F = np.nan_to_num(F, 0.0)

    for i in range(SEQ_LEN - 1, n - 6):
        if c[i] <= 0.01: continue
        fwd_raw = (c[i+3] - c[i]) / c[i]
        if abs(fwd_raw) > 2: continue
        fwd = np.clip(fwd_raw, -2, 2)

        # FFT features from close price window
        price_win = c[i-SEQ_LEN+1:i+1]
        fft_feats = extract_fft_features(price_win)

        # concatenate: technical features + FFT features
        seq_with_fft = np.zeros((SEQ_LEN, 17 + N_FFT*3), dtype=np.float32)
        seq_with_fft[:, :17] = F[i-SEQ_LEN+1:i+1]
        seq_with_fft[:, 17:] = fft_feats  # broadcast FFT features across all time steps

        seqs_list.append(seq_with_fft)
        ys_list.append(fwd)
        dates_list.append(g['date'].iloc[i])

seqs = np.array(seqs_list, dtype=np.float32); ys = np.array(ys_list, dtype=np.float32)
valid = ~np.isnan(ys) & ~np.isnan(seqs).any(axis=(1,2))
seqs = seqs[valid]; ys = ys[valid]
dates_list = [dates_list[i] for i in range(len(valid)) if valid[i]]
dates_arr = np.array(dates_list)

train_mask = (dates_arr >= '2015-01') & (dates_arr <= '2021-12')
val_mask = (dates_arr >= '2022-01') & (dates_arr <= '2023-12')
test_mask = (dates_arr >= '2024-01')
print(f'Data: {len(seqs):,} seqs, {seqs.shape[2]} features ({time.time()-t0:.0f}s)', flush=True)
print(f'Train: {train_mask.sum():,} Val: {val_mask.sum():,} Test: {test_mask.sum():,}', flush=True)

# Normalize
tr_seq = seqs[train_mask]
feat_mean = tr_seq.reshape(-1, seqs.shape[2]).mean(axis=0)
feat_std = tr_seq.reshape(-1, seqs.shape[2]).std(axis=0) + 1e-8
seqs = np.clip((seqs - feat_mean) / feat_std, -5, 5)

X_tr = torch.from_numpy(seqs[train_mask]).float().to(DEV); y_tr = torch.from_numpy(ys[train_mask]).float().to(DEV)
X_va = torch.from_numpy(seqs[val_mask]).float().to(DEV);   y_va = torch.from_numpy(ys[val_mask]).float().to(DEV)
X_te = torch.from_numpy(seqs[test_mask]).float().to(DEV);  y_te_np = ys[test_mask]
test_dates = dates_arr[test_mask]

# ======== 2. Model: LSTM + Multi-head Attention + Residual + MC Dropout ========
IN_DIM = seqs.shape[2]  # 17 + N_FFT*3
HIDDEN = 128

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads; self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3); self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out)

class LSTMv2(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, num_layers=3, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden)
        self.attn = MultiHeadAttention(hidden, heads=4, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        # head with residual
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln(lstm_out)
        # attention over time steps + residual
        attn_out = self.attn(lstm_out)
        out = self.ln2(lstm_out + attn_out)  # residual connection
        out = self.dropout(out[:, -1, :])  # last time step
        return self.head(out).squeeze(-1)

model = LSTMv2().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-5)
loss_fn = nn.HuberLoss(delta=0.5)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

# ======== 3. Training with early stopping ========
best_val_ic, best_state, patience = -99, None, 0
MAX_PATIENCE = 25

for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr), device=DEV)
    total_loss = 0
    for i in range(0, len(X_tr), BATCH):
        idx = perm[i:i+BATCH]
        pred = model(X_tr[idx])
        loss = loss_fn(pred, y_tr[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        p_list = [model(X_va[i:i+BATCH]).cpu().numpy() for i in range(0, len(X_va), BATCH)]
        val_ic = spearmanr(np.concatenate(p_list), ys[val_mask])[0]

    # MC Dropout validation check
    model.train()  # keep dropout on
    with torch.no_grad():
        mc_preds = []
        for _ in range(10):
            p_list = [model(X_va[i:i+BATCH]).cpu().numpy() for i in range(0, len(X_va), BATCH)]
            mc_preds.append(np.concatenate(p_list))
        mc_mean = np.mean(mc_preds, axis=0)
        mc_std = np.std(mc_preds, axis=0)
        mc_ic = spearmanr(mc_mean, ys[val_mask])[0]

    sched.step()

    if mc_ic > best_val_ic:
        best_val_ic = mc_ic
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1

    if (ep + 1) % 10 == 0:
        print(f'  epoch {ep+1:3d} loss={total_loss:.4f} val_IC={val_ic:+.4f} '
              f'MC_IC={mc_ic:+.4f} best={best_val_ic:+.4f} pat={patience}', flush=True)

    if patience >= MAX_PATIENCE:
        print(f'  Early stop at epoch {ep+1}', flush=True)
        break

# ======== 4. Evaluation (MC Dropout) ========
print('\n[Evaluating with MC Dropout...]', flush=True)
model.load_state_dict(best_state)
model.train()  # keep dropout ON for MC sampling

with torch.no_grad():
    mc_preds = []
    for s in range(MC_SAMPLES):
        p_list = [model(X_te[i:i+BATCH]).cpu().numpy() for i in range(0, len(X_te), BATCH)]
        mc_preds.append(np.concatenate(p_list))
    mc_preds = np.array(mc_preds)  # (MC_SAMPLES, N_test)
    p_te_mean = mc_preds.mean(axis=0)
    p_te_std = mc_preds.std(axis=0)

# ======== 5. Metrics ========
pred_dir = p_te_mean > 0
true_dir = y_te_np > 0
acc = np.mean(pred_dir == true_dir)

print()
print('='*65)
print(f'LSTM v2 Results (FFT + Attention + MC Dropout)')
print('='*65)
print(f'Direction accuracy: {acc:.2%} ({np.sum(pred_dir==true_dir)}/{len(p_te_mean):,})')

tp = np.sum(pred_dir & true_dir); fp = np.sum(pred_dir & ~true_dir)
tn = np.sum(~pred_dir & ~true_dir); fn = np.sum(~pred_dir & true_dir)
print(f'  Pred UP precision:  {tp/max(tp+fp,1):.2%} ({tp:,}/{tp+fp:,})')
print(f'  Pred DOWN precision:{tn/max(tn+fn,1):.2%} ({tn:,}/{tn+fn:,})')
print(f'  Recall(UP)={tp/max(tp+fn,1):.2%}  Recall(DOWN)={tn/max(tn+fp,1):.2%}')
print(f'  F1={2*tp/max(2*tp+fp+fn,1):.3f}')

# Top-K
for k in [10, 20]:
    n_k = max(1, int(len(p_te_mean)*k//100))
    top = np.argsort(p_te_mean)[-n_k:]; bot = np.argsort(p_te_mean)[:n_k]
    ls = np.mean(y_te_np[top]) - np.mean(y_te_np[bot])
    print(f'  Top-{k}% Long-Short: {ls:+.4f}')

# Monthly IC
ics = []
for mth in np.unique(test_dates):
    mask = test_dates == mth
    if mask.sum() >= 20:
        ics.append(spearmanr(p_te_mean[mask], y_te_np[mask])[0])
ics = np.array(ics)
print(f'  Monthly IC: {np.mean(ics):+.4f} ({np.sum(ics>0)}/{len(ics)} pos)')

# By year
for yr in ['2024', '2025', '2026']:
    m = np.array([str(d).startswith(yr) for d in test_dates])
    if m.sum() < 100: continue
    ic = spearmanr(p_te_mean[m], y_te_np[m])[0]
    hit = np.mean((p_te_mean[m]>0) == (y_te_np[m]>0))
    n_t = max(1, int(m.sum()*0.2))
    top20 = np.mean(y_te_np[m][np.argsort(p_te_mean[m])[-n_t:]])
    print(f'  {yr}: IC={ic:+.4f} Hit={hit:.2%} Top20={top20:+.4f} N={m.sum():,}')

# Overfitting check: MC uncertainty vs error
error = np.abs(p_te_mean - y_te_np)
high_conf = p_te_std < np.median(p_te_std)
low_conf = ~high_conf
hc_acc = np.mean((p_te_mean[high_conf]>0) == (y_te_np[high_conf]>0))
lc_acc = np.mean((p_te_mean[low_conf]>0) == (y_te_np[low_conf]>0))
print(f'  High-conf acc: {hc_acc:.2%}  Low-conf acc: {lc_acc:.2%}  '
      f'Delta: {hc_acc-lc_acc:+.2%} (MC uncertainty calibration)')

mse = np.mean((y_te_np - p_te_mean)**2)
naive = np.mean((y_te_np - np.mean(y_te_np))**2)
print(f'  MSE: {mse:.6f}  Naive: {naive:.6f}  Ratio: {mse/naive:.3f}')
print(f'  Val best MC_IC: {best_val_ic:+.4f}')
print('='*65)
