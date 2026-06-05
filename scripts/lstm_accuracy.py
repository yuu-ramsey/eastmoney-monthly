# LSTM prediction accuracy - detailed metrics
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time, json
from scipy.stats import spearmanr

DEV = torch.device('cuda')
SEQ_LEN, BATCH, LR, WD, EPOCHS = 60, 256, 3e-4, 1e-5, 50
DB = '.eastmoney-ai/db/klines-v2.sqlite'

print('[1/4] Loading data...', flush=True)
t0 = time.time()
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

seqs_list, ys_list, dates_list = [], [], []
for code in codes:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c = g['close'].values.astype(float); o = g['open'].values.astype(float)
    h = g['high'].values.astype(float); l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n = len(c)
    ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
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
        fwd = np.clip((c[i+3] - c[i]) / max(c[i], 0.01), -2, 2)
        if abs((c[i+3]-c[i])/c[i]) > 2: continue
        seqs_list.append(F[i-SEQ_LEN+1:i+1])
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
print(f'Data: {len(seqs):,} train={train_mask.sum():,} val={val_mask.sum():,} test={test_mask.sum():,} ({time.time()-t0:.0f}s)', flush=True)

# normalize
tr_seqs = seqs[train_mask]
feat_mean = tr_seqs.reshape(-1, 17).mean(axis=0)
feat_std = tr_seqs.reshape(-1, 17).std(axis=0) + 1e-8
seqs = np.clip((seqs - feat_mean) / feat_std, -5, 5)

X_tr = torch.from_numpy(seqs[train_mask]).float().to(DEV)
y_tr = torch.from_numpy(ys[train_mask]).float().to(DEV)
X_va = torch.from_numpy(seqs[val_mask]).float().to(DEV)
X_te = torch.from_numpy(seqs[test_mask]).float().to(DEV)
y_te_np = ys[test_mask]

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(17, 128, 2, batch_first=True, dropout=0.2)
        self.bn = nn.BatchNorm1d(128)
        self.head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.bn(out[:, -1, :])).squeeze(-1)

print('[2/4] Training LSTM...', flush=True)
model = LSTM().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
loss_fn = nn.HuberLoss(delta=0.5)

best_ic, best_state = -99, None
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr), device=DEV)
    total_loss = 0
    for i in range(0, len(X_tr), BATCH):
        idx = perm[i:i+BATCH]
        loss = loss_fn(model(X_tr[idx]), y_tr[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    model.eval()
    with torch.no_grad():
        p_list = [model(X_va[i:i+BATCH]).cpu().numpy() for i in range(0, len(X_va), BATCH)]
        ic = spearmanr(np.concatenate(p_list), ys[val_mask])[0]
    if ic > best_ic: best_ic = ic; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    sched.step()
    if (ep+1) % 10 == 0:
        print(f'  epoch {ep+1:3d} loss={total_loss:.4f} val_IC={ic:+.4f} best={best_ic:+.4f}', flush=True)

print('[3/4] Evaluating...', flush=True)
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    p_list = [model(X_te[i:i+BATCH]).cpu().numpy() for i in range(0, len(X_te), BATCH)]
    p_te = np.concatenate(p_list)

test_dates = dates_arr[test_mask]

# ===== ACCURACY METRICS =====
print()
print('='*65)
print('LSTM PREDICTION ACCURACY (Test 2024+, 46 months, 51K predictions)')
print('='*65)

# 1. Direction accuracy
pred_dir = p_te > 0
true_dir = y_te_np > 0
acc = np.mean(pred_dir == true_dir)
print(f'Direction accuracy: {acc:.2%} ({np.sum(pred_dir==true_dir)}/{len(p_te):,})')

# 2. Confusion matrix
tp = np.sum(pred_dir & true_dir)
fp = np.sum(pred_dir & ~true_dir)
tn = np.sum(~pred_dir & ~true_dir)
fn = np.sum(~pred_dir & true_dir)
print(f'Confusion matrix:')
print(f'  Predicted UP:   Really UP={tp:,}  Really DOWN={fp:,}  Precision={tp/max(tp+fp,1):.2%}')
print(f'  Predicted DOWN: Really UP={fn:,}  Really DOWN={tn:,}  Precision={tn/max(tn+fn,1):.2%}')
print(f'  Recall(UP)={tp/max(tp+fn,1):.2%}  Recall(DOWN)={tn/max(tn+fp,1):.2%}')
print(f'  F1={2*tp/max(2*tp+fp+fn,1):.3f}')

# 3. Distribution of predictions vs actual
print(f'\nPrediction distribution:')
print(f'  Mean={np.mean(p_te):+.4f}  Std={np.std(p_te):.4f}  Min={np.min(p_te):+.4f}  Max={np.max(p_te):+.4f}')
print(f'  Pred UP: {np.sum(pred_dir):,} ({np.mean(pred_dir):.1%})')
print(f'Actual distribution:')
print(f'  Mean={np.mean(y_te_np):+.4f}  Std={np.std(y_te_np):.4f}  Min={np.min(y_te_np):+.4f}  Max={np.max(y_te_np):+.4f}')
print(f'  True UP: {np.sum(true_dir):,} ({np.mean(true_dir):.1%})')

# 4. Top-K performance
for k_pct in [10, 20, 30]:
    n_k = max(1, int(len(p_te) * k_pct / 100))
    top_idx = np.argsort(p_te)[-n_k:]
    bottom_idx = np.argsort(p_te)[:n_k]
    top_ret = np.mean(y_te_np[top_idx])
    bottom_ret = np.mean(y_te_np[bottom_idx])
    top_hit = np.mean(y_te_np[top_idx] > 0)
    bottom_hit = np.mean(y_te_np[bottom_idx] > 0)
    print(f'Top-{k_pct}% long:  mean ret={top_ret:+.4f}  hit={top_hit:.1%}')
    print(f'Bottom-{k_pct}% short: mean ret={bottom_ret:+.4f}  hit={bottom_hit:.1%}')
    print(f'  Long-Short spread: {top_ret - bottom_ret:+.4f}')

# 5. By-year
print(f'\n{"Year":<8s} {"IC":>8s} {"Hit":>8s} {"Top20%ret":>10s} {"UpPrec":>8s} {"N":>8s}')
print('-'*56)
for yr in ['2024', '2025', '2026']:
    m = np.array([str(d).startswith(yr) for d in test_dates])
    if m.sum() < 100: continue
    ic = spearmanr(p_te[m], y_te_np[m])[0]
    hit = np.mean((p_te[m]>0) == (y_te_np[m]>0))
    n_t = max(1, int(m.sum()*0.2))
    top20 = np.mean(y_te_np[m][np.argsort(p_te[m])[-n_t:]])
    up_prec = np.sum((p_te[m]>0) & (y_te_np[m]>0)) / max(np.sum(p_te[m]>0), 1)
    print(f'{yr:<8s} {ic:+8.4f} {hit:8.2%} {top20:+10.4f} {up_prec:8.2%} {m.sum():>8,}')

# 6. Monthly IC stats
ics = []
for mth in np.unique(test_dates):
    mask = test_dates == mth
    if mask.sum() >= 20:
        ics.append(spearmanr(p_te[mask], y_te_np[mask])[0])
ics = np.array(ics)
pos = np.sum(ics > 0); neg = np.sum(ics < 0)
print(f'\n[4/4] Monthly IC distribution ({len(ics)} months):')
print(f'  Mean: {np.mean(ics):+.4f}  Median: {np.median(ics):+.4f}  Std: {np.std(ics):.4f}')
print(f'  Positive: {pos}/{len(ics)} ({pos/len(ics):.0%})  Negative: {neg}/{len(ics)} ({neg/len(ics):.0%})')
print(f'  IC>0.05: {np.sum(ics>0.05)}  IC<-0.05: {np.sum(ics<-0.05)}')
print(f'  Best: {np.max(ics):+.4f}  Worst: {np.min(ics):+.4f}')
print(f'  Win rate (>0): {pos/len(ics):.1%}')

# Compare with naive (always predict mean)
naive_mse = np.mean((y_te_np - np.mean(y_te_np))**2)
lstm_mse = np.mean((y_te_np - p_te)**2)
print(f'\nMSE: LSTM={lstm_mse:.6f}  Naive={naive_mse:.6f}  Ratio={lstm_mse/naive_mse:.3f}')
print('='*65)
