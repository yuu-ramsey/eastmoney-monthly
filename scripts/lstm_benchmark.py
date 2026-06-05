# LSTM benchmark - same features as Ridge/LightGBM for direct comparison
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time, json
from pathlib import Path
from scipy.stats import spearmanr

DEV = torch.device('cuda')
SEQ_LEN, BATCH, LR, WD, EPOCHS = 60, 256, 3e-4, 1e-5, 50
print(f'Device: {DEV} ({torch.cuda.get_device_name(0)})')

# ---- data ----
DB = '.eastmoney-ai/db/klines-v2.sqlite'
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute(
    'SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84'
).fetchall()]
ind_map = {r[0]: r[1] for r in conn.execute(
    'SELECT stock_code, industry_code FROM stock_industry_mapping'
)}
params = ','.join('?' * len(codes))
df = pd.read_sql_query(
    f"SELECT code,date,open,high,low,close,volume,turnover_rate "
    f"FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' "
    f"ORDER BY code,date", conn, params=codes
)
conn.close()

print(f'Loading {len(codes)} stocks...')
t0 = time.time()
seqs, ys, dates_list, metas = [], [], [], []
for code in codes:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 72: continue
    c = g['close'].values.astype(float); o = g['open'].values.astype(float)
    h = g['high'].values.astype(float); l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n = len(c)
    # same 17 features as cross_dataset.py
    ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    dif = e12 - e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif - dea) * 2
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100 - 100/(1 + avg_gain/np.maximum(avg_loss, 1e-8)), 50)
    bb_std = pd.Series(c).rolling(20).std().values
    bb_pos = np.nan_to_num((c - (ma20 - 2*bb_std)) / np.maximum(4*bb_std, 0.01), 0.5)
    trange = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
    atr14 = pd.Series(trange).rolling(14).mean().values

    F = np.zeros((n, 17), dtype=np.float32)
    for i in range(n):
        F[i, 0] = (c[i]-c[i-1])/max(abs(c[i-1]), 0.01) if i >= 1 else 0
        F[i, 1] = (c[i]-c[i-3])/max(abs(c[i-3]), 0.01) if i >= 3 else 0
        F[i, 2] = (c[i]-c[i-6])/max(abs(c[i-6]), 0.01) if i >= 6 else 0
        F[i, 3] = (c[i]-c[i-12])/max(abs(c[i-12]), 0.01) if i >= 12 else 0
        F[i, 4] = (c[i]-ma5[i])/max(abs(c[i]), 0.01) if not np.isnan(ma5[i]) else 0
        F[i, 5] = (c[i]-ma20[i])/max(abs(c[i]), 0.01) if not np.isnan(ma20[i]) else 0
        F[i, 6] = (c[i]-ma60[i])/max(abs(c[i]), 0.01) if not np.isnan(ma60[i]) else 0
        F[i, 7] = dif[i] if not np.isnan(dif[i]) else 0
        F[i, 8] = dea[i] if not np.isnan(dea[i]) else 0
        F[i, 9] = macd_hist[i] if not np.isnan(macd_hist[i]) else 0
        F[i, 10] = rsi14[i] if not np.isnan(rsi14[i]) else 50
        F[i, 11] = bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5
        F[i, 12] = np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]), 0.01)) if i >= 6 else 0
        F[i, 13] = atr14[i]/max(abs(c[i]), 0.01) if not np.isnan(atr14[i]) else 0
        F[i, 14] = (h[i]-l[i])/max(abs(c[i]), 0.01)
        F[i, 15] = 1.0 if c[i] > ma20[i] else 0.0
        F[i, 16] = 1.0 if c[i] > ma60[i] else 0.0

    F = np.nan_to_num(F, 0.0)
    dates = g['date'].tolist()
    for i in range(SEQ_LEN - 1, n - 6):
        if c[i] <= 0.01: continue
        fwd = np.clip((c[i+3] - c[i]) / max(c[i], 0.01), -2, 2)
        if abs((c[i+3]-c[i])/c[i]) > 2: continue
        seqs.append(F[i-SEQ_LEN+1:i+1])
        ys.append(fwd)
        dates_list.append(dates[i])

seqs = np.array(seqs, dtype=np.float32)
ys = np.array(ys, dtype=np.float32)
# filter NaN
valid = ~np.isnan(ys) & ~np.isnan(seqs).any(axis=(1,2))
seqs = seqs[valid]
ys = ys[valid]
dates_list = [dates_list[i] for i in range(len(valid)) if valid[i]]
print(f'Data: {len(seqs):,} sequences ({valid.sum()}/{len(valid)} valid, {time.time()-t0:.0f}s)')

# train/val/test split by date
dates_arr = np.array(dates_list)
train_mask = (dates_arr >= '2015-01') & (dates_arr <= '2021-12')
val_mask = (dates_arr >= '2022-01') & (dates_arr <= '2023-12')
test_mask = (dates_arr >= '2024-01')
print(f'Train: {train_mask.sum():,} Val: {val_mask.sum():,} Test: {test_mask.sum():,}')

# ---- normalize features ----
# compute mean/std on training data
tr_seqs = seqs[train_mask]
feat_mean = tr_seqs.reshape(-1, 17).mean(axis=0)
feat_std = tr_seqs.reshape(-1, 17).std(axis=0) + 1e-8
seqs = (seqs - feat_mean) / feat_std
seqs = np.clip(seqs, -5, 5)  # clip extreme outliers
print(f'Feature norm: mean={feat_mean[:5]} std={feat_std[:5]}')

X_tr = torch.from_numpy(seqs[train_mask]).float().to(DEV)
y_tr = torch.from_numpy(ys[train_mask]).float().to(DEV)
X_va = torch.from_numpy(seqs[val_mask]).float().to(DEV)
y_va = torch.from_numpy(ys[val_mask]).float().to(DEV)
X_te = torch.from_numpy(seqs[test_mask]).float().to(DEV)
y_te = torch.from_numpy(ys[test_mask]).float().to(DEV)
y_te_np = ys[test_mask]

# ---- model ----
class LSTM(nn.Module):
    def __init__(self, in_dim=17, hidden=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.bn(out[:, -1, :])
        return self.head(out).squeeze(-1)

model = LSTM().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
loss_fn = nn.HuberLoss(delta=0.5)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

# ---- train ----
best_ic, best_state = -99, None
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
    # validate
    model.eval()
    with torch.no_grad():
        # batch inference to avoid OOM
        p_list = []
        for i in range(0, len(X_va), BATCH):
            p_list.append(model(X_va[i:i+BATCH]).cpu().numpy())
        p = np.concatenate(p_list)
        ic = spearmanr(p, ys[val_mask])[0]
    if ic > best_ic:
        best_ic = ic
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    sched.step()
    if (ep + 1) % 10 == 0:
        print(f'  Epoch {ep+1:3d}/{EPOCHS}  loss={total_loss:.4f}  val_IC={ic:+.4f}  best={best_ic:+.4f}')

# ---- test ----
if best_state is not None:
    model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    p_list = []
    for i in range(0, len(X_te), BATCH):
        p_list.append(model(X_te[i:i+BATCH]).cpu().numpy())
    p_te = np.concatenate(p_list)
    # monthly cross-section IC
    test_dates = np.array(dates_list)[test_mask]
    ics = []
    for m in np.unique(test_dates):
        mask = test_dates == m
        if mask.sum() >= 20:
            ics.append(spearmanr(p_te[mask], y_te_np[mask])[0])
    avg_ic = np.mean(ics) if ics else np.nan

print(f'\n{"="*60}')
print(f'LSTM Results (RTX 5070)')
print(f'  Val best IC:   {best_ic:+.4f}')
print(f'  Test IC:       {avg_ic:+.4f} ({len(ics)} months)')
print(f'  Test ICIR:     {avg_ic/np.std(ics):+.4f}' if ics and np.std(ics) > 0 else '  Test ICIR: N/A')
print(f'  Qlib LSTM IC:  +0.0395 (CSI300 daily)')
print(f'  Ridge IC:      +0.0603 (same features)')
print(f'{"="*60}')
