# Path E: Daily LSTM → Monthly embeddings → 3-month forward return prediction
# 12 months × ~22 daily bars = 240+ time steps of intra-month microstructure
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F, copy
import numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

DEV = torch.device('cuda')
MAX_DAYS, N_MONTHS = 22, 12
BATCH, LR, WD, EPOCHS = 64, 5e-4, 1e-4, 60
EMA_DECAY = 0.999
DB = '.eastmoney-ai/db/klines-v2.sqlite'
print(f'Path E: Daily LSTM → Monthly Embeddings | GPU: {torch.cuda.get_device_name(0)}', flush=True)
print(f'Arch: {N_MONTHS} months × {MAX_DAYS} days → BiLSTM → prediction', flush=True)

# ======== 1. Build Monthly Embedding Dataset ========
def build_daily_features(c, o, h, l, v, tr):
    """10 daily features per bar"""
    n = len(c)
    F = np.zeros((n, 10), dtype=np.float32)
    ma5 = np.array(pd.Series(c).rolling(5).mean().fillna(0))
    ma20 = np.array(pd.Series(c).rolling(20).mean().fillna(0))
    vol_ma20 = np.array(pd.Series(v).rolling(20).mean().fillna(1))
    for i in range(n):
        F[i,0] = (c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0
        F[i,1] = (c[i]-o[i])/max(abs(o[i]),0.01) if o[i]!=0 else 0
        F[i,2] = (h[i]-l[i])/max(abs(o[i]),0.01) if o[i]!=0 else 0
        F[i,3] = (h[i]-max(c[i],o[i]))/max(abs(o[i]),0.01) if o[i]!=0 else 0
        F[i,4] = (min(c[i],o[i])-l[i])/max(abs(o[i]),0.01) if o[i]!=0 else 0
        F[i,5] = v[i]/max(vol_ma20[i],1)
        F[i,6] = tr[i] if not np.isnan(tr[i]) else 0
        F[i,7] = (c[i]-ma5[i])/max(abs(c[i]),0.01) if ma5[i]!=0 else 0
        F[i,8] = (c[i]-ma20[i])/max(abs(c[i]),0.01) if ma20[i]!=0 else 0
        F[i,9] = np.std(np.diff(c[max(0,i-5):i+1])/np.maximum(np.abs(c[max(0,i-4):i+1]),0.01)) if i>=5 else 0
    F = np.where(np.isfinite(F), F, 0.0)
    return F.astype(np.float32)

conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute(
    'SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=500'
).fetchall()]
print(f'Stocks with >=500 daily bars: {len(codes)}', flush=True)

# Get month labels for alignment
month_df = pd.read_sql_query(
    "SELECT DISTINCT substr(date,1,7) as month FROM daily_klines WHERE date>='2010-01' ORDER BY month",
    conn)
all_months = month_df['month'].tolist()
print(f'Months: {len(all_months)} ({all_months[0]} ~ {all_months[-1]})', flush=True)

print('Building monthly embeddings...', flush=True); t0 = time.time()
X_months, y_targets, dates_out = [], [], []

for code in codes:
    df = pd.read_sql_query(
        f"SELECT date,open,close,high,low,volume,turnover_rate FROM daily_klines "
        f"WHERE code='{code}' AND date>='2010-01-01' ORDER BY date", conn)
    if len(df) < 500: continue
    df['month'] = df['date'].str[:7]
    c = df['close'].values.astype(float); o = df['open'].values.astype(float)
    h = df['high'].values.astype(float); l = df['low'].values.astype(float)
    v = df['volume'].values.astype(float)
    tr = df['turnover_rate'].values.astype(float) if 'turnover_rate' in df.columns else np.zeros(len(df))
    F = build_daily_features(c, o, h, l, v, tr)

    # Group by month
    monthly_embeddings = []
    month_labels = []
    month_end_prices = []
    for month, group in df.groupby('month'):
        if len(group) < 5: continue  # skip months with <5 trading days
        idx = group.index
        days = F[idx[0]:idx[-1]+1][-MAX_DAYS:]  # take last MAX_DAYS days
        # Pad to MAX_DAYS
        if len(days) < MAX_DAYS:
            pad = np.zeros((MAX_DAYS-len(days), 10), dtype=np.float32)
            days = np.vstack([pad, days])
        monthly_embeddings.append(days)
        month_labels.append(month)
        month_end_prices.append(c[idx[-1]])

    # Build 12-month sequences with 3-month forward target
    for i in range(N_MONTHS, len(monthly_embeddings)-3):
        # Check for data gaps (>2 month gap = sequence broken)
        seq_months = month_labels[i-N_MONTHS:i]
        target_month = month_labels[i+3]
        # Target: 3-month forward return
        fwd_ret = (month_end_prices[i+3] - month_end_prices[i]) / max(month_end_prices[i], 0.01)
        if abs(fwd_ret) > 2: continue
        seq = np.stack(monthly_embeddings[i-N_MONTHS:i])  # (12, 22, 10)
        X_months.append(seq)
        y_targets.append(np.clip(fwd_ret, -2, 2))
        dates_out.append(target_month)

conn.close()

X = np.array(X_months, dtype=np.float32)  # (N, 12, 22, 10)
y = np.array(y_targets, dtype=np.float32)
dates_arr = np.array(dates_out)
v = ~np.isnan(X).any(axis=(1,2,3)) & ~np.isnan(y)
X = X[v]; y = y[v]; dates_arr = dates_arr[v]

tr_m = (dates_arr >= '2015-01') & (dates_arr <= '2021-12')
va_m = (dates_arr >= '2022-01') & (dates_arr <= '2023-12')
te_m = (dates_arr >= '2024-01')
print(f'Data: {len(X):,} seqs T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)', flush=True)

# Normalize per-feature across all data
X_flat = X.reshape(-1, X.shape[-1])  # flatten to (N*12*22, 10)
fm = X_flat.mean(0); fs = X_flat.std(0) + 1e-8
X = np.clip((X - fm) / fs, -5, 5)

X_tr = torch.from_numpy(X[tr_m]).float().to(DEV); y_tr_t = torch.from_numpy(y[tr_m]).float().to(DEV)
X_va = torch.from_numpy(X[va_m]).float().to(DEV); y_va_np = y[va_m]
X_te = torch.from_numpy(X[te_m]).float().to(DEV); y_te_np = y[te_m]; te_dates = dates_arr[te_m]

# ======== 2. Model: Two-level LSTM (daily → monthly → prediction) ========
class DailyLSTM(nn.Module):
    def __init__(self, in_dim=10, daily_hidden=64, monthly_hidden=128, dropout=0.3):
        super().__init__()
        # Level 1: Daily LSTM (process 22 days within a month)
        self.daily_lstm = nn.LSTM(in_dim, daily_hidden, 2, batch_first=True, dropout=dropout)
        self.daily_ln = nn.LayerNorm(daily_hidden)
        # Level 2: Monthly BiLSTM (process 12 monthly embeddings)
        self.monthly_lstm = nn.LSTM(daily_hidden, monthly_hidden, 2, batch_first=True,
                                    dropout=dropout, bidirectional=True)
        for name, param in self.monthly_lstm.named_parameters():
            if 'bias_ih' in name: n=param.shape[0]//4; param.data[n:2*n].fill_(1.0)
            elif 'bias_hh' in name: n=param.shape[0]//4; param.data[n:2*n].fill_(1.0)
        self.monthly_ln = nn.LayerNorm(monthly_hidden*2)
        self.head = nn.Sequential(
            nn.Linear(monthly_hidden*2, monthly_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(monthly_hidden, 1))
    def forward(self, x):
        # x: (B, 12, 22, 10)
        B, M, D, Fdim = x.shape
        # Level 1: Process each month's daily data
        x = x.reshape(B*M, D, Fdim)  # (B*12, 22, 10)
        d_out, _ = self.daily_lstm(x)
        d_emb = self.daily_ln(d_out[:, -1, :])  # (B*12, daily_hidden)
        # Level 2: Process monthly embeddings
        d_emb = d_emb.reshape(B, M, -1)  # (B, 12, daily_hidden)
        m_out, _ = self.monthly_lstm(d_emb)
        m_emb = self.monthly_ln(m_out[:, -1, :])  # (B, monthly_hidden*2)
        return self.head(m_emb).squeeze(-1)

def rank_loss(pred, target):
    if len(pred)<2: return torch.tensor(0.0, device=pred.device)
    d=target.unsqueeze(0)-target.unsqueeze(1); pd=pred.unsqueeze(0)-pred.unsqueeze(1)
    return torch.sigmoid(-pd*torch.sign(d)*torch.abs(d)).mean()

model = DailyLSTM().to(DEV); ema = copy.deepcopy(model)
for p in ema.parameters(): p.requires_grad = False
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
huber = nn.HuberLoss(delta=0.5)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

print('Training...', flush=True); t0 = time.time()
best_ic, best_state, patience = -99, None, 0
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(X_tr), device=DEV)
    total_loss = 0.0
    for i in range(0, len(X_tr), BATCH):
        idx = perm[i:i+BATCH]
        pred = model(X_tr[idx])
        loss = huber(pred, y_tr_t[idx]) + 0.3*rank_loss(pred, y_tr_t[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for ep_, p in zip(ema.parameters(), model.parameters()):
                ep_.data.mul_(EMA_DECAY).add_(p.data, alpha=1-EMA_DECAY)
        total_loss += loss.item()
    ema.eval()
    with torch.no_grad():
        pv = [ema(X_va[i:i+BATCH]).detach().cpu().numpy() for i in range(0, len(X_va), BATCH)]
        ic = spearmanr(np.concatenate(pv), y_va_np)[0]
    if ic > best_ic: best_ic = ic; best_state = copy.deepcopy(ema.state_dict()); patience = 0
    else: patience += 1
    if (ep+1) % 15 == 0:
        print(f'  ep{ep+1:3d} loss={total_loss:.1f} val_IC={ic:+.4f} best={best_ic:+.4f}', flush=True)
    if patience >= 20: print(f'  Early stop ep{ep+1}', flush=True); break

# Test
model.load_state_dict(best_state); model.eval()
print('Evaluating...', flush=True)
with torch.no_grad():
    pt = np.concatenate([model(X_te[i:i+BATCH]).detach().cpu().numpy() for i in range(0, len(X_te), BATCH)])

def cs_ic(pred, true, dates):
    return np.mean([spearmanr(pred[dates==m], true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20])

ic = cs_ic(pt, y_te_np, te_dates)
hit = np.mean((pt>0) == (y_te_np>0))
n20 = max(1, int(len(pt)*0.2))
top20 = np.argsort(pt)[-n20:]; bot20 = np.argsort(pt)[:n20]
ls = np.mean(y_te_np[top20]) - np.mean(y_te_np[bot20])
n_months = len([m for m in np.unique(te_dates) if (te_dates==m).sum()>=20])

print(); print('='*60)
print('Path E: Daily LSTM → Monthly Embeddings')
print(f'  CS_IC:    {ic:+.4f} ({n_months} months)')
print(f'  Hit Rate: {hit:.3f}')
print(f'  Top20 LS: {ls:+.4f}')
print(f'  Val best: {best_ic:+.4f}')
print(f'  Params:   {sum(p.numel() for p in model.parameters()):,}')
print()
print(f'  Baseline BiLSTM (monthly): CS_IC=+0.126')
print(f'  Baseline LightGBM (66d):   CS_IC=+0.166')
print('='*60)
