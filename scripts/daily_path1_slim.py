# Daily LSTM Path 1: Architecture slim-down (Qlib-standard)
# 2 layers, hidden=64, lookback=60, 6 raw features, 5 seeds
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr
from pathlib import Path

DEV = torch.device('cuda')
LOOKBACK, BATCH = 60, 1024
HIDDEN, N_LAYERS, DROPOUT = 64, 2, 0.0
LR, WD, EPOCHS = 0.001, 0, 200
N_SEEDS = 5
DB = '.eastmoney-ai/db/klines-v2.sqlite'
print(f'Daily Path 1: {N_LAYERS}Lx{HIDDEN}h, {LOOKBACK}d lookback, {N_SEEDS} seeds', flush=True)
print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)

# ======== 1. Alpha360-style features (6 raw ratios) ========
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute(
    'SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=500'
).fetchall()]
n_stocks = len(codes)
print(f'Stocks: {n_stocks} with >=500 daily bars', flush=True)

# Load all daily data
params = ','.join('?' * len(codes))
df_all = pd.read_sql_query(
    f"SELECT code,date,open,close,high,low,volume,turnover_rate FROM daily_klines "
    f"WHERE code IN ({params}) AND date>='2010-01-01' ORDER BY code,date",
    conn, params=codes)
conn.close()
print(f'Daily rows: {len(df_all):,}', flush=True)

# Build sequences
print('Building sequences...', flush=True); t0 = time.time()
X_list, y_list, date_list = [], [], []

for code in codes:
    g = df_all[df_all['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 66: continue  # need 60 lookback + ~66 for 3m forward target
    c = g['close'].values.astype(float)
    o = g['open'].values.astype(float)
    h = g['high'].values.astype(float)
    l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    dates = g['date'].values
    n = len(c)
    vol_ma20 = pd.Series(v).rolling(20).mean().fillna(1).values

    # 6 raw ratio features (Alpha360 style)
    F = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        pc = max(abs(c[i-1]), 0.01) if i >= 1 else c[i]
        F[i,0] = o[i] / max(pc, 0.01) - 1       # open / prev_close
        F[i,1] = h[i] / max(c[i], 0.01) - 1       # high / close
        F[i,2] = l[i] / max(c[i], 0.01) - 1       # low / close
        F[i,3] = c[i] / max(pc, 0.01) - 1         # close / prev_close (return)
        F[i,4] = v[i] / max(vol_ma20[i], 1) - 1   # volume ratio
        F[i,5] = tr[i] if not np.isnan(tr[i]) and tr[i] < 100 else 0  # turnover rate
        F[i,6] = (h[i]-l[i]) / max(c[i], 0.01)     # amplitude
        F[i,7] = (c[i]-o[i]) / max(o[i], 0.01) if o[i] > 0 else 0  # intraday return

    F = np.nan_to_num(F, 0.0).astype(np.float32)

    # Build 60-day windows, predict 3-month forward (~63 trading days)
    target_offset = 63  # ~3 months of trading days
    for i in range(LOOKBACK - 1, n - target_offset):
        fwd = (c[i+target_offset] - c[i]) / max(c[i], 0.01)
        if abs(fwd) > 3: continue
        X_list.append(F[i-LOOKBACK+1:i+1])
        y_list.append(np.clip(fwd, -3, 3))
        date_list.append(str(dates[i])[:7])  # year-month

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
dates_arr = np.array(date_list)
v = ~np.isnan(X).any(axis=(1,2)) & ~np.isnan(y)
X = X[v]; y = y[v]; dates_arr = dates_arr[v]

# Strict time split
tr_m = (dates_arr >= '2015-01') & (dates_arr <= '2021-12')
va_m = (dates_arr >= '2022-01') & (dates_arr <= '2023-12')
te_m = (dates_arr >= '2024-01')
print(f'Data: {len(X):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({X.shape[2]}feat) {time.time()-t0:.0f}s', flush=True)

# Normalize per-feature
fm = X[tr_m].reshape(-1, X.shape[-1]).mean(0)
fs = X[tr_m].reshape(-1, X.shape[-1]).std(0) + 1e-8
X = np.clip((X - fm) / fs, -5, 5)

X_tr = torch.from_numpy(X[tr_m]).float().to(DEV); y_tr = torch.from_numpy(y[tr_m]).float().to(DEV)
X_va = torch.from_numpy(X[va_m]).float().to(DEV); y_va = y[va_m]
X_te = torch.from_numpy(X[te_m]).float().to(DEV); y_te_np = y[te_m]; te_dates = dates_arr[te_m]

# ======== 2. Slim LSTM (Qlib-standard) ========
class SlimLSTM(nn.Module):
    def __init__(self, in_dim=8, hidden=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)

# ======== 3. Train with N seeds ========
def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates==m], true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan

print(f'\nTraining {N_SEEDS} seeds...', flush=True)
results = []
for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SlimLSTM(X.shape[2], HIDDEN, N_LAYERS, DROPOUT).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    best_va, best_state, patience = -99, None, 0

    t_s = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr), device=DEV)
        for i in range(0, len(X_tr), BATCH):
            idx = perm[i:i+BATCH]
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(X_va[i:i+BATCH]).cpu().numpy() for i in range(0, len(X_va), BATCH)])
            ic = spearmanr(pv, y_va)[0]

        if ic > best_va:
            best_va = ic; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= 20: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = np.concatenate([model(X_te[i:i+BATCH]).cpu().numpy() for i in range(0, len(X_te), BATCH)])

    ic_test = cs_ic(pt, y_te_np, te_dates)
    ic_raw = spearmanr(pt, y_te_np)[0]
    hit = np.mean((pt>0) == (y_te_np>0))
    n20 = max(1,int(len(pt)*0.2))
    ls = np.mean(y_te_np[np.argsort(pt)[-n20:]]) - np.mean(y_te_np[np.argsort(pt)[:n20]])
    params_count = sum(p.numel() for p in model.parameters())
    elapsed = time.time() - t_s

    results.append({'seed':seed, 'IC':ic_test, 'IC_raw':ic_raw, 'Hit':hit, 'Top20LS':ls,
                    'val_best':best_va, 'params':params_count, 'time':elapsed, 'epochs':ep+1})
    print(f'  seed {seed}: CS_IC={ic_test:+.4f} Hit={hit:.3f} LS={ls:+.4f} '
          f'val={best_va:+.4f} ep={ep+1} params={params_count:,} ({elapsed:.0f}s)', flush=True)

# ======== 4. Summary ========
ics = [r['IC'] for r in results]
hits = [r['Hit'] for r in results]
print(); print('='*65)
print('Path 1 Results: Slim LSTM (2-layer, 64-hidden, 60-day lookback)')
print('='*65)
print(f'  {"Seed":<8s} {"CS_IC":>8s} {"Hit":>8s} {"Top20LS":>8s} {"Params":>10s} {"Time":>6s}')
print('-'*55)
for r in results:
    print(f'  {r["seed"]:<8d} {r["IC"]:+8.4f} {r["Hit"]:8.3f} {r["Top20LS"]:+8.4f} '
          f'{r["params"]:>10,} {r["time"]:>5.0f}s')
print('-'*55)
print(f'  {"Mean":<8s} {np.mean(ics):+8.4f} {np.mean(hits):8.3f}')
print(f'  {"Std":<8s} {np.std(ics):8.4f} {np.std(hits):8.3f}')
print(f'  {"Best":<8s} {np.max(ics):+8.4f}')
print(f'  {"Worst":<8s} {np.min(ics):+8.4f}')
print()
print(f'  Target: std < 0.005 (Qlib level)')
print(f'  Actual std: {np.std(ics):.4f}')
print(f'  Qlib LSTM IC: 0.045 +/- 0.00 (Alpha360, CSI300)')
print(f'  Qlib ALSTM IC: 0.050 +/- 0.00 (Alpha360, CSI300)')
print(f'  Old 7-layer LSTM IC: {0.014}-{0.062} (seed sensitive)')
print('='*65)
