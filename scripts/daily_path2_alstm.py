# Daily LSTM Path 2+4: ALSTM + Multi-seed Ensemble
# Input Attention + Temporal Attention, 10 seeds, ensemble evaluation
import warnings; warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, pandas as pd, sqlite3, time
from scipy.stats import spearmanr

DEV = torch.device('cuda')
LOOKBACK, BATCH = 60, 1024
HIDDEN, N_LAYERS, DROPOUT = 64, 2, 0.0
LR, WD, EPOCHS = 0.001, 0, 200
N_SEEDS = 10
DB = '.eastmoney-ai/db/klines-v2.sqlite'
print(f'Path 2+4: ALSTM {N_LAYERS}Lx{HIDDEN}h, {LOOKBACK}d, {N_SEEDS} seeds', flush=True)
print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)

# ======== 1. Data (same as Path 1) ========
conn = sqlite3.connect(DB)
codes = [r[0] for r in conn.execute('SELECT code FROM daily_klines GROUP BY code HAVING COUNT(*)>=500').fetchall()]
params = ','.join('?' * len(codes))
df_all = pd.read_sql_query(
    f"SELECT code,date,open,close,high,low,volume,turnover_rate FROM daily_klines "
    f"WHERE code IN ({params}) AND date>='2010-01-01' ORDER BY code,date", conn, params=codes)
conn.close()
print(f'Stocks: {len(codes)}, rows: {len(df_all):,}', flush=True)

print('Building sequences...', flush=True); t0 = time.time()
X_list, y_list, date_list = [], [], []
for code in codes:
    g = df_all[df_all['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 66: continue
    c=g['close'].values.astype(float); o=g['open'].values.astype(float)
    h=g['high'].values.astype(float); l=g['low'].values.astype(float)
    v=g['volume'].values.astype(float)
    tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
    n=len(c); dates=g['date'].values
    vol_ma20=pd.Series(v).rolling(20).mean().fillna(1).values
    F=np.zeros((n,8),dtype=np.float32)
    for i in range(n):
        pc=max(abs(c[i-1]),0.01) if i>=1 else c[i]
        F[i,0]=o[i]/max(pc,0.01)-1; F[i,1]=h[i]/max(c[i],0.01)-1
        F[i,2]=l[i]/max(c[i],0.01)-1; F[i,3]=c[i]/max(pc,0.01)-1
        F[i,4]=v[i]/max(vol_ma20[i],1)-1
        F[i,5]=tr[i] if not np.isnan(tr[i]) and tr[i]<100 else 0
        F[i,6]=(h[i]-l[i])/max(c[i],0.01)
        F[i,7]=(c[i]-o[i])/max(o[i],0.01) if o[i]>0 else 0
    F=np.nan_to_num(F,0.0).astype(np.float32)
    for i in range(LOOKBACK-1, n-63):
        fwd=(c[i+63]-c[i])/max(c[i],0.01)
        if abs(fwd)>3: continue
        X_list.append(F[i-LOOKBACK+1:i+1]); y_list.append(np.clip(fwd,-3,3))
        date_list.append(str(dates[i])[:7])

X=np.array(X_list,dtype=np.float32); y=np.array(y_list,dtype=np.float32)
dates_arr=np.array(date_list)
v=~np.isnan(X).any(axis=(1,2))&~np.isnan(y); X=X[v]; y=y[v]; dates_arr=dates_arr[v]
tr_m=(dates_arr>='2015-01')&(dates_arr<='2021-12')
va_m=(dates_arr>='2022-01')&(dates_arr<='2023-12')
te_m=(dates_arr>='2024-01')
print(f'Data: {len(X):,} T={tr_m.sum():,} V={va_m.sum():,} Te={te_m.sum():,} ({time.time()-t0:.0f}s)', flush=True)

fm=X[tr_m].reshape(-1,X.shape[-1]).mean(0); fs=X[tr_m].reshape(-1,X.shape[-1]).std(0)+1e-8
X=np.clip((X-fm)/fs,-5,5)
X_tr=torch.from_numpy(X[tr_m]).float().to(DEV); y_tr_t=torch.from_numpy(y[tr_m]).float().to(DEV)
X_va=torch.from_numpy(X[va_m]).float().to(DEV); y_va=y[va_m]
X_te=torch.from_numpy(X[te_m]).float().to(DEV); y_te_np=y[te_m]; te_dates=dates_arr[te_m]

# ======== 2. ALSTM Architecture (Qlib-standard) ========
class ALSTM(nn.Module):
    """Qlib-style ALSTM: Feature Gate + LSTM + Temporal Attention"""
    def __init__(self, in_dim=8, hidden=64, num_layers=2):
        super().__init__()
        self.hidden = hidden
        # Feature-wise linear + sigmoid gate (per-sample, no cross-batch leak)
        self.feat_gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        # LSTM encoder
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True, dropout=0.0)
        # Temporal Attention
        self.attn = nn.Sequential(nn.Linear(hidden, hidden//2), nn.Tanh(), nn.Linear(hidden//2, 1))
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # Feature gate: compute per-sample gate (same gate across time steps)
        gate = self.feat_gate(x.mean(dim=1))  # (B, T, F) -> (B, F) -> gate (B, F)
        x = x * gate.unsqueeze(1)  # broadcast to (B, T, F)
        # LSTM
        o, _ = self.lstm(x)  # (B, T, H)
        # Temporal attention
        w = self.attn(o).squeeze(-1)  # (B, T)
        w = torch.nn.functional.softmax(w, dim=1)
        context = (w.unsqueeze(-1) * o).sum(dim=1)  # (B, H)
        return self.fc(context).squeeze(-1)

# ======== 3. Train with N seeds ========
def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates==m],true[dates==m])[0] for m in np.unique(dates) if (dates==m).sum()>=20]
    return np.mean(ics) if ics else np.nan

print(f'\nTraining {N_SEEDS} seeds...', flush=True)
results = []
all_preds = []

for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ALSTM(X.shape[2], HIDDEN, N_LAYERS).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    best_va, best_state, patience = -99, None, 0
    t_s = time.time()

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr), device=DEV)
        total_loss = 0.0
        for i in range(0, len(X_tr), BATCH):
            idx = perm[i:i+BATCH]
            pred = model(X_tr[idx])
            loss = loss_fn(pred, y_tr_t[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(X_va[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_va),BATCH)])
            ic = spearmanr(pv, y_va)[0]

        if ic > best_va:
            best_va = ic; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= 20: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = np.concatenate([model(X_te[i:i+BATCH]).cpu().numpy() for i in range(0,len(X_te),BATCH)])

    all_preds.append(pt)
    ic_test = cs_ic(pt, y_te_np, te_dates)
    ic_raw = spearmanr(pt, y_te_np)[0]
    hit = np.mean((pt>0)==(y_te_np>0))
    n20 = max(1,int(len(pt)*0.2))
    ls = np.mean(y_te_np[np.argsort(pt)[-n20:]]) - np.mean(y_te_np[np.argsort(pt)[:n20]])
    params_count = sum(p.numel() for p in model.parameters())
    elapsed = time.time()-t_s
    results.append({'seed':seed,'IC':ic_test,'IC_raw':ic_raw,'Hit':hit,'Top20LS':ls,
                    'val_best':best_va,'params':params_count,'time':elapsed,'epochs':ep+1})
    print(f'  s{seed}: IC={ic_test:+.4f} Hit={hit:.3f} LS={ls:+.4f} val={best_va:+.4f} ep={ep+1} ({elapsed:.0f}s)', flush=True)

# ======== 4. Ensemble Evaluation ========
all_preds = np.array(all_preds)  # (N_SEEDS, N_test)
ens_pred = all_preds.mean(axis=0)
ens_ic = cs_ic(ens_pred, y_te_np, te_dates)
ens_hit = np.mean((ens_pred>0)==(y_te_np>0))
n20 = max(1,int(len(ens_pred)*0.2))
ens_ls = np.mean(y_te_np[np.argsort(ens_pred)[-n20:]]) - np.mean(y_te_np[np.argsort(ens_pred)[:n20]])

# Top-K ensemble (pick K best seeds by val)
sorted_seeds = sorted(results, key=lambda r: r['val_best'], reverse=True)
for k in [3, 5, 7]:
    top_seeds = [r['seed'] for r in sorted_seeds[:k]]
    sub_pred = all_preds[top_seeds].mean(axis=0)
    sub_ic = cs_ic(sub_pred, y_te_np, te_dates)
    sub_hit = np.mean((sub_pred>0)==(y_te_np>0))
    print(f'  Top-{k} ensemble: IC={sub_ic:+.4f} Hit={sub_hit:.3f}')

# ======== 5. Summary ========
ics = [r['IC'] for r in results]
print(); print('='*65)
print('Path 2+4 Results: ALSTM + Multi-seed Ensemble')
print('='*65)
print(f'  Seeds: {N_SEEDS}, Architecture: ALSTM {N_LAYERS}Lx{HIDDEN}h')
print(f'  Mean CS_IC: {np.mean(ics):+.4f} +/- {np.std(ics):.4f}')
print(f'  Best seed:  {np.max(ics):+.4f}')
print(f'  Worst seed: {np.min(ics):+.4f}')
print(f'  Ensemble:   {ens_ic:+.4f} (Hit={ens_hit:.3f}, LS={ens_ls:+.4f})')
print()
print(f'  Qlib ALSTM IC: 0.050 +/- 0.00')
print(f'  Qlib LSTM IC:  0.045 +/- 0.00')
print(f'  Our LSTM (path1) IC: 0.044 +/- 0.011')
print(f'  Our ALSTM IC:        {np.mean(ics):.3f} +/- {np.std(ics):.3f}')
print(f'  Improvement:          {(np.mean(ics)-0.0435):+.4f} vs path1 LSTM')
print('='*65)
