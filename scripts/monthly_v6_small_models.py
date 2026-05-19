"""Monthly v6: smaller models for small dataset (16K seqs).
LSTM-7 is 938K params on 16K sequences → severe overparameterization."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEV = torch.device('cuda')
B, LR, WD, EPOCHS = 128, 5e-4, 1e-4, 30

# Build monthly strictly (same as before but with file, no f-string issues)
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
params_str = ','.join('?' * len(stocks))
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({params_str}) AND date >= '2010-01' ORDER BY code, date", conn, params=stocks)
conn.close()

all_seq, all_y, all_dates = [], [], []
for code in stocks:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 84: continue
    n = len(g); dates = g['date'].tolist()
    c = g['close'].values.astype(float); o = g['open'].values.astype(float)
    h = g['high'].values.astype(float); l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    feats = np.zeros((n, 21), dtype=np.float32)
    for j, arr in enumerate([c, o, h, l, v]):
        s = pd.Series(arr); m = s.rolling(60, min_periods=60).mean(); std = s.rolling(60, min_periods=60).std()
        feats[:, j] = ((arr - m) / std.replace(0, 1)).fillna(0).values
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    dif = np.nan_to_num(e12 - e26, 0); dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:, 5] = dif; feats[:, 6] = dea; feats[:, 7] = (dif - dea) * 2
    delta = np.diff(c, prepend=c[0]); gs = np.where(delta > 0, delta, 0); ls = np.where(delta < 0, -delta, 0)
    feats[:, 8] = np.nan_to_num(100 - 100 / (1 + pd.Series(gs).ewm(alpha=1/14).mean().values / np.maximum(pd.Series(ls).ewm(alpha=1/14).mean().values, 1e-8)), 50)
    ma20 = pd.Series(c).rolling(20).mean().values; m60 = pd.Series(c).rolling(60).mean().values
    feats[:, 18] = np.nan_to_num((c - ma20) / np.maximum(c, 0.01), 0)
    feats[:, 19] = np.nan_to_num((c - m60) / np.maximum(c, 0.01), 0)
    feats[:, 20] = hash(code) % 31 / 31.0
    feats = np.nan_to_num(feats, 0.0)

    LOOKBACK = 24  # shorter lookback for monthly
    for i in range(LOOKBACK - 1, n - 12):
        if c[i] <= 0.01: continue
        seq = feats[i - LOOKBACK + 1 : i + 1]
        y3 = np.clip((c[i+3] - c[i]) / max(c[i], 0.01), -2, 2) if i + 3 < n else 0
        y6 = np.clip((c[i+6] - c[i]) / max(c[i], 0.01), -2, 2) if i + 6 < n else 0
        all_seq.append(seq); all_y.append([y3, y6]); all_dates.append(dates[i])

X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
dates_arr = np.array(all_dates)
train_m = np.array([d >= '2015-01' and d <= '2021-12' for d in dates_arr])
test_m = np.array([d >= '2024-01' for d in dates_arr])
Xtr, ytr = X[train_m], y[train_m]; Xtr, ytr = Xtr[~np.isnan(ytr).any(axis=1)], ytr[~np.isnan(ytr).any(axis=1)]
Xte, yte = X[test_m], y[test_m]; Xte, yte = Xte[~np.isnan(yte).any(axis=1)], yte[~np.isnan(yte).any(axis=1)]
print(f"Monthly v6 (lookback=24): train={Xtr.shape}, test={Xte.shape}")

def train_test(arch, dim):
    torch.manual_seed(456); np.random.seed(456)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    ld = torch.utils.data.DataLoader(ds, min(B, len(Xtr)//4), shuffle=True, pin_memory=True)
    model = create_model(arch, dim).to(DEV)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
    for ep in range(1, EPOCHS+1):
        for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
        model.train()
        for Xb, yb in ld:
            Xb, yb = Xb.to(DEV), yb.to(DEV); opt.zero_grad()
            p = model(Xb)
            loss = 0.5*nn.MSELoss()(p[:,0], yb[:,0]) + 0.5*nn.MSELoss()(p[:,1], yb[:,1])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    test_ld = torch.utils.data.DataLoader(test_ds, B, pin_memory=True)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for Xb, yb in test_ld:
            Xb, yb = Xb.to(DEV), yb.to(DEV); p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
    tp = np.concatenate(tp); tt = np.concatenate(tt)
    ic3 = spearmanr(tp[:,0], tt[:,0])[0]
    n = len(tp); cut = int(n*0.3); idx = np.argsort(tp[:,0])
    ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
    sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0
    print(f"  {arch:12s} params={params:>8,} | Test IC3={ic3:+.4f} SR3={sr3:+.3f}")
    return ic3

for arch in ['LSTM-1', 'LSTM-2', 'LSTM-3']:
    train_test(arch, 21)

# Daily baseline
print(f"\nDaily v6 baseline: IC3=+0.141 (LSTM-7, 381K seqs)")
print(f"Monthly: {len(Xtr)} seqs vs Daily: 381K seqs (24x more data)")
best_ic = -1
for arch in ['LSTM-1', 'LSTM-2']:
    ic = train_test(arch, 21)
    if ic > best_ic: best_ic = ic
print(f"\nBest monthly IC3: {best_ic:+.4f}")
if best_ic > 0.03: print("Monthly signal FOUND with smaller model!")
else: print("Monthly signal NOT FOUND even with small model. Data volume is the bottleneck.")
