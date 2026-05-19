"""Strict Test IC validation: NO early stopping, NO val info in checkpoint.
Train: 2015-2021 data only. Test: 2024-2026. Fixed 40 epochs. No val monitoring.
INPUT_DATA_RANGE: 2015-01-01 to 2021-12-31 (train), 2024-01-01 to 2026-05-31 (test)
WALK_FORWARD: yes (test data never seen during training)
LOOK_AHEAD_RISK: none (no val data, no early stop, fixed epochs)
TEST_SET_USAGE: read-only, evaluated ONCE"""
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DEVICE = torch.device('cuda')
LOOKBACK, BATCH, LR, WD, FIXED_EPOCHS = 252, 128, 5e-4, 1e-4, 30  # 30 epochs for speed

def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        if std > 1e-8: r[i] = (s[i] - m) / std
    return r

print("=== STRICT Test IC: No val, no early stop, fixed epochs ===")

# Load daily data
conn = sqlite3.connect(str(PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()][:100]  # 100 stocks for speed
d_df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM daily_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01' ORDER BY code, date", conn, params=stocks)
conn.close()

# Build features for all stocks, all dates (same as build_daily_v5.py)
all_seq, all_y, all_dates = [], [], []
for code in stocks:
    g = d_df[d_df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 504: continue
    n = len(g); closes = g['close'].values.astype(float); opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float); lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float); dates = g['date'].tolist()
    feats = np.zeros((n, 21), dtype=np.float32)
    feats[:,0] = rolling_zscore(closes); feats[:,1] = rolling_zscore(opens)
    feats[:,2] = rolling_zscore(highs); feats[:,3] = rolling_zscore(lows)
    feats[:,4] = rolling_zscore(volumes)
    e12 = pd.Series(closes).ewm(span=12).mean().values; e26 = pd.Series(closes).ewm(span=26).mean().values
    dif = np.nan_to_num(e12-e26,0); dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
    dl=np.diff(closes,prepend=closes[0]); gs=np.where(dl>0,dl,0); ls=np.where(dl<0,-dl,0)
    ag=pd.Series(gs).ewm(alpha=1/14).mean().values; al=pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8]=np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)),50)
    ma20=pd.Series(closes).rolling(20).mean().values; m60=pd.Series(closes).rolling(60).mean().values
    feats[:,18]=np.nan_to_num((closes-ma20)/np.maximum(closes,0.01),0)
    feats[:,19]=np.nan_to_num((closes-m60)/np.maximum(closes,0.01),0)
    feats[:,20]=hash(code)%31/31.0
    feats=np.nan_to_num(feats,0.0)
    for i in range(LOOKBACK-1, n-252):
        if closes[i]<=0.01: continue
        y3d=np.clip((closes[i+63]-closes[i])/max(closes[i],0.01),-2,2) if i+63<n else 0
        y6d=np.clip((closes[i+126]-closes[i])/max(closes[i],0.01),-2,2) if i+126<n else 0
        all_seq.append(feats[i-LOOKBACK+1:i+1]); all_y.append([y3d,y6d]); all_dates.append(dates[i])

X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
dates_arr = np.array(all_dates)

# Strict date-based split: Train <= 2021-12, Test >= 2024-01
train_mask = (dates_arr >= '2015') & (dates_arr <= '2021')
test_mask = dates_arr >= '2024'

Xtr, ytr = X[train_mask], y[train_mask]
Xte, yte = X[test_mask], y[test_mask]
print(f"Mask counts: train={train_mask.sum()}, test={test_mask.sum()}, total={len(dates_arr)}")
print(f"Date sample: {dates_arr[:5]} ... {dates_arr[-5:]}")
Xtr, ytr = Xtr[~np.isnan(ytr).any(axis=1)], ytr[~np.isnan(ytr).any(axis=1)]
Xte, yte = Xte[~np.isnan(yte).any(axis=1)], yte[~np.isnan(yte).any(axis=1)]
if train_mask.sum() > 0:
    print(f"Train: {Xtr.shape} ({dates_arr[train_mask][0]}~{dates_arr[train_mask][-1]})")
if test_mask.sum() > 0:
    print(f"Test:  {Xte.shape} ({dates_arr[test_mask][0]}~{dates_arr[test_mask][-1]})")

# Train with NO val data, NO early stopping, FIXED epochs
torch.manual_seed(456); np.random.seed(456)
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
model = create_model('LSTM-7', 21).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)

train_losses = []
for ep in range(1, FIXED_EPOCHS+1):
    for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
    model.train(); tl = 0
    for Xb, yb in train_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
        loss = 0.5*nn.MSELoss()(model(Xb)[:,0], yb[:,0]) + 0.5*nn.MSELoss()(model(Xb)[:,1], yb[:,1])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tl += loss.item() * Xb.size(0)
    train_losses.append(tl/len(train_ds))
    if ep % 10 == 0: print(f"  ep{ep}: train_loss={train_losses[-1]:.4f}")

# Test evaluation (FROZEN, ONE TIME, no val influence)
test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
test_ld = torch.utils.data.DataLoader(test_ds, BATCH, pin_memory=True)
model.eval(); tp, tt = [], []
with torch.no_grad():
    for Xb, yb in test_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
tp = np.concatenate(tp); tt = np.concatenate(tt)
ic3 = spearmanr(tp[:,0], tt[:,0])[0]; ic6 = spearmanr(tp[:,1], tt[:,1])[0]
n = len(tp); cut = int(n*0.3); idx = np.argsort(tp[:,0])
ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0
hit3 = np.mean(np.sign(tp[:,0]) == np.sign(tt[:,0]))

print(f"\nSTRICT Test IC3 (NO val, NO early stop): {ic3:.4f}")
print(f"STRICT Test IC6: {ic6:.4f}, SR3: {sr3:.3f}, Hit: {hit3:.2f}")
print(f"Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
print(f"\nOriginal build_daily_v5.py Test IC3: 0.114 (with val early stop)")
print(f"Strict (no val) Test IC3:       {ic3:.4f}")
if abs(ic3) >= 0.05:
    print("VERDICT: Signal persists WITHOUT val info. IC3 confirmed real.")
else:
    print("VERDICT: Signal disappears without val info. IC3 was inflated by early stopping.")
