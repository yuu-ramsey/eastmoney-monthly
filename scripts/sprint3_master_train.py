"""Sprint 3: MASTER-inspired cross-stock model training + Test eval"""
import torch, torch.nn as nn, numpy as np, pandas as pd, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from master import MASTERBaseline

PROJECT = Path(__file__).parent.parent
DATA = PROJECT / '.eastmoney-ai' / 'lstm'
DEVICE = torch.device('cuda')
BATCH, LR, WD, EP, LOOKBACK = 128, 5e-4, 1e-4, 80, 60

print("Loading data...")
Xtr = np.load(DATA/'train.npz')['X'].astype(np.float32)
ytr = np.load(DATA/'train.npz')['y'].astype(np.float32)
Xva = np.load(DATA/'val.npz')['X'].astype(np.float32)
yva = np.load(DATA/'val.npz')['y'].astype(np.float32)
Xte = np.load(DATA/'test.npz')['X'].astype(np.float32)
yte = np.load(DATA/'test.npz')['y'].astype(np.float32)
print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

# Market features: for each sample, compute market context from the price data
# Market = avg close, vol, return of last 12 months
def compute_market_features(X, lookback=12):
    """Extract market context from OHLCV data"""
    n = len(X)
    mf = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        closes = X[i, :, 0]  # first feature is close z-score
        # Last 12 months of close data
        recent = closes[-min(lookback, len(closes)):]
        mf[i, 0] = recent.mean()  # avg close
        mf[i, 1] = recent.std()   # volatility
        if len(recent) >= 2:
            mf[i, 2] = recent[-1] - recent[0]  # trend
        mf[i, 3] = recent[-1] - recent[-min(3,len(recent))] if len(recent)>=3 else 0  # short mom
        mf[i, 4] = recent[-1]     # latest close
    return mf

mkt_tr = compute_market_features(Xtr)
mkt_va = compute_market_features(Xva)
mkt_te = compute_market_features(Xte)

# Pad y to 3 targets (y1, y3, y6)
ytr3 = np.column_stack([ytr[:,0]/3, ytr[:,0], ytr[:,1]])  # y1 = y3/3 approx
yva3 = np.column_stack([yva[:,0]/3, yva[:,0], yva[:,1]])
yte3 = np.column_stack([yte[:,0]/3, yte[:,0], yte[:,1]])

# Train
torch.manual_seed(456); np.random.seed(456)
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(mkt_tr), torch.from_numpy(ytr3))
val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(mkt_va), torch.from_numpy(yva3))
train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)

model = MASTERBaseline(stock_input_dim=21, market_input_dim=5, hidden=64, heads=4).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MASTER params: {n_params:,}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
best_ic3, best_vl, no_imp = -1.0, float('inf'), 0

for ep in range(1, EP+1):
    for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
    model.train()
    for Xb, mb, yb in train_ld:
        Xb, mb, yb = Xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        p = model(Xb, mb)
        loss = 0.2*nn.MSELoss()(p[:,0], yb[:,0]) + 0.5*nn.MSELoss()(p[:,1], yb[:,1]) + 0.3*nn.MSELoss()(p[:,2], yb[:,2])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval(); vl=0; preds,targs=[],[]
    with torch.no_grad():
        for Xb, mb, yb in val_ld:
            Xb, mb, yb = Xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            p = model(Xb, mb); preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
            vl += (0.2*nn.MSELoss()(p[:,0],yb[:,0])+0.5*nn.MSELoss()(p[:,1],yb[:,1])+0.3*nn.MSELoss()(p[:,2],yb[:,2])).item()*Xb.size(0)
    vl/=len(val_ds); ic3 = spearmanr(np.concatenate(preds)[:,1], np.concatenate(targs)[:,1])[0]
    best_ic3 = max(best_ic3, ic3)
    if vl < best_vl: best_vl, no_imp = vl, 0
    else: no_imp += 1
    if ep%10==0 or ep==1: print(f"  ep{ep}: vl={vl:.4f} ic3={ic3:.4f}")
    if no_imp >= 15 and ep > 20: break

# Test (FROZEN)
test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(mkt_te), torch.from_numpy(yte3))
test_ld = torch.utils.data.DataLoader(test_ds, BATCH, pin_memory=True)
model.eval(); tp, tt = [], []
with torch.no_grad():
    for Xb, mb, yb in test_ld:
        Xb, mb, yb = Xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
        p = model(Xb, mb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
tp = np.concatenate(tp); tt = np.concatenate(tt)
ic3_t = spearmanr(tp[:,1], tt[:,1])[0]
ic6_t = spearmanr(tp[:,2], tt[:,2])[0]
hit3 = np.mean(np.sign(tp[:,1]) == np.sign(tt[:,1]))
n=len(tp); cut=int(n*0.3); idx=np.argsort(tp[:,1])
ls = tt[idx[-cut:],1] - tt[idx[:cut],1]
sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0

print(f"\n{'='*60}")
print(f"SPRINT 3: MASTER RESULTS")
print(f"{'='*60}")
print(f"Params: {n_params:,} | Val IC3={best_ic3:.4f} | Test IC3={ic3_t:.4f} | IC6={ic6_t:.4f} | SR3={sr3:.3f} | Hit={hit3:.2f}")
print(f"\nBaseline LSTM-7 (21d): Val=0.180 Test=0.019")
print(f"MASTER (21d+market):  Val={best_ic3:.4f} Test={ic3_t:.4f}")
delta = ic3_t - 0.019
print(f"Δ vs baseline Test: {delta:+.4f}")
if ic3_t > 0.10: print("PASS > 0.10 → Phase 19 v5")
elif ic3_t > 0.08: print("MARGINAL 0.08-0.10 → Sprint 4")
else: print("FAIL < 0.08")
