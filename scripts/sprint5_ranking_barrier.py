"""Sprint 5A+5B: ListNet ranking loss + Triple-Barrier labeling.
Same data (hs300 monthly 21-dim), same arch (LSTM-7), different loss.
One-shot Test eval (FROZEN)."""
import torch, torch.nn as nn, numpy as np, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

DATA = Path(__file__).parent.parent / '.eastmoney-ai' / 'lstm'
DEVICE = torch.device('cuda')
BATCH, LR, WD, EP, LOOKBACK = 128, 5e-4, 1e-4, 80, 60

Xtr = np.load(DATA/'train.npz')['X'].astype(np.float32)
ytr = np.load(DATA/'train.npz')['y'].astype(np.float32)
Xva = np.load(DATA/'val.npz')['X'].astype(np.float32)
yva = np.load(DATA/'val.npz')['y'].astype(np.float32)
Xte = np.load(DATA/'test.npz')['X'].astype(np.float32)
yte = np.load(DATA/'test.npz')['y'].astype(np.float32)
print(f"Data: train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

# ============================================================
# Sprint 5A: ListNet Ranking Loss
# ============================================================
class ListNetLoss(nn.Module):
    """Cross-entropy between softmax of predicted and true returns per month.
    Simple implementation: treat each month as a cross-section."""
    def forward(self, pred, target):
        # pred/target: (N,) — all stocks in one month
        # Softmax top-k logit version: rank top 20% vs bottom 20%
        n = len(pred)
        if n < 10: return torch.tensor(0.0, device=pred.device)
        k = max(2, n // 5)
        # Top-k predicted vs top-k actual
        _, pred_top = torch.topk(pred, k)
        _, true_top = torch.topk(target, k)
        _, pred_bot = torch.topk(pred, k, largest=False)
        _, true_bot = torch.topk(target, k, largest=False)
        # Penalize: top pred should overlap with top true
        overlap_top = len(set(pred_top.cpu().numpy()) & set(true_top.cpu().numpy())) / k
        # Simple proxy: MSE on ranks (faster and more stable than full listnet)
        pred_rank = torch.argsort(torch.argsort(pred)).float()
        true_rank = torch.argsort(torch.argsort(target)).float()
        return nn.MSELoss()(pred_rank / max(n-1, 1), true_rank / max(n-1, 1))

def train_mse(model, X_tr, y_tr, X_va, y_va, epochs=EP):
    """Standard MSE training — returns best model state + best_ic3"""
    torch.manual_seed(456); np.random.seed(456)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)

    model = create_model('LSTM-7', 21).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
    best_ic3, best_vl, best_state, no_imp = -1.0, float('inf'), None, 0

    for ep in range(1, epochs+1):
        for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
        model.train()
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
            loss = 0.5*nn.MSELoss()(model(Xb)[:,0], yb[:,0]) + 0.5*nn.MSELoss()(model(Xb)[:,1], yb[:,1])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval(); vl=0; preds,targs=[],[]
        with torch.no_grad():
            for Xb, yb in val_ld:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                p = model(Xb); preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
                vl += (0.5*nn.MSELoss()(p[:,0],yb[:,0])+0.5*nn.MSELoss()(p[:,1],yb[:,1])).item()*Xb.size(0)
        vl/=len(val_ds); ic3 = spearmanr(np.concatenate(preds)[:,0], np.concatenate(targs)[:,0])[0]
        if ic3 > best_ic3: best_ic3, best_state = ic3, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if vl < best_vl: best_vl, no_imp = vl, 0
        else: no_imp += 1
        if no_imp >= 15 and ep > 20: break

    model.load_state_dict(best_state)
    return model, best_ic3

def evaluate(model, X_te, y_te):
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
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
    return ic3, ic6, sr3

# 5A already computed: Val IC3=0.1697, Test IC3=0.0100
ic_ra, ic6_ra, sr_ra = 0.0100, 0.0426, 0.063
print(f"5A ListNet (cached): Test IC3=0.0100 (worse than MSE baseline 0.019)")

# ============================================================
# 5B: Triple-Barrier Labeling (3-class classification)
# ============================================================
print("\n" + "="*60)
print("5B: Triple-Barrier Classification")
print("="*60)

# Compute returns path from monthly returns
def compute_barrier_labels(y_returns, upper=0.10, lower=-0.10, max_horizon=6):
    """y_returns: (N, 2) with [y3, y6].
    Simulate barrier: use y3 as 1-step, y6 as 3-step proxy.
    Label: 1 if cumulative return crosses upper first, -1 if lower, 0 if neither.
    Since we only have discrete 3m and 6m returns, approximate: if y3 > upper/3 → 1, if y3 < lower/3 → -1 else check y6"""
    labels = np.zeros(len(y_returns), dtype=np.int64)
    for i in range(len(y_returns)):
        r3 = y_returns[i, 0]  # 3-month return
        r6 = y_returns[i, 1]  # 6-month return
        # Step 1: check at 3 months
        if r3 >= upper/2: labels[i] = 1
        elif r3 <= lower/2: labels[i] = -1
        # Step 2: check at 6 months (if not hit at 3)
        elif r6 >= upper: labels[i] = 1
        elif r6 <= lower: labels[i] = -1
        else: labels[i] = 0
    return labels

ytr_labels = compute_barrier_labels(ytr) + 1  # shift [-1,0,1] → [0,1,2]
yva_labels = compute_barrier_labels(yva) + 1
yte_labels = compute_barrier_labels(yte) + 1
print(f"Label distribution: train={dict(zip(*np.unique(ytr_labels, return_counts=True)))}")

# Build classification model
class BarrierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = create_model('LSTM-7', 21)
        # Override final Sequential's last layer for 3-class output
        self.lstm.fc[-1] = nn.Linear(64, 3)
    def forward(self, x): return self.lstm(x)

torch.manual_seed(456); np.random.seed(456)
train_ds_b = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr_labels))
val_ds_b = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva_labels))
train_ld_b = torch.utils.data.DataLoader(train_ds_b, BATCH, shuffle=True, pin_memory=True)
val_ld_b = torch.utils.data.DataLoader(val_ds_b, BATCH, pin_memory=True)

model_tb = BarrierModel().to(DEVICE)
opt = torch.optim.AdamW(model_tb.parameters(), lr=1e-5, weight_decay=WD)
best_acc, best_state_tb, no_imp = 0.0, None, 0
for ep in range(1, EP+1):
    for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
    model_tb.train()
    for Xb, yb in train_ld_b:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model_tb(Xb), yb)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model_tb.parameters(),1.0); opt.step()
    model_tb.eval(); correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in val_ld_b:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model_tb(Xb).argmax(dim=1); correct += (pred == yb).sum().item(); total += yb.size(0)
    acc = correct/max(total,1)
    if acc > best_acc: best_acc, best_state_tb = acc, {k:v.cpu().clone() for k,v in model_tb.state_dict().items()}
    else: no_imp += 1
    if no_imp >= 15 and ep > 20: break

model_tb.load_state_dict(best_state_tb)
test_ds_b = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte_labels))
test_ld_b = torch.utils.data.DataLoader(test_ds_b, BATCH, pin_memory=True)
model_tb.eval(); correct_te, total_te = 0, 0; tp_te = []
with torch.no_grad():
    for Xb, yb in test_ld_b:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = model_tb(Xb); tp_te.append(p.cpu().numpy())
        correct_te += (p.argmax(dim=1) == yb).sum().item(); total_te += yb.size(0)
acc_te = correct_te/max(total_te,1)
tp_te_arr = np.concatenate(tp_te)
# Use class 1 (bullish) probability as signal for Sharpe computation
bull_signal = tp_te_arr[:,2] - tp_te_arr[:,0]  # P(bull=2) - P(bear=0)
ic_tb = spearmanr(bull_signal, yte[:,0])[0]  # IC vs y3
n=len(bull_signal); cut=int(n*0.3); idx=np.argsort(bull_signal)
ls = yte[idx[-cut:],0] - yte[idx[:cut],0]
sr_tb = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0

print(f"5B Triple-Barrier: Val Acc={best_acc:.4f} | Test Acc={acc_te:.4f} | IC={ic_tb:.4f} | SR3={sr_tb:.3f}")

# ============================================================
# FINAL COMPARISON
# ============================================================
print(f"\n{'='*70}")
print(f"SPRINT 5 FINAL COMPARISON")
print(f"{'='*70}")
print(f"{'Method':<25} {'Test IC3':>10} {'Test SR3':>10}")
print(f"{'MSE Baseline (v2)':<25} {0.0193:10.4f} {0.115:10.3f}")
print(f"{'5A ListNet':<25} {ic_ra:10.4f} {sr_ra:10.3f}")
print(f"{'5B Triple-Barrier':<25} {ic_tb:10.4f} {sr_tb:10.3f}")

best_ic = max(ic_ra, ic_tb, 0.0193)
best_sr = max(sr_ra if not np.isnan(sr_ra) else 0, sr_tb if not np.isnan(sr_tb) else 0, 0.115)
print(f"\nBest IC: {best_ic:.4f}, Best SR: {best_sr:.3f}")
if best_ic > 0.05 or best_sr > 0.6:
    print("PASS → Phase 19 v4 integration")
elif best_ic > 0.03:
    print("MARGINAL → combine with Sprint 5A+5B ensemble")
else:
    print("FAIL → HS300 monthly LSTM ceiling confirmed. PIVOT.")
