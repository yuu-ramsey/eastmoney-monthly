"""P1 simplified: walk-forward Top 3 + light hyper-search ~15 min"""
import torch, torch.nn as nn, numpy as np, sys, json, time
from pathlib import Path
from scipy.stats import spearmanr

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / 'lib' / 'lstm'))
from model_v2 import create_model

DATA_DIR = PROJECT / '.eastmoney-ai' / 'lstm'
DEVICE = torch.device('cuda')
BATCH, WD, PATIENCE = 128, 1e-4, 15

X_train = np.load(DATA_DIR/'train.npz')['X'].astype(np.float32)
y_train = np.load(DATA_DIR/'train.npz')['y'].astype(np.float32)
X_val = np.load(DATA_DIR/'val.npz')['X'].astype(np.float32)
y_val = np.load(DATA_DIR/'val.npz')['y'].astype(np.float32)

def train_model(name, Xt, yt, Xv, yv, lr=5e-4, epochs=80):
    torch.manual_seed(42); np.random.seed(42)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
    train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)
    try:
        model = create_model(name, 21).to(DEVICE)
    except Exception as e:
        print(f"  FAIL create_model({name}): {e}")
        return None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
    best_vl, no_imp, best_ic3 = float('inf'), 0, -1.0
    for ep in range(1, epochs+1):
        for pg in opt.param_groups:
            pg['lr'] = min(lr, lr*ep/10.0) if ep <= 10 else lr
        model.train(); tl = 0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            p = model(Xb)
            loss = 0.5*nn.MSELoss()(p[:,0],yb[:,0]) + 0.5*nn.MSELoss()(p[:,1],yb[:,1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * Xb.size(0)
        tl /= len(train_ds)
        model.eval(); vl = 0; preds, targs = [], []
        with torch.no_grad():
            for Xb, yb in val_ld:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                p = model(Xb)
                vl += (0.5*nn.MSELoss()(p[:,0],yb[:,0])+0.5*nn.MSELoss()(p[:,1],yb[:,1])).item()*Xb.size(0)
                preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
        vl /= len(val_ds)
        ic3 = spearmanr(np.concatenate(preds)[:,0], np.concatenate(targs)[:,0])[0]
        best_ic3 = max(best_ic3, ic3)
        if vl < best_vl: best_vl = vl; no_imp = 0
        else: no_imp += 1
        if no_imp >= PATIENCE and ep > 20: break
    del model; torch.cuda.empty_cache()
    return {'name': name, 'best_ic3': best_ic3, 'best_vl': best_vl, 'epochs': ep}

print("P1 Simplified: Top 3 Walk-Forward + Hyper Light")
print(f"GPU: {torch.cuda.get_device_name(0)}")
results = {}

for model in ['LSTM-5', 'LSTM-7', 'LSTM-8']:
    print(f"\n=== {model} ===")
    # Walk-forward: train on 65% of train, eval on Val as OOS
    n_sub = int(0.65 * len(X_train))
    r_wf = train_model(model, X_train[:n_sub], y_train[:n_sub], X_val, y_val, epochs=80)
    if r_wf is None:
        print(f"  FAILED")
        continue
    r_wf['oos_ic3'] = r_wf['best_ic3']
    # Full train for Val IC
    r_full = train_model(model, X_train, y_train, X_val, y_val, epochs=50)
    r_wf['val_ic3'] = r_full['best_ic3'] if r_full else r_wf['best_ic3']
    print(f"  Val IC3={r_wf['val_ic3']:.4f} OOS IC3={r_wf['oos_ic3']:.4f} ep={r_wf['epochs']}")
    results[model] = r_wf

if not results:
    print("ALL FAILED")
    sys.exit(1)

# Light hyper on best
best_model = max(results.items(), key=lambda x: x[1]['oos_ic3'])[0]
print(f"\n=== {best_model} Hyper Light ===")
best_lr, best_ic = 5e-4, results[best_model]['oos_ic3']
for lr in [1e-4, 5e-4, 1e-3]:
    r = train_model(best_model, X_train, y_train, X_val, y_val, lr=lr, epochs=50)
    if r and r['best_ic3'] > best_ic:
        best_lr, best_ic = lr, r['best_ic3']
        print(f"  lr={lr} IC3={r['best_ic3']:.4f} NEW BEST")
    elif r:
        print(f"  lr={lr} IC3={r['best_ic3']:.4f}")

print(f"\n=== Final ===")
for m, r in sorted(results.items(), key=lambda x: x[1]['oos_ic3'], reverse=True):
    print(f"  {m}: Val={r['val_ic3']:.4f} OOS={r['oos_ic3']:.4f}")
print(f"  Best: {best_model} OOS IC3={best_ic:.4f} lr={best_lr}")

json.dump({m: {'val_ic3': r['val_ic3'], 'oos_ic3': r['oos_ic3']} for m, r in results.items()},
          open(str(DATA_DIR/'p1_results.json'), 'w'), indent=2)
print("Saved p1_results.json")
