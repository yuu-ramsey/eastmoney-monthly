"""Test evaluation — FROZEN, opened ONCE. Top 3 models."""
import torch, torch.nn as nn, numpy as np, json, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

DATA = Path(__file__).parent.parent / '.eastmoney-ai' / 'lstm'
MODEL_DIR = DATA / 'models_v2'
DEVICE = torch.device('cuda')
BATCH = 128

# Load data
X_train = np.load(DATA/'train.npz')['X'].astype(np.float32)
y_train = np.load(DATA/'train.npz')['y'].astype(np.float32)
X_val = np.load(DATA/'val.npz')['X'].astype(np.float32)
y_val = np.load(DATA/'val.npz')['y'].astype(np.float32)
X_test = np.load(DATA/'test.npz')['X'].astype(np.float32)
y_test = np.load(DATA/'test.npz')['y'].astype(np.float32)

# Top 3 configs from overnight
TOP3 = [
    {'name': 'LSTM-7', 'seed': 456, 'lr': 5e-4, 'epochs': 80, 'label': 'LSTM-7_s456'},
    {'name': 'LSTM-5', 'seed': 123, 'lr': 5e-4, 'epochs': 80, 'label': 'LSTM-5_s123'},
    {'name': 'LSTM-8', 'seed': 42,  'lr': 1e-3, 'epochs': 50, 'label': 'LSTM-8_s42'},
]

results = {}

for cfg in TOP3:
    name = cfg['name']; seed = cfg['seed']; lr = cfg['lr']; epochs = cfg['epochs']
    label = cfg['label']
    print(f"\n=== Training {label} ===")

    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)

    model = create_model(name, 21).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    best_vl, no_imp, best_ic3 = float('inf'), 0, -1.0

    for ep in range(1, epochs+1):
        for pg in opt.param_groups:
            pg['lr'] = min(lr, lr * ep / 10.0) if ep <= 10 else lr
        model.train()
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            p = model(Xb)
            loss = 0.5*nn.MSELoss()(p[:,0],yb[:,0]) + 0.5*nn.MSELoss()(p[:,1],yb[:,1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
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
        if vl < best_vl: best_vl, no_imp = vl, 0
        else: no_imp += 1
        if no_imp >= 15 and ep > 20: break

    # Save checkpoint
    ckpt_path = MODEL_DIR / f'{label}.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Val IC3={best_ic3:.4f} ep={ep} saved={ckpt_path.name}")

    # ---- TEST EVALUATION (FROZEN, ONE-TIME) ----
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_ld = torch.utils.data.DataLoader(test_ds, BATCH, pin_memory=True)
    model.eval(); t_preds, t_targs = [], []
    with torch.no_grad():
        for Xb, yb in test_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            p = model(Xb)
            t_preds.append(p.cpu().numpy()); t_targs.append(yb.cpu().numpy())
    tp = np.concatenate(t_preds); tt = np.concatenate(t_targs)

    ic3_test = spearmanr(tp[:,0], tt[:,0])[0]
    ic6_test = spearmanr(tp[:,1], tt[:,1])[0]

    # Sharpe (Long-Short top30% vs bottom30%)
    n = len(tp); cut = int(n * 0.3)
    idx = np.argsort(tp[:, 0])
    ls3 = tt[idx[-cut:], 0] - tt[idx[:cut], 0]
    sr3 = ls3.mean() / ls3.std() * np.sqrt(12/3) if ls3.std() > 0 else 0
    idx6 = np.argsort(tp[:, 1])
    ls6 = tt[idx6[-cut:], 1] - tt[idx6[:cut], 1]
    sr6 = ls6.mean() / ls6.std() * np.sqrt(12/6) if ls6.std() > 0 else 0

    hit3 = np.mean(np.sign(tp[:,0]) == np.sign(tt[:,0]))
    hit6 = np.mean(np.sign(tp[:,1]) == np.sign(tt[:,1]))

    print(f"  TEST IC y3={ic3_test:.4f} IC y6={ic6_test:.4f}")
    print(f"  TEST Sharpe y3={sr3:.3f} y6={sr6:.3f} Hit y3={hit3:.2f} y6={hit6:.2f}")

    results[label] = {'val_ic3': best_ic3, 'test_ic3': ic3_test, 'test_ic6': ic6_test,
                      'test_sr3': sr3, 'test_sr6': sr6, 'test_hit3': hit3, 'test_hit6': hit6,
                      'preds': tp, 'targets': tt, 'model': model}
    del model; torch.cuda.empty_cache()

# ---- Ensemble ----
print(f"\n=== Ensemble ===")
weights = [1/3, 1/3, 1/3]
ens_pred = sum(w * r['preds'] for w, r in zip(weights, results.values())) / sum(weights)
ens_ic3 = spearmanr(ens_pred[:,0], list(results.values())[0]['targets'][:,0])[0]
ens_ic6 = spearmanr(ens_pred[:,1], list(results.values())[0]['targets'][:,1])[0]
print(f"Ensemble (equal): IC3={ens_ic3:.4f} IC6={ens_ic6:.4f}")

# IC-weighted ensemble
ic_weights = [max(0.001, r['val_ic3']) for r in results.values()]
icw_pred = sum(w * r['preds'] for w, r in zip(ic_weights, results.values())) / sum(ic_weights)
icw_ic3 = spearmanr(icw_pred[:,0], list(results.values())[0]['targets'][:,0])[0]
print(f"Ensemble (IC-w): IC3={icw_ic3:.4f}")

# ---- Report ----
print(f"\n{'='*70}")
print(f"FINAL TEST REPORT")
print(f"{'='*70}")
print(f"{'Model':<18} {'Val IC3':>8} {'Test IC3':>8} {'Test IC6':>8} {'SR3':>7} {'Hit3':>6}")
print(f"{'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
best_test = None; best_ic = -1
for label, r in results.items():
    print(f"{label:<18} {r['val_ic3']:8.4f} {r['test_ic3']:8.4f} {r['test_ic6']:8.4f} {r['test_sr3']:7.3f} {r['test_hit3']:6.2f}")
    if r['test_ic3'] > best_ic: best_ic, best_test = r['test_ic3'], label
print(f"{'Ensemble EQ':<18} {'—':>8} {ens_ic3:8.4f} {ens_ic6:8.4f}")
print(f"{'Ensemble ICW':<18} {'—':>8} {icw_ic3:8.4f}")

# Gap analysis
print(f"\nVal→Test Gap:")
for label, r in results.items():
    gap = r['test_ic3'] - r['val_ic3']
    print(f"  {label}: {r['val_ic3']:.4f} → {r['test_ic3']:.4f} (gap={gap:+.4f})")

# Decision
v1_test_ic = 0.025
print(f"\nvs Phase 17 v1 Test IC3: {v1_test_ic:.4f}")
delta = best_ic - v1_test_ic
print(f"Δ best - v1: {delta:+.4f} ({(delta/abs(v1_test_ic)*100):+.0f}%)")
if best_ic > 0.08:
    print("PASS: Test IC3 > 0.08, LSTM beats all Phase 11-15 prompt engineering")
elif best_ic > 0.05:
    print("MARGINAL: Test IC3 > 0.05, weak but usable signal")
else:
    print("FAIL: Test IC3 < 0.05, LSTM path insufficient")

# Save report
report = {
    'models': {l: {k: v for k, v in r.items() if k not in ['preds', 'targets', 'model']} for l, r in results.items()},
    'ensemble_eq_ic3': float(ens_ic3), 'ensemble_icw_ic3': float(icw_ic3),
    'v1_test_ic3': v1_test_ic, 'best_test_ic3': float(best_ic), 'best_model': best_test
}
json.dump(report, open(str(DATA/'test_evaluation_report.json'), 'w'), indent=2)
print(f"\nSaved: test_evaluation_report.json")
