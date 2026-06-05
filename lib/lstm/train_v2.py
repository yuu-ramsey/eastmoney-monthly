"""Phase 17 v2: Train 3 architectures on GPU, compare Val IC"""
import torch, torch.nn as nn, numpy as np, json, time, sys
from pathlib import Path
from scipy.stats import spearmanr
from model_v2 import create_model, count_params, DATA_DIR, MODEL_DIR

torch.manual_seed(42); np.random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH, LR, WD, PATIENCE, MAX_EPOCHS = 128, 5e-4, 1e-4, 15, 100

def load_data():
    Xt, yt = np.load(DATA_DIR / 'train.npz')['X'], np.load(DATA_DIR / 'train.npz')['y']
    Xv, yv = np.load(DATA_DIR / 'val.npz')['X'], np.load(DATA_DIR / 'val.npz')['y']
    # Dataset has y3, y6 (2 targets). y1 to be added with macro features.
    y_train = yt.astype(np.float32)  # (N, 2): y3, y6
    y_val = yv.astype(np.float32)
    return Xt.astype(np.float32), y_train, Xv.astype(np.float32), y_val

def two_head_loss(pred, target):
    mse = nn.MSELoss()
    return 0.5 * mse(pred[:, 0], target[:, 0]) + 0.5 * mse(pred[:, 1], target[:, 1])

def train_one_model(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    X_train, y_train, X_val, y_val = load_data()
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)

    model = create_model(name, X_train.shape[2]).to(DEVICE)
    n_params = count_params(model)
    print(f"Params: {n_params:,}, Device: {DEVICE}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'ic_y3': [], 'epoch_time': []}
    best_val_loss, best_ep, no_improve = float('inf'), 0, 0

    for ep in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train(); tl = 0.0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = two_head_loss(model(Xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item() * Xb.size(0)
        tl /= len(train_ds)

        model.eval(); vl = 0.0; preds, targs = [], []
        with torch.no_grad():
            for Xb, yb in val_ld:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                p = model(Xb); loss = two_head_loss(p, yb)
                vl += loss.item() * Xb.size(0)
                preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
        vl /= len(val_ds)
        preds = np.concatenate(preds); targs = np.concatenate(targs)
        ic3 = spearmanr(preds[:, 0], targs[:, 0])[0]

        et = time.time() - t0
        history['epoch'].append(ep); history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['ic_y3'].append(float(ic3)); history['epoch_time'].append(et)

        marker = ''
        if vl < best_val_loss:
            best_val_loss, best_ep, no_improve = vl, ep, 0
            torch.save(model.state_dict(), MODEL_DIR / f'{name}.pt'); marker = ' *'
        else:
            no_improve += 1

        if ep <= 3 or ep % 10 == 0 or marker:
            print(f"Ep {ep:3d}: TL={tl:.4f} VL={vl:.4f} IC3={ic3:.4f} {et:.1f}s{marker}")

        if no_improve >= PATIENCE and ep > 20:
            print(f"Early stop at {ep}")
            break

    best_ic = max(history['ic_y3'])
    print(f"Best VL={best_val_loss:.4f} (ep{best_ep}), Best IC3={best_ic:.4f}")
    return {'name': name, 'params': n_params, 'best_vl': best_val_loss, 'best_ic3': best_ic,
            'avg_epoch_time': np.mean(history['epoch_time']), 'epochs': ep, 'history': history}

def main():
    models = ['LSTM-2', 'GRU-2', 'Transformer-2']
    results = []
    for m in models:
        r = train_one_model(m)
        results.append(r)
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"{'Model':<18} {'Params':>8} {'Val IC y3':>10} {'Val Loss':>10} {'Epochs':>7} {'Time/ep':>8}")
    print(f"{'-'*18} {'-'*8} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")
    for r in results:
        print(f"{r['name']:<18} {r['params']:>8,} {r['best_ic3']:>10.4f} {r['best_vl']:>10.4f} {r['epochs']:>7} {r['avg_epoch_time']:>7.1f}s")

    best = max(results, key=lambda r: r['best_ic3'])
    print(f"\nBest: {best['name']} IC3={best['best_ic3']:.4f}")

    # Kill switch
    v1_ic = 0.095
    if best['best_ic3'] > 0.12:
        print(f"PASS: IC3 {best['best_ic3']:.4f} > 0.12 → expand to 14 architectures")
    elif best['best_ic3'] >= 0.10:
        print(f"MARGINAL: IC3 {best['best_ic3']:.4f} ∈ [0.10, 0.12] → add macro features")
    else:
        print(f"FAIL: all IC3 < 0.10")

    with open(MODEL_DIR / 'v2_results.json', 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'history'} for r in results], f, indent=2)

if __name__ == '__main__':
    main()
