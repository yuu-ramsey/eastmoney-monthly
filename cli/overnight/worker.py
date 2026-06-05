"""Worker: isolated single-experiment training. Exit 0 on success, 1 on failure."""
import torch, torch.nn as nn, numpy as np, sys, sqlite3, json, time
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT / 'lib' / 'lstm'))
from model_v2 import create_model

DATA_DIR = PROJECT / '.eastmoney-ai' / 'lstm'
DB_PATH = DATA_DIR / 'overnight.db'
DEVICE = torch.device('cuda')

def train(config):
    Xt = np.load(DATA_DIR/'train.npz')['X'].astype(np.float32)
    yt = np.load(DATA_DIR/'train.npz')['y'].astype(np.float32)
    Xv = np.load(DATA_DIR/'val.npz')['X'].astype(np.float32)
    yv = np.load(DATA_DIR/'val.npz')['y'].astype(np.float32)

    task = config.get('task', 'train')
    model_name = config['model']
    seed = config.get('seed', 42)
    epochs = config.get('epochs', 80)
    patience = config.get('patience', 15)
    lr = config.get('lr', 5e-4)
    wd = config.get('wd', 1e-4)
    batch = config.get('batch', 128)
    w3 = config.get('loss_w3', 0.5)
    w6 = config.get('loss_w6', 0.5)

    # Walk-forward: use sub-train
    if task == 'walk_forward':
        year = config.get('retrain_year', 2020)
        cutoff_idx = int(0.55 * len(Xt) + (year - 2020) * 0.02 * len(Xt))
        cutoff_idx = min(cutoff_idx, len(Xt) - 100)
        Xt_use, yt_use = Xt[:cutoff_idx], yt[:cutoff_idx]
    else:
        Xt_use, yt_use = Xt, yt

    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xt_use), torch.from_numpy(yt_use))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
    train_ld = torch.utils.data.DataLoader(train_ds, batch, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, batch, pin_memory=True)

    model = create_model(model_name, 21).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=wd)
    best_vl, no_imp, best_ic3, best_ic6 = float('inf'), 0, -1.0, -1.0
    t0 = time.time()

    for ep in range(1, epochs+1):
        for pg in opt.param_groups:
            pg['lr'] = min(lr, lr * ep / 10.0) if ep <= 10 else lr

        model.train(); tl = 0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            p = model(Xb)
            loss = w3 * nn.MSELoss()(p[:,0], yb[:,0]) + w6 * nn.MSELoss()(p[:,1], yb[:,1])
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
                vl += (w3*nn.MSELoss()(p[:,0],yb[:,0]) + w6*nn.MSELoss()(p[:,1],yb[:,1])).item() * Xb.size(0)
                preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
        vl /= len(val_ds)
        pr = np.concatenate(preds); tg = np.concatenate(targs)
        ic3 = float(spearmanr(pr[:,0], tg[:,0])[0]) if len(pr) > 10 else 0.0
        ic6 = float(spearmanr(pr[:,1], tg[:,1])[0]) if len(pr) > 10 else 0.0
        best_ic3 = max(best_ic3, ic3); best_ic6 = max(best_ic6, ic6)
        if vl < best_vl: best_vl, no_imp = vl, 0
        else: no_imp += 1
        if no_imp >= patience and ep > 20: break

    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model; torch.cuda.empty_cache()
    return {'ic_y3': best_ic3, 'ic_y6': best_ic6, 'val_loss': best_vl, 'epochs': ep, 'elapsed_s': elapsed, 'params': params}

def main():
    if len(sys.argv) < 2:
        print("Usage: worker.py <exp_id>")
        sys.exit(1)
    exp_id = int(sys.argv[1])
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute("SELECT id, config_json FROM experiments WHERE id=?", (exp_id,)).fetchone()
    if not row:
        print(f"Experiment {exp_id} not found")
        sys.exit(1)
    config = json.loads(row[1])
    conn.execute("UPDATE experiments SET status='running', start_time=? WHERE id=?", (datetime.now().isoformat(), exp_id))
    conn.commit()

    print(f"[Worker {exp_id}] {config['task']} {config.get('model','?')} epochs={config.get('epochs','?')}")

    try:
        result = train(config)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"[Worker {exp_id}] OOM, retrying with half batch")
            config['batch'] = config.get('batch', 128) // 2
            try:
                result = train(config)
            except Exception as e2:
                conn.execute("UPDATE experiments SET status='failed', error=?, end_time=? WHERE id=?",
                           (str(e2)[:500], datetime.now().isoformat(), exp_id))
                conn.commit(); conn.close()
                sys.exit(1)
        else:
            conn.execute("UPDATE experiments SET status='failed', error=?, end_time=? WHERE id=?",
                       (str(e)[:500], datetime.now().isoformat(), exp_id))
            conn.commit(); conn.close()
            sys.exit(1)
    except Exception as e:
        conn.execute("UPDATE experiments SET status='failed', error=?, end_time=? WHERE id=?",
                   (str(e)[:500], datetime.now().isoformat(), exp_id))
        conn.commit(); conn.close()
        sys.exit(1)

    conn.execute("UPDATE experiments SET status='done', ic_y3=?, ic_y6=?, val_loss=?, end_time=? WHERE id=?",
               (result['ic_y3'], result['ic_y6'], result['val_loss'], datetime.now().isoformat(), exp_id))
    conn.commit(); conn.close()
    print(f"[Worker {exp_id}] Done: IC3={result['ic_y3']:.4f} IC6={result['ic_y6']:.4f} {result['elapsed_s']:.0f}s")
    sys.exit(0)

if __name__ == '__main__':
    main()
