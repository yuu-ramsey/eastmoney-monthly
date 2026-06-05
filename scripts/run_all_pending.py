"""Run all pending experiments in single process. No subprocess isolation needed."""
import torch, torch.nn as nn, numpy as np, sqlite3, json, time
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

DB = Path(__file__).parent.parent / '.eastmoney-ai' / 'lstm' / 'overnight.db'
DATA = Path(__file__).parent.parent / '.eastmoney-ai' / 'lstm'
DEVICE = torch.device('cuda')

X_train = np.load(DATA/'train.npz')['X'].astype(np.float32)
y_train = np.load(DATA/'train.npz')['y'].astype(np.float32)
X_val = np.load(DATA/'val.npz')['X'].astype(np.float32)
y_val = np.load(DATA/'val.npz')['y'].astype(np.float32)
print(f"Data: train={X_train.shape}, val={X_val.shape}")

conn = sqlite3.connect(str(DB))

def train_one(exp_id, config):
    model_name = config['model']
    seed = config.get('seed', 42)
    epochs = config.get('epochs', 80)
    lr = config.get('lr', 5e-4)
    wd = config.get('wd', 1e-4)
    batch = config.get('batch', 128)
    w3 = config.get('loss_w3', 0.5)
    w6 = config.get('loss_w6', 0.5)
    patience = config.get('patience', 15)

    # Walk-forward: use sub-train
    Xt, yt = X_train, y_train
    if config.get('task') == 'walk_forward':
        year = config.get('retrain_year', 2020)
        cutoff = int(0.55 * len(Xt) + (year - 2020) * 0.02 * len(Xt))
        Xt, yt = Xt[:min(cutoff, len(Xt)-100)], yt[:min(cutoff, len(Xt)-100)]

    torch.manual_seed(seed); np.random.seed(seed)
    ds_train = torch.utils.data.TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
    ds_val = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    ld_train = torch.utils.data.DataLoader(ds_train, batch, shuffle=True, pin_memory=True)
    ld_val = torch.utils.data.DataLoader(ds_val, batch, pin_memory=True)

    try:
        model = create_model(model_name, 21).to(DEVICE)
    except:
        return None

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=wd)
    best_vl, no_imp, best_ic3, best_ic6 = float('inf'), 0, -1.0, -1.0
    t0 = time.time()

    for ep in range(1, epochs+1):
        for pg in opt.param_groups:
            pg['lr'] = min(lr, lr * ep / 10.0) if ep <= 10 else lr
        model.train()
        for Xb, yb in ld_train:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            p = model(Xb)
            loss = w3 * nn.MSELoss()(p[:,0], yb[:,0]) + w6 * nn.MSELoss()(p[:,1], yb[:,1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); vl = 0; preds, targs = [], []
        with torch.no_grad():
            for Xb, yb in ld_val:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                p = model(Xb)
                vl += (w3*nn.MSELoss()(p[:,0],yb[:,0]) + w6*nn.MSELoss()(p[:,1],yb[:,1])).item() * Xb.size(0)
                preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
        vl /= len(ds_val)
        pr = np.concatenate(preds); tg = np.concatenate(targs)
        ic3 = float(spearmanr(pr[:,0], tg[:,0])[0])
        ic6 = float(spearmanr(pr[:,1], tg[:,1])[0])
        best_ic3 = max(best_ic3, ic3); best_ic6 = max(best_ic6, ic6)
        if vl < best_vl: best_vl, no_imp = vl, 0
        else: no_imp += 1
        if no_imp >= patience and ep > 20: break

    elapsed = time.time() - t0
    del model; torch.cuda.empty_cache()
    return {'ic_y3': best_ic3, 'ic_y6': best_ic6, 'val_loss': best_vl, 'elapsed_s': elapsed}

# Main loop
total = conn.execute("SELECT COUNT(*) FROM experiments WHERE status='pending'").fetchone()[0]
done = 0
t_start = time.time()

while True:
    row = conn.execute(
        "SELECT id, config_json FROM experiments WHERE status='pending' ORDER BY priority, id LIMIT 1"
    ).fetchone()
    if not row:
        break
    exp_id, cfg_json = row
    cfg = json.loads(cfg_json)
    conn.execute("UPDATE experiments SET status='running', start_time=? WHERE id=?", (datetime.now().isoformat(), exp_id))
    conn.commit()

    result = train_one(exp_id, cfg)
    if result:
        conn.execute("UPDATE experiments SET status='done', ic_y3=?, ic_y6=?, val_loss=?, end_time=? WHERE id=?",
                   (result['ic_y3'], result['ic_y6'], result['val_loss'], datetime.now().isoformat(), exp_id))
    else:
        conn.execute("UPDATE experiments SET status='failed', error='model_create_failed', end_time=? WHERE id=?",
                   (datetime.now().isoformat(), exp_id))
    conn.commit()

    done += 1
    elapsed = time.time() - t_start
    eta = elapsed / done * (total - done) if done > 0 else 0
    tag = f"IC3={result['ic_y3']:.4f}" if result else "FAIL"
    print(f"[{done}/{total}] [{exp_id}] {cfg.get('task')}/{cfg.get('model')} {tag} {elapsed:.0f}s ETA{eta:.0f}s")

# Summary
rows = conn.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status").fetchall()
print(f"\nDone: {dict(rows)}")
top = conn.execute("SELECT id, phase, config_json, ic_y3 FROM experiments WHERE status='done' AND ic_y3 IS NOT NULL ORDER BY ic_y3 DESC LIMIT 5").fetchall()
print("Top 5:")
for r in top:
    cfg = json.loads(r[2])
    print(f"  [{r[0]}] {r[1]} {cfg.get('model')} IC3={r[3]:.4f}")
conn.close()
print(f"Total: {time.time()-t_start:.0f}s")
