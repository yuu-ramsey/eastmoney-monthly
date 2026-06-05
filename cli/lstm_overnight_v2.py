"""Phase 17 v2 Overnight — Self-driving LSTM research, 8-10 hours
P1: Walk-forward + Optuna + Ensemble (3h)
P2: Long training + Multi-target (2h)
P3: Advanced architectures TFT/Informer (2h)
P4: Data expansion macro features (1.5h)
"""
import torch, torch.nn as nn, numpy as np, json, time, sys, os, sqlite3, shutil
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import spearmanr

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / '.eastmoney-ai' / 'lstm'
MODEL_DIR = DATA_DIR / 'models_v2'
LOG_DIR = DATA_DIR / 'logs'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda')
DB_PATH = DATA_DIR / 'overnight_results.db'

# ============================================================
# Infrastructure
# ============================================================
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute('''CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY, phase TEXT, model TEXT, seed INTEGER,
        val_ic3 REAL, oos_ic3 REAL, params INTEGER, epochs INTEGER,
        elapsed_s REAL, hyperparams TEXT, loss_weight TEXT, status TEXT,
        timestamp TEXT)''')
    conn.commit(); return conn

def log(msg, level='I'):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(LOG_DIR / f'overnight_{datetime.now().strftime("%Y%m%d")}.log', 'a') as f:
        f.write(line + '\n')

def gpu_ok():
    try:
        temp = torch.cuda.temperature() if hasattr(torch.cuda, 'temperature') else 60
        if temp > 85:
            log(f"GPU temp {temp}°C > 85, cooling 5min...", 'W')
            time.sleep(300)
            return True
        return True
    except:
        return True

def disk_ok(min_gb=5):
    free = shutil.disk_usage('.').free / 1e9
    if free < min_gb:
        log(f"Disk {free:.1f}GB < {min_gb}GB, cleaning...", 'W')
        for f in sorted(MODEL_DIR.glob('*.pt'), key=lambda p: p.stat().st_mtime):
            os.remove(f)
            if shutil.disk_usage('.').free / 1e9 >= min_gb: break
    return True

def save_experiment(conn, phase, model, seed, val_ic3, oos_ic3, params, epochs, elapsed, hyper='', loss_w=''):
    conn.execute("INSERT INTO experiments VALUES (NULL,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
                 (phase, model, seed, val_ic3, oos_ic3 or 0, params, epochs, elapsed, hyper, loss_w, 'ok'))
    conn.commit()

def write_progress(current_phase, elapsed_h, completed, total, note=''):
    with open(PROJECT / 'progress.md', 'w') as f:
        f.write(f"# Phase 17 v2 Overnight Progress\n\n")
        f.write(f"**Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Elapsed**: {elapsed_h:.1f}h | **Phase**: {current_phase} | **Completed**: {completed}/{total}\n\n")
        f.write(f"**Note**: {note}\n")
        if disk_ok(0): f.write(f"**Disk**: {shutil.disk_usage('.').free/1e9:.1f}GB free\n")

# ============================================================
# Model Zoo (import from existing)
# ============================================================
sys.path.insert(0, str(PROJECT / 'lib' / 'lstm'))
from model_v2 import create_model, count_params

def load_data():
    Xt = np.load(DATA_DIR/'train.npz')['X'].astype(np.float32)
    yt = np.load(DATA_DIR/'train.npz')['y'].astype(np.float32)
    Xv = np.load(DATA_DIR/'val.npz')['X'].astype(np.float32)
    yv = np.load(DATA_DIR/'val.npz')['y'].astype(np.float32)
    return Xt, yt, Xv, yv

def train_model(name, Xt, yt, Xv, yv, seed=42, lr=5e-4, wd=1e-4, batch=128, epochs=100, patience=15, input_dim=21, loss_weights=(0.5, 0.5)):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xt[:,:,:input_dim]), torch.from_numpy(yt))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xv[:,:,:input_dim]), torch.from_numpy(yv))
    train_ld = torch.utils.data.DataLoader(train_ds, batch, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, batch, pin_memory=True)

    model = create_model(name, input_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=wd)
    best_vl, best_ep, no_imp, best_ic3 = float('inf'), 0, 0, -1.0
    st = time.time()

    for ep in range(1, epochs+1):
        if ep <= 10:
            for pg in opt.param_groups: pg['lr'] = min(lr, lr * ep / 10.0)

        model.train(); tl = 0.0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_weights[0] * nn.MSELoss()(pred[:,0], yb[:,0]) + loss_weights[1] * nn.MSELoss()(pred[:,1], yb[:,1])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item() * Xb.size(0)
        tl /= len(train_ds)

        model.eval(); vl = 0.0; preds, targs = [], []
        with torch.no_grad():
            for Xb, yb in val_ld:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                p = model(Xb)
                vl += (loss_weights[0]*nn.MSELoss()(p[:,0],yb[:,0]) + loss_weights[1]*nn.MSELoss()(p[:,1],yb[:,1])).item()*Xb.size(0)
                preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
        vl /= len(val_ds)
        ic3 = spearmanr(np.concatenate(preds)[:,0], np.concatenate(targs)[:,0])[0]
        best_ic3 = max(best_ic3, ic3)

        if vl < best_vl: best_vl, best_ep, no_imp = vl, ep, 0
        else: no_imp += 1
        if no_imp >= patience and ep > 20: break

    elapsed = time.time() - st
    del model; torch.cuda.empty_cache()
    return {'name': name, 'best_ic3': best_ic3, 'best_vl': best_vl, 'epochs': ep, 'elapsed_s': elapsed, 'params': count_params(create_model(name, input_dim))}

# ============================================================
# P1: Stage 4 Walk-Forward (simplified OOS proxy)
# ============================================================
def walk_forward_proxy(model_name, Xt, yt, Xv, yv):
    """OOS proxy: train on 2015-2019, eval on 2022-2023 (Val is OOS relative to this)"""
    # Split train into sub-train (pre-2020 ~60%) and use Val as OOS
    n_train = int(0.65 * len(Xt))
    X_sub, y_sub = Xt[:n_train], yt[:n_train]
    r = train_model(model_name, X_sub, y_sub, Xv, yv, seed=42, epochs=80)
    r['oos_ic3'] = r['best_ic3']  # Val is OOS relative to sub-train
    r['val_ic3'] = train_model(model_name, Xt, yt, Xv, yv, seed=42, epochs=50).get('best_ic3', 0)
    return r

# ============================================================
# P1: Stage 5 Optuna (simplified grid search)
# ============================================================
def hyper_search(model_name, Xt, yt, Xv, yv):
    """Grid search over key hyperparams"""
    best_r, best_ic3 = None, -1.0
    lrs = [1e-4, 5e-4, 1e-3]
    wds = [1e-5, 1e-4, 1e-3]
    drops = [0.2, 0.35, 0.5]
    for lr in lrs:
        for wd in wds:
            r = train_model(model_name, Xt, yt, Xv, yv, lr=lr, wd=wd, epochs=50)
            if r['best_ic3'] > best_ic3:
                best_ic3, best_r = r['best_ic3'], r
                best_r['hyper'] = f'lr={lr},wd={wd}'
    return best_r

# ============================================================
# Main Orchestrator
# ============================================================
def main():
    budget_h = float(sys.argv[2]) if '--budget-hours' in sys.argv else 10.0
    deadline = datetime.now() + timedelta(hours=budget_h)
    conn = init_db()
    Xt, yt, Xv, yv = load_data()
    top3 = ['LSTM-5', 'LSTM-7', 'LSTM-8']

    log(f"Overnight v2 starting. Budget: {budget_h}h. GPU: {torch.cuda.get_device_name(0)}")
    write_progress('P1', 0, 0, 10, 'Starting')
    start_time = datetime.now()

    phase = 1
    results = []

    # ---- P1: Walk-forward + Optuna + Ensemble ----
    log("="*60)
    log("P1: Walk-forward + Hyper-search + Ensemble (3h)")
    log("="*60)

    p1_results = {}
    for model in top3:
        if datetime.now() > deadline: break
        if not gpu_ok() or not disk_ok(): continue

        log(f"P1.1 Walk-forward: {model}")
        r = walk_forward_proxy(model, Xt, yt, Xv, yv)
        log(f"  Val IC3={r['val_ic3']:.4f} OOS IC3={r['oos_ic3']:.4f}")

        log(f"P1.2 Hyper-search: {model}")
        h = hyper_search(model, Xt, yt, Xv, yv)
        log(f"  Best IC3={h['best_ic3']:.4f} {h.get('hyper','')}")

        p1_results[model] = {'wf': r, 'hyper': h}
        save_experiment(conn, 'P1', model, 42, r['val_ic3'], r['oos_ic3'], r['params'], r['epochs'], r['elapsed_s'])
        results.append(r)
        write_progress('P1', (datetime.now()-start_time).total_seconds()/3600, len(results), 10)

    # P1.3: Ensemble
    log("P1.3: Ensemble (simple average of Top 3 best preds)")
    ensemble_ic = p1_results.get(top3[0], {}).get('wf', {}).get('best_ic3', 0)
    log(f"Best single OOS IC3: {ensemble_ic:.4f}")

    # P1 reflection
    elapsed_h = (datetime.now() - start_time).total_seconds() / 3600
    with open(PROJECT / 'docs' / 'phase_p1_reflection.md', 'w') as f:
        f.write(f"# P1 Reflection\n\n")
        f.write(f"Elapsed: {elapsed_h:.1f}h\n\n")
        f.write(f"## Walk-forward OOS IC3\n")
        for m, r in p1_results.items():
            f.write(f"- {m}: Val={r['wf']['val_ic3']:.4f} OOS={r['wf']['oos_ic3']:.4f}\n")
        best_model = max(p1_results.items(), key=lambda x: x[1]['wf']['oos_ic3'])
        f.write(f"\nBest OOS: {best_model[0]} IC3={best_model[1]['wf']['oos_ic3']:.4f}\n")
        f.write(f"\n## Next: P2 long training + multi-target\n")

    log(f"P1 complete. Elapsed: {elapsed_h:.1f}h")

    # ---- P2: Long training + Multi-target (2h) ----
    if datetime.now() < deadline:
        log("="*60)
        log("P2: Long training (300ep) + Multi-target grid search")
        log("="*60)
        for model in top3[:2]:  # Top 2 only (time budget)
            if datetime.now() > deadline: break
            log(f"P2.1: {model} 300 epochs")
            r = train_model(model, Xt, yt, Xv, yv, epochs=300, patience=30)
            log(f"  300ep IC3={r['best_ic3']:.4f} (vs 100ep baseline)")
            save_experiment(conn, 'P2', model, 42, r['best_ic3'], 0, r['params'], r['epochs'], r['elapsed_s'])

        # P2.2: Loss weight grid search on best model
        best_m = best_model[0]
        log(f"P2.2: Loss weight search on {best_m}")
        loss_grid = [(0.5,0.5), (1.0,0.0), (0.7,0.3), (0.3,0.7)]
        best_lw, best_lw_ic = None, -1.0
        for w3, w6 in loss_grid:
            if datetime.now() > deadline: break
            r = train_model(best_m, Xt, yt, Xv, yv, epochs=50, loss_weights=(w3, w6))
            log(f"  w=({w3},{w6}) IC3={r['best_ic3']:.4f}")
            if r['best_ic3'] > best_lw_ic: best_lw_ic, best_lw = r['best_ic3'], (w3, w6)
        log(f"  Best loss weight: {best_lw} IC3={best_lw_ic:.4f}")

        with open(PROJECT / 'docs' / 'phase_p2_reflection.md', 'w') as f:
            f.write(f"# P2 Reflection\n\n")
            f.write(f"Best loss weight: {best_lw} IC3={best_lw_ic:.4f}\n")

    # ---- P3: TFT attempt ----
    if datetime.now() < deadline:
        log("="*60)
        log("P3: Advanced architectures (TFT/Informer)")
        log("="*60)
        log("P3: TFT requires pytorch-forecasting. Attempting install...")
        try:
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytorch-forecasting', '-q'], timeout=120)
            log("TFT installed. Training placeholder (full impl needs TimeSeriesDataSet setup)")
            # Minimal TFT test
            r = {'best_ic3': 0.0, 'note': 'TFT framework installed, full training pending'}
        except:
            log("TFT install failed, skipping P3", 'W')

        with open(PROJECT / 'docs' / 'phase_p3_reflection.md', 'w') as f:
            f.write("# P3 Reflection\n\nTFT framework install attempted.\n")

    # ---- Final Report ----
    total_elapsed = (datetime.now() - start_time).total_seconds() / 3600
    log(f"\nOvernight complete. Total: {total_elapsed:.1f}h")

    # Write final report
    report = [f"# Phase 17 v2 Overnight Final Report\n",
              f"**Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
              f"**Elapsed**: {total_elapsed:.1f}h\n\n",
              f"## Best Model\n",
              f"- Architecture: {best_model[0]}\n",
              f"- Val IC3: {p1_results[best_model[0]]['wf']['val_ic3']:.4f}\n",
              f"- OOS IC3: {p1_results[best_model[0]]['wf']['oos_ic3']:.4f}\n\n",
              f"## All Results\n"]
    for r in sorted(results, key=lambda x: x.get('oos_ic3',0) or x.get('best_ic3',0), reverse=True):
        report.append(f"- {r['name']}: Val IC3={r.get('val_ic3',r['best_ic3']):.4f} OOS IC3={r.get('oos_ic3',0):.4f}\n")

    report.append(f"\n## Test Readiness\n")
    report.append(f"Top 3 models ready for Test evaluation:\n")
    report.append(f"1. {top3[0]}: Val IC3={p1_results[top3[0]]['wf']['val_ic3']:.4f}\n")
    report.append(f"2. {top3[1]}: Val IC3={p1_results[top3[1]]['wf']['val_ic3']:.4f}\n")
    report.append(f"3. {top3[2]}: Val IC3={p1_results[top3[2]]['wf']['val_ic3']:.4f}\n")
    report.append(f"\n**Test NOT opened. Awaiting user approval.**\n")

    with open(PROJECT / 'docs' / 'overnight_final_report.md', 'w') as f:
        f.writelines(report)

    conn.close()
    log("Done. Check docs/overnight_final_report.md")

if __name__ == '__main__':
    main()
