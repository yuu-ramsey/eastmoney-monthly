"""Phase 17 v2: Overnight GPU training — Stage 1-6 automated"""
import torch, torch.nn as nn, numpy as np, json, time, sys, os, shutil
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / '.eastmoney-ai' / 'lstm'
MODEL_DIR = DATA_DIR / 'models_v2'
LOG_DIR = DATA_DIR / 'logs'
for d in [MODEL_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda')
torch.manual_seed(42); np.random.seed(42)

BATCH_BASE, LR, WD, PATIENCE, MAX_EPOCHS = 128, 5e-4, 1e-4, 15, 100
LOG_FILE = LOG_DIR / f'overnight_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log'

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f: f.write(line + '\n')

# ============================================================
# Model Zoo
# ============================================================
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, use_attn=True):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.residual = num_layers >= 4
        self.layer_norms = nn.ModuleList() if self.residual else None
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.lstm_layers.append(nn.LSTM(in_dim, hidden_dim, 1, batch_first=True))
            if self.residual:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.attn = nn.Linear(hidden_dim, 1) if use_attn else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        h = x
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(h)
            if self.residual and i > 0 and out.shape == h.shape:
                out = self.layer_norms[i](out + h)
            h = out
        if self.attn:
            w = torch.softmax(self.attn(h), dim=1)
            h = (w * h).sum(dim=1)
        else:
            h = h[:, -1, :]
        return self.fc(self.dropout(h))

class GRUStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))

class TransformerStack(nn.Module):
    def __init__(self, input_dim, d_model=128, num_layers=2, n_heads=4, dropout=0.3, max_len=60):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        return self.fc(self.dropout(self.encoder(x).mean(dim=1)))

def create_model(name, input_dim=21):
    if name.startswith('LSTM-'):
        n = int(name.split('-')[1])
        h = {1:128, 2:128, 3:128, 4:96, 5:64, 6:64}[n]
        d = 0.2 + n * 0.04
        return ResidualLSTM(input_dim, h, n, min(d, 0.5))
    elif name.startswith('GRU-'):
        n = int(name.split('-')[1])
        h = {1:128, 2:128, 3:96, 4:96}[n]
        return GRUStack(input_dim, h, n, 0.3)
    elif name.startswith('Transformer-'):
        n = int(name.split('-')[1])
        d = 0.2 + n * 0.03
        return TransformerStack(input_dim, 128, n, 4, min(d, 0.4))
    else:
        raise ValueError(name)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ============================================================
# Data
# ============================================================
def load_data():
    Xt = np.load(DATA_DIR/'train.npz')['X'].astype(np.float32)
    yt = np.load(DATA_DIR/'train.npz')['y'].astype(np.float32)
    Xv = np.load(DATA_DIR/'val.npz')['X'].astype(np.float32)
    yv = np.load(DATA_DIR/'val.npz')['y'].astype(np.float32)
    return Xt, yt, Xv, yv

def two_head_loss(pred, target):
    return 0.5 * nn.MSELoss()(pred[:,0], target[:,0]) + 0.5 * nn.MSELoss()(pred[:,1], target[:,1])

# ============================================================
# Training
# ============================================================
def train_one(name, input_dim, Xt, yt, Xv, yv, batch_size=None):
    if batch_size is None: batch_size = BATCH_BASE
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size, pin_memory=True)

    try:
        model = create_model(name, input_dim).to(DEVICE)
    except Exception as e:
        log(f"  {name}: model creation failed: {e}")
        return None

    n_p = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best_vl, best_ep, no_imp = float('inf'), 0, 0
    best_ic3 = -1.0
    hist = {'train_loss': [], 'val_loss': [], 'ic_y3': []}
    t0 = time.time()

    for ep in range(1, MAX_EPOCHS+1):
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
                p = model(Xb); vl += two_head_loss(p, yb).item() * Xb.size(0)
                preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
        vl /= len(val_ds)
        pr = np.concatenate(preds); tg = np.concatenate(targs)
        ic3 = spearmanr(pr[:,0], tg[:,0])[0] if len(pr) > 10 else 0.0

        hist['train_loss'].append(tl); hist['val_loss'].append(vl); hist['ic_y3'].append(float(ic3))

        if vl < best_vl:
            best_vl, best_ep, no_imp = vl, ep, 0
            torch.save(model.state_dict(), MODEL_DIR / f'{name}_{input_dim}d.pt')
        else:
            no_imp += 1
        best_ic3 = max(best_ic3, ic3)

        if no_imp >= PATIENCE and ep > 20:
            break

    elapsed = time.time() - t0
    # Free GPU memory
    del model; torch.cuda.empty_cache()

    return {'name': name, 'input_dim': input_dim, 'params': n_p, 'epochs': ep,
            'best_vl': best_vl, 'best_ic3': best_ic3, 'best_ep': best_ep,
            'elapsed_s': elapsed, 'batch_size': batch_size,
            'ic3_final': hist['ic_y3'][-1], 'ic3_max': max(hist['ic_y3'])}

# ============================================================
# Stage 1: 14 arch, 21 features
# ============================================================
def stage1():
    log("="*60)
    log("STAGE 1: 14 architectures, 21 features, seed=42")
    log("="*60)
    Xt, yt, Xv, yv = load_data()
    models = [f'LSTM-{i}' for i in range(1,7)] + [f'GRU-{i}' for i in range(1,5)] + \
             [f'Transformer-{i}' for i in [2,4,6]]
    results = []
    for i, name in enumerate(models):
        log(f"  [{i+1}/14] {name}...")
        r = train_one(name, 21, Xt, yt, Xv, yv)
        if r:
            log(f"    IC3={r['best_ic3']:.4f} VL={r['best_vl']:.4f} ep={r['epochs']} {r['elapsed_s']:.0f}s")
            results.append(r)
        else:
            log(f"    FAILED")

    results.sort(key=lambda r: r['best_ic3'], reverse=True)
    with open(DATA_DIR / 'stage1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    log(f"\nStage 1 Top 5:")
    for r in results[:5]:
        log(f"  {r['name']:15s} IC3={r['best_ic3']:.4f} params={r['params']:,}")
    return results

# ============================================================
# Stage 2: Macro data + 26d retrain
# ============================================================
def fetch_macro():
    """Fetch M2/CPI/SHIBOR/PMI/Northbound from akshare"""
    log("Fetching macro data...")
    import pandas as pd
    try:
        import akshare as ak

        # M2 (monthly)
        m2 = ak.macro_china_m2_yearly()
        m2_df = pd.DataFrame({'date': pd.to_datetime(m2['日期']), 'm2_yoy': m2['同比增长']})
        m2_df['date'] = m2_df['date'].dt.strftime('%Y-%m')

        # CPI
        cpi = ak.macro_china_cpi_monthly()
        cpi_df = pd.DataFrame({'date': pd.to_datetime(cpi['日期']), 'cpi_yoy': cpi['同比增长']})
        cpi_df['date'] = cpi_df['date'].dt.strftime('%Y-%m')

        # PMI
        try:
            pmi = ak.macro_china_pmi()
            pmi_df = pd.DataFrame({'date': pd.to_datetime(pmi['日期']), 'pmi': pmi['制造业']})
            pmi_df['date'] = pmi_df['date'].dt.strftime('%Y-%m')
        except:
            pmi_df = pd.DataFrame(columns=['date', 'pmi'])

        # SHIBOR (monthly average)
        try:
            shi = ak.rate_interbank(market='上海银行间同业拆放利率', indicator='隔夜')
            shi_df = pd.DataFrame(shi)
            if '日期' in shi_df.columns:
                shi_df['date'] = pd.to_datetime(shi_df['日期']).dt.strftime('%Y-%m')
                shi_df = shi_df.groupby('date')['利率'].mean().reset_index()
                shi_df.columns = ['date', 'shibor']
            else:
                shi_df = pd.DataFrame(columns=['date', 'shibor'])
        except:
            shi_df = pd.DataFrame(columns=['date', 'shibor'])

        # Northbound capital flow
        try:
            north = ak.stock_hsgt_north_net_flow_in_em()
            north_df = pd.DataFrame(north)
            if '日期' in north_df.columns:
                north_df['date'] = pd.to_datetime(north_df['日期']).dt.strftime('%Y-%m')
                north_df = north_df.groupby('date')['净流入'].sum().reset_index()
                north_df.columns = ['date', 'north_flow']
            else:
                north_df = pd.DataFrame(columns=['date', 'north_flow'])
        except:
            north_df = pd.DataFrame(columns=['date', 'north_flow'])

        # Merge
        macro = m2_df.merge(cpi_df, on='date', how='outer')
        for df in [pmi_df, shi_df, north_df]:
            if len(df) > 0:
                macro = macro.merge(df, on='date', how='outer')
        macro = macro.sort_values('date').ffill().fillna(0)
        macro.to_parquet(DATA_DIR / 'macro_features.parquet')
        log(f"Macro data saved: {len(macro)} months, columns={list(macro.columns)}")
        return macro
    except Exception as e:
        log(f"Macro fetch error: {e}, using fallback (zeros)")
        return None

def add_macro_to_npz(X_old, macro_df):
    """Add 5 macro features to existing 21-dim npz data. Needs date alignment."""
    # Simplified: replicate macro features for all samples (no date alignment in npz)
    # Full implementation would need to store dates per sequence
    # For now: use mean/std of macro features as 5 extra dims (constant across samples)
    n_samples, seq_len, n_feats = X_old.shape
    X_new = np.zeros((n_samples, seq_len, n_feats + 5), dtype=np.float32)
    X_new[:, :, :n_feats] = X_old
    # Add macro as constant features (simplified — full version needs date alignment)
    if macro_df is not None and len(macro_df) > 0:
        cols = ['m2_yoy', 'cpi_yoy', 'pmi', 'shibor', 'north_flow']
        for j, c in enumerate(cols):
            if c in macro_df.columns:
                val = macro_df[c].mean()
                std = macro_df[c].std() + 1e-8
                X_new[:, :, n_feats + j] = val / max(std, 1)
    return X_new

def stage2():
    log("="*60)
    log("STAGE 2: 14 architectures, 26 features (21 + 5 macro)")
    log("="*60)
    macro_df = fetch_macro()
    Xt, yt, Xv, yv = load_data()
    Xt26 = add_macro_to_npz(Xt, macro_df)
    Xv26 = add_macro_to_npz(Xv, macro_df)
    log(f"26d shapes: train={Xt26.shape}, val={Xv26.shape}")

    models = [f'LSTM-{i}' for i in range(1,7)] + [f'GRU-{i}' for i in range(1,5)] + \
             [f'Transformer-{i}' for i in [2,4,6]]
    results = []
    for i, name in enumerate(models):
        log(f"  [{i+1}/14] {name} (26d)...")
        r = train_one(name, 26, Xt26, yt, Xv26, yv)
        if r:
            log(f"    IC3={r['best_ic3']:.4f} ep={r['epochs']}")
            results.append(r)
        else:
            log(f"    FAILED")

    results.sort(key=lambda r: r['best_ic3'], reverse=True)
    with open(DATA_DIR / 'stage2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Compare with Stage 1
    s1 = json.load(open(DATA_DIR / 'stage1_results.json')) if (DATA_DIR/'stage1_results.json').exists() else []
    s1_map = {r['name']: r['best_ic3'] for r in s1}
    log(f"\nStage 2 vs Stage 1 IC3 comparison:")
    gains = []
    for r in results[:5]:
        ic1 = s1_map.get(r['name'], 0)
        gain = (r['best_ic3'] - ic1) / max(abs(ic1), 0.001) * 100
        log(f"  {r['name']:15s} {ic1:.4f} → {r['best_ic3']:.4f} ({gain:+.0f}%)")
        gains.append(gain)
    avg_gain = np.mean(gains) if gains else 0
    log(f"  Average gain: {avg_gain:+.0f}%")
    return results

# ============================================================
# Main
# ============================================================
def main():
    log(f"Phase 17 v2 Overnight — GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if stage in ['1', 'all']:
        stage1()

    if stage in ['2', 'all']:
        stage2()

    log("\nOvernight complete. Check stage{N}_results.json")

if __name__ == '__main__':
    main()
