"""Stage 1.5 (extra deep) + Stage 3 (multi-seed) — parallel GPU training"""
import torch, torch.nn as nn, numpy as np, json, time, sys, copy
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

PROJECT = Path(__file__).parent.parent
DATA_DIR = PROJECT / '.eastmoney-ai' / 'lstm'
MODEL_DIR = DATA_DIR / 'models_v2'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda')
BATCH, LR, WD, PATIENCE, MAX_EPOCHS = 128, 5e-4, 1e-4, 15, 100

# ============================================================
# Highway LSTM Cell
# ============================================================
class HighwayLSTM(nn.Module):
    """LSTM with highway gate: y = T*LSTM(x) + (1-T)*x"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        if isinstance(self.proj, nn.Identity):
            skip = x
        else:
            skip = self.proj(x)
        t = self.gate(x)
        return self.ln(t * lstm_out + (1 - t) * skip)

# ============================================================
# DenseNet LSTM (concatenation with bottleneck)
# ============================================================
class DenseLSTMLayer(nn.Module):
    def __init__(self, input_dim, growth_rate, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.bottleneck = nn.Linear(hidden_dim, growth_rate)
        self.ln = nn.LayerNorm(growth_rate)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.ln(self.bottleneck(out))

# ============================================================
# ResNet Block LSTM
# ============================================================
class ResLSTMBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.ln1(out + x)
        out2, _ = self.lstm2(out)
        return self.ln2(out2 + out)

# ============================================================
# Model Factory (extends Stage 1)
# ============================================================
class DeepLSTM(nn.Module):
    """Configurable deep LSTM with residual + layernorm"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, use_highway=False):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.use_highway = use_highway
        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.gates = nn.ModuleList() if use_highway else None
        for _ in range(num_layers):
            self.layers.append(nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True))
            self.lns.append(nn.LayerNorm(hidden_dim))
            if use_highway:
                self.gates.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()))
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        fc_hidden = max(32, hidden_dim // 2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, fc_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_hidden, 2))

    def forward(self, x):
        h = self.proj(x)
        for i, lstm in enumerate(self.layers):
            o, _ = lstm(h)
            if self.use_highway:
                t = self.gates[i](h)
                h = self.lns[i](t * o + (1 - t) * h)
            else:
                h = self.lns[i](o + h)
        w = torch.softmax(self.attn(h), dim=1)
        return self.fc(self.dropout((w * h).sum(1)))

def create_deep_model(name, input_dim=21):
    """Create extra deep architectures"""
    if name == 'LSTM-7':
        return DeepLSTM(input_dim, 128, 7, 0.35)
    elif name == 'LSTM-8':
        return DeepLSTM(input_dim, 128, 8, 0.35)
    elif name == 'LSTM-10':
        return DeepLSTM(input_dim, 96, 10, 0.4)
    elif name == 'LSTM-12':
        return DeepLSTM(input_dim, 64, 12, 0.4)
    elif name == 'LSTM-Highway-10':
        return DeepLSTM(input_dim, 64, 10, 0.4, use_highway=True)
        m = nn.ModuleList([nn.LSTM(input_dim, 128, 1, batch_first=True)] +
                          [nn.LSTM(128, 128, 1, batch_first=True) for _ in range(6)])
        lns = nn.ModuleList([nn.LayerNorm(128) for _ in range(7)])
        proj = nn.Linear(input_dim, 128) if input_dim != 128 else None
        attn = nn.Linear(128, 1)
        fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.35), nn.Linear(64, 2))
        class Model(nn.Module):
            def __init__(s):
                super().__init__(); s.m, s.lns, s.proj, s.attn, s.fc, s.d = m, lns, proj, attn, fc, nn.Dropout(0.35)
            def forward(s, x):
                h = s.proj(x) if s.proj else x
                for i in range(7):
                    o, _ = s.m[i](h)
                    h = s.lns[i](o + (h if i > 0 and (s.proj or i > 0) else h if s.proj else h))
                w = torch.softmax(s.attn(h), dim=1); h = (w * h).sum(1)
                return s.fc(s.d(h))
        return Model()

    elif name == 'LSTM-8':
        # 8 layers, residual
        layers = [nn.LSTM(input_dim if i==0 else 128, 128, 1, batch_first=True) for i in range(8)]
        class Model(nn.Module):
            def __init__(s):
                super().__init__()
                s.layers = nn.ModuleList(layers); s.lns = nn.ModuleList([nn.LayerNorm(128) for _ in range(8)])
                s.proj = nn.Linear(input_dim, 128)
                s.attn = nn.Linear(128, 1); s.d = nn.Dropout(0.35)
                s.fc = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.35), nn.Linear(64,2))
            def forward(s, x):
                h = s.proj(x)
                for i in range(8):
                    o, _ = s.layers[i](h); h = s.lns[i](o + h)
                w = torch.softmax(s.attn(h), dim=1)
                return s.fc(s.d((w * h).sum(1)))
        return Model()

    elif name == 'LSTM-10':
        # 10 highway layers
        layers = [HighwayLSTM(input_dim if i==0 else 64, 64) for i in range(10)]
        class Model(nn.Module):
            def __init__(s):
                super().__init__(); s.layers = nn.ModuleList(layers)
                s.proj = nn.Linear(input_dim, 64)
                s.attn = nn.Linear(64, 1); s.d = nn.Dropout(0.4)
                s.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), s.d, nn.Linear(32, 2))
            def forward(s, x):
                h = s.proj(x)
                for l in s.layers: h = l(h)
                w = torch.softmax(s.attn(h), dim=1)
                return s.fc(s.d((w * h).sum(1)))
        return Model()

    elif name == 'LSTM-Dense-8':
        # DenseNet style: each 2-layer block concatenates
        class Model(nn.Module):
            def __init__(s):
                super().__init__()
                s.proj = nn.Linear(input_dim, 64)
                s.block1 = nn.LSTM(64, 64, 2, batch_first=True)
                s.ln1 = nn.LayerNorm(64)
                s.block2 = nn.LSTM(128, 64, 2, batch_first=True)
                s.ln2 = nn.LayerNorm(64)
                s.block3 = nn.LSTM(192, 64, 2, batch_first=True)
                s.ln3 = nn.LayerNorm(64)
                s.block4 = nn.LSTM(256, 64, 1, batch_first=True)
                s.ln4 = nn.LayerNorm(64)
                s.attn = nn.Linear(64, 1); s.d = nn.Dropout(0.4)
                s.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
            def forward(s, x):
                h0 = s.proj(x); h1, _ = s.block1(h0); h1 = s.ln1(h1)
                h1c = torch.cat([h0, h1], dim=-1); h2, _ = s.block2(h1c); h2 = s.ln2(h2)
                h2c = torch.cat([h0, h1, h2], dim=-1); h3, _ = s.block3(h2c); h3 = s.ln3(h3)
                h3c = torch.cat([h0, h1, h2, h3], dim=-1); h4, _ = s.block4(h3c); h4 = s.ln4(h4)
                w = torch.softmax(s.attn(h4), dim=1)
                return s.fc(s.d((w * h4).sum(1)))
        return Model()

    elif name == 'Transformer-8':
        return _make_transformer(input_dim, 128, 8, 4, 0.35)
    elif name == 'Transformer-12':
        return _make_transformer(input_dim, 128, 12, 4, 0.4)
    elif name == 'ResLSTM-8':
        return _make_reslstm(input_dim, 128, 4, 0.3)  # 4 blocks × 2 layers = 8
    elif name == 'ResLSTM-12':
        return _make_reslstm(input_dim, 96, 6, 0.35)  # 6 blocks × 2 layers = 12
    else:
        raise ValueError(name)

def _make_transformer(input_dim, d_model, layers, heads, dropout):
    proj = nn.Linear(input_dim, d_model)
    pos = nn.Parameter(torch.randn(1, 60, d_model) * 0.02)
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model, heads, 512, dropout, batch_first=True), layers)
    fc = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    class M(nn.Module):
        def __init__(s): super().__init__(); s.proj, s.pos, s.enc, s.fc = proj, pos, enc, fc
        def forward(s, x): return s.fc(s.enc(s.proj(x) + s.pos[:, :x.size(1), :]).mean(1))
    return M()

def _make_reslstm(input_dim, hidden_dim, blocks, dropout):
    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.proj = nn.Linear(input_dim, hidden_dim)
            s.blocks = nn.ModuleList([ResLSTMBlock(hidden_dim) for _ in range(blocks)])
            s.attn = nn.Linear(hidden_dim, 1); s.d = nn.Dropout(dropout)
            s.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        def forward(s, x):
            h = s.proj(x)
            for b in s.blocks: h = b(h)
            w = torch.softmax(s.attn(h), dim=1)
            return s.fc(s.d((w * h).sum(1)))
    return M()

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ============================================================
# Training
# ============================================================
def load_data():
    Xt = np.load(DATA_DIR/'train.npz')['X'].astype(np.float32)
    yt = np.load(DATA_DIR/'train.npz')['y'].astype(np.float32)
    Xv = np.load(DATA_DIR/'val.npz')['X'].astype(np.float32)
    yv = np.load(DATA_DIR/'val.npz')['y'].astype(np.float32)
    return Xt, yt, Xv, yv

def train_one(name, input_dim, seed, Xt, yt, Xv, yv):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
    train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)

    # lr warmup (first 5 epochs)
    try:
        model = create_deep_model(name, input_dim).to(DEVICE)
    except:
        sys.path.insert(0, str(PROJECT / 'lib' / 'lstm'))
        from model_v2 import create_model
        model = create_model(name, input_dim).to(DEVICE)

    n_p = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)  # start low for warmup
    best_vl, best_ep, no_imp = float('inf'), 0, 0
    best_ic3 = -1.0
    hist = {'ic_y3': []}

    for ep in range(1, MAX_EPOCHS+1):
        # Warmup
        if ep <= 10:
            for pg in opt.param_groups:
                pg['lr'] = min(LR, LR * ep / 10.0)

        model.train(); tl = 0.0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = 0.5 * nn.MSELoss()(pred[:,0], yb[:,0]) + 0.5 * nn.MSELoss()(pred[:,1], yb[:,1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * Xb.size(0)
        tl /= len(train_ds)

        model.eval(); vl = 0.0; preds, targs = [], []
        with torch.no_grad():
            for Xb, yb in val_ld:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                p = model(Xb)
                vl += (0.5*nn.MSELoss()(p[:,0],yb[:,0])+0.5*nn.MSELoss()(p[:,1],yb[:,1])).item()*Xb.size(0)
                preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
        vl /= len(val_ds)
        pr = np.concatenate(preds); tg = np.concatenate(targs)
        ic3 = spearmanr(pr[:,0], tg[:,0])[0] if len(pr) > 10 else 0.0
        hist['ic_y3'].append(float(ic3))

        if vl < best_vl:
            best_vl, best_ep, no_imp = vl, ep, 0
            torch.save(model.state_dict(), MODEL_DIR / f'{name}_s{seed}.pt')
        else:
            no_imp += 1
        best_ic3 = max(best_ic3, ic3)

        # Early exit if loss explodes
        if tl > 1e3:
            break
        if no_imp >= PATIENCE and ep > 20:
            break

    del model; torch.cuda.empty_cache()
    return {'name': name, 'seed': seed, 'params': n_p, 'epochs': ep,
            'best_ic3': best_ic3, 'best_vl': best_vl, 'best_ep': best_ep,
            'ic3_final': hist['ic_y3'][-1], 'gradient_ok': not (best_ic3 < -0.05)}

# ============================================================
# Stage 1.5: 8 extra deep architectures
# ============================================================
def stage1_5():
    print("STAGE 1.5: 8 extra deep architectures (21 features)")
    Xt, yt, Xv, yv = load_data()
    models = ['LSTM-7', 'LSTM-8', 'LSTM-10', 'LSTM-12', 'LSTM-Dense-8', 'LSTM-Highway-10',
              'Transformer-8', 'Transformer-12']
    results = []
    for name in models:
        print(f"  {name}...")
        r = train_one(name, 21, 42, Xt, yt, Xv, yv)
        r.pop('seed', None)
        print(f"    IC3={r['best_ic3']:.4f} VL={r['best_vl']:.4f} ep={r['epochs']} params={r['params']:,} grad_ok={r['gradient_ok']}")
        results.append(r)

    results.sort(key=lambda r: r['best_ic3'], reverse=True)
    with open(DATA_DIR / 'stage1_5_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nStage 1.5 Top:")
    for r in results[:5]:
        print(f"  {r['name']:20s} IC3={r['best_ic3']:.4f}")
    return results

# ============================================================
# Stage 3: Multi-seed stability
# ============================================================
def stage3():
    print("\nSTAGE 3: Top 5 multi-seed stability")
    Xt, yt, Xv, yv = load_data()
    seeds = [42, 123, 456, 789, 1024]

    # Get top 5 from combined Stage 1 + 1.5
    all_prev = []
    for f in ['stage1_results.json', 'stage1_5_results.json']:
        p = DATA_DIR / f
        if p.exists():
            all_prev.extend(json.load(open(p)))

    all_prev.sort(key=lambda r: r['best_ic3'], reverse=True)
    top5 = [r['name'] for r in all_prev[:5]]
    print(f"Top 5: {top5}")

    multi_results = {}
    for name in top5:
        model_seeds = []
        for s in seeds:
            print(f"  {name} seed={s}...")
            r = train_one(name, 21, s, Xt, yt, Xv, yv)
            model_seeds.append(r['best_ic3'])
            print(f"    IC3={r['best_ic3']:.4f}")

        mean_ic = np.mean(model_seeds)
        std_ic = np.std(model_seeds)
        cv = std_ic / abs(mean_ic) if abs(mean_ic) > 0 else 999
        stability = 'stable' if cv < 0.3 else 'marginal' if cv < 0.5 else 'unstable'
        multi_results[name] = {'seeds': model_seeds, 'mean': float(mean_ic), 'std': float(std_ic),
                               'cv': float(cv), 'stability': stability}
        print(f"    mean={mean_ic:.4f} std={std_ic:.4f} cv={cv:.2f} [{stability}]")

    with open(DATA_DIR / 'stage3_results.json', 'w') as f:
        json.dump(multi_results, f, indent=2)

    print("\nMulti-seed summary:")
    for name, m in sorted(multi_results.items(), key=lambda x: x[1]['mean'], reverse=True):
        print(f"  {name:20s} {m['mean']:.4f} ± {m['std']:.4f} [{m['stability']}]")
    return multi_results

if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ['1.5', 'all']:
        stage1_5()
    if stage in ['3', 'all']:
        stage3()
