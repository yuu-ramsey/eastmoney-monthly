"""Weekly GPU deep learning v2: time-safe version
Fixes:
  - 日线信号 ds_* 特征已弃用（v5 比例分割泄漏，v6 仅在 2022+ 有效）
  - 使用 16 维安全特征（动量 + MA + 波动率 + 成交量）
  - 折扣多月目标 (γ=0.9)
  - 容量控制 (27K 有效样本)
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import sys

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

sys.path.insert(0, str(PROJECT / 'lib' / 'lstm'))

DEV = torch.device('cuda')
print(f"Device: {torch.cuda.get_device_name(0)}")

# ======== 1. 构建安全特征 + 折扣目标 ========
print("\n1/3 Building 16-dim safe features with discounted target...")

conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
w_raw = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume FROM weekly_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01'
    ORDER BY code, date
""", conn, params=stocks)
conn.close()
w_raw['date'] = w_raw['date'].astype(str)

FEATURE_DIM = 16  # 安全特征（不含 ds_*）
LOOKBACK = 104  # 2 years

def build_safe_sequences(df_merged):
    """构建16维安全特征序列 + 折扣多月目标"""
    all_seqs, all_targets, all_dates = [], [], []

    for code in sorted(df_merged['code'].unique()):
        g = df_merged[df_merged['code'] == code].sort_values('date').reset_index(drop=True)
        n = len(g)
        if n < LOOKBACK + 52:
            continue

        closes = g['close'].values.astype(float)
        opens = g['open'].values.astype(float)
        highs = g['high'].values.astype(float)
        lows = g['low'].values.astype(float)
        vols = g['volume'].values.astype(float)
        dates = g['date'].tolist()

        F = np.zeros((n, FEATURE_DIM), dtype=np.float32)

        # [0:5] OHLCV z-scores (60-week rolling, past-only)
        for j, arr in enumerate([closes, opens, highs, lows, vols]):
            s = pd.Series(arr)
            m = s.rolling(60, min_periods=60).mean()
            std = s.rolling(60, min_periods=60).std()
            F[:, j] = ((arr - m) / std.replace(0, 1)).fillna(0).values

        # [5:8] MACD
        e12 = pd.Series(closes).ewm(span=12).mean().values
        e26 = pd.Series(closes).ewm(span=26).mean().values
        dif = np.nan_to_num(e12 - e26, 0)
        dea = pd.Series(dif).ewm(span=9).mean().values
        F[:, 5] = dif; F[:, 6] = dea; F[:, 7] = (dif - dea) * 2

        # [8] RSI
        delta = np.diff(closes, prepend=closes[0])
        gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
        ag = pd.Series(gain).ewm(alpha=1/14).mean().values
        al = pd.Series(loss).ewm(alpha=1/14).mean().values
        F[:, 8] = np.nan_to_num(100 - 100/(1 + ag/np.maximum(al, 1e-8)), 50)

        # [9:13] 周线动量 (1w, 4w, 13w, 26w, 52w)
        for j, w in enumerate([1, 4, 13, 26, 52]):
            for i in range(n):
                if i >= w and closes[i-w] > 0.01:
                    F[i, 9+j] = np.clip((closes[i] - closes[i-w]) / closes[i-w], -2, 2)

        # [14] 52周位置
        for i in range(n):
            lo = max(0, i-52)
            h52 = highs[lo:i+1].max(); l52 = lows[lo:i+1].min()
            F[i, 14] = (closes[i] - l52) / max(h52 - l52, 0.01)

        # [15] MA20 位置
        ma20 = pd.Series(closes).rolling(20).mean().values
        F[:, 15] = np.nan_to_num((closes - ma20) / np.maximum(closes, 0.01), 0)

        F = np.nan_to_num(F, 0.0)

        # 构建序列 + 折扣目标 (γ=0.9, 13w/26w/39w/52w)
        for i in range(LOOKBACK-1, n - 52):
            if closes[i] <= 0.01:
                continue
            seq = F[i-LOOKBACK+1:i+1]
            disc = 0.0; gamma = 1.0
            for horizon in [13, 26, 39, 52]:
                if i + horizon < n:
                    r = np.clip((closes[i+horizon] - closes[i]) / closes[i], -2, 2)
                    disc += gamma * r
                    gamma *= 0.9
            all_seqs.append(seq)
            all_targets.append(disc)
            all_dates.append(dates[i])

    return np.array(all_seqs, dtype=np.float32), np.array(all_targets, dtype=np.float32), np.array(all_dates)

X, Y, dates = build_safe_sequences(w_raw)
print(f"  Sequences: {X.shape}, Target: {Y.shape}")

# 先清理NaN再统计
mask_all = np.isfinite(Y) & ~np.isnan(X).any(axis=(1,2))
X, Y, dates = X[mask_all], Y[mask_all], dates[mask_all]
print(f"  Clean sequences: {X.shape}, Target: mean={Y.mean():.4f}, std={Y.std():.4f}")

# 时间分割
train_m = (dates >= '2015-01') & (dates <= '2021-12')
val_m   = (dates >= '2022-01') & (dates <= '2023-12')
test_m  = (dates >= '2024-01')

Xtr, Ytr = X[train_m], Y[train_m]
Xva, Yva = X[val_m], Y[val_m]
Xte, Yte = X[test_m], Y[test_m]

# 标准化（训练集统计量）
X_mean = Xtr.mean(axis=(0,1), keepdims=True)
X_std = Xtr.std(axis=(0,1), keepdims=True) + 1e-8
Xtr = (Xtr - X_mean) / X_std
Xva = (Xva - X_mean) / X_std
Xte = (Xte - X_mean) / X_std

Y_mean_tr = Ytr.mean(); Y_std_tr = Ytr.std() + 1e-8
Ytr_n = (Ytr - Y_mean_tr) / Y_std_tr
Yva_n = (Yva - Y_mean_tr) / Y_std_tr
Yte_n = (Yte - Y_mean_tr) / Y_std_tr

print(f"  Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")

# ======== 2. 模型 + 训练 ========
print("\n2/3 Training models (capacity-controlled, safe features)...")

class WeeklyLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

class WeeklyGRU(nn.Module):
    def __init__(self, input_dim=16, hidden=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

def train_and_eval(model, name, Xtr, Ytr, Xva, Yva, Xte, Yte, epochs=100):
    model = model.to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n_params:,} params", end="", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Ytr_t = torch.from_numpy(Ytr).float().reshape(-1, 1)
    Yva_t = torch.from_numpy(Yva).float().reshape(-1, 1)
    Yte_t = torch.from_numpy(Yte).float().reshape(-1, 1)

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), Ytr_t)
    val_ds   = torch.utils.data.TensorDataset(torch.from_numpy(Xva), Yva_t)
    test_ds  = torch.utils.data.TensorDataset(torch.from_numpy(Xte), Yte_t)
    train_ld = torch.utils.data.DataLoader(train_ds, 128, shuffle=True, pin_memory=True)
    val_ld   = torch.utils.data.DataLoader(val_ds, 256, pin_memory=True)
    test_ld  = torch.utils.data.DataLoader(test_ds, 256, pin_memory=True)

    best_val_ic = -999; best_state = None
    patience = 20; no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            p = model(Xb)
            loss = nn.MSELoss()(p, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            p_all, y_all = [], []
            with torch.no_grad():
                for Xb, yb in val_ld:
                    p_all.append(model(Xb.to(DEV)).cpu().numpy())
                    y_all.append(yb.numpy())
            pv = np.concatenate(p_all).flatten()
            yv = np.concatenate(y_all).flatten()
            ic = spearmanr(pv, yv)[0]

            if ic > best_val_ic:
                best_val_ic = ic
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    p_all, y_all = [], []
    with torch.no_grad():
        for Xb, yb in test_ld:
            p_all.append(model(Xb.to(DEV)).cpu().numpy())
            y_all.append(yb.numpy())
    pt = np.concatenate(p_all).flatten()
    yt = np.concatenate(y_all).flatten()

    test_ic = spearmanr(pt, yt)[0]
    n = len(pt); cut = int(n * 0.3)
    idx = np.argsort(pt)
    ls = yt[idx[-cut:]] - yt[idx[:cut]]
    sr = ls.mean() / ls.std() * np.sqrt(52/13) if ls.std() > 0 else 0

    print(f" → Test IC={test_ic:.4f}, SR={sr:.3f}, val_best={best_val_ic:.4f}")
    return {'name': name, 'params': n_params, 'test_ic': test_ic, 'sr': sr, 'best_val_ic': best_val_ic}

# 模型候选（容量递增）
model_configs = [
    ('LSTM-tiny',  WeeklyLSTM, dict(hidden=64,  num_layers=1, dropout=0.5)),
    ('LSTM-small', WeeklyLSTM, dict(hidden=128, num_layers=1, dropout=0.4)),
    ('LSTM-mid',   WeeklyLSTM, dict(hidden=128, num_layers=2, dropout=0.4)),
    ('GRU-small',  WeeklyGRU,  dict(hidden=128, num_layers=1, dropout=0.4)),
    ('GRU-mid',    WeeklyGRU,  dict(hidden=128, num_layers=2, dropout=0.4)),
]

results = []
for name, cls, kwargs in model_configs:
    t0 = time.time()
    r = train_and_eval(cls(**kwargs), name, Xtr, Ytr_n, Xva, Yva_n, Xte, Yte_n)
    r['time_s'] = time.time() - t0
    results.append(r)

# ======== 3. 最终报告 ========
print(f"\n{'='*70}")
print("FINAL: Weekly GPU DL Results (safe features, discounted target)")
print(f"{'='*70}")
print(f"{'Model':<15} {'Params':>8} {'Test IC':>10} {'SR':>8} {'Val IC':>10} {'Time':>8}")
print(f"{'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

for r in sorted(results, key=lambda x: x['test_ic'], reverse=True):
    print(f"  {r['name']:<15} {r['params']:>8,} {r['test_ic']:10.4f} {r['sr']:8.3f} {r['best_val_ic']:10.4f} {r['time_s']:>7.0f}s")

print(f"\n  --- Baselines (discounted target) ---")
print(f"  Ridge (safe feats):       IC≈0.063  (walk-forward)")
print(f"  LightGBM-small (safe):    IC≈0.070  (walk-forward)")
print(f"  Old LSTM-7 (raw target):  IC=0.007")
print(f"  Old LGB (raw target):     IC=0.045")

if results:
    best = max(results, key=lambda x: x['test_ic'])
    print(f"\n  Best DL: {best['name']} IC={best['test_ic']:.4f}")
    print(f"  vs Ridge baseline: {best['test_ic']/0.063:.1f}x")
    print(f"  vs old LSTM-7:     {best['test_ic']/0.007:.1f}x")

print(f"\n{'='*70}")
print("Done.")
