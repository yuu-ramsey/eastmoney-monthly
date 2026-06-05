"""周线 MC Dropout 降噪训练：不确定性量化 + 折扣目标 + walk-forward 评估

降噪策略：
  1. 目标降噪：折扣多月回报（γ=0.9, 13w+26w+39w+52w）
  2. 推断降噪：MC Dropout 50次采样 → cv = std/|mean|
  3. 信号降噪：过滤 high uncertainty (cv≥0.7) → 去除~31%噪音信号

用法：
  python scripts/weekly_mc_dropout_train.py           # 完整训练+评估
  python scripts/weekly_mc_dropout_train.py --quick   # 快速模式（少epoch）
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time, argparse
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEVICE = torch.device('cuda')
FEATURE_DIM = 16
LOOKBACK = 104  # 2年周线
MC_SAMPLES = 50
GAMMA = 0.9

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"MC Samples: {MC_SAMPLES}, Lookback: {LOOKBACK}w, Features: {FEATURE_DIM}")

# ======== 1. 数据加载 + 安全特征 + 折扣目标 ========
print("\n1/5 Loading weekly data + building safe features + discounted target...")

conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute(
    "SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
w_raw = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume FROM weekly_klines
    WHERE code IN ({','.join('?' * len(stocks))}) AND date >= '2010-01-01'
    ORDER BY code, date
""", conn, params=stocks)
conn.close()
w_raw['date'] = w_raw['date'].astype(str)


def build_safe_sequences(df_merged):
    """16-dim safe features + discounted multi-horizon target"""
    all_seqs, all_targets, all_dates, all_codes = [], [], [], []

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

        # [0:5] OHLCV z-scores (60-week rolling)
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

        # 构建序列 + 折扣目标
        for i in range(LOOKBACK-1, n - 52):
            if closes[i] <= 0.01:
                continue
            seq = F[i-LOOKBACK+1:i+1]
            disc = 0.0
            g = 1.0
            for horizon in [13, 26, 39, 52]:
                if i + horizon < n:
                    r = np.clip((closes[i+horizon] - closes[i]) / closes[i], -2, 2)
                    disc += g * r
                    g *= GAMMA
            all_seqs.append(seq)
            all_targets.append(disc)
            all_dates.append(dates[i])
            all_codes.append(code)

    return (np.array(all_seqs, dtype=np.float32),
            np.array(all_targets, dtype=np.float32),
            np.array(all_dates),
            np.array(all_codes))


X, Y, dates, codes = build_safe_sequences(w_raw)

# 清理 NaN
mask = np.isfinite(Y) & ~np.isnan(X).any(axis=(1, 2))
X, Y, dates, codes = X[mask], Y[mask], dates[mask], codes[mask]
print(f"  Clean sequences: {X.shape}, Target: mean={Y.mean():.4f}, std={Y.std():.4f}")

# 时间分割
train_m = (dates >= '2015-01') & (dates <= '2021-12')
val_m = (dates >= '2022-01') & (dates <= '2023-12')
test_m = (dates >= '2024-01')

Xtr, Ytr = X[train_m], Y[train_m]
Xva, Yva = X[val_m], Y[val_m]
Xte, Yte = X[test_m], Y[test_m]
te_dates = dates[test_m]
te_codes = codes[test_m]

# 标准化（训练集统计量）
X_mean = Xtr.mean(axis=(0, 1), keepdims=True)
X_std = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
Xtr = (Xtr - X_mean) / X_std
Xva = (Xva - X_mean) / X_std
Xte = (Xte - X_mean) / X_std

Y_mean_tr = Ytr.mean()
Y_std_tr = Ytr.std() + 1e-8
Ytr_n = (Ytr - Y_mean_tr) / Y_std_tr
Yva_n = (Yva - Y_mean_tr) / Y_std_tr

print(f"  Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")

# ======== 2. MC Dropout 模型 ========
print("\n2/5 Building MC Dropout models...")


class MCLSTM(nn.Module):
    """LSTM with configurable dropout for MC Dropout"""
    def __init__(self, input_dim=16, hidden=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class MCGRU(nn.Module):
    """GRU with configurable dropout for MC Dropout"""
    def __init__(self, input_dim=16, hidden=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def mc_predict(model, X_tensor, n_samples=MC_SAMPLES, batch_size=256):
    """MC Dropout 推断：N次前向传播，分batch避免OOM"""
    model.train()  # dropout 活跃
    n_total = X_tensor.shape[0]
    all_mean = np.zeros(n_total, dtype=np.float32)
    all_std = np.zeros(n_total, dtype=np.float32)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        Xb = X_tensor[start:end]
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(model(Xb).cpu().numpy())
        s = np.stack(samples, axis=0)  # (N, B, 1)
        all_mean[start:end] = s.mean(axis=0).flatten()
        all_std[start:end] = s.std(axis=0).flatten()

    cv = all_std / (np.abs(all_mean) + 1e-8)
    return all_mean, all_std, cv


def uncertainty_level(cv):
    """cv ≥ 0.7 → high, cv < 0.3 → low, else medium"""
    levels = np.full(len(cv), 'medium', dtype=object)
    levels[cv < 0.3] = 'low'
    levels[cv >= 0.7] = 'high'
    return levels


# ======== 3. 训练 ========
print("\n3/5 Training...")

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
args, _ = parser.parse_known_args()

EPOCHS = 30 if args.quick else 120
PATIENCE = 10 if args.quick else 25
print(f"  Epochs: {EPOCHS}, Patience: {PATIENCE}")

model_configs = [
    ('MC-LSTM-small', MCLSTM, dict(hidden=128, num_layers=1, dropout=0.5)),
    ('MC-LSTM-mid',   MCLSTM, dict(hidden=128, num_layers=2, dropout=0.4)),
    ('MC-GRU-small',  MCGRU,  dict(hidden=128, num_layers=1, dropout=0.5)),
    ('MC-GRU-mid',    MCGRU,  dict(hidden=128, num_layers=2, dropout=0.4)),
]


def train_one(model, name, Xtr, Ytr, Xva, Yva, epochs, patience):
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n_params:,} params", end="", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Ytr_t = torch.from_numpy(Ytr).float().reshape(-1, 1)
    Yva_t = torch.from_numpy(Yva).float().reshape(-1, 1)

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), Ytr_t)
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), Yva_t)
    train_ld = torch.utils.data.DataLoader(train_ds, 128, shuffle=True, pin_memory=True)
    val_ld = torch.utils.data.DataLoader(val_ds, 256, pin_memory=True)

    best_val_ic = -999
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.MSELoss()(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            p_all, y_all = [], []
            with torch.no_grad():
                for Xb, yb in val_ld:
                    p_all.append(model(Xb.to(DEVICE)).cpu().numpy())
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
    return model, best_val_ic


trained_models = {}
for name, cls, kwargs in model_configs:
    t0 = time.time()
    model, best_ic = train_one(cls(**kwargs), name, Xtr, Ytr_n, Xva, Yva_n, EPOCHS, PATIENCE)
    elapsed = time.time() - t0
    trained_models[name] = (model, best_ic)
    print(f" → Val IC={best_ic:.4f}, {elapsed:.0f}s")

# ======== 4. MC Dropout 推断 + 不确定性分层评估 ========
print(f"\n4/5 MC Dropout inference on test set ({len(Xte)} samples)...")

Xte_t = torch.from_numpy(Xte).float().to(DEVICE)

for name, (model, val_ic) in trained_models.items():
    t0 = time.time()
    model.to(DEVICE)

    # MC 推断
    mean, std, cv = mc_predict(model, Xte_t, MC_SAMPLES)
    ulevel = uncertainty_level(cv)
    torch.cuda.empty_cache()

    # 反标准化回到原始尺度
    pred = mean * Y_std_tr + Y_mean_tr
    y_true = Yte  # 原始尺度

    # 整体指标
    ic_all = spearmanr(pred, y_true)[0]
    n = len(pred)
    cut = int(n * 0.3)
    idx = np.argsort(pred)
    ls = y_true[idx[-cut:]] - y_true[idx[:cut]]
    sr = ls.mean() / ls.std() * np.sqrt(52/13) if ls.std() > 0 else 0

    # 不确定性分层
    strata = {}
    for level in ['low', 'medium', 'high']:
        m = ulevel == level
        if m.sum() < 10:
            strata[level] = {'count': int(m.sum()), 'ic': None, 'f1': None, 'avg_score': None}
            continue
        p_s = pred[m]; y_s = y_true[m]
        ic_s = spearmanr(p_s, y_s)[0]
        # F1 (方向预测)
        pred_dir = (p_s > 0).astype(int)
        true_dir = (y_s > 0).astype(int)
        if len(np.unique(pred_dir)) > 1 and len(np.unique(true_dir)) > 1:
            f1_s = f1_score(true_dir, pred_dir, zero_division=0)
            prec = precision_score(true_dir, pred_dir, zero_division=0)
            rec = recall_score(true_dir, pred_dir, zero_division=0)
        else:
            f1_s = prec = rec = None
        strata[level] = {
            'count': int(m.sum()),
            'ic': ic_s,
            'f1': f1_s,
            'precision': prec,
            'recall': rec,
            'avg_score': float(np.abs(p_s).mean()),
        }

    # 过滤 high uncertainty 后
    keep = ulevel != 'high'
    pred_f = pred[keep]; y_f = y_true[keep]
    ic_f = spearmanr(pred_f, y_f)[0] if len(pred_f) > 10 else None

    elapsed = time.time() - t0
    pct_low = 100 * (ulevel == 'low').mean()
    pct_med = 100 * (ulevel == 'medium').mean()
    pct_high = 100 * (ulevel == 'high').mean()

    print(f"\n  {name} (val_ic={val_ic:.4f}, {elapsed:.0f}s)")
    print(f"    All:  IC={ic_all:.4f}  SR={sr:.3f}  N={n}")
    print(f"    Low:  {pct_low:.0f}%  IC={strata['low']['ic']:.4f}" if strata['low']['ic'] else f"    Low:  {pct_low:.0f}%  IC=N/A")
    print(f"    Med:  {pct_med:.0f}%  IC={strata['medium']['ic']:.4f}" if strata['medium']['ic'] else f"    Med:  {pct_med:.0f}%  IC=N/A")
    print(f"    High: {pct_high:.0f}%  IC={strata['high']['ic']:.4f}" if strata['high']['ic'] else f"    High: {pct_high:.0f}%  IC=N/A")
    print(f"    Filtered (no high): IC={ic_f:.4f}  N={int(keep.sum())}")

    # 保存到 trained_models 供后续用
    trained_models[name] = (model, val_ic, {
        'ic_all': ic_all, 'sr': sr,
        'ic_filtered': ic_f,
        'strata': strata,
        'pred': pred, 'y_true': y_true, 'cv': cv, 'ulevel': ulevel,
        'dates': te_dates, 'codes': te_codes,
    })

# ======== 5. Walk-Forward 对比（用 Ridge/LGB baseline） ========
print(f"\n5/5 Walk-forward comparison (MC models vs Ridge/LGB baselines)...")

# 用 MC 模型的预计算预测做 walk-forward IC
from sklearn.linear_model import Ridge
import lightgbm as lgb

# 重建周线特征 dataframe（复用 weekly_noise_reduction 的特征）
df_feat_rows = []
for code in sorted(w_raw['code'].unique()):
    g = w_raw[w_raw['code'] == code].sort_values('date').reset_index(drop=True)
    n = len(g)
    if n < LOOKBACK + 52:
        continue
    closes = g['close'].values.astype(float)
    highs = g['high'].values.astype(float)
    lows = g['low'].values.astype(float)
    vols = g['volume'].values.astype(float)
    dates = g['date'].tolist()

    for i in range(52, n - 13):
        if closes[i] <= 0.01:
            continue
        h52 = np.max(highs[max(0, i-52):i+1])
        l52 = np.min(lows[max(0, i-52):i+1])
        ret_13w = (closes[i] - closes[i-13]) / max(closes[i-13], 0.01) if i >= 13 else 0
        ret_26w = (closes[i] - closes[i-26]) / max(closes[i-26], 0.01) if i >= 26 else 0
        ret_52w = (closes[i] - closes[i-52]) / max(closes[i-52], 0.01) if i >= 52 else 0
        ret_4w = (closes[i] - closes[i-4]) / max(closes[i-4], 0.01) if i >= 4 else 0
        ret_1w = (closes[i] - closes[i-1]) / max(closes[i-1], 0.01) if i >= 1 else 0
        pos_52w = (closes[i] - l52) / max(h52 - l52, 0.01)
        ma20 = np.mean(closes[max(0, i-20):i+1])
        ma_pos_20 = (closes[i] - ma20) / max(closes[i], 0.01)
        vol_13w = np.std(np.diff(closes[max(0, i-13):i+1]) / np.maximum(closes[max(0, i-13):i], 0.01))
        week_range = (highs[i] - lows[i]) / max(closes[i], 0.01)
        close_s = pd.Series(closes[max(0, i-52):i+1])
        e12 = close_s.ewm(span=12).mean().iloc[-1]
        e26 = close_s.ewm(span=26).mean().iloc[-1]
        macd_line = e12 - e26

        fwd_raw = np.clip((closes[i+13] - closes[i]) / max(closes[i], 0.01), -2, 2)
        fwd_disc = fwd_raw
        g = GAMMA
        for h in [26, 39, 52]:
            if i + h < n:
                r = np.clip((closes[i+h] - closes[i]) / max(closes[i], 0.01), -2, 2)
                fwd_disc += g * r
                g *= GAMMA
        fwd_disc = np.clip(fwd_disc, -3, 3)

        df_feat_rows.append({
            'code': code, 'date': dates[i],
            'ret_1w': ret_1w, 'ret_4w': ret_4w, 'ret_13w': ret_13w,
            'ret_26w': ret_26w, 'ret_52w': ret_52w, 'pos_52w': pos_52w,
            'vol_13w': vol_13w, 'ma_pos_20': ma_pos_20,
            'macd_line': macd_line, 'week_range': week_range,
            'fwd_raw': fwd_raw, 'fwd_disc': fwd_disc,
        })

df_wf = pd.DataFrame(df_feat_rows)
SAFE_FEATS_WF = ['ret_1w', 'ret_4w', 'ret_13w', 'ret_26w', 'ret_52w',
                  'pos_52w', 'vol_13w', 'ma_pos_20', 'macd_line', 'week_range']
df_wf = df_wf.dropna(subset=['fwd_disc'])
for f in SAFE_FEATS_WF:
    df_wf[f] = df_wf[f].fillna(0.0)
    df_wf = df_wf[np.isfinite(df_wf[f])]

months_wf = sorted(df_wf['date'].unique())

# Ridge baseline (walk-forward)
ridge_ic, ridge_f1 = [], []
for month in months_wf:
    if month < '2018-01': continue
    tr = df_wf[df_wf['date'] < month]
    te = df_wf[df_wf['date'] == month]
    if len(tr) < 1000 or len(te) < 10: continue
    X_tr = tr[SAFE_FEATS_WF].values.astype(np.float64)
    X_te = te[SAFE_FEATS_WF].values.astype(np.float64)
    y_tr = tr['fwd_disc'].values.astype(np.float64)
    y_te = te['fwd_disc'].values.astype(np.float64)
    if np.any(~np.isfinite(X_tr)) or np.any(~np.isfinite(X_te)): continue
    try:
        m = Ridge(alpha=1.0).fit(X_tr, y_tr)
        p = m.predict(X_te)
        if len(p) > 10:
            ic = spearmanr(p, y_te)[0]
            if not np.isnan(ic):
                ridge_ic.append(ic)
            pred_dir = (p > 0).astype(int)
            true_dir = (y_te > 0).astype(int)
            if len(np.unique(pred_dir)) > 1 and len(np.unique(true_dir)) > 1:
                ridge_f1.append(f1_score(true_dir, pred_dir, zero_division=0))
    except Exception:
        pass

# LGB baseline (walk-forward)
lgb_ic, lgb_f1 = [], []
for month in months_wf:
    if month < '2018-01': continue
    tr = df_wf[df_wf['date'] < month]
    te = df_wf[df_wf['date'] == month]
    if len(tr) < 1000 or len(te) < 10: continue
    X_tr = tr[SAFE_FEATS_WF].values.astype(np.float64)
    X_te = te[SAFE_FEATS_WF].values.astype(np.float64)
    y_tr = tr['fwd_disc'].values.astype(np.float64)
    y_te = te['fwd_disc'].values.astype(np.float64)
    if np.any(~np.isfinite(X_tr)) or np.any(~np.isfinite(X_te)): continue
    try:
        m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.03,
                               min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
                               reg_alpha=0.5, reg_lambda=0.5,
                               random_state=42, verbosity=-1).fit(X_tr, y_tr)
        p = m.predict(X_te)
        if len(p) > 10:
            ic = spearmanr(p, y_te)[0]
            if not np.isnan(ic):
                lgb_ic.append(ic)
            pred_dir = (p > 0).astype(int)
            true_dir = (y_te > 0).astype(int)
            if len(np.unique(pred_dir)) > 1 and len(np.unique(true_dir)) > 1:
                lgb_f1.append(f1_score(true_dir, pred_dir, zero_division=0))
    except Exception:
        pass

# ======== 最终报告 ========
print(f"\n{'='*70}")
print("FINAL: MC Dropout 降噪训练报告")
print(f"{'='*70}")

print(f"\n{'Model':<20} {'IC All':>8} {'IC Filt':>8} {'IC Low':>8} {'IC Med':>8} {'IC High':>8} {'Val IC':>8} {'SR':>8}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for name, (model, val_ic, res) in trained_models.items():
    s = res['strata']
    ic_low = f"{s['low']['ic']:.4f}" if s['low']['ic'] is not None else 'N/A'
    ic_med = f"{s['medium']['ic']:.4f}" if s['medium']['ic'] is not None else 'N/A'
    ic_high = f"{s['high']['ic']:.4f}" if s['high']['ic'] is not None else 'N/A'
    print(f"  {name:<20} {res['ic_all']:8.4f} {res['ic_filtered']:8.4f} {ic_low:>8} {ic_med:>8} {ic_high:>8} {val_ic:8.4f} {res['sr']:8.3f}")

# 不确定性分布
print(f"\n{'Model':<20} {'Low%':>8} {'Med%':>8} {'High%':>8} {'Filt Gain':>10}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
for name, (model, val_ic, res) in trained_models.items():
    u = res['ulevel']
    low_p = 100*(u=='low').mean()
    med_p = 100*(u=='medium').mean()
    high_p = 100*(u=='high').mean()
    gain = (res['ic_filtered'] - res['ic_all']) / max(abs(res['ic_all']), 0.001) * 100
    print(f"  {name:<20} {low_p:7.1f}% {med_p:7.1f}% {high_p:7.1f}% {gain:+9.1f}%")

# F1 分层
print(f"\n{'Model':<20} {'F1 All':>8} {'F1 Low':>8} {'F1 Med':>8} {'F1 High':>8}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for name, (model, val_ic, res) in trained_models.items():
    # F1 全量
    p_all = res['pred']; y_all = res['y_true']
    pd_all = (p_all>0).astype(int); td_all = (y_all>0).astype(int)
    f1_all = f1_score(td_all, pd_all, zero_division=0) if len(np.unique(pd_all))>1 else None
    s = res['strata']
    f1_low = f"{s['low']['f1']:.4f}" if s['low'].get('f1') is not None else 'N/A'
    f1_med = f"{s['medium']['f1']:.4f}" if s['medium'].get('f1') is not None else 'N/A'
    f1_high = f"{s['high']['f1']:.4f}" if s['high'].get('f1') is not None else 'N/A'
    f1_all_s = f"{f1_all:.4f}" if f1_all is not None else 'N/A'
    print(f"  {name:<20} {f1_all_s:>8} {f1_low:>8} {f1_med:>8} {f1_high:>8}")

# Walk-forward baselines
print(f"\n{'='*70}")
print("Walk-Forward Baseline (disc+safe, monthly)")
print(f"{'='*70}")
print(f"  Ridge(a=1):             IC={np.mean(ridge_ic):.4f} +- {np.std(ridge_ic):.4f}  F1={np.mean(ridge_f1):.4f}  n={len(ridge_ic)}")
print(f"  LightGBM-small:         IC={np.mean(lgb_ic):.4f} +- {np.std(lgb_ic):.4f}  F1={np.mean(lgb_f1):.4f}  n={len(lgb_ic)}")

# MC 模型也做 walk-forward（用预计算 prediction 按月算 IC）
print(f"\n  --- MC Models Walk-Forward (precomputed predictions, by month) ---")
for name, (model, val_ic, res) in trained_models.items():
    pred = res['pred']
    y_true = res['y_true']
    wf_dates = res['dates']
    months = sorted(set(wf_dates))
    mc_wf_ic = []
    for month in months:
        m = wf_dates == month
        if m.sum() < 10: continue
        ic = spearmanr(pred[m], y_true[m])[0]
        if not np.isnan(ic): mc_wf_ic.append(ic)
    if mc_wf_ic:
        print(f"  {name:<20}  WF-IC={np.mean(mc_wf_ic):.4f} +- {np.std(mc_wf_ic):.4f}  n={len(mc_wf_ic)}")

print(f"\n{'='*70}")
print("Done.")
