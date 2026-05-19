"""Retest monthly and weekly LSTM with strict v6 pipeline.
INPUT_DATA_RANGE: 2010-01 to 2026-05 (monthly), 2010-01 to 2026-05 (weekly)
WALK_FORWARD: yes (date-based split, no cross-period contamination)
LOOK_AHEAD_RISK: none
TEST_SET_USAGE: read-only, evaluated ONCE per frequency"""
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEV = torch.device('cuda')
B, LR, WD, FIXED_EPOCHS = 128, 5e-4, 1e-4, 30

def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        r[i] = (s[i]-m)/max(std,1e-8)
    return r

def build_features(df, lookback, forward_steps, name):
    """Build 21-dim features with strict date split"""
    print(f"\n=== {name}: lookback={lookback}, forward={forward_steps} ===")
    stocks = sorted(df['code'].unique())
    all_seq, all_y, all_dates = [], [], []

    for code in stocks:
        g = df[df['code']==code].sort_values('date').reset_index(drop=True)
        min_len = lookback + forward_steps * 2
        if len(g) < min_len: continue
        n = len(g); dates = g['date'].tolist()
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)

        feats = np.zeros((n, 21), dtype=np.float32)
        for j, arr in enumerate([c, o, h, l, v]):
            s = pd.Series(arr)
            m = s.rolling(60, min_periods=60).mean(); std = s.rolling(60, min_periods=60).std()
            feats[:,j] = ((arr-m)/std.replace(0,1)).fillna(0).values
        e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
        dif = np.nan_to_num(e12-e26,0); dea = pd.Series(dif).ewm(span=9).mean().values
        feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
        delta = np.diff(c,prepend=c[0]); gs=np.where(delta>0,delta,0); ls=np.where(delta<0,-delta,0)
        feats[:,8]=np.nan_to_num(100-100/(1+pd.Series(gs).ewm(alpha=1/14).mean().values/np.maximum(pd.Series(ls).ewm(alpha=1/14).mean().values,1e-8)),50)
        ma20 = pd.Series(c).rolling(20).mean().values; m60 = pd.Series(c).rolling(60).mean().values
        feats[:,18]=np.nan_to_num((c-ma20)/np.maximum(c,0.01),0)
        feats[:,19]=np.nan_to_num((c-m60)/np.maximum(c,0.01),0)
        feats[:,20]=hash(code)%31/31.0
        feats=np.nan_to_num(feats,0.0)

        # Sequences
        for i in range(lookback-1, n - forward_steps * 2):
            if c[i]<=0.01: continue
            seq = feats[i-lookback+1:i+1]
            y3 = np.clip((c[i+forward_steps] - c[i])/max(c[i],0.01), -2, 2) if i+forward_steps<n else 0
            y6 = np.clip((c[i+forward_steps*2] - c[i])/max(c[i],0.01), -2, 2) if i+forward_steps*2<n else 0
            all_seq.append(seq); all_y.append([y3,y6]); all_dates.append(dates[i])

    X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
    dates_arr = np.array(all_dates)
    print(f"  Seqs: {len(X)} | Dates: {dates_arr[0]}~{dates_arr[-1]}")

    # Strict date split
    train_m = np.array([d >= '2015-01' and d <= '2021-12' for d in dates_arr])
    val_m   = np.array([d >= '2022-01' and d <= '2023-12' for d in dates_arr])
    test_m  = np.array([d >= '2024-01' for d in dates_arr])

    Xtr, ytr = X[train_m], y[train_m]
    Xva, yva = X[val_m], y[val_m]
    Xte, yte = X[test_m], y[test_m]
    Xtr, ytr = Xtr[~np.isnan(ytr).any(axis=1)], ytr[~np.isnan(ytr).any(axis=1)]
    Xva, yva = Xva[~np.isnan(yva).any(axis=1)], yva[~np.isnan(yva).any(axis=1)]
    Xte, yte = Xte[~np.isnan(yte).any(axis=1)], yte[~np.isnan(yte).any(axis=1)]

    print(f"  Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
    return Xtr, ytr, Xva, yva, Xte, yte

def train_and_test(Xtr, ytr, Xte, yte, name):
    """Train LSTM-7, fixed epochs, no val info, Test once"""
    torch.manual_seed(456); np.random.seed(456)
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    train_ld = torch.utils.data.DataLoader(train_ds, B, shuffle=True, pin_memory=True)
    model = create_model('LSTM-7', 21).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)

    for ep in range(1, FIXED_EPOCHS+1):
        for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
        model.train()
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEV), yb.to(DEV); opt.zero_grad()
            p = model(Xb)
            loss = 0.5*nn.MSELoss()(p[:,0],yb[:,0]) + 0.5*nn.MSELoss()(p[:,1],yb[:,1])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    test_ld = torch.utils.data.DataLoader(test_ds, B, pin_memory=True)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for Xb, yb in test_ld:
            Xb, yb = Xb.to(DEV), yb.to(DEV); p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
    tp = np.concatenate(tp); tt = np.concatenate(tt)
    ic3 = spearmanr(tp[:,0], tt[:,0])[0]; ic6 = spearmanr(tp[:,1], tt[:,1])[0]
    n = len(tp); cut = int(n*0.3); idx = np.argsort(tp[:,0])
    ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
    sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0

    print(f"  {name}: Test IC3={ic3:.4f}, IC6={ic6:.4f}, SR3={sr3:.3f}")
    return ic3, ic6, sr3

# Load data
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]

# Monthly
m_df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01' ORDER BY code, date", conn, params=stocks)
print(f"Monthly: {len(m_df)} rows, {m_df['code'].nunique()} stocks")

# Weekly
w_df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM weekly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01' ORDER BY code, date", conn, params=stocks)
print(f"Weekly: {len(w_df)} rows, {w_df['code'].nunique()} stocks")
conn.close()

# Build + train
print(f"\n{'='*60}")
print("FINAL COMPARISON: Monthly vs Weekly vs Daily (strict v6)")
print(f"{'='*60}")

results = {}

for name, df, lookback, fwd in [('Monthly', m_df, 60, 3), ('Weekly', w_df, 200, 13)]:
    Xtr, ytr, Xva, yva, Xte, yte = build_features(df, lookback, fwd, name)
    ic3, ic6, sr3 = train_and_test(Xtr, ytr, Xte, yte, name)
    results[name] = {'ic3': ic3, 'ic6': ic6, 'sr3': sr3}

# Daily result from v6
results['Daily'] = {'ic3': 0.1409, 'ic6': 0.0983, 'sr3': 0.443}

print(f"\n{'='*60}")
print(f"{'Frequency':<12} {'Test IC3':>10} {'Test IC6':>10} {'SR3':>8}")
print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8}")
for name, r in results.items():
    print(f"{name:<12} {r['ic3']:10.4f} {r['ic6']:10.4f} {r['sr3']:8.3f}")

best = max(results.items(), key=lambda x: x[1]['ic3'])
print(f"\nBest: {best[0]} IC3={best[1]['ic3']:.4f}")
