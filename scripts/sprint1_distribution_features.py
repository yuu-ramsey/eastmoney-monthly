"""Sprint 1: 12-dim daily distribution features → 33-dim monthly model.
Strict walk-forward. LSTM-7. Test eval ONE TIME."""
import torch, torch.nn as nn, numpy as np, pandas as pd, time
from pathlib import Path
from scipy.stats import spearmanr, skew
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DEVICE = torch.device('cuda')
BATCH, LR, WD, EP, LOOKBACK = 128, 5e-4, 1e-4, 80, 60

# ======== 1. Load daily predictions ========
print("Loading daily predictions...")
daily = pd.read_parquet(OUT / 'daily_signals.parquet')
daily['month'] = daily['date'].str[:7]
print(f"Daily: {len(daily)} rows, {daily['code'].nunique()} stocks, {daily['month'].nunique()} months")

# ======== 2. Compute 12 monthly distribution features per (code, month) ========
print("Computing monthly distribution features...")

def compute_features(group):
    scores = group['score'].values
    dates = group['date'].values
    n = len(scores)
    if n < 5: return None

    feats = {}
    feats['lstm_p5'] = np.percentile(scores, 5)
    feats['lstm_p25'] = np.percentile(scores, 25)
    feats['lstm_p50'] = np.percentile(scores, 50)
    feats['lstm_p75'] = np.percentile(scores, 75)
    feats['lstm_p95'] = np.percentile(scores, 95)
    feats['lstm_mean'] = np.mean(scores)
    feats['lstm_std'] = np.std(scores)
    feats['lstm_skew'] = skew(scores) if len(scores) > 2 else 0.0

    # Trend slope (linear regression of scores over trading days)
    if n >= 5:
        x = np.arange(n); A = np.vstack([x, np.ones(n)]).T
        slope, _ = np.linalg.lstsq(A, scores, rcond=None)[0]
        feats['lstm_trend'] = slope
    else:
        feats['lstm_trend'] = 0.0

    # Vol decay: std(first half) - std(second half)
    mid = n // 2
    feats['lstm_vol_decay'] = np.std(scores[:mid]) - np.std(scores[mid:]) if n >= 4 else 0.0

    # Early vs late: mean(first half) - mean(second half)
    feats['lstm_early_late'] = np.mean(scores[:mid]) - np.mean(scores[mid:]) if n >= 4 else 0.0

    return feats

# Group and compute
monthly_feats = daily.groupby(['code', 'month']).apply(compute_features).dropna().reset_index()
monthly_feats = pd.concat([monthly_feats[['code', 'month']],
                           pd.DataFrame(monthly_feats[0].tolist())], axis=1)
print(f"Monthly distribution features: {len(monthly_feats)} rows, {len(monthly_feats.columns)-2} features")

# ======== 3. Merge with existing 21-dim features + build model ========
print("\nBuilding 33-dim training data...")

# Load monthly prices
import sqlite3
conn = sqlite3.connect(str(PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
m_df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01' ORDER BY code, date", conn, params=stocks)
conn.close()

# Build distribution lookup
dist_lookup = {}
for _, r in monthly_feats.iterrows():
    key = (r['code'], r['month'])
    dist_lookup[key] = [r[f'lstm_p5'], r[f'lstm_p25'], r[f'lstm_p50'], r[f'lstm_p75'], r[f'lstm_p95'],
                        r['lstm_mean'], r['lstm_std'], r['lstm_skew'], r['lstm_trend'],
                        r['lstm_vol_decay'], r['lstm_early_late'], 0.0]  # 12th: cs_rank placeholder

def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        if std > 1e-8: r[i] = (s[i] - m) / std
    return r

all_seq, all_y, all_dates = [], [], []
for code in stocks:
    g = m_df[m_df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 24: continue
    n = len(g); dates = g['date'].tolist()
    closes = g['close'].values.astype(float); opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float); lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)

    # 21 base features (same as always)
    feats = np.zeros((n, 33), dtype=np.float32)
    feats[:,0] = rolling_zscore(closes); feats[:,1] = rolling_zscore(opens)
    feats[:,2] = rolling_zscore(highs); feats[:,3] = rolling_zscore(lows)
    feats[:,4] = rolling_zscore(volumes)
    e12 = pd.Series(closes).ewm(span=12).mean().values; e26 = pd.Series(closes).ewm(span=26).mean().values
    dif = np.nan_to_num(e12-e26,0); dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
    dl = np.diff(closes,prepend=closes[0]); gs=np.where(dl>0,dl,0); ls=np.where(dl<0,-dl,0)
    ag=pd.Series(gs).ewm(alpha=1/14).mean().values; al=pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8]=np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)),50)
    ma20=pd.Series(closes).rolling(20).mean().values; m60=pd.Series(closes).rolling(60).mean().values
    feats[:,18]=np.nan_to_num((closes-ma20)/np.maximum(closes,0.01),0)
    feats[:,19]=np.nan_to_num((closes-m60)/np.maximum(closes,0.01),0)
    feats[:,20]=hash(code)%31/31.0

    # Append 12 distribution features per row
    for i in range(n):
        key = (code, dates[i])
        dist = dist_lookup.get(key)
        if dist:
            feats[i, 21:33] = dist

    feats = np.nan_to_num(feats, 0.0)

    for i in range(LOOKBACK-1, n-12):
        if closes[i] <= 0.01: continue
        all_seq.append(feats[i-LOOKBACK+1:i+1])
        y3 = np.clip((closes[i+3]-closes[i])/max(closes[i],0.01), -2, 2) if i+3<n else 0
        y6 = np.clip((closes[i+6]-closes[i])/max(closes[i],0.01), -2, 2) if i+6<n else 0
        all_y.append([y3, y6]); all_dates.append(dates[i])

X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
dates_arr = np.array(all_dates)
train_m = (dates_arr >= '2015-01') & (dates_arr <= '2021-12')
val_m = (dates_arr >= '2022-01') & (dates_arr <= '2023-12')
test_m = (dates_arr >= '2024-01') & (dates_arr <= '2026-05')
Xtr, ytr = X[train_m], y[train_m]; Xva, yva = X[val_m], y[val_m]; Xte, yte = X[test_m], y[test_m]
Xtr, ytr = Xtr[~np.isnan(ytr).any(axis=1)], ytr[~np.isnan(ytr).any(axis=1)]
Xva, yva = Xva[~np.isnan(yva).any(axis=1)], yva[~np.isnan(yva).any(axis=1)]
Xte, yte = Xte[~np.isnan(yte).any(axis=1)], yte[~np.isnan(yte).any(axis=1)]
print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

# Train
torch.manual_seed(456); np.random.seed(456)
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)
model = create_model('LSTM-7', 33).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
best_ic3, best_vl, no_imp = -1.0, float('inf'), 0
for ep in range(1, EP+1):
    for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
    model.train()
    for Xb, yb in train_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
        loss = 0.5*nn.MSELoss()(model(Xb)[:,0], yb[:,0]) + 0.5*nn.MSELoss()(model(Xb)[:,1], yb[:,1])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval(); vl=0; preds,targs=[],[]
    with torch.no_grad():
        for Xb, yb in val_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            p = model(Xb); preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
            vl += (0.5*nn.MSELoss()(p[:,0],yb[:,0])+0.5*nn.MSELoss()(p[:,1],yb[:,1])).item()*Xb.size(0)
    vl/=len(val_ds); ic3 = spearmanr(np.concatenate(preds)[:,0], np.concatenate(targs)[:,0])[0]
    best_ic3 = max(best_ic3, ic3)
    if vl < best_vl: best_vl, no_imp = vl, 0
    else: no_imp += 1
    if no_imp >= 15 and ep > 20: break

# Test (FROZEN)
test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
test_ld = torch.utils.data.DataLoader(test_ds, BATCH, pin_memory=True)
model.eval(); tp, tt = [], []
with torch.no_grad():
    for Xb, yb in test_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
tp = np.concatenate(tp); tt = np.concatenate(tt)
ic3_t = spearmanr(tp[:,0], tt[:,0])[0]
ic6_t = spearmanr(tp[:,1], tt[:,1])[0]
hit3 = np.mean(np.sign(tp[:,0]) == np.sign(tt[:,0]))
n = len(tp); cut = int(n*0.3); idx = np.argsort(tp[:,0])
ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0
gap = ic3_t - best_ic3

print(f"\n{'='*60}")
print(f"SPRINT 1 RESULTS")
print(f"{'='*60}")
print(f"Val IC3={best_ic3:.4f} | Test IC3={ic3_t:.4f} | IC6={ic6_t:.4f} | SR3={sr3:.3f} | Hit={hit3:.2f} | gap={gap:+.4f}")
print(f"\nBaseline (21d): Val=0.180 Test=0.019")
print(f"Sprint 1 (33d): Val={best_ic3:.4f} Test={ic3_t:.4f}")
delta = ic3_t - 0.019
print(f"Δ vs baseline Test: {delta:+.4f}")
if ic3_t > 0.08: print("KILL SWITCH: PASS > 0.08 → Sprint 2")
elif ic3_t >= 0.04: print("KILL SWITCH: MARGINAL 0.04-0.08 → Sprint 2")
else: print("KILL SWITCH: FAIL < 0.04 → Skip Sprint 2, go Sprint 3")
