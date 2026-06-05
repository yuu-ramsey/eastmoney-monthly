"""Phase 17 v5: Daily pipeline — 21-dim features + Test eval. LSTM-7, one-shot."""
import torch, torch.nn as nn, numpy as np, sqlite3, pandas as pd, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DEVICE = torch.device('cuda')
LOOKBACK_DAILY = 252  # ~1 year of trading days
BATCH, LR, WD, EP = 128, 5e-4, 1e-4, 80

print("Loading daily data...")
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
df = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume, turnover_rate FROM daily_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01'
    ORDER BY code, date
""", conn, params=stocks)
conn.close()
print(f"{len(df)} rows, {df['code'].nunique()} stocks")

def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        if std > 1e-8: r[i] = (s[i] - m) / std
    return r

all_seq, all_y = [], []
stock_count = 0
for code in stocks:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK_DAILY + 504: continue  # need 1yr lookback + 2yr forward
    n = len(g)
    closes = g['close'].values.astype(float); opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float); lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)

    feats = np.zeros((n, 21), dtype=np.float32)
    feats[:,0] = rolling_zscore(closes, 60); feats[:,1] = rolling_zscore(opens, 60)
    feats[:,2] = rolling_zscore(highs, 60); feats[:,3] = rolling_zscore(lows, 60)
    feats[:,4] = rolling_zscore(volumes, 60)
    e12 = pd.Series(closes).ewm(span=12).mean().values; e26 = pd.Series(closes).ewm(span=26).mean().values
    dif = np.nan_to_num(e12-e26,0); dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:,5] = dif; feats[:,6] = dea; feats[:,7] = (dif-dea)*2
    dl = np.diff(closes, prepend=closes[0]); gs = np.where(dl>0,dl,0); ls = np.where(dl<0,-dl,0)
    ag = pd.Series(gs).ewm(alpha=1/14).mean().values; al = pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8] = np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)), 50)
    k, d = np.full(n,50.0), np.full(n,50.0)
    for i in range(8, n):
        hh, ll = highs[i-8:i+1].max(), lows[i-8:i+1].min()
        rsv = (closes[i]-ll)/max(hh-ll,0.01)*100
        k[i] = k[i-1]*2/3+rsv*1/3; d[i] = d[i-1]*2/3+k[i]*1/3
    feats[:,9] = k; feats[:,10] = 3*k-2*d
    ma20 = pd.Series(closes).rolling(20).mean().values; s20 = pd.Series(closes).rolling(20).std().values
    feats[:,11] = np.nan_to_num((closes-(ma20-2*s20))/np.maximum(4*s20,0.01), 0.5)
    tr = np.maximum(highs-lows, np.abs(highs-np.roll(closes,1)))
    atr14 = pd.Series(tr).rolling(14).mean().values
    feats[:,12] = np.nan_to_num(atr14/closes, 0)
    obv = np.zeros(n); obv[0] = volumes[0]
    for i in range(1, n):
        if closes[i]>closes[i-1]: obv[i]=obv[i-1]+volumes[i]
        elif closes[i]<closes[i-1]: obv[i]=obv[i-1]-volumes[i]
        else: obv[i]=obv[i-1]
    feats[:,13] = obv/np.maximum(pd.Series(volumes).cumsum().values, 1)
    tp = (highs+lows+closes)/3; ma_tp = pd.Series(tp).rolling(20).mean().values
    md = pd.Series(np.abs(tp-ma_tp)).rolling(20).mean().values
    feats[:,14] = np.nan_to_num((tp-ma_tp)/np.maximum(md*0.015,0.001), 0)
    m60 = pd.Series(closes).rolling(60).mean().values
    feats[:,18] = np.nan_to_num((closes-ma20)/np.maximum(closes,0.01), 0)
    feats[:,19] = np.nan_to_num((closes-m60)/np.maximum(closes,0.01), 0)
    feats[:,20] = hash(code) % 31 / 31.0
    feats = np.nan_to_num(feats, 0.0)

    # Daily targets: 63 days ≈ 3 months, 126 days ≈ 6 months
    for i in range(LOOKBACK_DAILY-1, n-252):
        if closes[i] <= 0.01: continue
        seq = feats[i-LOOKBACK_DAILY+1:i+1]
        y3d = np.clip((closes[i+63]-closes[i])/max(closes[i],0.01), -2, 2) if i+63<n else 0
        y6d = np.clip((closes[i+126]-closes[i])/max(closes[i],0.01), -2, 2) if i+126<n else 0
        all_seq.append(seq); all_y.append([y3d, y6d])

    stock_count += 1
    if stock_count % 50 == 0:
        print(f"  {stock_count}/{len(stocks)} stocks, {len(all_seq)} seqs")

X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
print(f"X: {X.shape}, y: {y.shape}")

# Walk-forward split by proportional index (daily dates vary)
n = len(X)
n_tr = int(n * 0.55); n_va = int(n * 0.72)
Xtr, ytr = X[:n_tr], y[:n_tr]; Xva, yva = X[n_tr:n_va], y[n_tr:n_va]; Xte, yte = X[n_va:], y[n_va:]
m = ~np.isnan(ytr).any(axis=1); Xtr, ytr = Xtr[m], ytr[m]
m = ~np.isnan(yva).any(axis=1); Xva, yva = Xva[m], yva[m]
m = ~np.isnan(yte).any(axis=1); Xte, yte = Xte[m], yte[m]
print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

# Train LSTM-7
torch.manual_seed(456); np.random.seed(456)
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)
model = create_model('LSTM-7', 21).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
best_vl, no_imp, best_ic3 = float('inf'), 0, -1.0
t0 = time.time()
for ep in range(1, EP+1):
    for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
    model.train()
    for Xb, yb in train_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); p = model(Xb)
        loss = 0.5*nn.MSELoss()(p[:,0],yb[:,0]) + 0.5*nn.MSELoss()(p[:,1],yb[:,1])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval(); vl=0; preds,targs=[],[]
    with torch.no_grad():
        for Xb, yb in val_ld:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            p = model(Xb); preds.append(p.cpu().numpy()); targs.append(yb.cpu().numpy())
            vl += (0.5*nn.MSELoss()(p[:,0],yb[:,0])+0.5*nn.MSELoss()(p[:,1],yb[:,1])).item()*Xb.size(0)
    vl/=len(val_ds)
    ic3 = spearmanr(np.concatenate(preds)[:,0], np.concatenate(targs)[:,0])[0]
    best_ic3 = max(best_ic3, ic3)
    if vl < best_vl: best_vl, no_imp = vl, 0
    else: no_imp += 1
    if no_imp >= 15 and ep > 20: break
print(f"Train: {time.time()-t0:.0f}s, Val IC3={best_ic3:.4f} ep={ep}")

# Test (FROZEN, ONE TIME)
test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
test_ld = torch.utils.data.DataLoader(test_ds, BATCH, pin_memory=True)
model.eval(); tp, tt = [], []
with torch.no_grad():
    for Xb, yb in test_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
tp = np.concatenate(tp); tt = np.concatenate(tt)
ic3_t = spearmanr(tp[:,0], tt[:,0])[0]; ic6_t = spearmanr(tp[:,1], tt[:,1])[0]
n = len(tp); cut = int(n*0.3); idx = np.argsort(tp[:,0])
ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0
hit3 = np.mean(np.sign(tp[:,0]) == np.sign(tt[:,0]))
gap = ic3_t - best_ic3

print(f"\nDAILY Test: IC3={ic3_t:.4f} IC6={ic6_t:.4f} SR3={sr3:.3f} Hit={hit3:.2f} gap={gap:+.4f}")
print(f"\nFINAL COMPARISON:")
print(f"  HS300 monthly:  Val=0.180 Test=0.019 gap=-0.161")
print(f"  HS300 weekly:   Val=0.114 Test=0.008 gap=-0.106")
print(f"  CSI500 monthly: Val=0.172 Test=-0.080 gap=-0.252")
print(f"  HS300 daily:    Val={best_ic3:.4f} Test={ic3_t:.4f} gap={gap:+.4f}")
if ic3_t < 0.05: print("KILL SWITCH: daily < 0.05. LSTM path permanently closed.")
else: print("PASS: daily > 0.05!")
