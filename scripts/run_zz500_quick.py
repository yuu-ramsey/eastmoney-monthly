"""CSI 500 quick test — 27 stocks with >=84 months of data"""
import torch, torch.nn as nn, numpy as np, sqlite3, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

DATA = Path(__file__).parent.parent / '.eastmoney-ai' / 'lstm'
DB = Path(__file__).parent.parent / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEVICE = torch.device('cuda')
LOOKBACK, BATCH, LR, WD, EP = 60, 64, 5e-4, 1e-4, 80

# Get CSI 500 codes with >=84 months
conn = sqlite3.connect(str(DB))
import akshare as ak
csi500_codes = ak.index_stock_cons('000905')['品种代码'].tolist()
valid = []
for c in csi500_codes:
    n = conn.execute("SELECT COUNT(*) FROM monthly_klines WHERE code=?", (c,)).fetchone()[0]
    if n >= 84: valid.append(c)
print(f"CSI 500 with >=84 months: {len(valid)}")

df = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume FROM monthly_klines
    WHERE code IN ({','.join('?'*len(valid))}) AND date >= '2010-01'
    ORDER BY code, date
""", conn, params=valid)
conn.close()
print(f"Klines: {len(df)} rows, {df['code'].nunique()} stocks")

# Build features
def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        if std > 1e-8: r[i] = (s[i] - m) / std
    return r

all_seq, all_y, all_dates = [], [], []
for code in df['code'].unique():
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 24: continue
    n = len(g)
    dates = g['date'].tolist()
    closes = g['close'].values.astype(float); opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float); lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)

    feats = np.zeros((n, 21), dtype=np.float32)
    feats[:,0] = rolling_zscore(closes); feats[:,1] = rolling_zscore(opens)
    feats[:,2] = rolling_zscore(highs); feats[:,3] = rolling_zscore(lows)
    feats[:,4] = rolling_zscore(volumes)
    e12 = pd.Series(closes).ewm(span=12).mean().values; e26 = pd.Series(closes).ewm(span=26).mean().values
    dif = np.nan_to_num(e12-e26,0); dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
    dl = np.diff(closes, prepend=closes[0])
    gs = np.where(dl>0,dl,0); ls = np.where(dl<0,-dl,0)
    ag = pd.Series(gs).ewm(alpha=1/14).mean().values; al = pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8] = np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)), 50)
    k, d = np.full(n,50.0), np.full(n,50.0)
    for i in range(8, n):
        hh, ll = highs[i-8:i+1].max(), lows[i-8:i+1].min()
        rsv = (closes[i]-ll)/max(hh-ll,0.01)*100
        k[i] = k[i-1]*2/3+rsv*1/3; d[i] = d[i-1]*2/3+k[i]*1/3
    feats[:,9]=k; feats[:,10]=3*k-2*d
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
    tp = (highs+lows+closes)/3
    ma_tp = pd.Series(tp).rolling(20).mean().values; md = pd.Series(np.abs(tp-ma_tp)).rolling(20).mean().values
    feats[:,14] = np.nan_to_num((tp-ma_tp)/np.maximum(md*0.015,0.001), 0)
    m60 = pd.Series(closes).rolling(60).mean().values
    feats[:,18] = np.nan_to_num((closes-ma20)/np.maximum(closes,0.01), 0)
    feats[:,19] = np.nan_to_num((closes-m60)/np.maximum(closes,0.01), 0)
    feats[:,20] = hash(code) % 31 / 31.0
    feats = np.nan_to_num(feats, 0.0)

    for i in range(LOOKBACK-1, n-12):
        if closes[i] <= 0.01: continue
        seq = feats[i-LOOKBACK+1:i+1]
        y3 = np.clip((closes[i+3]-closes[i])/max(closes[i],0.01), -2, 2) if i+3<n else 0
        y6 = np.clip((closes[i+6]-closes[i])/max(closes[i],0.01), -2, 2) if i+6<n else 0
        all_seq.append(seq); all_y.append([y3,y6]); all_dates.append(dates[i])

X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
dates_arr = np.array(all_dates)
print(f"Sequences: {len(X)}")

train_m = (dates_arr >= '2015-01') & (dates_arr <= '2021-12')
val_m = (dates_arr >= '2022-01') & (dates_arr <= '2023-12')
test_m = (dates_arr >= '2024-01') & (dates_arr <= '2026-05')
Xtr, ytr = X[train_m], y[train_m]
Xva, yva = X[val_m], y[val_m]
Xte, yte = X[test_m], y[test_m]
m = lambda Xx, yy: (~np.isnan(yy).any(axis=1),)
Xtr, ytr = Xtr[~np.isnan(ytr).any(axis=1)], ytr[~np.isnan(ytr).any(axis=1)]
Xva, yva = Xva[~np.isnan(yva).any(axis=1)], yva[~np.isnan(yva).any(axis=1)]
Xte, yte = Xte[~np.isnan(yte).any(axis=1)], yte[~np.isnan(yte).any(axis=1)]
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

# Test
test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
test_ld = torch.utils.data.DataLoader(test_ds, BATCH, pin_memory=True)
model.eval(); tp, tt = [], []
with torch.no_grad():
    for Xb, yb in test_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
tp = np.concatenate(tp); tt = np.concatenate(tt)
ic3_t = spearmanr(tp[:,0], tt[:,0])[0]; ic6_t = spearmanr(tp[:,1], tt[:,1])[0]
n = len(tp); cut = int(n*0.3)
idx = np.argsort(tp[:,0])
ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0
hit3 = np.mean(np.sign(tp[:,0]) == np.sign(tt[:,0]))
gap = ic3_t - best_ic3

print(f"CSI500: Val IC3={best_ic3:.4f} Test IC3={ic3_t:.4f} IC6={ic6_t:.4f} SR3={sr3:.3f} Hit={hit3:.2f} gap={gap:+.4f} ep={ep}")
print(f"\nFINAL COMPARISON:")
print(f"  HS300 monthly: Val=0.180 Test=0.019 gap=-0.161")
print(f"  HS300 weekly:  Val=0.114 Test=0.008 gap=-0.106")
print(f"  CSI500 monthly: Val={best_ic3:.4f} Test={ic3_t:.4f} gap={gap:+.4f}")
if ic3_t < 0.05:
    print("KILL SWITCH: ALL < 0.05. A-share monthly/weekly/HS300/CSI500 LSTM path exhausted.")
