"""Sprint 4: CSI 1000 full pipeline — Baidu API → DB → train → Test.
Complete in one script."""
import requests, sqlite3, time, json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEVICE = torch.device('cuda')
LOOKBACK, BATCH, LR, WD, EP = 60, 128, 5e-4, 1e-4, 80
BAIDU_URL = 'https://finance.pae.baidu.com/selfselect/getstockquotation'

# ======== 1. Fetch CSI 1000 klines from Baidu API ========
print("Fetching CSI 1000 constituents...")
import akshare as ak
zz1000 = ak.index_stock_cons('000852')
codes = zz1000['品种代码'].tolist()
print(f"CSI 1000: {len(codes)} stocks")

conn = sqlite3.connect(str(DB))
existing = set(r[0] for r in conn.execute("SELECT DISTINCT code FROM monthly_klines").fetchall())
to_fetch = [c for c in codes if c not in existing]
print(f"New to fetch: {len(to_fetch)} (existing: {len(existing)})")

def fetch_baidu(code):
    """Fetch monthly klines from Baidu API"""
    params = {
        'all': '1', 'isIndex': 'false', 'isBk': 'false', 'isBlock': 'false',
        'isFutures': 'false', 'isStock': 'true', 'newFormat': '1',
        'group': 'quotation_kline_ab', 'finClientType': 'pc',
        'code': code, 'ktype': '3',
        'start_time': '2010-01-01 00:00:00'
    }
    try:
        r = requests.get(BAIDU_URL, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            market_data = data.get('Result', {}).get('newMarketData', {}).get('marketData', '')
            if market_data:
                return market_data.split(';')
    except:
        pass
    return None

inserted = 0
for i, code in enumerate(to_fetch):
    klines = fetch_baidu(code)
    if klines and len(klines) >= 12:
        for line in klines:
            parts = line.split(',')
            if len(parts) >= 11:
                try:
                    ts = int(parts[0]); date_str = parts[1]
                    o = float(parts[2]); c = float(parts[3]); v = float(parts[4])
                    h = float(parts[5]); l = float(parts[6]); amt = float(parts[7])
                    chg = float(parts[9]) if len(parts) > 9 and parts[9] else 0
                    tr = float(parts[10]) if len(parts) > 10 and parts[10] else 0
                    conn.execute("""INSERT OR IGNORE INTO monthly_klines
                        (code, date, open, close, high, low, volume, amount, change_percent, turnover_rate, source)
                        VALUES (?,?,?,?,?,?,?,?,?,?,'baidu')""",
                        (code, date_str, o, c, h, l, v, amt, chg, tr))
                except (ValueError, IndexError):
                    pass
        conn.commit()
        inserted += 1
    if i % 50 == 0:
        print(f"  {i}/{len(to_fetch)} ({inserted} inserted)")
    time.sleep(0.15)

conn.commit()
print(f"Fetched: {inserted} new stocks")
total = conn.execute("SELECT COUNT(DISTINCT code) FROM monthly_klines").fetchone()[0]
print(f"DB total: {total} stocks")

# ======== 2. Filter CSI 1000 stocks with enough data ========
csi1000_codes = [c for c in codes if conn.execute("SELECT COUNT(*) FROM monthly_klines WHERE code=?", (c,)).fetchone()[0] >= 84]
print(f"CSI 1000 with >=84 months: {len(csi1000_codes)}")

if len(csi1000_codes) < 50:
    print("INSUFFICIENT DATA for training")
    conn.close()
    sys.exit(1)

# ======== 3. Build features ========
print("\nBuilding features...")
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({','.join('?'*len(csi1000_codes))}) AND date >= '2010-01' ORDER BY code, date", conn, params=csi1000_codes)
conn.close()

def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        if std > 1e-8: r[i] = (s[i] - m) / std
    return r

all_seq, all_y, all_dates = [], [], []
for code in csi1000_codes:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 24: continue
    n = len(g); dates = g['date'].tolist()
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
    dl=np.diff(closes,prepend=closes[0]); gs=np.where(dl>0,dl,0); ls=np.where(dl<0,-dl,0)
    ag=pd.Series(gs).ewm(alpha=1/14).mean().values; al=pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8]=np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)),50)
    ma20=pd.Series(closes).rolling(20).mean().values; m60=pd.Series(closes).rolling(60).mean().values
    feats[:,18]=np.nan_to_num((closes-ma20)/np.maximum(closes,0.01),0)
    feats[:,19]=np.nan_to_num((closes-m60)/np.maximum(closes,0.01),0)
    feats[:,20]=hash(code)%31/31.0
    feats=np.nan_to_num(feats,0.0)

    for i in range(LOOKBACK-1, n-12):
        if closes[i]<=0.01: continue
        y3=np.clip((closes[i+3]-closes[i])/max(closes[i],0.01),-2,2) if i+3<n else 0
        y6=np.clip((closes[i+6]-closes[i])/max(closes[i],0.01),-2,2) if i+6<n else 0
        all_seq.append(feats[i-LOOKBACK+1:i+1]); all_y.append([y3,y6]); all_dates.append(dates[i])

X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
dates_arr = np.array(all_dates)
train_m = (dates_arr>='2015-01') & (dates_arr<='2021-12')
val_m = (dates_arr>='2022-01') & (dates_arr<='2023-12')
test_m = (dates_arr>='2024-01') & (dates_arr<='2026-05')
Xtr, ytr = X[train_m], y[train_m]; Xva, yva = X[val_m], y[val_m]; Xte, yte = X[test_m], y[test_m]
Xtr, ytr = Xtr[~np.isnan(ytr).any(axis=1)], ytr[~np.isnan(ytr).any(axis=1)]
Xva, yva = Xva[~np.isnan(yva).any(axis=1)], yva[~np.isnan(yva).any(axis=1)]
Xte, yte = Xte[~np.isnan(yte).any(axis=1)], yte[~np.isnan(yte).any(axis=1)]
print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

# ======== 4. Train + Test ========
torch.manual_seed(456); np.random.seed(456)
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
val_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
val_ld = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)
model = create_model('LSTM-7', 21).to(DEVICE)
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
ic3_t = spearmanr(tp[:,0], tt[:,0])[0]; ic6_t = spearmanr(tp[:,1], tt[:,1])[0]
n=len(tp); cut=int(n*0.3); idx=np.argsort(tp[:,0])
ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0

print(f"\n{'='*60}")
print(f"SPRINT 4: CSI 1000 RESULTS")
print(f"{'='*60}")
print(f"Stocks: {len(csi1000_codes)} | Val IC3={best_ic3:.4f} | Test IC3={ic3_t:.4f} | IC6={ic6_t:.4f} | SR3={sr3:.3f}")
print(f"\nFull comparison:")
print(f"  HS300 monthly:   Val=0.180 Test=0.019")
print(f"  CSI500 monthly:  Val=0.172 Test=-0.080")
print(f"  CSI1000 monthly: Val={best_ic3:.4f} Test={ic3_t:.4f}")
if ic3_t > 0.05: print("PASS > 0.05 → universe extension works!")
elif ic3_t >= 0: print("MARGINAL → edge, pivot decision")
else: print("FAIL < 0 → small-cap monthly LSTM definitive ceiling")
