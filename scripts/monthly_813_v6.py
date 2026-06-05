"""Monthly LSTM with 813 stocks (all available in DB). Strict v6 pipeline."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

DEV = torch.device('cuda')
B, LR, WD, EPOCHS = 128, 5e-4, 1e-4, 30

conn = sqlite3.connect('.eastmoney-ai/db/klines-v2.sqlite')
codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
print(f"Training on {len(codes)} stocks (all DB with >=84 months)")
params_str = ','.join('?' * len(codes))
df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM monthly_klines WHERE code IN ({params_str}) AND date >= '2010-01' ORDER BY code, date", conn, params=codes)
conn.close()
print(f"Stocks: {len(codes)}, Rows: {len(df)}")

all_seq, all_y, all_dates = [], [], []
for code in codes:
    g = df[df['code']==code].sort_values('date').reset_index(drop=True)
    if len(g) < 84: continue
    n = len(g); dates = g['date'].tolist()
    c = g['close'].values.astype(float); o = g['open'].values.astype(float)
    h = g['high'].values.astype(float); l = g['low'].values.astype(float)
    v = g['volume'].values.astype(float)
    feats = np.zeros((n, 21), dtype=np.float32)
    for j, arr in enumerate([c,o,h,l,v]):
        s = pd.Series(arr); m = s.rolling(60,min_periods=60).mean(); std = s.rolling(60,min_periods=60).std()
        feats[:,j] = ((arr-m)/std.replace(0,1)).fillna(0).values
    e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
    dif=np.nan_to_num(e12-e26,0); dea=pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
    delta=np.diff(c,prepend=c[0]); gs=np.where(delta>0,delta,0); ls=np.where(delta<0,-delta,0)
    feats[:,8]=np.nan_to_num(100-100/(1+pd.Series(gs).ewm(alpha=1/14).mean().values/np.maximum(pd.Series(ls).ewm(alpha=1/14).mean().values,1e-8)),50)
    ma20=pd.Series(c).rolling(20).mean().values; m60=pd.Series(c).rolling(60).mean().values
    feats[:,18]=np.nan_to_num((c-ma20)/np.maximum(c,0.01),0)
    feats[:,19]=np.nan_to_num((c-m60)/np.maximum(c,0.01),0)
    feats[:,20]=hash(code)%31/31.0
    feats=np.nan_to_num(feats,0.0)
    for i in range(59, n-12):
        if c[i]<=0.01: continue
        seq=feats[i-59:i+1]
        y3=np.clip((c[i+3]-c[i])/max(c[i],0.01),-2,2) if i+3<n else 0
        y6=np.clip((c[i+6]-c[i])/max(c[i],0.01),-2,2) if i+6<n else 0
        all_seq.append(seq); all_y.append([y3,y6]); all_dates.append(dates[i])
    if len(all_seq) % 20000 < len(codes):
        pass

X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
d_arr = np.array(all_dates)
train_m = np.array([d>='2015-01' and d<='2021-12' for d in d_arr])
test_m = np.array([d>='2024-01' for d in d_arr])
Xtr, ytr = X[train_m], y[train_m]; Xtr, ytr = Xtr[~np.isnan(ytr).any(axis=1)], ytr[~np.isnan(ytr).any(axis=1)]
Xte, yte = X[test_m], y[test_m]; Xte, yte = Xte[~np.isnan(yte).any(axis=1)], yte[~np.isnan(yte).any(axis=1)]
print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

if len(Xtr) > 300000:
    idx = np.random.choice(len(Xtr), 300000, replace=False)
    Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"Subsampled to 300K")

torch.manual_seed(456); np.random.seed(456)
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
train_ld = torch.utils.data.DataLoader(train_ds, B, shuffle=True, pin_memory=True)
model = create_model('LSTM-7', 21).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
t0 = time.time()
for ep in range(1, EPOCHS+1):
    for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
    model.train()
    for Xb, yb in train_ld:
        Xb, yb = Xb.to(DEV), yb.to(DEV); opt.zero_grad()
        p = model(Xb)
        loss = 0.5*nn.MSELoss()(p[:,0], yb[:,0]) + 0.5*nn.MSELoss()(p[:,1], yb[:,1])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if ep%10==0: print(f"  ep{ep} loss={loss.item():.4f}")

test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
test_ld = torch.utils.data.DataLoader(test_ds, B, pin_memory=True)
model.eval(); tp, tt = [], []
with torch.no_grad():
    for Xb, yb in test_ld:
        Xb, yb = Xb.to(DEV), yb.to(DEV); p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
tp = np.concatenate(tp); tt = np.concatenate(tt)
ic3 = spearmanr(tp[:,0], tt[:,0])[0]; ic6 = spearmanr(tp[:,1], tt[:,1])[0]
n = len(tp); cut = int(n*0.3); idx = np.argsort(tp[:,0])
ls = tt[idx[-cut:],0] - tt[idx[:cut],0]; sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0
print(f"\n813-stock Monthly Test IC3={ic3:+.4f}, IC6={ic6:+.4f}, SR3={sr3:+.3f} ({time.time()-t0:.0f}s)")
print(f"vs HS300-only: IC3=-0.027 (18K seqs)")
print(f"vs Daily:      IC3=+0.141 (381K seqs)")
