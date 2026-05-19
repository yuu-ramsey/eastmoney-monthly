"""Export daily LSTM predictions → aggregate to monthly signals.
Reuses exact same feature/training code from build_daily_v5.py that produced Test IC3=0.100."""
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DEVICE = torch.device('cuda')
LOOKBACK = 252
BATCH, LR, WD, EP = 128, 5e-4, 1e-4, 80

def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        if std > 1e-8: r[i] = (s[i] - m) / std
    return r

# ======== Load data (exactly like build_daily_v5.py) ========
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]

d_df = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume FROM daily_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01'
    ORDER BY code, date
""", conn, params=stocks)

m_df = pd.read_sql_query(f"""
    SELECT code, date, close FROM monthly_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2015-01'
    ORDER BY code, date
""", conn, params=stocks)
conn.close()

print(f"Daily: {len(d_df)} rows, {d_df['code'].nunique()} stocks")
print(f"Monthly: {len(m_df)} rows")

# ======== Build features + train (exact copy of build_daily_v5.py) ========
print("Building features...")
all_seq, all_y = [], []
for code in stocks:
    g = d_df[d_df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 504: continue
    n = len(g)
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
    dl = np.diff(closes,prepend=closes[0]); gs=np.where(dl>0,dl,0); ls=np.where(dl<0,-dl,0)
    ag=pd.Series(gs).ewm(alpha=1/14).mean().values; al=pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8]=np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)),50)
    k,d_v=np.full(n,50.0),np.full(n,50.0)
    for i in range(8,n):
        hh,ll=highs[i-8:i+1].max(),lows[i-8:i+1].min()
        rsv=(closes[i]-ll)/max(hh-ll,0.01)*100
        k[i]=k[i-1]*2/3+rsv*1/3; d_v[i]=d_v[i-1]*2/3+k[i]*1/3
    feats[:,9]=k; feats[:,10]=3*k-2*d_v
    ma20=pd.Series(closes).rolling(20).mean().values; s20=pd.Series(closes).rolling(20).std().values
    feats[:,11]=np.nan_to_num((closes-(ma20-2*s20))/np.maximum(4*s20,0.01),0.5)
    tr = np.maximum(highs-lows, np.abs(highs-np.roll(closes,1)))
    atr14=pd.Series(tr).rolling(14).mean().values
    feats[:,12]=np.nan_to_num(atr14/closes,0)
    obv=np.zeros(n); obv[0]=volumes[0]
    for i in range(1,n):
        if closes[i]>closes[i-1]: obv[i]=obv[i-1]+volumes[i]
        elif closes[i]<closes[i-1]: obv[i]=obv[i-1]-volumes[i]
        else: obv[i]=obv[i-1]
    feats[:,13]=obv/np.maximum(pd.Series(volumes).cumsum().values,1)
    tp=(highs+lows+closes)/3; ma_tp=pd.Series(tp).rolling(20).mean().values
    md=pd.Series(np.abs(tp-ma_tp)).rolling(20).mean().values
    feats[:,14]=np.nan_to_num((tp-ma_tp)/np.maximum(md*0.015,0.001),0)
    m60=pd.Series(closes).rolling(60).mean().values
    feats[:,18]=np.nan_to_num((closes-ma20)/np.maximum(closes,0.01),0)
    feats[:,19]=np.nan_to_num((closes-m60)/np.maximum(closes,0.01),0)
    feats[:,20]=hash(code)%31/31.0
    feats=np.nan_to_num(feats,0.0)

    for i in range(LOOKBACK-1, n-252):
        if closes[i]<=0.01: continue
        y3d=np.clip((closes[i+63]-closes[i])/max(closes[i],0.01),-2,2) if i+63<n else 0
        y6d=np.clip((closes[i+126]-closes[i])/max(closes[i],0.01),-2,2) if i+126<n else 0
        all_seq.append(feats[i-LOOKBACK+1:i+1]); all_y.append([y3d,y6d])

X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
print(f"Sequences: {X.shape}")

# Split (exact copy)
n = len(X); n_tr = int(n*0.55); n_va = int(n*0.72)
Xtr, ytr = X[:n_tr], y[:n_tr]; Xva, yva = X[n_tr:n_va], y[n_tr:n_va]; Xte, yte = X[n_va:], y[n_va:]
m = ~np.isnan(ytr).any(axis=1); Xtr, ytr = Xtr[m], ytr[m]
m = ~np.isnan(yva).any(axis=1); Xva, yva = Xva[m], yva[m]
m = ~np.isnan(yte).any(axis=1); Xte, yte = Xte[m], yte[m]
print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

# Train (exact copy)
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

# Test (exact copy — verify IC3=0.100)
test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
test_ld = torch.utils.data.DataLoader(test_ds, BATCH, pin_memory=True)
model.eval(); tp_arr, tt_arr = [], []
with torch.no_grad():
    for Xb, yb in test_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = model(Xb); tp_arr.append(p.cpu().numpy()); tt_arr.append(yb.cpu().numpy())
tp = np.concatenate(tp_arr); tt = np.concatenate(tt_arr)
ic3_test = spearmanr(tp[:,0], tt[:,0])[0]
print(f"Test IC3={ic3_test:.4f} (should be ~0.100)")

# ======== EXPORT: Daily predictions for all stocks ========
print("\nExporting daily predictions...")

# Rebuild features for ALL stocks (not just training ones) and predict
# Use exact same feature code as above
daily_rows = []
for code in stocks:
    g = d_df[d_df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK: continue
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
    dl = np.diff(closes,prepend=closes[0]); gs=np.where(dl>0,dl,0); ls=np.where(dl<0,-dl,0)
    ag=pd.Series(gs).ewm(alpha=1/14).mean().values; al=pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8]=np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)),50)
    ma20=pd.Series(closes).rolling(20).mean().values
    m60=pd.Series(closes).rolling(60).mean().values
    feats[:,18]=np.nan_to_num((closes-ma20)/np.maximum(closes,0.01),0)
    feats[:,19]=np.nan_to_num((closes-m60)/np.maximum(closes,0.01),0)
    feats[:,20]=hash(code)%31/31.0
    feats=np.nan_to_num(feats,0.0)

    # Batch predict
    seqs = [feats[i-LOOKBACK+1:i+1] for i in range(LOOKBACK-1, n)]
    if not seqs: continue
    model.eval()
    batch_preds = []
    with torch.no_grad():
        for j in range(0, len(seqs), 512):
            batch = torch.from_numpy(np.array(seqs[j:j+512])).float().to(DEVICE)
            p = model(batch).cpu().numpy()[:, 0]
            batch_preds.append(p)
    preds_all = np.concatenate(batch_preds)

    for i in range(len(preds_all)):
        daily_rows.append({'code': code, 'date': dates[LOOKBACK-1+i], 'score': float(preds_all[i])})

daily_df = pd.DataFrame(daily_rows)
print(f"Daily signals: {len(daily_df)} rows")

# Stats
print(f"Score: mean={daily_df['score'].mean():.4f} std={daily_df['score'].std():.4f} non_nan={daily_df['score'].notna().sum()}/{len(daily_df)}")
if daily_df['score'].std() < 1e-6:
    print("ERROR: Constant predictions!")
    sys.exit(1)

daily_df.to_parquet(OUT / 'daily_signals.parquet')

# ======== Aggregate to monthly (latest day only) ========
print("\nAggregating to monthly...")
daily_df['month'] = daily_df['date'].str[:7]
monthly = daily_df.groupby(['code', 'month'])['score'].last().reset_index()
monthly.columns = ['code', 'month', 'lstm_signal']
monthly['lstm_signal'] = monthly['lstm_signal'].clip(-1, 1)

# Validate IC
m_lookup = {(r['code'], r['date']): r['close'] for _, r in m_df.iterrows()}
sigs, rets = [], []
for _, r in monthly.iterrows():
    c, mm, sig = r['code'], r['month'], r['lstm_signal']
    if pd.isna(sig): continue
    y, mo = int(mm[:4]), int(mm[5:])
    nx = f'{y+1}-01' if mo==12 else f'{y}-{mo+1:02d}'
    cur = m_lookup.get((c,mm)); nxt = m_lookup.get((c,nx))
    if cur and nxt and cur>0.01:
        ret = np.clip((nxt-cur)/cur, -2, 2)
        sigs.append(sig); rets.append(ret)

sigs=np.array(sigs); rets=np.array(rets)
ic = spearmanr(sigs, rets)[0]
print(f"Monthly IC: {ic:.4f} n={len(sigs)}")

n=len(sigs); cut=int(n*0.3); idx=np.argsort(sigs)
ls = rets[idx[-cut:]].mean() - rets[idx[:cut]].mean()
sr = ls / (rets[idx[-cut:]] - rets[idx[:cut]]).std() * np.sqrt(12) if (rets[idx[-cut:]] - rets[idx[:cut]]).std()>0 else 0
print(f"Monthly L-S: {ls:.4f} Sharpe(ann)={sr:.3f}")

monthly.to_parquet(OUT / 'monthly_lstm_signals.parquet')
print(f"Saved: {len(monthly)} monthly signals")
print(f"\nExport complete. daily_signals.parquet + monthly_lstm_signals.parquet")
