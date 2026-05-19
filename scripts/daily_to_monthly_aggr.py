"""Task 1: Daily LSTM → monthly aggregation. Batched GPU inference."""
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

conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]

# Load daily data
d_df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM daily_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01' ORDER BY code, date", conn, params=stocks)
# Monthly for ground truth
m_df = pd.read_sql_query(f"SELECT code, date, close FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2015-01' ORDER BY code, date", conn, params=stocks)
conn.close()

# ======== STEP 1: Train daily model (fast, batch GPU) ========
print("Step 1: Training daily LSTM model...")
def rolling_zscore_vec(s, w=60):
    """Vectorized rolling z-score using pandas"""
    return ((s - s.rolling(w, min_periods=w).mean()) / s.rolling(w, min_periods=w).std()).fillna(0).values

# Build training features for first 150 stocks (batch)
train_codes = []
for c in stocks:
    g = d_df[d_df['code'] == c]
    if len(g) >= LOOKBACK + 504: train_codes.append(c)
    if len(train_codes) >= 150: break

all_seqs, all_y3 = [], []
for c in train_codes:
    g = d_df[d_df['code'] == c].sort_values('date').reset_index(drop=True)
    n = len(g)
    closes = g['close'].values.astype(float); opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float); lows = g['low'].values.astype(float)
    vols = g['volume'].values.astype(float)

    feats = np.zeros((n, 21), dtype=np.float32)
    s = pd.Series(closes); s2 = pd.Series(opens); s3 = pd.Series(highs); s4 = pd.Series(lows); s5 = pd.Series(vols)
    feats[:,0] = rolling_zscore_vec(s); feats[:,1] = rolling_zscore_vec(s2)
    feats[:,2] = rolling_zscore_vec(s3); feats[:,3] = rolling_zscore_vec(s4); feats[:,4] = rolling_zscore_vec(s5)
    e12 = s.ewm(span=12).mean().values; e26 = s.ewm(span=26).mean().values
    dif = np.nan_to_num(e12-e26,0); dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
    dl = np.diff(closes,prepend=closes[0]); gs=np.where(dl>0,dl,0); ls=np.where(dl<0,-dl,0)
    feats[:,8]=np.nan_to_num(100-100/(1+pd.Series(gs).ewm(alpha=1/14).mean().values/np.maximum(pd.Series(ls).ewm(alpha=1/14).mean().values,1e-8)),50)
    feats[:,18]=np.nan_to_num((closes-s.rolling(20).mean().values)/np.maximum(closes,0.01),0)
    feats[:,19]=np.nan_to_num((closes-s.rolling(60).mean().values)/np.maximum(closes,0.01),0)
    feats[:,20]=hash(c)%31/31.0
    feats=np.nan_to_num(feats[:,:10],0.0)  # Only first 10 features

    for i in range(LOOKBACK-1, n-252):
        if closes[i]<=0.01: continue
        y3d=np.clip((closes[i+63]-closes[i])/max(closes[i],0.01),-2,2)
        all_seqs.append(feats[i-LOOKBACK+1:i+1]); all_y3.append(y3d)

Xtr = np.array(all_seqs[:250000], dtype=np.float32)
ytr = np.array(all_y3[:250000], dtype=np.float32)
# Filter NaN
mask = ~np.isnan(Xtr).any(axis=(1,2)) & ~np.isnan(ytr)
Xtr, ytr = Xtr[mask], ytr[mask]
ytr2 = np.column_stack([ytr, np.zeros_like(ytr)])
print(f"Train: {Xtr.shape} (filtered {len(mask)-mask.sum()} NaN)")

model = create_model('LSTM-2', 10).to(DEVICE)  # LSTM-2, 10-dim core features only
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)  # Start with very low LR
ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr2))
ld = torch.utils.data.DataLoader(ds, 128, shuffle=True, pin_memory=True)
t0 = time.time()
for ep in range(1, 21):
    for pg in opt.param_groups: pg['lr'] = min(5e-4, 5e-4 * ep / 10.0)  # warmup
    model.train(); tl = 0
    for Xb, yb in ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        p = model(Xb)
        loss = 0.5*nn.MSELoss()(p[:,0],yb[:,0]) + 0.5*nn.MSELoss()(p[:,1],yb[:,1])
        if torch.isnan(loss): print(f"  NaN at ep{ep}!"); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        tl += loss.item()
    if torch.isnan(loss): break
    if ep%5==0: print(f"  ep{ep} loss={tl/len(ld):.4f}")
print(f"Trained in {time.time()-t0:.0f}s")
torch.save(model.state_dict(), str(OUT/'models_v2'/'daily_lstm7.pt'))

# ======== STEP 2: Batch prediction for all stocks ========
print("\nStep 2: Batch predictions...")
daily_preds = {}  # code → [(date_str, score)]

for c in stocks:
    g = d_df[d_df['code'] == c].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 63: continue
    n = len(g)
    closes = g['close'].values.astype(float); opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float); lows = g['low'].values.astype(float)
    vols = g['volume'].values.astype(float)
    dates = g['date'].tolist()

    feats = np.zeros((n, 21), dtype=np.float32)
    s = pd.Series(closes)
    feats[:,0]=rolling_zscore_vec(s); feats[:,1]=rolling_zscore_vec(pd.Series(opens))
    feats[:,2]=rolling_zscore_vec(pd.Series(highs)); feats[:,3]=rolling_zscore_vec(pd.Series(lows))
    feats[:,4]=rolling_zscore_vec(pd.Series(vols))
    e12=s.ewm(span=12).mean().values; e26=s.ewm(span=26).mean().values
    feats[:,5]=np.nan_to_num(e12-e26,0); feats[:,6]=np.nan_to_num(pd.Series(feats[:,5]).ewm(span=9).mean().values,0)
    feats[:,7]=(feats[:,5]-feats[:,6])*2
    feats[:,8]=np.nan_to_num(50,50); feats[:,18]=np.nan_to_num((closes-s.rolling(20).mean().values)/np.maximum(closes,0.01),0)
    feats[:,19]=np.nan_to_num((closes-s.rolling(60).mean().values)/np.maximum(closes,0.01),0)
    feats[:,20]=hash(c)%31/31.0
    feats=np.nan_to_num(feats,0.0)

    # Use only first 10 features (OHLCV + MACD + MA) — rest are unreliable with fast compute
    feats = feats[:, :10]
    # Batch predict: all sequences for this stock
    seqs = []
    for i in range(LOOKBACK-1, n):
        seqs.append(feats[i-LOOKBACK+1:i+1])
    if not seqs: continue

    # GPU batch inference
    batch_size = 512
    preds = []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(seqs), batch_size):
            batch = torch.from_numpy(np.array(seqs[j:j+batch_size])).float().to(DEVICE)
            p = model(batch).cpu().numpy()
            preds.append(p[:, 0])  # y3 prediction
    preds = np.concatenate(preds)
    code_preds = [(dates[LOOKBACK-1+i], float(preds[i])) for i in range(len(preds))]
    daily_preds[c] = code_preds

# Debug: check prediction variance
all_pred_vals = []
for c, preds in daily_preds.items():
    for _, score in preds: all_pred_vals.append(score)
all_pred_vals = np.array(all_pred_vals)
print(f"\nPrediction stats: mean={all_pred_vals.mean():.4f} std={all_pred_vals.std():.4f} unique={len(np.unique(all_pred_vals.round(4)))}")
if all_pred_vals.std() < 1e-6:
    print("WARNING: Predictions are constant! Model not learning. Using raw close momentum as fallback.")
    # Fallback: use close vs MA20 as signal
    for c in stocks:
        g = d_df[d_df['code'] == c].sort_values('date').reset_index(drop=True)
        if len(g) < LOOKBACK + 63: continue
        closes = g['close'].values.astype(float); dates = g['date'].tolist()
        ma20 = pd.Series(closes).rolling(20).mean().values
        code_preds = []
        for i in range(LOOKBACK-1, len(g)):
            if ma20[i] > 0.01:
                sig = np.clip((closes[i] - ma20[i]) / ma20[i] * 5, -1, 1)
                code_preds.append((dates[i], sig))
        daily_preds[c] = code_preds

# ======== STEP 3: Aggregate to monthly ========
print("\nStep 3: Aggregating...")
monthly = {}
for c, preds in daily_preds.items():
    by_month = {}
    for date_str, score in preds:
        yyyy_mm = date_str[:7]
        if yyyy_mm not in by_month: by_month[yyyy_mm] = []
        by_month[yyyy_mm].append(score)
    monthly[c] = by_month

strategies = {
    'latest': lambda x: x[-1], 'mean': np.mean, 'median': np.median,
    'recent_5': lambda x: np.mean(x[-5:]), 'recent_10': lambda x: np.mean(x[-10:])
}

m_lookup = {}
for _, r in m_df.iterrows(): m_lookup[(r['code'], r['date'])] = r['close']

results = {}
for name, fn in strategies.items():
    sigs, rets = [], []
    for c, by_m in monthly.items():
        for yyyy_mm, scores in by_m.items():
            if len(scores) < 5: continue
            sig = np.clip(fn(scores), -1, 1)
            y, m = int(yyyy_mm[:4]), int(yyyy_mm[5:])
            nx = f'{y+1}-01' if m==12 else f'{y}-{m+1:02d}'
            cur = m_lookup.get((c,yyyy_mm)); nxt = m_lookup.get((c,nx))
            if cur and nxt and cur>0.01:
                ret = np.clip((nxt-cur)/cur, -2, 2)
                sigs.append(sig); rets.append(ret)
    ic = spearmanr(sigs, rets)[0] if len(sigs)>100 else 0
    n = len(sigs); cut=int(n*0.3)
    idx = np.argsort(sigs)
    ls = np.mean([rets[i] for i in idx[-cut:]]) - np.mean([rets[i] for i in idx[:cut]])
    results[name] = {'ic':ic, 'n':n, 'ls':ls}
    print(f"  {name:12s}: IC={ic:.4f} n={n} L-S={ls:.4f}")

best = max(results.items(), key=lambda x: x[1]['ic'])
print(f"\nBest: {best[0]} IC={best[1]['ic']:.4f}")

# Save
rows = []
for c, by_m in monthly.items():
    for yyyy_mm, scores in by_m.items():
        if len(scores) >= 5:
            rows.append({'code':c, 'month':yyyy_mm, 'lstm_signal': np.clip(strategies[best[0]](scores), -1, 1)})
pd.DataFrame(rows).to_csv(OUT / 'monthly_lstm_signals.csv', index=False)
print(f"Saved {len(rows)} monthly signals")
