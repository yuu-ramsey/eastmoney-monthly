"""Phase 17 v6: Daily pipeline with STRICT TIME-BASED SPLIT (fixes v5 leak)
v5 bug: proportional index split (n_tr = n * 0.55) leaked cross-time info
v6 fix: train=2015-2021, val=2022-2023, test=2024+ (same as monthly/weekly)

Outputs:
  - .eastmoney-ai/lstm/models_v2/daily_lstm_v6.pt  (trained model)
  - .eastmoney-ai/lstm/daily_signals_v6.parquet    (clean out-of-sample signals)
"""
import torch, torch.nn as nn, numpy as np, sqlite3, pandas as pd, time
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
MODEL_OUT = OUT / 'models_v2' / 'daily_lstm_v6.pt'
DEVICE = torch.device('cuda')
LOOKBACK = 252
BATCH, LR, WD, EP = 128, 5e-4, 1e-4, 80

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Output model: {MODEL_OUT}")
print(f"Output signals: {OUT / 'daily_signals_v6.parquet'}")

# ======== 1. Load data ========
print("\n1/4 Loading daily data...")
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute(
    "SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
df = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume FROM daily_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01'
    ORDER BY code, date
""", conn, params=stocks)
conn.close()
print(f"  {len(df)} rows, {df['code'].nunique()} stocks")

# ======== 2. Build features with dates ========
print("\n2/4 Building features (21-dim, with dates)...")

def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        if std > 1e-8: r[i] = (s[i] - m) / std
    return r

all_seq, all_y, all_dates = [], [], []
stock_count = 0

for code in stocks:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 504:
        continue
    n = len(g)
    closes = g['close'].values.astype(float); opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float); lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)
    dates = g['date'].tolist()

    feats = np.zeros((n, 21), dtype=np.float32)
    feats[:,0] = rolling_zscore(closes, 60); feats[:,1] = rolling_zscore(opens, 60)
    feats[:,2] = rolling_zscore(highs, 60); feats[:,3] = rolling_zscore(lows, 60)
    feats[:,4] = rolling_zscore(volumes, 60)
    e12 = pd.Series(closes).ewm(span=12).mean().values
    e26 = pd.Series(closes).ewm(span=26).mean().values
    dif = np.nan_to_num(e12-e26,0); dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
    delta = np.diff(closes, prepend=closes[0])
    gs=np.where(delta>0,delta,0); ls=np.where(delta<0,-delta,0)
    ag=pd.Series(gs).ewm(alpha=1/14).mean().values
    al=pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8]=np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)), 50)
    k,d = np.full(n,50.0), np.full(n,50.0)
    for i in range(8, n):
        hh,ll=highs[i-8:i+1].max(), lows[i-8:i+1].min()
        rsv=(closes[i]-ll)/max(hh-ll,0.01)*100
        k[i]=k[i-1]*2/3+rsv*1/3; d[i]=d[i-1]*2/3+k[i]*1/3
    feats[:,9]=k; feats[:,10]=3*k-2*d
    ma20=pd.Series(closes).rolling(20).mean().values
    s20=pd.Series(closes).rolling(20).std().values
    feats[:,11]=np.nan_to_num((closes-(ma20-2*s20))/np.maximum(4*s20,0.01), 0.5)
    tr=np.maximum(highs-lows, np.abs(highs-np.roll(closes,1)))
    atr14=pd.Series(tr).rolling(14).mean().values
    feats[:,12]=np.nan_to_num(atr14/closes, 0)
    obv=np.zeros(n); obv[0]=volumes[0]
    for i in range(1,n):
        if closes[i]>closes[i-1]: obv[i]=obv[i-1]+volumes[i]
        elif closes[i]<closes[i-1]: obv[i]=obv[i-1]-volumes[i]
        else: obv[i]=obv[i-1]
    feats[:,13]=obv/np.maximum(pd.Series(volumes).cumsum().values, 1)
    tp=(highs+lows+closes)/3; ma_tp=pd.Series(tp).rolling(20).mean().values
    md=pd.Series(np.abs(tp-ma_tp)).rolling(20).mean().values
    feats[:,14]=np.nan_to_num((tp-ma_tp)/np.maximum(md*0.015,0.001), 0)
    m60=pd.Series(closes).rolling(60).mean().values
    feats[:,18]=np.nan_to_num((closes-ma20)/np.maximum(closes,0.01), 0)
    feats[:,19]=np.nan_to_num((closes-m60)/np.maximum(closes,0.01), 0)
    feats[:,20]=hash(code)%31/31.0
    feats=np.nan_to_num(feats, 0.0)

    for i in range(LOOKBACK-1, n-252):
        if closes[i] <= 0.01: continue
        seq = feats[i-LOOKBACK+1:i+1]
        y3d = np.clip((closes[i+63]-closes[i])/max(closes[i],0.01), -2, 2) if i+63<n else 0
        y6d = np.clip((closes[i+126]-closes[i])/max(closes[i],0.01), -2, 2) if i+126<n else 0
        all_seq.append(seq); all_y.append([y3d, y6d])
        all_dates.append(dates[i])

    stock_count += 1
    if stock_count % 50 == 0:
        print(f"  {stock_count}/{len(stocks)} stocks, {len(all_seq)} seqs")

X = np.array(all_seq, dtype=np.float32)
y = np.array(all_y, dtype=np.float32)
dates_arr = np.array(all_dates)
print(f"  X: {X.shape}, y: {y.shape}, dates: {dates_arr[0]}~{dates_arr[-1]}")

# ======== 3. STRICT TIME-BASED SPLIT ========
print("\n3/4 Time-based split: train=2015-2021, val=2022-2023, test=2024+")
train_m = (dates_arr >= '2015-01-01') & (dates_arr <= '2021-12-31')
val_m   = (dates_arr >= '2022-01-01') & (dates_arr <= '2023-12-31')
test_m  = (dates_arr >= '2024-01-01')

Xtr, ytr = X[train_m], y[train_m]
Xva, yva = X[val_m], y[val_m]
Xte, yte = X[test_m], y[test_m]

m_tr = ~np.isnan(ytr).any(axis=1); Xtr, ytr = Xtr[m_tr], ytr[m_tr]
m_va = ~np.isnan(yva).any(axis=1); Xva, yva = Xva[m_va], yva[m_va]
m_te = ~np.isnan(yte).any(axis=1); Xte, yte = Xte[m_te], yte[m_te]

print(f"  Train: {Xtr.shape}, Val: {Xva.shape}, Test: {Xte.shape}")
tr_dates = dates_arr[train_m][m_tr]; va_dates = dates_arr[val_m][m_va]; te_dates = dates_arr[test_m][m_te]
print(f"  Train dates: {tr_dates[0]} ~ {tr_dates[-1]}")
print(f"  Val dates:   {va_dates[0]} ~ {va_dates[-1]}")
print(f"  Test dates:  {te_dates[0]} ~ {te_dates[-1]}")

# ======== 4. Train ========
print("\n4/4 Training...")
torch.manual_seed(456); np.random.seed(456)
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
val_ds   = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
test_ds  = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
train_ld = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, pin_memory=True)
val_ld   = torch.utils.data.DataLoader(val_ds, BATCH, pin_memory=True)
test_ld  = torch.utils.data.DataLoader(test_ds, BATCH, pin_memory=True)

model = create_model('LSTM-7', 21).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
best_vl, no_imp, best_ic3 = float('inf'), 0, -1.0
best_state = None
t0 = time.time()

for ep in range(1, EP+1):
    for pg in opt.param_groups:
        pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
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
    vl /= len(val_ds)
    ic3 = spearmanr(np.concatenate(preds)[:,0], np.concatenate(targs)[:,0])[0]
    if ic3 > best_ic3:
        best_ic3 = ic3
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if vl < best_vl: best_vl, no_imp = vl, 0
    else: no_imp += 1
    if no_imp >= 15 and ep > 20: break

print(f"  Train: {time.time()-t0:.0f}s, Val IC3={best_ic3:.4f}, ep={ep}")

# Test
model.load_state_dict(best_state)
model.eval(); tp, tt = [], []
with torch.no_grad():
    for Xb, yb in test_ld:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
tp = np.concatenate(tp); tt = np.concatenate(tt)
ic3_t = spearmanr(tp[:,0], tt[:,0])[0]
ic6_t = spearmanr(tp[:,1], tt[:,1])[0]
n = len(tp); cut = int(n*0.3); idx = np.argsort(tp[:,0])
ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0
hit3 = np.mean(np.sign(tp[:,0]) == np.sign(tt[:,0]))
gap = ic3_t - best_ic3

print(f"\n  DAILY Test (v6, time-split): IC3={ic3_t:.4f} IC6={ic6_t:.4f} SR3={sr3:.3f} Hit={hit3:.2f} gap={gap:+.4f}")
print(f"  v5 (prop-split): IC3=0.1409 → v6 (time-split): IC3={ic3_t:.4f}")

# Save model
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save(best_state, MODEL_OUT)
print(f"\n  Model saved: {MODEL_OUT}")

# ======== 5. Generate clean out-of-sample signals for ALL stocks/dates ========
print("\n5/4 Generating clean daily signals (ALL dates, time-safe model)...")
model.load_state_dict(best_state)
model.eval()

signal_rows = []
batch_size = 512

for code in stocks:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK: continue
    n = len(g); dates = g['date'].tolist()
    closes = g['close'].values.astype(float); opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float); lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)

    feats = np.zeros((n, 21), dtype=np.float32)
    feats[:,0] = rolling_zscore(closes, 60); feats[:,1] = rolling_zscore(opens, 60)
    feats[:,2] = rolling_zscore(highs, 60); feats[:,3] = rolling_zscore(lows, 60)
    feats[:,4] = rolling_zscore(volumes, 60)
    e12 = pd.Series(closes).ewm(span=12).mean().values
    e26 = pd.Series(closes).ewm(span=26).mean().values
    dif = np.nan_to_num(e12-e26,0); dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:,5]=dif; feats[:,6]=dea; feats[:,7]=(dif-dea)*2
    delta = np.diff(closes, prepend=closes[0])
    gs=np.where(delta>0,delta,0); ls=np.where(delta<0,-delta,0)
    ag=pd.Series(gs).ewm(alpha=1/14).mean().values
    al=pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8]=np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)), 50)
    k,d = np.full(n,50.0), np.full(n,50.0)
    for i in range(8, n):
        hh,ll=highs[i-8:i+1].max(), lows[i-8:i+1].min()
        rsv=(closes[i]-ll)/max(hh-ll,0.01)*100
        k[i]=k[i-1]*2/3+rsv*1/3; d[i]=d[i-1]*2/3+k[i]*1/3
    feats[:,9]=k; feats[:,10]=3*k-2*d
    ma20=pd.Series(closes).rolling(20).mean().values
    s20=pd.Series(closes).rolling(20).std().values
    feats[:,11]=np.nan_to_num((closes-(ma20-2*s20))/np.maximum(4*s20,0.01), 0.5)
    tr=np.maximum(highs-lows, np.abs(highs-np.roll(closes,1)))
    atr14=pd.Series(tr).rolling(14).mean().values
    feats[:,12]=np.nan_to_num(atr14/closes, 0)
    m60=pd.Series(closes).rolling(60).mean().values
    feats[:,18]=np.nan_to_num((closes-ma20)/np.maximum(closes,0.01), 0)
    feats[:,19]=np.nan_to_num((closes-m60)/np.maximum(closes,0.01), 0)
    feats[:,20]=hash(code)%31/31.0
    feats=np.nan_to_num(feats, 0.0)

    seqs, seq_dates = [], []
    for i in range(LOOKBACK-1, n):
        seqs.append(feats[i-LOOKBACK+1:i+1])
        seq_dates.append(dates[i])

    if not seqs: continue

    # Batch inference
    all_preds = []
    for j in range(0, len(seqs), batch_size):
        batch = torch.from_numpy(np.array(seqs[j:j+batch_size])).float().to(DEVICE)
        with torch.no_grad():
            p = model(batch)
        all_preds.append(p.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)

    for i in range(len(seq_dates)):
        signal_rows.append({
            'code': code,
            'date': seq_dates[i],
            'score': float(preds[i, 0]),  # y3 prediction
            'in_sample': seq_dates[i] < '2022-01-01',  # True if train period
        })

df_sig = pd.DataFrame(signal_rows)
df_sig.to_parquet(OUT / 'daily_signals_v6.parquet', index=False)
print(f"  Signals saved: {len(df_sig)} rows, {df_sig.code.nunique()} stocks")
print(f"  In-sample (pre-2022): {df_sig.in_sample.sum():,} ({100*df_sig.in_sample.mean():.0f}%)")
print(f"  Out-of-sample (2022+): {(~df_sig.in_sample).sum():,} ({100*(1-df_sig.in_sample.mean()):.0f}%)")

print(f"\n{'='*60}")
print(f"v6 complete. Model: {MODEL_OUT}")
print(f"  Test IC3={ic3_t:.4f} (time-split, clean)")
print(f"  vs v5 IC3=0.1409 (prop-split, contaminated)")
print(f"  Clean signals: {OUT / 'daily_signals_v6.parquet'}")
print(f"{'='*60}")
