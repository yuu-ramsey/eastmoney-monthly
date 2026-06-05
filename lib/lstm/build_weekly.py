"""Build 21-dim weekly feature dataset. LOOKBACK=200 weeks (~4 years)."""
import sqlite3, numpy as np, pandas as pd
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
LOOKBACK = 200  # weeks

print("Loading weekly data...")
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
df = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume FROM weekly_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01'
    ORDER BY code, date
""", conn, params=stocks)
conn.close()
print(f"{len(df)} rows, {df['code'].nunique()} stocks, {df['date'].min()}~{df['date'].max()}")

def rolling_zscore(s, w=60):
    r = np.zeros(len(s))
    for i in range(w, len(s)):
        x = s[max(0,i-w):i].astype(float); m, std = x.mean(), x.std()
        if std > 1e-8: r[i] = (s[i] - m) / std
    return r

all_seq, all_y, all_dates = [], [], []

for code in stocks:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 52: continue  # need 4yr lookback + 1yr forward
    n = len(g)
    dates = g['date'].tolist()
    closes = g['close'].values.astype(float)
    opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float)
    lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)

    feats = np.zeros((n, 21), dtype=np.float32)
    feats[:,0] = rolling_zscore(closes); feats[:,1] = rolling_zscore(opens)
    feats[:,2] = rolling_zscore(highs); feats[:,3] = rolling_zscore(lows)
    feats[:,4] = rolling_zscore(volumes)

    # MACD
    e12 = pd.Series(closes).ewm(span=12).mean().values
    e26 = pd.Series(closes).ewm(span=26).mean().values
    dif = np.nan_to_num(e12 - e26, 0)
    dea = pd.Series(dif).ewm(span=9).mean().values
    feats[:,5] = dif; feats[:,6] = dea; feats[:,7] = (dif - dea) * 2

    # RSI
    dl = np.diff(closes, prepend=closes[0])
    gs = np.where(dl>0,dl,0); ls = np.where(dl<0,-dl,0)
    ag = pd.Series(gs).ewm(alpha=1/14).mean().values
    al = pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8] = np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)), 50)

    # KDJ
    k, d = np.full(n,50.0), np.full(n,50.0)
    for i in range(8, n):
        hh, ll = highs[i-8:i+1].max(), lows[i-8:i+1].min()
        rsv = (closes[i]-ll)/max(hh-ll,0.01)*100
        k[i] = k[i-1]*2/3+rsv*1/3; d[i] = d[i-1]*2/3+k[i]*1/3
    feats[:,9] = k; feats[:,10] = 3*k-2*d

    # Bollinger position
    ma20 = pd.Series(closes).rolling(20).mean().values
    s20 = pd.Series(closes).rolling(20).std().values
    feats[:,11] = np.nan_to_num((closes - (ma20-2*s20)) / np.maximum(4*s20, 0.01), 0.5)

    # ATR ratio
    tr = np.maximum(highs-lows, np.abs(highs-np.roll(closes,1)), np.abs(lows-np.roll(closes,1)))
    atr14 = pd.Series(tr).rolling(14).mean().values
    feats[:,12] = np.nan_to_num(atr14 / closes, 0)

    # OBV ratio
    obv = np.zeros(n); obv[0] = volumes[0]
    for i in range(1, n):
        if closes[i] > closes[i-1]: obv[i] = obv[i-1] + volumes[i]
        elif closes[i] < closes[i-1]: obv[i] = obv[i-1] - volumes[i]
        else: obv[i] = obv[i-1]
    feats[:,13] = obv / np.maximum(pd.Series(volumes).cumsum().values, 1)

    tp = (highs+lows+closes)/3
    ma_tp = pd.Series(tp).rolling(20).mean().values
    md = pd.Series(np.abs(tp - ma_tp)).rolling(20).mean().values
    feats[:,14] = np.nan_to_num((tp - ma_tp) / np.maximum(md*0.015, 0.001), 0)  # CCI
    for i in range(13, n):
        feats[i,15] = (highs[i-13:i+1].max() - closes[i]) / max(highs[i-13:i+1].max()-lows[i-13:i+1].min(), 0.01) * -100
    feats[:,16] = np.nan_to_num(feats[:,15], 50)  # MFI placeholder
    for i in range(8, n):
        feats[i,17] = (closes[i] - lows[i-8:i+1].min()) / max(highs[i-8:i+1].max()-lows[i-8:i+1].min(), 0.01) * 100
    feats[:,18] = np.nan_to_num((closes - ma20) / np.maximum(closes, 0.01), 0)
    m60 = pd.Series(closes).rolling(60).mean().values
    feats[:,19] = np.nan_to_num((closes - m60) / np.maximum(closes, 0.01), 0)
    feats[:,20] = hash(code) % 31 / 31.0

    feats = np.nan_to_num(feats, 0.0)

    # Generate sequences (weekly: target in weeks, not months)
    for i in range(LOOKBACK-1, n-52):
        if closes[i] <= 0.01: continue
        seq = feats[i-LOOKBACK+1:i+1]
        # 13 weeks ≈ 3 months forward, 26 weeks ≈ 6 months
        y3w = np.clip((closes[i+13] - closes[i]) / max(closes[i],0.01), -2, 2) if i+13 < n else 0
        y6w = np.clip((closes[i+26] - closes[i]) / max(closes[i],0.01), -2, 2) if i+26 < n else 0
        all_seq.append(seq); all_y.append([y3w, y6w]); all_dates.append(dates[i])

    if len(stocks) > 0 and len(all_seq) % 2000 < len(all_seq):
        pass

print(f"Sequences: {len(all_seq)}")
X = np.array(all_seq, dtype=np.float32)
y = np.array(all_y, dtype=np.float32)
dates_arr = np.array(all_dates)

train_m = (dates_arr >= '2015-01-01') & (dates_arr <= '2021-12-31')
val_m = (dates_arr >= '2022-01-01') & (dates_arr <= '2023-12-31')
test_m = (dates_arr >= '2024-01-01') & (dates_arr <= '2026-05-31')

Xtr, ytr = X[train_m], y[train_m]
Xva, yva = X[val_m], y[val_m]
Xte, yte = X[test_m], y[test_m]

def clean(Xx, yy):
    m = ~np.isnan(yy).any(axis=1)
    return Xx[m], yy[m]
Xtr, ytr = clean(Xtr, ytr); Xva, yva = clean(Xva, yva); Xte, yte = clean(Xte, yte)

print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape}")
for name, (Xx, yy) in [('train',(Xtr,ytr)),('val',(Xva,yva)),('test',(Xte,yte))]:
    print(f"  {name}: NaN_X={np.any(np.isnan(Xx))} NaN_Y={np.any(np.isnan(yy))} y=[{yy.min():.2f},{yy.max():.2f}]")

np.savez_compressed(OUT / 'weekly_train.npz', X=Xtr, y=ytr)
np.savez_compressed(OUT / 'weekly_val.npz', X=Xva, y=yva)
np.savez_compressed(OUT / 'weekly_test.npz', X=Xte, y=yte)
print("Saved weekly_*.npz")
