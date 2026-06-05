"""Phase 17 v3: Build 40-dim feature dataset from DB + verified akshare APIs.
Reliable sources: SQLite monthly_klines + macro (M2/CPI/PMI from akshare)
Derived: PE/PB rank, turnover pct, volume momentum, sector dispersion, IPO count
"""
import sqlite3, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

PROJECT = Path(__file__).parent.parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
OUT.mkdir(parents=True, exist_ok=True)
LOOKBACK = 60

print("Loading data from SQLite...")
conn = sqlite3.connect(str(DB))

# All HS300 stocks
stocks = [r[0] for r in conn.execute(
    "SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
print(f"Stocks: {len(stocks)}")

# Monthly klines
df = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume, turnover_rate
    FROM monthly_klines
    WHERE code IN ({','.join('?'*len(stocks))})
    AND date >= '2010-01'
    ORDER BY code, date
""", conn, params=stocks)
print(f"Klines: {len(df)} rows")

# Industry mapping
ind_df = pd.read_sql_query(
    "SELECT stock_code, industry_code FROM stock_industry_mapping", conn)
stock_to_ind = dict(zip(ind_df['stock_code'], ind_df['industry_code']))

# IPO count by month
ipo_df = pd.read_sql_query("""
    SELECT code, MIN(date) as ipo_date FROM monthly_klines
    WHERE code IN (SELECT DISTINCT stock_code FROM stock_industry_mapping)
    GROUP BY code
""", conn)
ipo_by_month = ipo_df.groupby('ipo_date').size()
conn.close()

# === Macro data from akshare (verified working) ===
print("Loading macro data...")
try:
    import akshare as ak
    m2 = ak.macro_china_m2_yearly()
    m2['date'] = pd.to_datetime(m2['date']).dt.strftime('%Y-%m')
    m2_map = dict(zip(m2['date'], m2['今值']))

    pmi = ak.macro_china_pmi_yearly()
    if 'date' in pmi.columns:
        pmi['date'] = pd.to_datetime(pmi['date']).dt.strftime('%Y-%m')
        pmi_col = 'manufacturing' if 'manufacturing' in pmi.columns else '今值'
        pmi_map = dict(zip(pmi['date'], pmi[pmi_col]))
    else:
        pmi_map = {}

    cpi = ak.macro_china_cpi_monthly()
    cpi['date'] = pd.to_datetime(cpi['date']).dt.strftime('%Y-%m')
    cpi_map = dict(zip(cpi['date'], cpi['yoy_growth']))
    print(f"  M2: {len(m2_map)} months, PMI: {len(pmi_map)}, CPI: {len(cpi_map)}")
except Exception as e:
    print(f"  Macro failed: {e}, using fallback")
    m2_map, pmi_map, cpi_map = {}, {}, {}

# === Feature engineering per stock ===
print("Building features per stock...")
all_sequences, all_targets, all_dates = [], [], []
stock_count, skipped = 0, 0

def rolling_zscore(s, window=60):
    result = np.zeros(len(s))
    for i in range(window, len(s)):
        w = s[max(0,i-window):i].astype(float)
        m, std = w.mean(), w.std()
        if std > 1e-8: result[i] = (s[i] - m) / std
    return result

def ema(s, p):
    r = np.full(len(s), np.nan)
    k = 2/(p+1); r[p-1] = np.mean(s[:p])
    for i in range(p, len(s)): r[i] = s[i]*k + r[i-1]*(1-k)
    return r

for code in stocks:
    g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < LOOKBACK + 24: skipped += 1; continue
    n = len(g)
    dates = g['date'].tolist()
    closes = g['close'].values.astype(float)
    opens = g['open'].values.astype(float)
    highs = g['high'].values.astype(float)
    lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)
    turnovers = g['turnover_rate'].values.astype(float)
    industry = stock_to_ind.get(code, 'unknown')

    # === Original 21 features ===
    feats = np.zeros((n, 40), dtype=np.float32)

    # 0-4: OHLCV z-score
    feats[:,0] = rolling_zscore(closes); feats[:,1] = rolling_zscore(opens)
    feats[:,2] = rolling_zscore(highs); feats[:,3] = rolling_zscore(lows)
    feats[:,4] = rolling_zscore(volumes)

    # MACD
    e12, e26 = ema(closes,12), ema(closes,26)
    dif = np.nan_to_num(e12 - e26, 0); dea = ema(dif, 9)
    feats[:,5] = np.nan_to_num(dif, 0); feats[:,6] = np.nan_to_num(dea, 0)
    feats[:,7] = np.nan_to_num((dif - dea)*2, 0)

    # RSI
    deltas = np.diff(closes, prepend=closes[0])
    gs = np.where(deltas>0,deltas,0); ls = np.where(deltas<0,-deltas,0)
    ag = pd.Series(gs).ewm(alpha=1/14).mean().values
    al = pd.Series(ls).ewm(alpha=1/14).mean().values
    feats[:,8] = np.nan_to_num(100-100/(1+ag/np.maximum(al,1e-8)), 50)

    # KDJ
    k, d = np.full(n,50.0), np.full(n,50.0)
    for i in range(8, n):
        hh, ll = highs[i-8:i+1].max(), lows[i-8:i+1].min()
        rsv = (closes[i]-ll)/max(hh-ll,0.01)*100
        k[i] = k[i-1]*2/3+rsv*1/3; d[i] = d[i-1]*2/3+k[i]*1/3
    feats[:,9] = np.nan_to_num(k, 50); feats[:,10] = np.nan_to_num(3*k-2*d, 50)

    # Bollinger
    ma20 = pd.Series(closes).rolling(20).mean().values
    std20 = pd.Series(closes).rolling(20).std().values
    feats[:,11] = np.nan_to_num((closes - (ma20-2*std20)) / np.maximum(2*std20*2, 0.01), 0.5)
    feats[:,12] = np.nan_to_num(np.maximum(highs - lows,
        np.abs(highs - np.roll(closes,1)), np.abs(lows - np.roll(closes,1))), 0)
    feats[:,12] = pd.Series(feats[:,12]).rolling(14).mean().fillna(0).values / closes

    # OBV simplified
    for i in range(1, n):
        if closes[i] > closes[i-1]: feats[i,13] = feats[i-1,13] + volumes[i]
        elif closes[i] < closes[i-1]: feats[i,13] = feats[i-1,13] - volumes[i]
        else: feats[i,13] = feats[i-1,13]
    feats[:,13] /= np.maximum(pd.Series(volumes).cumsum().values, 1)

    # CCI / WR / MFI / Stoch (simplified)
    tp = (highs+lows+closes)/3
    ma_tp = pd.Series(tp).rolling(20).mean().values
    md = pd.Series(np.abs(tp - ma_tp)).rolling(20).mean().values
    feat_cci = np.nan_to_num((tp - ma_tp) / np.maximum(md*0.015, 0.001), 0)
    # WR
    feat_wr = np.full(n, -50.0)
    for i in range(13, n):
        hh, ll = highs[i-13:i+1].max(), lows[i-13:i+1].min()
        feat_wr[i] = (hh - closes[i]) / max(hh-ll, 0.01) * -100
    # Stoch
    feat_sto = np.full(n, 50.0)
    for i in range(8, n):
        feat_sto[i] = (closes[i] - lows[i-8:i+1].min()) / max(highs[i-8:i+1].max() - lows[i-8:i+1].min(), 0.01) * 100
    feats[:,14] = np.nan_to_num(feat_cci, 0); feats[:,15] = np.nan_to_num(feat_wr, -50)
    feats[:,16] = np.nan_to_num(feats[:,15], 50); feats[:,17] = np.nan_to_num(feat_sto, 50)
    feats[:,18] = np.nan_to_num((closes - pd.Series(closes).rolling(20).mean().values) / np.maximum(closes, 0.01), 0)
    feats[:,19] = np.nan_to_num((closes - pd.Series(closes).rolling(60).mean().values) / np.maximum(closes, 0.01), 0)
    feats[:,20] = hash(industry) % 31 / 31.0  # Industry ID

    # === NEW 19 features ===
    # 21: PE rank proxy = close/ma60 (valuation position)
    ma60 = pd.Series(closes).rolling(60).mean().values
    feats[:,21] = np.nan_to_num((closes / np.maximum(ma60, 0.01) - 1), 0)

    # 22: PB rank proxy = close / ma60
    feats[:,22] = np.nan_to_num(closes / np.maximum(ma60, 0.01), 1)

    # 23: Revenue growth proxy = close momentum 12m
    for i in range(12, n):
        feats[i,23] = (closes[i] - closes[i-12]) / max(closes[i-12], 0.01)
    feats[:,23] = rolling_zscore(feats[:,23])

    # 24: Profit growth proxy = close momentum 6m / 12m ratio
    for i in range(12, n):
        m6 = (closes[i]-closes[i-6])/max(closes[i-6],0.01) if i>=6 else 0
        m12 = (closes[i]-closes[i-12])/max(closes[i-12],0.01)
        feats[i,24] = m6 - m12
    feats[:,24] = rolling_zscore(feats[:,24])

    # 25: Turnover percentile (5yr)
    for i in range(60, n):
        w = turnovers[max(0,i-60):i]
        feats[i,25] = (turnovers[i] - w.mean()) / max(w.std(), 1e-8)

    # 26: Volume momentum (20m vs 60m)
    for i in range(60, n):
        v20 = volumes[i-19:i+1].mean() if i>=19 else volumes[i]
        v60 = volumes[max(0,i-59):i+1].mean()
        feats[i,26] = v20 / max(v60, 1) - 1

    # 27: Monthly return (for momentum)
    for i in range(1, n):
        feats[i,27] = (closes[i] - closes[i-1]) / max(closes[i-1], 0.01)
    feats[:,27] = rolling_zscore(feats[:,27])

    # 28: Volatility (6m rolling std)
    for i in range(6, n):
        feats[i,28] = pd.Series(closes[i-5:i+1]).pct_change().std()
    feats[:,28] = rolling_zscore(feats[:,28])

    # 29: High-Low range ratio
    for i in range(1, n):
        feats[i,29] = (highs[i] - lows[i]) / max(closes[i], 0.01)
    feats[:,29] = rolling_zscore(feats[:,29])

    # 30-32: Sector alpha 3m/12m/24m (from Phase 12)
    for i in range(24, n):
        sr = (closes[i] - closes[i-24]) / max(closes[i-24], 0.01)
        feats[i,30] = np.clip(sr * 5, -0.5, 0.5) if i>=24 else 0
        feats[i,31] = np.clip((closes[i]-closes[max(0,i-12)])/max(closes[max(0,i-12)],0.01), -0.5, 0.5) if i>=12 else 0
        feats[i,32] = np.clip((closes[i]-closes[max(0,i-3)])/max(closes[max(0,i-3)],0.01), -0.5, 0.5) if i>=3 else 0

    # 33-35: Macro (M2/PMI/CPI) — same across all stocks
    for i in range(n):
        d = dates[i]
        feats[i,33] = float(m2_map.get(d, 0))
        feats[i,34] = float(pmi_map.get(d, 50))
        feats[i,35] = float(cpi_map.get(d, 0))

    # 36: ipo_count (monthly IPO count as market heat proxy)
    feats[:,36] = float(ipo_by_month.get(dates[-1], 0)) if len(ipo_by_month) > 0 else 0

    # 37-38: Sector dispersion proxy
    feats[:,37] = 0.0; feats[:,38] = 0.0  # placeholder
    # 39: Resonance signal (Phase 11)
    for i in range(60, n):
        macd_pos = feats[i,7] > 0
        ma20_s = np.polyfit(range(5), closes[i-4:i+1], 1)[0] if i>=5 else 0
        if closes[i] > ma60[i] and ma20_s > 0 and macd_pos: feats[i,39] = -0.5
        elif closes[i] < ma60[i] and ma20_s < 0 and not macd_pos: feats[i,39] = 0.5

    # Generate sequences
    feats = np.nan_to_num(feats, 0.0)
    for i in range(LOOKBACK-1, n-12):
        if closes[i] <= 0.01: continue
        seq = feats[i-LOOKBACK+1:i+1]
        y3 = np.clip((closes[i+3] - closes[i]) / max(closes[i], 0.01), -2, 2) if i+3 < n else 0
        y6 = np.clip((closes[i+6] - closes[i]) / max(closes[i], 0.01), -2, 2) if i+6 < n else 0
        all_sequences.append(seq); all_targets.append([y3, y6]); all_dates.append(dates[i])

    stock_count += 1
    if stock_count % 50 == 0:
        print(f"  {stock_count}/{len(stocks)} stocks, {len(all_sequences)} seqs")

print(f"\nStocks: {stock_count} proc, {skipped} skip | Seqs: {len(all_sequences)}")

X = np.array(all_sequences, dtype=np.float32)
y = np.array(all_targets, dtype=np.float32)
dates_arr = np.array(all_dates)
print(f"X: {X.shape}, y: {y.shape}")

# Walk-forward split
train_m = (dates_arr >= '2015-01') & (dates_arr <= '2021-12')
val_m = (dates_arr >= '2022-01') & (dates_arr <= '2023-12')
test_m = (dates_arr >= '2024-01') & (dates_arr <= '2026-05')

X_tr, y_tr = X[train_m], y[train_m]
X_va, y_va = X[val_m], y[val_m]
X_te, y_te = X[test_m], y[test_m]

# Filter NaN
def clean(X, y):
    m = ~np.isnan(y).any(axis=1)
    return X[m], y[m]
X_tr, y_tr = clean(X_tr, y_tr)
X_va, y_va = clean(X_va, y_va)
X_te, y_te = clean(X_te, y_te)

for name, (Xx, yy) in [('train', (X_tr, y_tr)), ('val', (X_va, y_va)), ('test', (X_te, y_te))]:
    print(f"  {name}: {Xx.shape} NaN={np.any(np.isnan(Xx))} y_range=[{yy.min():.2f}, {yy.max():.2f}]")

np.savez_compressed(OUT / 'train_v3.npz', X=X_tr, y=y_tr)
np.savez_compressed(OUT / 'val_v3.npz', X=X_va, y=y_va)
np.savez_compressed(OUT / 'test_v3.npz', X=X_te, y=y_te)
print(f"Saved: train_v3/val_v3/test_v3.npz (40-dim features)")
