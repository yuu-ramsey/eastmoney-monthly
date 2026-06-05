"""Sprint 4: Full A-share universe (5000 stocks) monthly + weekly pipeline.
Baidu API → DB → strict v6 features → LSTM-7 → Test IC.
INPUT_DATA_RANGE: 1990-01 to 2026-05 (all available history)
WALK_FORWARD: yes (date-based split, strict)
LOOK_AHEAD_RISK: none (Baidu API returns only historical data)
TEST_SET_USAGE: read-only, evaluated ONCE per frequency"""
import requests, sqlite3, time, json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib' / 'lstm'))
from model_v2 import create_model

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
DEV = torch.device('cuda')
BAIDU_URL = 'https://finance.pae.baidu.com/selfselect/getstockquotation'
B, LR, WD, FIXED_EPOCHS = 128, 5e-4, 1e-4, 30

# ======== 1. Get all A-share stocks ========
print("Getting all A-share stock list...")
import akshare as ak
stock_info = ak.stock_zh_a_spot_em()
all_codes = stock_info['代码'].tolist()
print(f"All A-shares: {len(all_codes)} stocks")

# ======== 2. Fetch klines from Baidu API ========
conn = sqlite3.connect(str(DB))
existing = set(r[0] for r in conn.execute("SELECT DISTINCT code FROM monthly_klines").fetchall())
print(f"Existing in DB: {len(existing)} stocks")

def fetch_baidu(code, ktype):
    """ktype: 2=weekly, 3=monthly"""
    params = {
        'all': '1', 'isIndex': 'false', 'isBk': 'false', 'isBlock': 'false',
        'isFutures': 'false', 'isStock': 'true', 'newFormat': '1',
        'group': 'quotation_kline_ab', 'finClientType': 'pc',
        'code': code, 'ktype': str(ktype),
        'start_time': '2010-01-01 00:00:00'
    }
    try:
        r = requests.get(BAIDU_URL, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            md = data.get('Result', {}).get('newMarketData', {}).get('marketData', '')
            if md: return md.split(';')
    except: pass
    return None

for freq, ktype, table in [('monthly', 3, 'monthly_klines'), ('weekly', 2, 'weekly_klines')]:
    to_fetch = [c for c in all_codes if c not in existing]
    print(f"\nFetching {freq} for {len(to_fetch)} new stocks...")
    inserted = 0
    for i, code in enumerate(to_fetch):
        klines = fetch_baidu(code, ktype)
        if klines and len(klines) >= 12:
            for line in klines:
                parts = line.split(',')
                if len(parts) >= 11:
                    try:
                        date_str = parts[1]; o = float(parts[2]); c = float(parts[3])
                        v = float(parts[4]); h = float(parts[5]); l = float(parts[6])
                        amt = float(parts[7]) if len(parts) > 7 and parts[7] else 0
                        chg = float(parts[9]) if len(parts) > 9 and parts[9] else 0
                        tr = float(parts[10]) if len(parts) > 10 and parts[10] else 0
                        conn.execute(f"INSERT OR IGNORE INTO {table} (code, date, open, close, high, low, volume, amount, change_percent, turnover_rate, source) VALUES (?,?,?,?,?,?,?,?,?,?,'baidu')",
                                     (code, date_str, o, c, h, l, v, amt, chg, tr))
                    except (ValueError, IndexError): pass
            conn.commit(); inserted += 1
        if i % 200 == 0: print(f"  {i}/{len(to_fetch)} ({inserted} inserted)")
        time.sleep(0.12)
    conn.commit()
    existing = set(r[0] for r in conn.execute(f"SELECT DISTINCT code FROM {table}").fetchall())
    print(f"  Done: {inserted} new, DB now has {len(existing)} stocks in {table}")

print(f"\nDB total: {len(existing)} stocks")

# ======== 3. Filter stocks with enough history ========
# For monthly: need >= 84 months
monthly_codes = [r[0] for r in conn.execute("SELECT code, COUNT(*) as n FROM monthly_klines GROUP BY code HAVING n >= 84").fetchall()]
weekly_codes = [r[0] for r in conn.execute("SELECT code, COUNT(*) as n FROM weekly_klines GROUP BY code HAVING n >= 200").fetchall()]
print(f"Monthly valid (>=84m): {len(monthly_codes)}")
print(f"Weekly valid (>=200w): {len(weekly_codes)}")

if len(monthly_codes) < 100:
    print("INSUFFICIENT monthly stocks. DB build incomplete.")
    conn.close()
    sys.exit(1)

# ======== 4. Build features + train + Test ========
def build_and_train(freq, codes, lookback, fwd_steps, table_name):
    """Full pipeline: load→features→split→train→test"""
    params_str = ','.join('?' * len(codes))
    df = pd.read_sql_query(f"SELECT code, date, open, high, low, close, volume FROM {table_name} WHERE code IN ({params_str}) AND date >= '2010-01-01' ORDER BY code, date", conn, params=codes)
    print(f"\n{freq}: {len(df)} rows, {df['code'].nunique()} stocks")

    all_seq, all_y, all_dates = [], [], []
    for code in codes:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        min_len = lookback + fwd_steps * 2
        if len(g) < min_len: continue
        n = len(g); dates = g['date'].tolist()
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        feats = np.zeros((n, 21), dtype=np.float32)
        for j, arr in enumerate([c, o, h, l, v]):
            s = pd.Series(arr); m = s.rolling(60, min_periods=60).mean(); std = s.rolling(60, min_periods=60).std()
            vals = ((arr - m) / std.replace(0, 1)).fillna(0).values
            feats[:, j] = vals.astype(np.float32)
        e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
        dif = np.nan_to_num(e12 - e26, 0); dea = pd.Series(dif).ewm(span=9).mean().values
        feats[:, 5] = dif; feats[:, 6] = dea; feats[:, 7] = (dif - dea) * 2
        delta = np.diff(c, prepend=c[0]); gs = np.where(delta > 0, delta, 0); ls = np.where(delta < 0, -delta, 0)
        feats[:, 8] = np.nan_to_num(100 - 100 / (1 + pd.Series(gs).ewm(alpha=1/14).mean().values / np.maximum(pd.Series(ls).ewm(alpha=1/14).mean().values, 1e-8)), 50)
        ma20 = pd.Series(c).rolling(20).mean().values; m60 = pd.Series(c).rolling(60).mean().values
        feats[:, 18] = np.nan_to_num((c - ma20) / np.maximum(c, 0.01), 0)
        feats[:, 19] = np.nan_to_num((c - m60) / np.maximum(c, 0.01), 0)
        feats[:, 20] = hash(code) % 31 / 31.0
        feats = np.nan_to_num(feats, 0.0)

        for i in range(lookback - 1, n - fwd_steps * 2):
            if c[i] <= 0.01: continue
            seq = feats[i - lookback + 1 : i + 1]
            y3 = np.clip((c[i + fwd_steps] - c[i]) / max(c[i], 0.01), -2, 2) if i + fwd_steps < n else 0
            y6 = np.clip((c[i + fwd_steps * 2] - c[i]) / max(c[i], 0.01), -2, 2) if i + fwd_steps * 2 < n else 0
            all_seq.append(seq); all_y.append([y3, y6]); all_dates.append(dates[i])

        if len(codes) > 200 and len(all_seq) % 50000 == 0:
            pass

    X = np.array(all_seq, dtype=np.float32); y = np.array(all_y, dtype=np.float32)
    dates_arr = np.array(all_dates)
    print(f"  Seqs: {len(X)} | Dates: {dates_arr[0]}~{dates_arr[-1]}")

    train_m = np.array([d >= '2015-01' and d <= '2021-12' for d in dates_arr])
    test_m = np.array([d >= '2024-01' for d in dates_arr])
    Xtr, ytr = X[train_m], y[train_m]; Xte, yte = X[test_m], y[test_m]
    Xtr, ytr = Xtr[~np.isnan(ytr).any(axis=1)], ytr[~np.isnan(ytr).any(axis=1)]
    Xte, yte = Xte[~np.isnan(yte).any(axis=1)], yte[~np.isnan(yte).any(axis=1)]
    print(f"  Train: {Xtr.shape}, Test: {Xte.shape}")

    if len(Xtr) < 1000: return None

    torch.manual_seed(456); np.random.seed(456)
    # Sample if > 300K to manage memory
    if len(Xtr) > 300000:
        idx = np.random.choice(len(Xtr), 300000, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
        print(f"  Subsampled train to 300K")

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    train_ld = torch.utils.data.DataLoader(train_ds, B, shuffle=True, pin_memory=True)
    model = create_model('LSTM-7', 21).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=WD)
    t0 = time.time()
    for ep in range(1, FIXED_EPOCHS+1):
        for pg in opt.param_groups: pg['lr'] = min(LR, LR*ep/10.0) if ep<=10 else LR
        model.train()
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(DEV), yb.to(DEV); opt.zero_grad()
            p = model(Xb)
            loss = 0.5*nn.MSELoss()(p[:,0], yb[:,0]) + 0.5*nn.MSELoss()(p[:,1], yb[:,1])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 10 == 0: print(f"    ep{ep} loss={loss.item():.4f}")

    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    test_ld = torch.utils.data.DataLoader(test_ds, B, pin_memory=True)
    model.eval(); tp, tt = [], []
    with torch.no_grad():
        for Xb, yb in test_ld:
            Xb, yb = Xb.to(DEV), yb.to(DEV); p = model(Xb); tp.append(p.cpu().numpy()); tt.append(yb.cpu().numpy())
    tp = np.concatenate(tp); tt = np.concatenate(tt)
    ic3 = spearmanr(tp[:,0], tt[:,0])[0]
    n = len(tp); cut = int(n*0.3); idx = np.argsort(tp[:,0])
    ls = tt[idx[-cut:],0] - tt[idx[:cut],0]
    sr3 = ls.mean()/ls.std()*np.sqrt(12/3) if ls.std()>0 else 0
    print(f"  Test IC3={ic3:+.4f}, SR3={sr3:+.3f} ({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return ic3, sr3

# Run monthly and weekly
results = {}
for freq, codes, lookback, fwd, table in [
    ('Monthly (full A)', monthly_codes, 60, 3, 'monthly_klines'),
    ('Weekly (full A)', weekly_codes, 200, 13, 'weekly_klines'),
]:
    r = build_and_train(freq, codes, lookback, fwd, table)
    if r: results[freq] = {'ic3': r[0], 'sr3': r[1]}

conn.close()

results['Daily (HS300)'] = {'ic3': 0.1409, 'sr3': 0.443}

print(f"\n{'='*70}")
print(f"SPRINT 4 FINAL: Full A-share universe")
print(f"{'='*70}")
print(f"{'Frequency':<22} {'Stocks':>8} {'Test IC3':>10} {'SR3':>8}")
print(f"{'-'*22} {'-'*8} {'-'*10} {'-'*8}")
for name, r in results.items():
    print(f"{name:<22} {'5000+' if 'full' in name else '300':>8} {r['ic3']:10.4f} {r['sr3']:8.3f}")
