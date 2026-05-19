"""Debug: why monthly IC is NaN despite daily IC=0.114"""
import pandas as pd, numpy as np, sqlite3
from scipy.stats import spearmanr

daily = pd.read_parquet('.eastmoney-ai/lstm/daily_signals.parquet')
print(f"Daily: {len(daily)} rows, score_nan={daily.score.isna().sum()}, std={daily.score.std():.4f}")

monthly = pd.read_parquet('.eastmoney-ai/lstm/monthly_lstm_signals.parquet')
print(f"Monthly: {len(monthly)} rows, signal_nan={monthly.lstm_signal.isna().sum()}")
print(f"Monthly stats: mean={monthly.lstm_signal.mean():.4f} std={monthly.lstm_signal.std():.4f}")

conn = sqlite3.connect('.eastmoney-ai/db/klines-v2.sqlite')
m = pd.read_sql_query("SELECT code, date, close FROM monthly_klines WHERE date >= '2015-01'", conn)
conn.close()
print(f"Monthly klines: {len(m)} rows")

m_lookup = {(r['code'], r['date']): r['close'] for _, r in m.iterrows()}

# Test each aggregation strategy
daily['month'] = daily['date'].str[:7]

strategies = {
    'latest': lambda g: g['score'].iloc[-1],
    'mean': lambda g: g['score'].mean(),
    'median': lambda g: g['score'].median(),
    'recent_5': lambda g: g['score'].iloc[-5:].mean(),
    'recent_10': lambda g: g['score'].iloc[-10:].mean(),
}

results = {}
for name, fn in strategies.items():
    # Aggregate
    monthly_agg = daily.groupby(['code', 'month']).apply(fn).reset_index()
    monthly_agg.columns = ['code', 'month', 'lstm_signal']
    monthly_agg['lstm_signal'] = monthly_agg['lstm_signal'].clip(-1, 1)

    # Validate IC (Val period 2022-2023 only)
    sigs, rets = [], []
    for _, r in monthly_agg.iterrows():
        c, mm, sig = r['code'], r['month'], r['lstm_signal']
        if pd.isna(sig): continue
        if not ('2022' <= mm[:4] <= '2023'): continue  # Val only!
        y, mo = int(mm[:4]), int(mm[5:])
        nx = f'{y+1}-01' if mo == 12 else f'{y}-{mo+1:02d}'
        cur = m_lookup.get((c, mm)); nxt = m_lookup.get((c, nx))
        if cur and nxt and cur > 0.01:
            ret = np.clip((nxt - cur) / cur, -2, 2)
            sigs.append(sig); rets.append(ret)

    sigs_arr = np.array(sigs); rets_arr = np.array(rets)
    if len(sigs_arr) > 100 and sigs_arr.std() > 1e-8:
        ic = spearmanr(sigs_arr, rets_arr)[0]
        n = len(sigs_arr); cut = int(n * 0.3)
        idx = np.argsort(sigs_arr)
        ls = rets_arr[idx[-cut:]].mean() - rets_arr[idx[:cut]].mean()
        results[name] = {'ic': ic, 'n': n, 'ls': ls, 'sig_std': sigs_arr.std()}
        print(f"  {name:12s}: IC={ic:.4f} n={n} L-S={ls:.4f} std={sigs_arr.std():.4f}")
    else:
        print(f"  {name:12s}: FAIL (n={len(sigs_arr)} std={sigs_arr.std():.6f})")

if results:
    best = max(results.items(), key=lambda x: x[1]['ic'])
    print(f"\nBest: {best[0]} IC={best[1]['ic']:.4f}")
else:
    print("\nALL FAILED — checking raw daily signal quality:")
    # Sample daily predictions for 1 stock
    sample_code = daily['code'].iloc[0]
    s = daily[daily['code'] == sample_code].head(20)
    print(f"\n{sample_code} first 20 days:")
    print(s[['date', 'score']].to_string())
