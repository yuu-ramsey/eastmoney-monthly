"""Phase 19 v4: Full comparison — timing v1/v2 × portfolio methods.
One-shot Live Test (FROZEN)."""
import numpy as np, pandas as pd, sqlite3, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.backtest.timing_v2 import compute_improved_timing, compute_timing_v1
from lib.portfolio.optimizer import black_litterman, risk_parity, multi_strategy

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
TOP_K, SLIPPAGE, COMMISSION = 20, 0.003, 0.00025

print("Loading data...")
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
prices = pd.read_sql_query(f"SELECT code, date, close FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2014-12'", conn, params=stocks)
conn.close()
price_matrix = prices.pivot(index='date', columns='code', values='close').sort_index().ffill()

# Load pre-computed signals (with LSTM)
lstm = pd.read_parquet(OUT / 'monthly_lstm_signals_v2.parquet')
signal_df = pd.pivot_table(lstm, index='month', columns='code', values='lstm_signal')
signal_df = signal_df.reindex(sorted(signal_df.index)).fillna(0.0)

print(f"Signal matrix: {signal_df.shape}, Price matrix: {price_matrix.shape}")

def run(signals, timing, portfolio_method, label):
    """Run backtest with given timing and portfolio method. Returns metrics."""
    common = sorted(set(price_matrix.index) & set(signals.index))
    if len(common) < 2:
        return None

    # Timing — scale signals
    if timing is not None:
        mt = timing.reindex(common).fillna(0.5)
    else:
        mt = pd.Series(1.0, index=common)

    port_rets, eq = [], [1.0]
    for i in range(len(common)-1):
        d, nd = common[i], common[i+1]
        sig = signals.loc[d].dropna()
        if len(sig) < TOP_K: continue

        # Portfolio weights
        if portfolio_method == 'ew':
            top = sig.nlargest(TOP_K)
            w = pd.Series(1.0/TOP_K, index=top.index)
        elif portfolio_method == 'bl':
            w_all = black_litterman(signals.loc[d:d], price_matrix)[d:d].iloc[0] if d in price_matrix.index else pd.Series()
            w = w_all.nlargest(TOP_K) if len(w_all) > TOP_K else w_all
            if len(w) == 0: w = pd.Series(1.0/TOP_K, index=sig.nlargest(TOP_K).index)
        elif portfolio_method == 'rp':
            w_all = risk_parity(signals.loc[:d], price_matrix.loc[:d])
            if d in w_all.index:
                w_all_row = w_all.loc[d].dropna()
                w = w_all_row.nlargest(TOP_K) if len(w_all_row) > TOP_K else w_all_row
            else:
                w = pd.Series(1.0/TOP_K, index=sig.nlargest(TOP_K).index)
            if len(w) == 0: w = pd.Series(1.0/TOP_K, index=sig.nlargest(TOP_K).index)
        else:
            top = sig.nlargest(TOP_K)
            w = pd.Series(1.0/TOP_K, index=top.index)

        if len(w) == 0: continue
        w = w / w.sum()  # Normalize

        # Apply timing
        pos = mt[d]
        w = w * pos

        # Compute return
        ret = 0.0
        for c in w.index:
            p0 = price_matrix.at[d, c] if c in price_matrix.columns and d in price_matrix.index else np.nan
            p1 = price_matrix.at[nd, c] if c in price_matrix.columns and nd in price_matrix.index else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret += w[c] * (p1-p0)/p0
        # Costs
        if i > 0:
            prev_sig = signals.loc[common[max(0,i-1)]].dropna()
            if len(prev_sig) >= TOP_K:
                turnover = len(set(w.index) - set(prev_sig.nlargest(TOP_K).index)) / TOP_K
                ret -= turnover * (SLIPPAGE + COMMISSION*2) * pos
        port_rets.append(ret); eq.append(eq[-1]*(1+ret))

    if len(port_rets) < 5: return None
    pr = np.array(port_rets)
    sr = pr.mean()*12/(pr.std()*np.sqrt(12)) if pr.std()>0 else 0
    ann_r = pr.mean()*12
    eq_arr = np.array(eq)
    mdd = float((eq_arr/np.maximum.accumulate(eq_arr)-1).min())
    calmar = ann_r/abs(mdd) if abs(mdd)>0 else 0

    # IS (0-75%) vs OOS (75-100%)
    split = int(0.75*len(pr))
    sr_is = pr[:split].mean()*12/(pr[:split].std()*np.sqrt(12)) if pr[:split].std()>0 else 0
    sr_oos = pr[split:].mean()*12/(pr[split:].std()*np.sqrt(12)) if pr[split:].std()>0 else 0

    print(f"  {label:30s}: SR={sr:.3f} IS={sr_is:.3f} OOS={sr_oos:.3f} DD={mdd:.1%} Calmar={calmar:.3f} Ret={ann_r:.1%}")
    return {'sharpe': sr, 'sr_is': sr_is, 'sr_oos': sr_oos, 'max_dd': mdd, 'calmar': calmar, 'ann_ret': ann_r}

# Compute both timing versions
timing_old = compute_timing_v1(price_matrix.loc['2015-01':])
timing_new = compute_improved_timing(price_matrix.loc['2015-01':], signal_df.loc['2015-01':])

# Run all configs
configs = [
    ('v3 EW old-timing', None, None, 'ew'),
    ('v3 EW+old-timing', None, timing_old, 'ew'),
    ('v4 EW new-timing', None, timing_new, 'ew'),
    ('v4 EW+BL', None, None, 'bl'),
    ('v4 EW+RP', None, None, 'rp'),
    ('v4 new-timing+BL', None, timing_new, 'bl'),
    ('v4 new-timing+RP', None, timing_new, 'rp'),
]

print(f"\n{'='*70}")
print("PHASE 19 v4: Live Test Comparison (FROZEN)")
print(f"{'='*70}\n")

results = {}
for label, _, timing, method in configs:
    r = run(signal_df.loc['2015-01':], timing, method, label)
    if r: results[label] = r

# Summary
print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"{'Config':<30} {'Sharpe':>7} {'IS':>7} {'OOS':>7} {'MaxDD':>7} {'Calmar':>7}")
print(f"{'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for label, r in sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True):
    print(f"{label:<30} {r['sharpe']:7.3f} {r['sr_is']:7.3f} {r['sr_oos']:7.3f} {r['max_dd']:6.1%} {r['calmar']:7.3f}")

best = max(results.items(), key=lambda x: x[1]['sharpe'])
print(f"\nBest: {best[0]} Sharpe={best[1]['sharpe']:.3f}")
sr = best[1]['sharpe']
if sr > 1.0: print("PASS > 1.0 → Phase 21")
elif sr > 0.85: print("MARGINAL 0.85-1.0 → Phase 21 with caution")
elif sr > 0.7: print("BELOW 0.85 but > 0.7 → marginal")
else: print("FAIL < 0.7 → monthly HS300 framework ceiling confirmed")
