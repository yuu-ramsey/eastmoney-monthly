"""B3: Risk-First Strategy. Dynamic position sizing, stop-loss, vol targeting.
Goal: Calmar > 1.0 (vs baseline -60% MaxDD, Calmar=0.34)"""
import numpy as np, pandas as pd, sqlite3
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
TOP_K, SLIPPAGE, COMMISSION = 20, 0.003, 0.00025
TARGET_VOL = 0.15  # 15% annual vol target
STOP_LOSS = -0.20  # -20% per position stop
MAX_POSITION = 0.10  # max 10% per stock
DRAWDOWN_CAP = 0.25  # If DD > 25%, halve positions

print("Loading data...")
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
prices = pd.read_sql_query(f"SELECT code, date, close FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2014-12'", conn, params=stocks)
conn.close()
price_matrix = prices.pivot(index='date', columns='code', values='close').sort_index().ffill()

# Load LSTM signal
lstm = pd.read_parquet(OUT / 'monthly_lstm_signals_v2.parquet')
signal_df = pd.pivot_table(lstm, index='month', columns='code', values='lstm_signal', aggfunc='mean')
signal_df = signal_df.reindex(sorted(signal_df.index)).fillna(0.0)

def run_risk_first(signals, prices, label, use_stop_loss=False, use_vol_target=False, use_dd_cap=False, stop_loss_pct=STOP_LOSS):
    common = sorted(set(prices.index) & set(signals.index))
    if len(common) < 2: return None

    port_rets, eq = [], [1.0]
    trailing_rets = []
    stop_hits = 0

    for i in range(len(common)-1):
        d, nd = common[i], common[i+1]
        sig = signals.loc[d].dropna()
        if len(sig) < TOP_K: continue

        top = sig.nlargest(TOP_K)
        w = pd.Series(1.0 / TOP_K, index=top.index)

        # Cap individual position
        w = w.clip(upper=MAX_POSITION)
        w = w / w.sum() if w.sum() > 0 else w

        # Vol targeting: scale positions to hit target vol
        if use_vol_target and len(trailing_rets) >= 6:
            recent_vol = np.std(trailing_rets[-6:]) * np.sqrt(12)
            if recent_vol > 0.01:
                scale = TARGET_VOL / recent_vol
                w = w * np.clip(scale, 0.3, 2.0)  # cap scaling

        # Drawdown cap: halve positions if in deep DD
        if use_dd_cap and len(eq) > 1:
            peak = max(eq)
            dd = (eq[-1] - peak) / peak
            if dd < -DRAWDOWN_CAP:
                w = w * 0.5

        # Compute returns
        port_ret = 0.0
        for c in w.index:
            p0 = prices.at[d, c] if c in prices.columns and d in prices.index else np.nan
            p1 = prices.at[nd, c] if c in prices.columns and nd in prices.index else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret = (p1 - p0) / p0
                # Stop-loss per position
                if use_stop_loss and ret < stop_loss_pct:
                    ret = stop_loss_pct
                    stop_hits += 1
                port_ret += w[c] * ret

        # Turnover cost
        if i > 0:
            prev_sig = signals.loc[common[max(0,i-1)]].dropna()
            if len(prev_sig) >= TOP_K:
                to = len(set(w.index) - set(prev_sig.nlargest(TOP_K).index)) / TOP_K
                port_ret -= to * (SLIPPAGE + COMMISSION * 2)

        port_rets.append(port_ret)
        trailing_rets.append(port_ret)
        eq.append(eq[-1] * (1 + port_ret))

    if len(port_rets) < 5: return None
    pr = np.array(port_rets)
    sr = pr.mean()*12/(pr.std()*np.sqrt(12)) if pr.std()>0 else 0
    ann_r = pr.mean()*12
    eq_arr = np.array(eq)
    mdd = float((eq_arr / np.maximum.accumulate(eq_arr) - 1).min())
    calmar = ann_r / abs(mdd) if abs(mdd) > 0 else 0

    # OOS split
    split = int(0.75 * len(pr))
    sr_oos = pr[split:].mean()*12/(pr[split:].std()*np.sqrt(12)) if pr[split:].std()>0 else 0

    print(f"  {label:35s}: SR={sr:.3f} DD={mdd:.1%} Calmar={calmar:.3f} Ret={ann_r:.1%} OOS_SR={sr_oos:.3f} Stops={stop_hits}")
    return {'sharpe': sr, 'max_dd': mdd, 'calmar': calmar, 'ann_ret': ann_r, 'sr_oos': sr_oos}

# Run all risk configurations
configs = [
    ('v3 EW (baseline)', False, False, False),
    ('v3 + StopLoss', True, False, False),
    ('v3 + VolTarget', False, True, False),
    ('v3 + DDCap', False, False, True),
    ('v3 + StopLoss + VolTarget', True, True, False),
    ('v3 + ALL risk controls', True, True, True),
]

print(f"\n{'='*70}")
print("B3: Risk-First Strategy")
print(f"{'='*70}\n")

results = {}
for label, sl, vt, dc in configs:
    r = run_risk_first(signal_df.loc['2015-01':], price_matrix, label, use_stop_loss=sl, use_vol_target=vt, use_dd_cap=dc)
    if r: results[label] = r

print(f"\n{'='*70}")
print("B3 FINAL")
print(f"{'='*70}")
print(f"{'Config':<35} {'Sharpe':>7} {'MaxDD':>7} {'Calmar':>7} {'OOS_SR':>7}")
print(f"{'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for label, r in sorted(results.items(), key=lambda x: x[1]['calmar'], reverse=True):
    print(f"{label:<35} {r['sharpe']:7.3f} {r['max_dd']:6.1%} {r['calmar']:7.3f} {r['sr_oos']:7.3f}")

best = max(results.items(), key=lambda x: x[1]['calmar'])
print(f"\nBest Calmar: {best[0]} = {best[1]['calmar']:.3f}")
if best[1]['calmar'] > 1.0: print("KILL SWITCH: PASS > 1.0")
elif best[1]['calmar'] > 0.5: print("KILL SWITCH: MARGINAL 0.5-1.0")
else: print("KILL SWITCH: FAIL < 0.5")
