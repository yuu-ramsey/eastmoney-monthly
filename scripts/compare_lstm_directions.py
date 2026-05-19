"""Compare normal vs reversed LSTM signal direction in top-20 backtest"""
import pandas as pd, numpy as np, sqlite3
from pathlib import Path

PROJECT = Path(__file__).parent.parent
conn = sqlite3.connect(str(PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
prices = pd.read_sql_query(f"SELECT code, date, close FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2014-12'", conn, params=stocks)
conn.close()
price_matrix = prices.pivot(index='date', columns='code', values='close').sort_index().ffill()

def backtest(signal_file, label):
    lstm = pd.read_parquet(signal_file)
    months = sorted(set(lstm['month'].unique()) & set(price_matrix.index))
    port_rets = []
    for i, mm in enumerate(months[:-1]):
        sig = lstm[lstm['month'] == mm].set_index('code')['lstm_signal'].dropna()
        if len(sig) < 20: continue
        top = sig.nlargest(20)
        next_mm = months[i+1]
        ret_sum, cnt = 0, 0
        for c in top.index:
            if c in price_matrix.columns:
                p0 = price_matrix.at[mm, c] if mm in price_matrix.index else np.nan
                p1 = price_matrix.at[next_mm, c] if next_mm in price_matrix.index else np.nan
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    ret_sum += (p1-p0)/p0; cnt += 1
        if cnt > 0: port_rets.append(ret_sum/cnt)

    pr = np.array(port_rets)
    sr = pr.mean()*12/(pr.std()*np.sqrt(12)) if pr.std()>0 else 0
    ann_r = pr.mean()*12
    eq = np.cumprod(1+pr); mdd = float((eq/np.maximum.accumulate(eq)-1).min())

    # IS vs OOS split
    n_is = int(0.75 * len(pr))
    sr_is = pr[:n_is].mean()*12/(pr[:n_is].std()*np.sqrt(12)) if pr[:n_is].std()>0 else 0
    sr_oos = pr[n_is:].mean()*12/(pr[n_is:].std()*np.sqrt(12)) if pr[n_is:].std()>0 else 0

    print(f"{label:20s}: Sharpe={sr:.3f} AnnRet={ann_r:.1%} MaxDD={mdd:.1%} IS={sr_is:.3f} OOS={sr_oos:.3f}")
    return sr

# Swap signals
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
import shutil
# Current file = v2 (reversed). Backup it, then test non-reversed
shutil.copy(str(OUT/'monthly_lstm_signals_v2.parquet'), str(OUT/'monthly_lstm_signals_v2_bak.parquet'))

print("Running both LSTM signal directions...\n")
sr_rev = backtest(OUT/'monthly_lstm_signals_v2.parquet', "Reversed (v3)")
shutil.copy(str(OUT/'monthly_lstm_signals_nr.parquet'), str(OUT/'monthly_lstm_signals_v2.parquet'))
sr_norm = backtest(OUT/'monthly_lstm_signals_v2.parquet', "Normal (non-reversed)")

print(f"\nComparison: Reversed={sr_rev:.3f} Normal={sr_norm:.3f}")
if sr_rev > sr_norm:
    print("Reversed is better — keep v2 signals")
    shutil.copy(str(OUT/'monthly_lstm_signals_v2_bak.parquet'), str(OUT/'monthly_lstm_signals_v2.parquet'))
else:
    print("Normal is better — switch to non-reversed")
