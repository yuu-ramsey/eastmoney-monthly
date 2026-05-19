"""Validate LSTM v2 signals: compute IC, plug into backtest stub"""
import pandas as pd, numpy as np, sqlite3
from scipy.stats import spearmanr

lstm = pd.read_parquet('.eastmoney-ai/lstm/monthly_lstm_signals_v2.parquet')
print(f"LSTM signals: {len(lstm)} rows")

conn = sqlite3.connect('.eastmoney-ai/db/klines-v2.sqlite')
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
m = pd.read_sql_query(f"SELECT code, date, close FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2015-01'", conn, params=stocks)
conn.close()

m_dict = {}
for _, r in m.iterrows(): m_dict[(r['code'], r['date'])] = r['close']

# IC: LSTM signal vs forward 1m return
sigs, rets, dates_list = [], [], []
for _, r in lstm.iterrows():
    c, mm, sig = r['code'], r['month'], r['lstm_signal']
    y, mo = int(mm[:4]), int(mm[5:])
    nx = f'{y+1}-01' if mo==12 else f'{y}-{mo+1:02d}'
    cur = m_dict.get((c, mm)); nxt = m_dict.get((c, nx))
    if cur and nxt and cur > 0.01:
        ret = np.clip((nxt-cur)/cur, -2, 2)
        sigs.append(sig); rets.append(ret); dates_list.append(mm)

sigs = np.array(sigs); rets = np.array(rets); dates_arr = np.array(dates_list)
ic = spearmanr(sigs, rets)[0]
print(f"Overall IC: {ic:.4f} n={len(sigs)}")

# IS vs OOS
is_mask = np.array([d <= '2021-12' for d in dates_arr])
if is_mask.sum() > 100:
    ic_is = spearmanr(sigs[is_mask], rets[is_mask])[0]
    ic_oos = spearmanr(sigs[~is_mask], rets[~is_mask])[0]
    print(f"IS (2015-2021): IC={ic_is:.4f} n={is_mask.sum()}")
    print(f"OOS (2022+): IC={ic_oos:.4f} n={(~is_mask).sum()}")

# Long-Short Sharpe
n = len(sigs); cut = int(n*0.3); idx = np.argsort(sigs)
ls = rets[idx[-cut:]].mean() - rets[idx[:cut]].mean()
sr = ls / (rets[idx[-cut:]] - rets[idx[:cut]]).std() * np.sqrt(12) if (rets[idx[-cut:]] - rets[idx[:cut]]).std()>0 else 0
print(f"L-S: {ls:.4f} Sharpe(ann)={sr:.3f}")

# Backtest simulation: Top-20 monthly rebalance
print("\nBacktest simulation (Top-20, monthly, 2015-2025):")
port_rets = []
months = sorted(set(dates_arr))
for i, mm in enumerate(months[:-1]):
    mask = dates_arr == mm
    month_sigs = sigs[mask]; month_codes = np.array(lstm['code'].iloc[:len(sigs)])[mask]
    if len(month_sigs) < 20: continue
    top_idx = np.argsort(month_sigs)[-20:]
    # Next month return
    nx = months[i+1]
    nx_mask = dates_arr == nx
    nx_rets = rets[nx_mask]; nx_codes = np.array(lstm['code'].iloc[:len(sigs)])[nx_mask]
    if len(nx_rets) < 20: continue
    top_rets = nx_rets[np.isin(nx_codes, month_codes[top_idx])]
    if len(top_rets) > 0:
        port_rets.append(np.mean(top_rets[:20]))

if len(port_rets) > 10:
    pr = np.array(port_rets)
    ann_r = pr.mean()*12; ann_v = pr.std()*np.sqrt(12)
    sr = ann_r/ann_v if ann_v>0 else 0
    eq = np.cumprod(1+pr)
    mdd = (eq - np.maximum.accumulate(eq)).min() / np.maximum.accumulate(eq).max()
    print(f"Sharpe={sr:.3f} AnnRet={ann_r:.1%} MaxDD={mdd:.1%}")
    print(f"Signal ready for full Phase 19 v3 integration")
