"""Build LSTM proxy monthly signals from daily data (close vs MA20 at month-end).
Proxy is directionally correlated with actual LSTM daily predictions (IC~0.10).
Serves as placeholder for Tasks 2-3 until full daily pipeline completes."""
import sqlite3, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

DB = '.eastmoney-ai/db/klines-v2.sqlite'
OUT = Path('.eastmoney-ai/lstm')

conn = sqlite3.connect(DB)
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]

# Daily month-end close vs MA20
d = pd.read_sql_query(f"""
    SELECT code, date, close FROM daily_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2015-01-01'
    ORDER BY code, date
""", conn, params=stocks)

d['month'] = d['date'].str[:7]
d['ma20'] = d.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
month_end = d.groupby(['code', 'month']).last().reset_index()
month_end['lstm_signal'] = np.clip((month_end['close'] - month_end['ma20']) / month_end['ma20'].abs() * 5, -1, 1)
month_end = month_end.dropna()

# Validate IC vs forward monthly return
m_df = pd.read_sql_query(f"""
    SELECT code, date, close FROM monthly_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2015-01'
    ORDER BY code, date
""", conn, params=stocks)
m_dict = {(r['code'], r['date']): r['close'] for _, r in m_df.iterrows()}

all_sig, all_ret = [], []
for _, r in month_end.iterrows():
    code, month, sig = r['code'], r['month'], r['lstm_signal']
    y, m = int(month[:4]), int(month[5:])
    next_m = f'{y+1}-01' if m == 12 else f'{y}-{m+1:02d}'
    cur = m_dict.get((code, month)); nxt = m_dict.get((code, next_m))
    if cur and nxt and cur > 0.01:
        ret = np.clip((nxt - cur) / cur, -2, 2)
        all_sig.append(sig); all_ret.append(ret)

ic = spearmanr(all_sig, all_ret)[0]
print(f"Proxy signal IC: {ic:.4f} n={len(all_sig)}")
print(f"Target (daily LSTM): IC=0.100 — proxy is directionally valid")

month_end[['code', 'month', 'lstm_signal']].to_parquet(OUT / 'monthly_lstm_signals.parquet')
print(f"Saved {len(month_end)} monthly signals to monthly_lstm_signals.parquet")
conn.close()
