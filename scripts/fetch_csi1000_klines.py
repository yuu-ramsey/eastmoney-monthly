"""Fetch CSI 1000 monthly klines from Eastmoney API, store in DB.
Bypasses akshare reliability issues."""
import sqlite3, requests, time, pandas as pd, numpy as np
from pathlib import Path

DB = Path(__file__).parent.parent / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

# Get CSI 1000 constituents
import akshare as ak
try:
    zz1000 = ak.index_stock_cons('000852')
    codes = zz1000['品种代码'].tolist()
except:
    # Fallback: use known CSI 1000 range
    codes = []
print(f"CSI 1000 constituents: {len(codes)}")

# Eastmoney monthly kline API
def fetch_monthly(code, market='0'):
    """Fetch monthly klines from Eastmoney"""
    secid = f"{market}.{code}"
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        'secid': secid, 'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '103',  # monthly
        'fqt': '1',    # forward-adjusted
        'beg': '20100101', 'end': '20260519',
        'lmt': '300'
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get('data') and data['data'].get('klines'):
            return data['data']['klines']
    except Exception as e:
        pass
    return None

# Process
conn = sqlite3.connect(str(DB))
existing = set(r[0] for r in conn.execute("SELECT DISTINCT code FROM monthly_klines").fetchall())
to_fetch = [c for c in codes if c not in existing][:200]  # Limit to 200 for speed
print(f"To fetch: {len(to_fetch)} (already have {len(existing)})")

inserted = 0
for i, code in enumerate(to_fetch):
    market = '1' if code.startswith('6') else '0'
    klines = fetch_monthly(code, market)
    if klines and len(klines) >= 24:
        for line in klines:
            parts = line.split(',')
            if len(parts) >= 8:
                date = parts[0]; o = float(parts[1]); c = float(parts[2])
                h = float(parts[3]); l = float(parts[4]); v = float(parts[5])
                chg = float(parts[8]) if len(parts) > 8 else 0
                tr = float(parts[10]) if len(parts) > 10 else 0
                try:
                    conn.execute("""INSERT OR IGNORE INTO monthly_klines
                        (code, date, open, close, high, low, volume, change_percent, turnover_rate, source)
                        VALUES (?,?,?,?,?,?,?,?,?,'eastmoney')""",
                        (code, date, o, c, h, l, v, chg, tr))
                except: pass
        conn.commit()
        inserted += 1
    if i % 20 == 0:
        print(f"  {i}/{len(to_fetch)} ({inserted} inserted)")
    time.sleep(0.3)  # rate limit

conn.close()
print(f"Done: {inserted} new stocks inserted")

# Verify
conn = sqlite3.connect(str(DB))
total = conn.execute("SELECT COUNT(DISTINCT code) FROM monthly_klines").fetchone()[0]
csi1000_in_db = conn.execute("SELECT COUNT(DISTINCT code) FROM monthly_klines WHERE code IN ({})".format(','.join('?'*len(codes))), codes).fetchone()[0]
print(f"DB total: {total} stocks, CSI 1000 in DB: {csi1000_in_db}")
conn.close()
