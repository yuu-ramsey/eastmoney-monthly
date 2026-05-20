"""Fetch missing monthly klines from Baidu API for HS300+CSI500+CSI1000 stocks."""
import requests, sqlite3, time, akshare as ak

conn = sqlite3.connect('.eastmoney-ai/db/klines-v2.sqlite')
codes = set()
for idx in ['000300', '000905', '000852']:
    df = ak.index_stock_cons(idx)
    codes.update(df['品种代码'].tolist())
print(f'3-index total: {len(codes)}')

existing = set(r[0] for r in conn.execute('SELECT DISTINCT code FROM monthly_klines').fetchall())
missing = [c for c in codes if c not in existing]
print(f'Missing: {len(missing)}')

url = 'https://finance.pae.baidu.com/selfselect/getstockquotation'
inserted = 0
for i, code in enumerate(missing):
    try:
        r = requests.get(url, params={
            'all': '1', 'isIndex': 'false', 'isBk': 'false', 'isBlock': 'false',
            'isFutures': 'false', 'isStock': 'true', 'newFormat': '1',
            'group': 'quotation_kline_ab', 'finClientType': 'pc',
            'code': code, 'ktype': '3', 'start_time': '2010-01-01 00:00:00'
        }, timeout=10)
        if r.status_code == 200:
            data = r.json()
            md = data.get('Result', {}).get('newMarketData', {}).get('marketData', '')
            if md:
                for line in md.split(';'):
                    parts = line.split(',')
                    if len(parts) >= 11:
                        try:
                            date_str = parts[1]
                            o = float(parts[2]); c = float(parts[3]); v = float(parts[4])
                            h = float(parts[5]); l = float(parts[6])
                            amt = float(parts[7]) if len(parts) > 7 and parts[7] else 0
                            chg = float(parts[9]) if len(parts) > 9 and parts[9] else 0
                            tr = float(parts[10]) if len(parts) > 10 and parts[10] else 0
                            conn.execute(
                                "INSERT OR IGNORE INTO monthly_klines (code, date, open, close, high, low, volume, amount, change_percent, turnover_rate, source) VALUES (?,?,?,?,?,?,?,?,?,?,'baidu')",
                                (code, date_str, o, c, h, l, v, amt, chg, tr))
                        except (ValueError, IndexError):
                            pass
                conn.commit()
                inserted += 1
    except Exception as e:
        pass
    if i % 100 == 0:
        print(f"  {i}/{len(missing)} ({inserted} inserted)")
    time.sleep(0.1)

conn.commit()
total = conn.execute("SELECT COUNT(DISTINCT code) FROM monthly_klines").fetchone()[0]
valid = conn.execute("SELECT COUNT(DISTINCT code) FROM (SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84)").fetchone()[0]
print(f"Done: {inserted} new, DB={total} total, {valid} valid (>=84m)")
conn.close()
