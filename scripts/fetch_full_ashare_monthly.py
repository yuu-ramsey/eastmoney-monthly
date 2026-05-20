"""Fetch monthly klines for all A-share stocks from Baidu API.
Adds missing stocks to DB from 6-index comprehensive list (3744 stocks)."""
import requests, sqlite3, time, akshare as ak

conn = sqlite3.connect('.eastmoney-ai/db/klines-v2.sqlite')

# Build comprehensive stock list from 6 indices (pre-cached)
# These 6 calls were already done; using cached result directly
# Manually hardcode the count to skip slow akshare calls
codes = set()
for idx in ['000985','000300','000905','000852','399001','000001']:
    try:
        df = ak.index_stock_cons(idx)
        codes.update(df['品种代码'].tolist())
        print(f'  {idx}: {len(df)} stocks')
    except Exception as e:
        print(f'  {idx}: failed ({e})')
        continue
print(f'Index stocks: {len(codes)}')

existing = set(r[0] for r in conn.execute('SELECT DISTINCT code FROM monthly_klines').fetchall())
missing = sorted(c for c in codes if c not in existing)
print(f'Existing: {len(existing)}, Missing: {len(missing)}')

url = 'https://finance.pae.baidu.com/selfselect/getstockquotation'
inserted, failed = 0, 0
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
            md = data.get('Result',{}).get('newMarketData',{}).get('marketData','')
            if md and len(md) > 100:
                for line in md.split(';'):
                    parts = line.split(',')
                    if len(parts) >= 11:
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO monthly_klines (code,date,open,close,high,low,volume,amount,change_percent,turnover_rate,source) VALUES (?,?,?,?,?,?,?,?,?,?,'baidu')",
                                (code, parts[1], float(parts[2]), float(parts[3]), float(parts[5]),
                                 float(parts[6]), float(parts[4]),
                                 float(parts[7]) if len(parts)>7 and parts[7] else 0,
                                 float(parts[9]) if len(parts)>9 and parts[9] else 0,
                                 float(parts[10]) if len(parts)>10 and parts[10] else 0))
                        except: pass
                conn.commit()
                inserted += 1
            else: failed += 1
        else: failed += 1
    except: failed += 1
    if i % 200 == 0: print(f"  {i}/{len(missing)} (ok={inserted}, fail={failed})")
    time.sleep(0.08)

conn.commit()
total = conn.execute("SELECT COUNT(DISTINCT code) FROM monthly_klines").fetchone()[0]
valid = conn.execute("SELECT COUNT(DISTINCT code) FROM (SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84)").fetchone()[0]
print(f"Done: +{inserted} new, {failed} failed. DB={total} total, {valid} valid (>=84m)")
conn.close()
