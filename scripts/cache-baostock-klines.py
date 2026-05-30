"""Pull Baostock monthly klines for v2 pool stocks, save cache."""
import baostock as bs, json, os

with open('data/frozen-eval-lowpos-v2-baostock.json') as f:
    pool = json.load(f)

codes = sorted(set(tp['stockCode'] for tp in pool['testPoints']))
bcs = [f"sh.{c}" if c.startswith('6') else f"sz.{c}" for c in codes]
print(f"Unique: {len(codes)}")

bs.login()
cache = {}
for i, bc in enumerate(bcs):
    if i % 200 == 0: print(f"  {i}/{len(codes)}...")
    try:
        rs = bs.query_history_k_data_plus(bc, 'date,close,open,high,low,volume', start_date='2010-01-01', end_date='2026-05-31', frequency='m', adjustflag='2')
        rows = []
        while (rs.error_code == '0') and rs.next():
            d = rs.get_row_data()
            c = float(d[1]) if d[1] and d[1] != '' else None
            if c and c > 0.001:
                rows.append({'d': d[0][:7], 'c': c, 'o': float(d[2]) if d[2] and d[2]!='' else c,
                            'h': float(d[3]) if d[3] and d[3]!='' else c,
                            'l': float(d[4]) if d[4] and d[4]!='' else c,
                            'v': float(d[5]) if d[5] and d[5]!='' else 0})
        cache[bc.split('.')[-1]] = rows
    except: cache[bc.split('.')[-1]] = []

bs.logout()
os.makedirs('data', exist_ok=True)
with open('data/baostock-klines-cache.json', 'w') as f:
    json.dump(cache, f)
print(f"Saved {len(cache)} stocks")
