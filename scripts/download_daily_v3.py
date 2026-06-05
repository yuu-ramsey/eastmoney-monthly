# Download daily K-lines via Scrapling (Baidu API)
import sqlite3, time
from scrapling import Fetcher

DB = '.eastmoney-ai/db/klines-v2.sqlite'
API = 'https://finance.pae.baidu.com/selfselect/getstockquotation'

conn = sqlite3.connect(DB)
monthly = set(r[0] for r in conn.execute('SELECT DISTINCT code FROM monthly_klines'))
daily = set(r[0] for r in conn.execute('SELECT DISTINCT code FROM daily_klines'))
missing = sorted(monthly - daily)
print(f'Monthly: {len(monthly)}, Daily: {len(daily)}, Missing: {len(missing)}')
if not missing: print('All done!'); conn.close(); exit()

def get_market(c): return '1' if c.startswith(('60','68')) else '0'

f = Fetcher(auto_match=False)
done, err, skip = 0, 0, 0
t0 = time.time()

for i, code in enumerate(missing):
    market = get_market(code)
    url = (f'{API}?all=1&isIndex=false&isBk=false&isBlock=false'
           f'&isFutures=false&isStock=true&newFormat=1'
           f'&group=quotation_kline_ab&finClientType=pc'
           f'&code={code}&ktype=1&start_time=2010-01-01%2000:00:00')
    try:
        resp = f.get(url)
        data = resp.json()
        if data.get('ResultCode') != '0':
            err += 1
            if err <= 2: print(f'  {code} API err {data.get("ResultCode")}')
            continue

        md = data['Result']['newMarketData']
        raw = md['marketData']  # semicolon-separated CSV rows
        rows_text = raw.split(';')
        rows_data = []
        for row_str in rows_text:
            if not row_str.strip(): continue
            parts = row_str.split(',')
            if len(parts) < 8: continue
            try:
                ts = int(parts[0]); open_p = float(parts[2]); close_p = float(parts[3])
                volume = float(parts[4]); high_p = float(parts[5]); low_p = float(parts[6])
                amount = float(parts[7]) if parts[7] != '--' else 0
                turnover = float(parts[10]) if len(parts) > 10 and parts[10] != '--' else 0
                change_pct = float(parts[9]) if len(parts) > 9 and parts[9] != '--' else 0
                amplitude = (high_p - low_p) / max(close_p, 0.01) if close_p > 0 else 0
                date_str = parts[1]
                if date_str < '2010-01-01': continue
                rows_data.append((code, date_str, open_p, close_p, high_p, low_p,
                                  volume, amount, amplitude, change_pct, 0, turnover,
                                  1.0, 'baidu'))
            except (ValueError, IndexError): continue

        if rows_data:
            conn.executemany(
                'INSERT OR REPLACE INTO daily_klines '
                '(code,date,open,close,high,low,volume,amount,amplitude,'
                'change_percent,change_amount,turnover_rate,adjust,source) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows_data)
            conn.commit()
            done += 1
        else:
            skip += 1
    except Exception as e:
        err += 1
        if err <= 3: print(f'  {code} {type(e).__name__}: {str(e)[:60]}')
        time.sleep(1)

    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed/(i+1)*(len(missing)-i-1)
        print(f'  [{i+1}/{len(missing)}] done={done} err={err} skip={skip} {elapsed:.0f}s ETA {eta:.0f}s', flush=True)
        time.sleep(1)

conn.close()
print(f'\nDone: {done} downloaded, {err} errors, {skip} skipped ({time.time()-t0:.0f}s)')
