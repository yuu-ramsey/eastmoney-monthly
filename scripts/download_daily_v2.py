# Download daily K-line data from Baidu API using Scrapling
import sqlite3, json, time
from datetime import datetime
from scrapling import Fetcher

DB = '.eastmoney-ai/db/klines-v2.sqlite'
API = 'https://finance.pae.baidu.com/selfselect/getstockquotation'

conn = sqlite3.connect(DB)
monthly_codes = set(r[0] for r in conn.execute('SELECT DISTINCT code FROM monthly_klines').fetchall())
daily_codes = set(r[0] for r in conn.execute('SELECT DISTINCT code FROM daily_klines').fetchall())
missing = sorted(monthly_codes - daily_codes)
print(f'Monthly: {len(monthly_codes)}, Daily: {len(daily_codes)}, Missing: {len(missing)}')

if len(missing) == 0:
    print('All stocks have daily data!')
    conn.close()
    exit(0)

def get_market(code):
    return '1' if code.startswith(('60','68')) else '0'

f = Fetcher(auto_match=False)
downloaded, errors, skipped = 0, 0, 0

for idx, stock_code in enumerate(missing):
    market = get_market(stock_code)
    url = (f'{API}?all=1&isIndex=false&isBk=false&isBlock=false'
           f'&isFutures=false&isStock=true&newFormat=1'
           f'&group=quotation_kline_ab&finClientType=pc'
           f'&code={stock_code}&ktype=1'
           f'&start_time=2010-01-01%2000:00:00')

    try:
        resp = f.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = resp.json()
        if data.get('ResultCode') != '0':
            errors += 1
            if errors <= 3: print(f'  [{stock_code}] API error {data.get("ResultCode")}')
            continue

        market_data = data.get('Result', {}).get('newMarketData', {})
        rows = market_data.get('keys', [])
        if not rows:
            skipped += 1
            continue

        rows_data = []
        for row in rows:
            if len(row) < 8: continue
            try:
                ts = int(row[0])
                date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                if date_str < '2010-01-01': continue
                open_p = float(row[2]); close_p = float(row[3])
                volume = float(row[4]); high_p = float(row[5]); low_p = float(row[6])
                amount = float(row[7])
                turnover = float(row[10]) if len(row) > 10 and row[10] else 0
                amplitude = (high_p - low_p) / max(close_p, 0.01) if close_p > 0 else 0
                change_pct = float(row[9]) if len(row) > 9 and row[9] else 0

                rows_data.append((stock_code, date_str, open_p, close_p, high_p, low_p,
                                  volume, amount, amplitude, change_pct, 0, turnover,
                                  1.0, 'baidu'))
            except (ValueError, IndexError, TypeError):
                continue

        if rows_data:
            conn.executemany(
                'INSERT OR REPLACE INTO daily_klines '
                '(code, date, open, close, high, low, volume, amount, amplitude, '
                'change_percent, change_amount, turnover_rate, adjust, source) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows_data)
            conn.commit()
            downloaded += 1

    except Exception as e:
        errors += 1
        if errors <= 3: print(f'  [{stock_code}] {type(e).__name__}: {str(e)[:80]}')

    if (idx + 1) % 50 == 0:
        pct = (idx+1)/len(missing)*100
        print(f'  [{idx+1}/{len(missing)}] {pct:.0f}% downloaded={downloaded} errors={errors} skipped={skipped}...', flush=True)
        time.sleep(1)

conn.close()
print(f'\nDone: downloaded={downloaded} errors={errors} skipped={skipped}')
# Verify
conn2 = sqlite3.connect(DB)
new_daily = conn2.execute('SELECT COUNT(DISTINCT code) FROM daily_klines').fetchone()[0]
conn2.close()
print(f'Daily stocks now: {new_daily} (was {len(daily_codes)})')
