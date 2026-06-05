# Download daily K-line data from Baidu API for all monthly stocks
import sqlite3, urllib.request, json, time, sys
from datetime import datetime

DB = '.eastmoney-ai/db/klines-v2.sqlite'
API = 'https://finance.pae.baidu.com/selfselect/getstockquotation'
BATCH_SIZE = 50  # stocks per batch, pause between batches

conn = sqlite3.connect(DB)
# Get stocks that have monthly data but NO daily data
monthly_codes = set(r[0] for r in conn.execute(
    'SELECT DISTINCT code FROM monthly_klines').fetchall())
daily_codes = set(r[0] for r in conn.execute(
    'SELECT DISTINCT code FROM daily_klines').fetchall())
missing = sorted(monthly_codes - daily_codes)
print(f'Monthly stocks: {len(monthly_codes)}, with daily: {len(daily_codes)}, missing: {len(missing)}')

if len(missing) == 0:
    print('All stocks already have daily data!')
    conn.close()
    sys.exit(0)

# Determine market from code prefix
def get_market(code):
    if code.startswith(('60','68')): return '1'  # Shanghai
    return '0'  # Shenzhen

downloaded = 0
errors = 0
for stock_code in missing:
    market = get_market(stock_code)
    # Fetch all data from 2010
    url = (f'{API}?all=1&isIndex=false&isBk=false&isBlock=false'
           f'&isFutures=false&isStock=true&newFormat=1'
           f'&group=quotation_kline_ab&finClientType=pc'
           f'&code={stock_code}&ktype=1'
           f'&start_time=2010-01-01%2000:00:00')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        if data.get('ResultCode') != '0':
            errors += 1
            if errors < 5: print(f'  {stock_code}: API error {data.get("ResultCode")}')
            continue

        market_data = data.get('Result', {}).get('newMarketData', {})
        headers = market_data.get('headers', [])
        rows = market_data.get('keys', [])

        if not headers or not rows:
            continue

        # Parse and insert
        rows_data = []
        for i, row in enumerate(rows):
            if len(row) < 8: continue
            try:
                ts = int(row[0])
                date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                if date_str < '2010-01-01': continue
                open_p = float(row[2])
                close_p = float(row[3])
                volume = float(row[4])
                high_p = float(row[5])
                low_p = float(row[6])
                amount = float(row[7])
                turnover = float(row[10]) if len(row) > 10 else 0
                change_pct = float(row[9]) if len(row) > 9 else 0
                amplitude = (high_p - low_p) / max(close_p, 0.01) if close_p > 0 else 0

                rows_data.append((stock_code, date_str, open_p, close_p, high_p, low_p,
                                  volume, amount, amplitude, change_pct, 0, turnover,
                                  1.0, 'baidu'))
            except (ValueError, IndexError):
                continue

        if rows_data:
            conn.executemany(
                'INSERT OR REPLACE INTO daily_klines '
                '(code, date, open, close, high, low, volume, amount, amplitude, '
                'change_percent, change_amount, turnover_rate, adjust, source) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', rows_data)
            conn.commit()
            downloaded += 1

        if downloaded % 20 == 0:
            print(f'  Downloaded {downloaded}/{len(missing)} stocks ({rows_data[-1][1] if rows_data else "?"})...', flush=True)
            time.sleep(0.5)  # rate limit

    except Exception as e:
        errors += 1
        if errors < 5: print(f'  {stock_code}: {e}')
        time.sleep(2)

conn.close()
print(f'\nDone: {downloaded} downloaded, {errors} errors')
print(f'New daily stocks: {downloaded}')
