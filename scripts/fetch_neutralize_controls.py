"""
Fetch monthly turnover + one totalShare snapshot per stock.

For each stock in the 72tp pool:
  1. Monthly k-data: date, close, amount (CNY turnover)  — Amihud + size proxy
  2. One recent quarterly profit row: totalShare          — true market cap base

Market cap at cutoff = closeAtCutoff × totalShare_snapshot
(PIT approximation: totalShare from latest disclosed quarter, assumed stable.
 Corporate-action stocks are a small minority; the approximation is acceptable
 for cross-sectional size neutralization.)

Amihud_monthly = |R_monthly| / (amount_monthly / 1e8)
Size proxy = log(totalShare × closeAtCutoff)

Output: data/neutralize-controls.json
  {
    "monthly": {code: {"YYYY-MM": {"close": ..., "amount": ..., "return": ...}}},
    "total_share": {code: float},   # latest available quarterly snapshot
    "metadata": {...}
  }

Single-connection note: baostock enforces one session per IP.
Do NOT run concurrently with any other bs.login() script.
"""

import json
import os
import socket
import time
import sys
from datetime import datetime

import baostock as bs

# Global socket timeout prevents baostock API calls from hanging indefinitely.
# Without this, rs.next() can block forever on a stalled connection.
socket.setdefaulttimeout(30)

POOL_PATH = 'data/frozen-eval-lowpos-72tp.json'
OUT_PATH = 'data/neutralize-controls.json'
SLEEP = 0.12      # seconds between queries
SLEEP_ERR = 0.3   # back-off on error (was 2.0, reduced to avoid 72s/stock worst case)
CHECKPOINT_EVERY = 100  # incremental save interval

# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_pool():
    with open(POOL_PATH, 'r', encoding='utf-8') as f:
        d = json.load(f)
    pts = d.get('testPoints', d) if isinstance(d, dict) else d
    return pts


def extract_stocks(pts):
    """
    Returns dict: code → {'cutoffs': set, 'close_at_cutoff': {cutoff: price}}.
    """
    stocks = {}
    for r in pts:
        code = r['stockCode']
        cutoff = r['cutoffDate']
        if code not in stocks:
            stocks[code] = {'cutoffs': set(), 'close_at_cutoff': {}}
        stocks[code]['cutoffs'].add(cutoff)
        stocks[code]['close_at_cutoff'][cutoff] = r.get('closeAtCutoff')
    return stocks


def date_range_for(cutoffs):
    dates = sorted(cutoffs)
    start = dates[0][:4] + '-' + dates[0][5:7] + '-01'
    y, m = int(dates[-1][:4]), int(dates[-1][5:7])
    m += 2
    if m > 12:
        m -= 12
        y += 1
    return start, f'{y}-{m:02d}-28'


def fetch_monthly_kdata(code, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            rs = bs.query_history_k_data_plus(
                code, 'date,close,amount',
                start_date=start_date, end_date=end_date,
                frequency='m', adjustflag='3',
            )
            if rs.error_code != '0':
                if attempt < retries - 1:
                    time.sleep(SLEEP_ERR)
                    continue
                return None
            rows = []
            while rs.next():
                r = rs.get_row_data()
                rows.append({'date': r[0][:7], 'close': safe_float(r[1]), 'amount': safe_float(r[2])})
            # Compute monthly returns
            sorted_rows = sorted(rows, key=lambda x: x['date'])
            for i, row in enumerate(sorted_rows):
                if i == 0:
                    row['return'] = None
                else:
                    prev = sorted_rows[i-1]['close']
                    curr = row['close']
                    if prev and prev > 0 and curr:
                        row['return'] = (curr - prev) / prev
                    else:
                        row['return'] = None
            return {r['date']: {k: v for k, v in r.items() if k != 'date'} for r in sorted_rows}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(SLEEP_ERR)
            else:
                return None
    return None


def fetch_latest_total_share(code, probe_years, retries=3):
    """
    Query quarterly profit data in reverse chronological order.
    Return the first (most recent) totalShare found.
    Probe list: [(year, quarter)] in descending order.
    """
    for year in sorted(probe_years, reverse=True):
        for quarter in [4, 3, 2, 1]:
            for attempt in range(retries):
                try:
                    rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
                    if rs.error_code != '0':
                        if attempt < retries - 1:
                            time.sleep(SLEEP_ERR)
                            continue
                        break
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    if rows:
                        ts = safe_float(rows[0][9])   # index 9 = totalShare
                        if ts and ts > 0:
                            return ts
                    time.sleep(0.05)
                    break
                except Exception:
                    if attempt < retries - 1:
                        time.sleep(SLEEP_ERR)
                    else:
                        break
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def save_checkpoint(monthly_data, total_share_data, errors, n_stocks, start_date, end_date, probe_years, stock_list):
    """Write partial results so progress is not lost on interruption."""
    ts_ok = sum(1 for c in stock_list if total_share_data.get(c) is not None)
    mo_ok = sum(1 for c in stock_list if len(monthly_data.get(c, {})) > 0)
    out = {
        'metadata': {
            'fetched_at': datetime.now().isoformat(),
            'n_stocks': n_stocks,
            'pool': POOL_PATH,
            'date_range': f'{start_date} to {end_date}',
            'mc_coverage_pct': round(ts_ok / n_stocks * 100, 1) if n_stocks > 0 else 0,
            'monthly_coverage_pct': round(mo_ok / n_stocks * 100, 1) if n_stocks > 0 else 0,
            'n_errors': len(errors),
            'note': 'totalShare is latest snapshot (not PIT per cutoff). '
                    'market_cap = closeAtCutoff × totalShare is approximate. '
                    'Amihud = |monthly_return| / (monthly_amount / 1e8).',
        },
        'monthly': monthly_data,
        'total_share': total_share_data,
        'errors': errors[:20],
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, separators=(',', ':'))


def main():
    print("Loading pool...")
    pts = load_pool()
    stocks = extract_stocks(pts)
    n_stocks = len(stocks)
    print(f"Pool: {len(pts)} records, {n_stocks} unique stocks")

    all_cutoffs = set()
    for s in stocks.values():
        all_cutoffs.update(s['cutoffs'])
    dates = sorted(all_cutoffs)
    start_global = dates[0][:4] + '-' + dates[0][5:7] + '-01'
    y, m = int(dates[-1][:4]), int(dates[-1][5:7])
    m += 2
    if m > 12:
        m -= 12
        y += 1
    end_global = f'{y}-{m:02d}-28'

    # Years to probe for totalShare (last 2 years of cutoffDates should suffice)
    probe_years = sorted(set(int(d[:4]) for d in all_cutoffs))[-3:]
    print(f"Global date range: {start_global} to {end_global}")
    print(f"Probe years for totalShare: {probe_years}")
    print(f"\nStarting fetch (~{n_stocks * 2} API calls, ~{n_stocks * 2 * SLEEP / 60:.0f} min)...\n")

    stock_list = sorted(stocks.keys())

    # Resume support: load existing partial output and skip already-done stocks
    monthly_data = {}
    total_share_data = {}
    errors = []
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            monthly_data = existing.get('monthly', {})
            total_share_data = existing.get('total_share', {})
            errors = list(existing.get('errors', []))
            n_resume = sum(1 for c in stock_list if c in monthly_data and c in total_share_data)
            if n_resume > 0:
                print(f"Resuming from checkpoint: {n_resume}/{n_stocks} stocks already done, skipping.")
        except Exception as e:
            print(f"Could not load checkpoint ({e}), starting fresh.")

    bs.login()
    print("Baostock login OK\n")

    try:
        done = 0
        for i, code in enumerate(stock_list):
            # Skip if already fetched (resume support)
            if code in monthly_data and code in total_share_data:
                done += 1
                continue

            if done % 50 == 0:
                sys.stdout.write(f'\r  {done}/{n_stocks} ({done/n_stocks*100:.0f}%)  ')
                sys.stdout.flush()

            # Phase 1: monthly k-data
            s_start, s_end = date_range_for(stocks[code]['cutoffs'])
            mk = fetch_monthly_kdata(code, s_start, s_end)
            if mk is None:
                errors.append({'code': code, 'phase': 'monthly'})
                monthly_data[code] = {}
            else:
                monthly_data[code] = mk
            time.sleep(SLEEP)

            # Phase 2: latest totalShare
            ts = fetch_latest_total_share(code, probe_years)
            if ts is None:
                # Fallback: try one more year back
                ts = fetch_latest_total_share(code, [probe_years[0] - 1])
            total_share_data[code] = ts
            time.sleep(SLEEP)

            done += 1

            # Incremental checkpoint save
            if done % CHECKPOINT_EVERY == 0:
                save_checkpoint(monthly_data, total_share_data, errors, n_stocks,
                                start_global, end_global, probe_years, stock_list)
                sys.stdout.write(f'\r  {done}/{n_stocks} ({done/n_stocks*100:.0f}%) [saved]  ')
                sys.stdout.flush()

    finally:
        bs.logout()
        print("\n\nBaostock logout OK")

    print(f"\nFetch complete. Errors: {len(errors)}/{n_stocks}")

    # Coverage check
    mc_ok = sum(1 for c in stock_list if total_share_data.get(c) is not None)
    mo_ok = sum(1 for c in stock_list if len(monthly_data.get(c, {})) > 0)
    print(f"Market cap base (totalShare available): {mc_ok}/{n_stocks} = {mc_ok/n_stocks*100:.0f}%")
    print(f"Monthly data available: {mo_ok}/{n_stocks} = {mo_ok/n_stocks*100:.0f}%")

    save_checkpoint(monthly_data, total_share_data, errors, n_stocks,
                    start_global, end_global, probe_years, stock_list)

    print(f"\nSaved: {OUT_PATH}")
    print("✅ Script completed successfully")


if __name__ == '__main__':
    main()
