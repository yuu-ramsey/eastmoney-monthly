"""
Build monthly PIT ST history by polling query_all_stock() at each month's first trading day.

Insight: baostock query_all_stock(day=date) returns the HISTORICAL stock name as of
that date. Names containing 'ST' or '*ST' indicate the stock was ST on that date.
This gives a cheap (~200 queries) PIT ST history without needing isST from k-data
(which was found to reflect current status, not historical per-day status).

Coverage: 2008-01 to 2024-12, one query per month (~204 months).
For each month, we probe dates 1-7 until baostock returns a non-empty result.

ST detection rule:
  'ST' in code_name (index 2 of query_all_stock row, case-sensitive).
  Matches: 'ST', '*ST', '退市ST', etc.

Output tables in data/pead-baostock.sqlite:
  st_history   (code, year_month)   — sparse: only ST=1 rows stored
  st_query_log (year_month, query_date, n_total, n_st, queried_at)

Resume: months already in st_query_log are skipped.

Constraint: baostock single session per IP.
  Run AFTER 10_fetch_baostock_eps.py finishes, alongside §15/§16.

Validation target: sh.600070 (*ST富润)
  Confirmed ST at 2020-06-01 via query_all_stock → should appear in st_history 2020-06.
  At some later month after 摘帽 (ST removal) → should NOT appear in st_history.

Usage:
  .venv/Scripts/python.exe scripts/koc/17_st_history.py
  .venv/Scripts/python.exe scripts/koc/17_st_history.py --smoke   # 2020-2021 only
  .venv/Scripts/python.exe scripts/koc/17_st_history.py --verify  # sh.600070 spot-check
"""

import contextlib
import io
import os
import sqlite3
import sys
import time
import threading
from datetime import date, datetime, timedelta
from typing import Optional

import baostock as bs

# ── Config ────────────────────────────────────────────────────────────────────
SMOKE_MODE: bool = "--smoke" in sys.argv
VERIFY_MODE: bool = "--verify" in sys.argv
DB_PATH: str = "data/pead-baostock.sqlite"
DATE_START: str = "2008-01"
DATE_END: str = "2024-12"
SLEEP_PER_QUERY: float = 0.2    # 200 ms between queries (200 queries total = ~40s)
WATCHDOG_TIMEOUT: int = 60
MAX_DAY_PROBE: int = 8          # Try days 1-8 of each month to find a trading day


# ── Watchdog ──────────────────────────────────────────────────────────────────
class Watchdog:
    def __init__(self, timeout_sec: int = WATCHDOG_TIMEOUT) -> None:
        self._timeout = timeout_sec
        self._last_ping = time.time()
        self._stopped = False
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def ping(self) -> None:
        self._last_ping = time.time()

    def stop(self) -> None:
        self._stopped = True

    def _watch(self) -> None:
        while not self._stopped:
            time.sleep(5)
            if time.time() - self._last_ping > self._timeout:
                print(
                    f"\n[WATCHDOG] No activity for {self._timeout}s — killing for checkpoint/resume",
                    flush=True,
                )
                os._exit(2)


# ── DB setup ──────────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS st_history (
            code        TEXT NOT NULL,
            year_month  TEXT NOT NULL,   -- 'YYYY-MM'
            PRIMARY KEY (code, year_month)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS st_query_log (
            year_month  TEXT PRIMARY KEY,
            query_date  TEXT,            -- actual date passed to query_all_stock
            n_total     INTEGER,         -- total stocks in snapshot
            n_st        INTEGER,         -- number of ST stocks found
            queried_at  TEXT
        )
    """)
    conn.commit()


def get_completed_months(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT year_month FROM st_query_log").fetchall()
    return {r[0] for r in rows}


# ── Month generation ──────────────────────────────────────────────────────────
def all_year_months(start: str, end: str) -> list[str]:
    """Return list of 'YYYY-MM' strings from start to end inclusive."""
    months: list[str] = []
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    while (y, m) <= (ey, em):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def probe_dates_for_month(year_month: str) -> list[str]:
    """Return candidate dates to try for this month (days 1-MAX_DAY_PROBE)."""
    y, m = int(year_month[:4]), int(year_month[5:7])
    candidates: list[str] = []
    for day in range(1, MAX_DAY_PROBE + 1):
        try:
            d = date(y, m, day)
            candidates.append(d.strftime("%Y-%m-%d"))
        except ValueError:
            break  # month doesn't have this many days
    return candidates


# ── Query logic ───────────────────────────────────────────────────────────────
def query_st_for_month(
    year_month: str, watchdog: Watchdog
) -> Optional[tuple[str, list[str], int]]:
    """Query query_all_stock for the first successful date in this month.

    Returns (query_date, [st_codes], n_total) or None if all probes fail.
    """
    for candidate_date in probe_dates_for_month(year_month):
        watchdog.ping()
        try:
            noise = io.StringIO()
            with contextlib.redirect_stdout(noise):
                rs = bs.query_all_stock(day=candidate_date)
            if rs.error_code != "0":
                time.sleep(SLEEP_PER_QUERY)
                continue

            st_codes: list[str] = []
            n_total: int = 0
            while rs.next():
                row = rs.get_row_data()
                if len(row) < 3:
                    continue
                code = row[0]
                name = row[2]
                n_total += 1
                if "ST" in name:
                    st_codes.append(code)

            if n_total > 0:
                return candidate_date, st_codes, n_total

            time.sleep(SLEEP_PER_QUERY)

        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] query_all_stock({candidate_date}): {exc}", flush=True)
            time.sleep(SLEEP_PER_QUERY)

    return None  # all probes failed


def save_month(
    conn: sqlite3.Connection,
    year_month: str,
    query_date: str,
    st_codes: list[str],
    n_total: int,
) -> None:
    # Delete any prior data for this month (for idempotent re-runs)
    conn.execute("DELETE FROM st_history WHERE year_month = ?", (year_month,))
    if st_codes:
        conn.executemany(
            "INSERT OR REPLACE INTO st_history (code, year_month) VALUES (?, ?)",
            [(code, year_month) for code in st_codes],
        )
    conn.execute(
        "INSERT OR REPLACE INTO st_query_log "
        "(year_month, query_date, n_total, n_st, queried_at) VALUES (?,?,?,?,?)",
        (year_month, query_date, n_total, len(st_codes), datetime.now().isoformat()),
    )


# ── Verification ──────────────────────────────────────────────────────────────
def verify_st_history(conn: sqlite3.Connection, code: str = "sh.600070") -> None:
    """Print the monthly ST history for a known ST→non-ST transition stock."""
    print(f"\n  Verification: ST history for {code}", flush=True)
    rows = conn.execute(
        "SELECT year_month FROM st_history WHERE code=? ORDER BY year_month",
        (code,),
    ).fetchall()
    if not rows:
        print(f"  No ST rows found for {code} — either not ST or data missing", flush=True)
        return

    months = [r[0] for r in rows]
    print(f"  ST months ({len(months)} total): {months[:6]}{'...' if len(months) > 6 else ''}", flush=True)
    print(f"  First ST: {months[0]}, Last ST: {months[-1]}", flush=True)

    # Check 2020 specifically (should be ST)
    st_2020 = [m for m in months if m.startswith("2020")]
    print(f"  2020 ST months: {st_2020}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    watchdog = Watchdog(WATCHDOG_TIMEOUT)

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        init_db(conn)
        completed = get_completed_months(conn)

    if SMOKE_MODE:
        months = all_year_months("2019-06", "2022-06")
        print(f"  SMOKE MODE: {len(months)} months (2019-06 to 2022-06)", flush=True)
    else:
        months = all_year_months(DATE_START, DATE_END)

    pending = [m for m in months if m not in completed]
    print(f"  Months to query: {len(pending)} ({len(completed)} already done)", flush=True)

    if not pending:
        print("  Nothing to do.", flush=True)
        if VERIFY_MODE:
            with sqlite3.connect(DB_PATH, timeout=30) as conn:
                verify_st_history(conn)
        return

    # Baostock login
    watchdog.ping()
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise):
        lg = bs.login()
    if lg.error_code != "0":
        print(f"[FATAL] Baostock login failed: {lg.error_code} {lg.error_msg}", flush=True)
        sys.exit(1)
    print("  Baostock login: OK", flush=True)

    done = 0
    failed = 0

    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            for i, year_month in enumerate(pending):
                if i % 20 == 0:
                    print(
                        f"\r  {done}/{len(pending)} months done, {failed} failures  ",
                        end="", flush=True,
                    )
                watchdog.ping()
                result = query_st_for_month(year_month, watchdog)
                if result is None:
                    print(f"\n  [WARN] All probes failed for {year_month}", flush=True)
                    failed += 1
                    # Log failure so we don't retry indefinitely
                    conn.execute(
                        "INSERT OR REPLACE INTO st_query_log "
                        "(year_month, query_date, n_total, n_st, queried_at) VALUES (?,?,?,?,?)",
                        (year_month, None, 0, 0, datetime.now().isoformat()),
                    )
                else:
                    query_date, st_codes, n_total = result
                    save_month(conn, year_month, query_date, st_codes, n_total)

                done += 1
                time.sleep(SLEEP_PER_QUERY)

                if done % 20 == 0:
                    conn.commit()

            conn.commit()

    finally:
        bs.logout()
        print("\n  Baostock logout: OK", flush=True)

    print(f"\n  Done. {done} months queried, {failed} failures.", flush=True)

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        n_st_rows = conn.execute("SELECT COUNT(*) FROM st_history").fetchone()[0]
        n_months = conn.execute("SELECT COUNT(*) FROM st_query_log").fetchone()[0]
        avg_st = conn.execute(
            "SELECT AVG(n_st) FROM st_query_log WHERE n_st > 0"
        ).fetchone()[0]
        print(f"  st_history rows: {n_st_rows:,}", flush=True)
        print(f"  Months covered: {n_months}", flush=True)
        if avg_st:
            print(f"  Avg ST stocks/month: {avg_st:.0f}", flush=True)

        if VERIFY_MODE or True:
            verify_st_history(conn)

    print("[OK] 17_st_history.py completed", flush=True)


if __name__ == "__main__":
    main()
