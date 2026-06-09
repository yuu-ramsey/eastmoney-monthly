"""
Fetch Baostock daily k-data for all A-shares present in eps_baostock_raw.
Also supplements eps_baostock_raw with liqa_share (floating A-share count).

Output table: daily_kline in data/pead-baostock.sqlite
  code, date, close, turn, amount, pct_chg, tradestatus, is_st
  adjustflag='3' (unadjusted close — correct for market cap via close × total_share)

liqa_share supplement:
  Adds liqa_share REAL column to eps_baostock_raw (once, idempotent).
  Probes up to 3 recent years per stock to get the most recent liqaShare value.
  Stored as a per-stock snapshot; used as a fast approximation for float market cap.
  For time-series float market cap, §16 falls back to total_share if liqa_share is missing.

Constraint: baostock enforces one session per IP.
  Must run AFTER 10_fetch_baostock_eps.py finishes (single-session constraint).

Checkpoint/resume: daily_fetch_log table (same pattern as §10).
  Re-running the script skips stocks already marked complete.

Usage:
  .venv/Scripts/python.exe scripts/koc/15_fetch_baostock_daily.py
  .venv/Scripts/python.exe scripts/koc/15_fetch_baostock_daily.py --smoke   # 50 stocks
"""

import contextlib
import io
import os
import socket
import sqlite3
import sys
import time
import threading
from datetime import datetime
from typing import Optional

import baostock as bs

# ── Config ────────────────────────────────────────────────────────────────────
SMOKE_MODE: bool = "--smoke" in sys.argv
DB_PATH: str = "data/pead-baostock.sqlite"
DATE_START: str = "2008-01-01"
DATE_END: str = "2024-12-31"
SLEEP_PER_QUERY: float = 0.04   # 40 ms (matches §10 rate)
CHECKPOINT_EVERY: int = 20      # commit every N stocks
SOCKET_TIMEOUT: int = 15
WATCHDOG_TIMEOUT: int = 60

# Probe years for liqa_share snapshot (most recent available)
LIQA_PROBE_YEARS: list[int] = [2024, 2023, 2022]


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
        CREATE TABLE IF NOT EXISTS daily_kline (
            code        TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            close       REAL,       -- unadjusted (adjustflag='3'), for market cap
            turn        REAL,       -- turnover rate (%)
            amount      REAL,       -- CNY turnover
            pct_chg     REAL,       -- daily return (%)
            tradestatus INTEGER,    -- 1=trading 0=suspended
            is_st       INTEGER,    -- 1=ST stock
            PRIMARY KEY (code, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_fetch_log (
            code        TEXT PRIMARY KEY,
            status      TEXT,           -- 'complete' | 'no_data' | 'error'
            n_records   INTEGER,
            fetched_at  TEXT
        )
    """)
    # Add liqa_share column to eps_baostock_raw if it doesn't exist
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(eps_baostock_raw)").fetchall()
    }
    if "liqa_share" not in existing:
        conn.execute("ALTER TABLE eps_baostock_raw ADD COLUMN liqa_share REAL")
        print("  Added liqa_share column to eps_baostock_raw", flush=True)
    conn.commit()


def get_completed_codes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT code FROM daily_fetch_log WHERE status IN ('complete','no_data','error')"
    ).fetchall()
    return {r[0] for r in rows}


def get_eps_codes(conn: sqlite3.Connection) -> list[str]:
    """Return all distinct codes present in eps_baostock_raw (§10 output)."""
    rows = conn.execute("SELECT DISTINCT code FROM eps_baostock_raw ORDER BY code").fetchall()
    return [r[0] for r in rows]


# ── liqa_share helpers ────────────────────────────────────────────────────────
def _safe_float(s: object) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def fetch_liqa_share(code: str, watchdog: Watchdog) -> Optional[float]:
    """Probe recent quarters to get the most recent liqaShare for this stock.

    liqaShare is at index 10 in query_profit_data row.
    Returns None if no data found.
    """
    for year in LIQA_PROBE_YEARS:
        for quarter in (4, 3, 2, 1):
            watchdog.ping()
            try:
                rs = bs.query_profit_data(code=code, year=str(year), quarter=str(quarter))
                if rs.error_code != "0":
                    time.sleep(SLEEP_PER_QUERY)
                    continue
                if rs.data and len(rs.data) > 0:
                    row = rs.data[0]
                    if len(row) > 10:
                        val = _safe_float(row[10])  # index 10 = liqaShare
                        if val is not None and val > 0:
                            return val
                time.sleep(SLEEP_PER_QUERY)
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARN] liqa_share {code} {year}Q{quarter}: {exc}", flush=True)
                time.sleep(SLEEP_PER_QUERY)
    return None


def update_liqa_share(code: str, liqa: float, conn: sqlite3.Connection) -> None:
    """Set liqa_share for all rows of this stock where it is currently NULL."""
    conn.execute(
        "UPDATE eps_baostock_raw SET liqa_share = ? WHERE code = ? AND liqa_share IS NULL",
        (liqa, code),
    )


# ── Daily kline fetch ─────────────────────────────────────────────────────────
def fetch_daily_kline(
    code: str, conn: sqlite3.Connection, watchdog: Watchdog
) -> int:
    """Fetch full date range daily klines for one stock.

    Returns number of rows inserted.
    """
    watchdog.ping()
    fields = "date,close,turn,amount,pctChg,tradestatus,isST"
    try:
        rs = bs.query_history_k_data_plus(
            code, fields,
            start_date=DATE_START,
            end_date=DATE_END,
            frequency="d",
            adjustflag="3",   # unadjusted close — required for true market cap
        )
        if rs.error_code != "0":
            return 0

        buffer: list[tuple] = []
        while rs.next():
            watchdog.ping()
            row = rs.get_row_data()
            if len(row) < 7 or not row[0]:
                continue
            buffer.append((
                code,
                row[0],                     # date
                _safe_float(row[1]),        # close
                _safe_float(row[2]),        # turn
                _safe_float(row[3]),        # amount
                _safe_float(row[4]),        # pct_chg
                int(row[5]) if row[5] else None,   # tradestatus
                int(row[6]) if row[6] else None,   # is_st
            ))

        if buffer:
            conn.executemany(
                "INSERT OR REPLACE INTO daily_kline "
                "(code, date, close, turn, amount, pct_chg, tradestatus, is_st) "
                "VALUES (?,?,?,?,?,?,?,?)",
                buffer,
            )
        return len(buffer)

    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] daily_kline {code}: {exc}", flush=True)
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    watchdog = Watchdog(WATCHDOG_TIMEOUT)

    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        init_db(conn)
        completed = get_completed_codes(conn)
        all_codes = get_eps_codes(conn)

    print(f"  eps_baostock_raw codes: {len(all_codes)}", flush=True)
    print(f"  Checkpoint: {len(completed)} stocks already done", flush=True)

    pending = [c for c in all_codes if c not in completed]
    if SMOKE_MODE:
        pending = pending[:50]
        print(f"  SMOKE MODE: limited to {len(pending)} stocks", flush=True)

    if not pending:
        print("  Nothing to do — all stocks complete.", flush=True)
        return

    print(
        f"  Fetching {len(pending)} stocks: daily klines {DATE_START}→{DATE_END} + liqa_share",
        flush=True,
    )

    # Baostock login — single session, must not be called again
    watchdog.ping()
    noise = io.StringIO()
    with contextlib.redirect_stdout(noise):
        lg = bs.login()
    if lg.error_code != "0":
        print(f"[FATAL] Baostock login failed: {lg.error_code} {lg.error_msg}", flush=True)
        sys.exit(1)
    print("  Baostock login: OK", flush=True)

    total_rows = 0
    done = 0
    errors = 0

    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            for i, code in enumerate(pending):
                if i % 50 == 0:
                    print(f"\r  {done}/{len(pending)} done, {errors} errors  ", end="", flush=True)

                # Phase 1: supplement liqa_share (fast probe, ≤8 queries)
                liqa = fetch_liqa_share(code, watchdog)
                time.sleep(SLEEP_PER_QUERY)

                # Phase 2: daily klines (one big query, all years)
                n = fetch_daily_kline(code, conn, watchdog)
                time.sleep(SLEEP_PER_QUERY)

                # Write liqa_share if found
                if liqa is not None:
                    update_liqa_share(code, liqa, conn)

                status = "complete" if n > 0 else "no_data"
                if n == 0:
                    errors += 1
                    status = "error"
                conn.execute(
                    "INSERT OR REPLACE INTO daily_fetch_log (code, status, n_records, fetched_at) "
                    "VALUES (?,?,?,?)",
                    (code, status, n, datetime.now().isoformat()),
                )
                total_rows += n
                done += 1

                if done % CHECKPOINT_EVERY == 0:
                    conn.commit()

            conn.commit()

    finally:
        bs.logout()
        print("\n  Baostock logout: OK", flush=True)

    print(f"\n  Done. {done} stocks, {total_rows:,} daily rows, {errors} errors.", flush=True)

    # Quick sanity check
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        n_kline = conn.execute("SELECT COUNT(*) FROM daily_kline").fetchone()[0]
        n_liqa = conn.execute(
            "SELECT COUNT(*) FROM eps_baostock_raw WHERE liqa_share IS NOT NULL"
        ).fetchone()[0]
        n_total = conn.execute("SELECT COUNT(*) FROM eps_baostock_raw").fetchone()[0]
    print(f"  daily_kline rows:  {n_kline:,}", flush=True)
    print(f"  liqa_share filled: {n_liqa:,} / {n_total:,} eps rows", flush=True)
    print("[OK] 15_fetch_baostock_daily.py completed", flush=True)


if __name__ == "__main__":
    main()
