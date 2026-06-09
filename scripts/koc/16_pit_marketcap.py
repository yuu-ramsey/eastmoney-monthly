"""
Build PIT (point-in-time) daily market cap via as-of join.

Implements the spec in docs/pead-s16-pit-mktcap.md:
  market_cap = close(date) × total_share(latest pub_date ≤ date)
  float_mktcap = close(date) × liqa_share(latest pub_date ≤ date)

Time anchor: pub_date (disclosure date), NOT stat_date — prevents look-ahead.
Join method: pd.merge_asof(direction='backward') per code.
Units: total_share in 万股, close in CNY/share → market_cap in 万元 → ÷100 = 亿元.

Input tables (pead-baostock.sqlite):
  daily_kline      — code, date, close (unadjusted)
  eps_baostock_raw — code, pub_date, total_share, liqa_share

Output table:
  daily_mktcap — code, date, market_cap_yi, float_mktcap_yi, mktcap_missing, liqa_missing

Run after: 15_fetch_baostock_daily.py completes.
Usage: python scripts/koc/16_pit_marketcap.py [--smoke]
"""

import sqlite3
import sys
from typing import Optional

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
SMOKE_MODE: bool = "--smoke" in sys.argv
DB_PATH: str = "data/pead-baostock.sqlite"

# §01 filter threshold (Market cap >= 5 billion CNY = 50 亿元)
MKTCAP_FLOOR_YI: float = 50.0

# Validation benchmarks: (code, date, expected_market_cap_yi, tolerance_pct)
BENCHMARKS: list[tuple[str, str, float, float]] = [
    # 贵州茅台 2023-12-29 close ≈ 1711 CNY, total_share ≈ 125602 万股
    # market_cap ≈ 1711 × 125602 / 100 ≈ 21489 亿元
    ("sh.600519", "2023-12-29", 21000.0, 0.15),
    # 招商银行 2023-12-29 close ≈ 33 CNY, total_share ≈ 2519 千万股 = 251900 万股
    # market_cap ≈ 33 × 251900 / 100 ≈ 8313 亿元
    ("sh.600036", "2023-12-29", 8000.0, 0.20),
]


# ── DB helpers ─────────────────────────────────────────────────────────────────
def init_output_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_mktcap (
            code             TEXT    NOT NULL,
            date             TEXT    NOT NULL,
            market_cap_yi    REAL,       -- total market cap in 亿元
            float_mktcap_yi  REAL,       -- float market cap in 亿元 (may be NULL)
            mktcap_missing   INTEGER,    -- 1 if total_share not available at this date
            liqa_missing     INTEGER,    -- 1 if liqa_share not available at this date
            PRIMARY KEY (code, date)
        )
    """)
    conn.commit()


def load_shares(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load per-quarter share counts from eps_baostock_raw, keyed by pub_date."""
    df = pd.read_sql_query(
        "SELECT code, pub_date, total_share, liqa_share "
        "FROM eps_baostock_raw "
        "WHERE pub_date IS NOT NULL AND total_share IS NOT NULL "
        "ORDER BY code, pub_date",
        conn,
    )
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    return df


def load_daily(conn: sqlite3.Connection, codes: Optional[list[str]] = None) -> pd.DataFrame:
    if codes is not None:
        placeholders = ",".join("?" * len(codes))
        df = pd.read_sql_query(
            f"SELECT code, date, close FROM daily_kline "
            f"WHERE code IN ({placeholders}) AND close IS NOT NULL "
            f"ORDER BY code, date",
            conn,
            params=codes,
        )
    else:
        df = pd.read_sql_query(
            "SELECT code, date, close FROM daily_kline "
            "WHERE close IS NOT NULL ORDER BY code, date",
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── PIT join ──────────────────────────────────────────────────────────────────
def compute_pit_mktcap(
    daily: pd.DataFrame,
    shares: pd.DataFrame,
) -> pd.DataFrame:
    """Per-code as-of join: match each trading date to the most recent pub_date ≤ date.

    Returns DataFrame with market_cap_yi, float_mktcap_yi, mktcap_missing, liqa_missing.
    """
    # merge_asof requires both DataFrames sorted by the join key
    daily = daily.sort_values("date")
    shares = shares.sort_values("pub_date")

    # rename pub_date → date so merge_asof can match on the same column name
    shares_renamed = shares.rename(columns={"pub_date": "date"})

    merged = pd.merge_asof(
        daily,
        shares_renamed[["code", "date", "total_share", "liqa_share"]],
        on="date",
        by="code",
        direction="backward",   # only use pub_date ≤ trading date (no look-ahead)
    )

    # market_cap: close (CNY/share) × total_share (万股) / 100 = 亿元
    merged["market_cap_yi"] = merged["close"] * merged["total_share"] / 100.0
    merged["float_mktcap_yi"] = merged["close"] * merged["liqa_share"] / 100.0

    # missing flags
    merged["mktcap_missing"] = merged["total_share"].isna().astype(int)
    merged["liqa_missing"] = merged["liqa_share"].isna().astype(int)

    # Null out derived values where inputs are missing
    merged.loc[merged["mktcap_missing"] == 1, "market_cap_yi"] = None
    merged.loc[merged["liqa_missing"] == 1, "float_mktcap_yi"] = None

    return merged[["code", "date", "market_cap_yi", "float_mktcap_yi",
                   "mktcap_missing", "liqa_missing"]]


# ── Validation ────────────────────────────────────────────────────────────────
def validate_row_count(result: pd.DataFrame, daily: pd.DataFrame) -> None:
    """§5 check 1: join must not create extra rows (no cartesian product)."""
    assert len(result) == len(daily), (
        f"FAIL cartesian check: result has {len(result)} rows, daily has {len(daily)} rows. "
        "As-of join should preserve the daily row count."
    )
    print(f"  [PASS] Row count check: {len(result):,} rows (no cartesian product)", flush=True)


def validate_benchmarks(conn: sqlite3.Connection) -> bool:
    """§5 check 2: market cap magnitude sanity (spot-check known large caps)."""
    all_pass = True
    for code, date, expected_yi, tol in BENCHMARKS:
        row = conn.execute(
            "SELECT market_cap_yi FROM daily_mktcap WHERE code=? AND date=?",
            (code, date),
        ).fetchone()
        if row is None or row[0] is None:
            print(f"  [BENCHMARK MISS] {code} {date}: no data", flush=True)
            all_pass = False
            continue
        actual = row[0]
        rel_err = abs(actual - expected_yi) / expected_yi
        status = "PASS" if rel_err <= tol else "FAIL"
        print(
            f"  [BENCHMARK {status}] {code} {date}: "
            f"expected≈{expected_yi:.0f}亿, actual={actual:.0f}亿, err={rel_err:.1%}",
            flush=True,
        )
        if status == "FAIL":
            all_pass = False
    return all_pass


def validate_pit_jump(conn: sqlite3.Connection, code: str = "sh.600519") -> None:
    """§5 check 3: total_share used must be from pub_date ≤ date.

    Prints a sample of market cap values around a known pub_date to show the jump.
    """
    # Find a pub_date for this stock
    row = conn.execute(
        "SELECT pub_date, total_share FROM eps_baostock_raw "
        "WHERE code=? AND pub_date IS NOT NULL AND total_share IS NOT NULL "
        "ORDER BY pub_date DESC LIMIT 1",
        (code,),
    ).fetchone()
    if row is None:
        print(f"  [PIT CHECK] No pub_date found for {code}, skipping", flush=True)
        return

    pub_date, ts = row
    # Sample 3 days before and 3 days after pub_date
    sample = conn.execute(
        "SELECT m.date, m.market_cap_yi, e.total_share, e.pub_date "
        "FROM daily_mktcap m "
        "LEFT JOIN eps_baostock_raw e ON e.code = m.code "
        "  AND e.pub_date = ("
        "    SELECT MAX(e2.pub_date) FROM eps_baostock_raw e2 "
        "    WHERE e2.code = m.code AND e2.pub_date <= m.date AND e2.total_share IS NOT NULL"
        "  ) "
        "WHERE m.code=? AND m.date >= date(?, '-5 days') AND m.date <= date(?, '+5 days') "
        "ORDER BY m.date",
        (code, pub_date, pub_date),
    ).fetchall()
    if not sample:
        print(f"  [PIT CHECK] No rows found for {code} near {pub_date}", flush=True)
        return
    print(f"  [PIT CHECK] {code} around pub_date={pub_date}:", flush=True)
    for date_val, mc, ts_used, pd_used in sample:
        marker = " ← pub_date" if date_val >= pub_date and (not sample or True) else ""
        print(f"    {date_val}  mktcap={mc:.0f}亿  pub_date_used={pd_used}{marker}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        # Check prerequisites
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "daily_kline" not in tables:
            print("[FATAL] daily_kline table not found — run 15_fetch_baostock_daily.py first")
            sys.exit(1)
        n_daily = conn.execute("SELECT COUNT(*) FROM daily_kline").fetchone()[0]
        n_shares = conn.execute(
            "SELECT COUNT(DISTINCT code) FROM eps_baostock_raw WHERE total_share IS NOT NULL"
        ).fetchone()[0]
        print(f"  daily_kline rows: {n_daily:,}", flush=True)
        print(f"  eps codes with total_share: {n_shares:,}", flush=True)

        init_output_table(conn)

        if SMOKE_MODE:
            codes = conn.execute(
                "SELECT DISTINCT code FROM daily_kline LIMIT 200"
            ).fetchall()
            codes = [r[0] for r in codes]
            print(f"  SMOKE MODE: {len(codes)} stocks", flush=True)
            shares = load_shares(conn)
            shares = shares[shares["code"].isin(codes)]
            daily = load_daily(conn, codes)
        else:
            print("  Loading shares from eps_baostock_raw...", flush=True)
            shares = load_shares(conn)
            print(f"  Loading {n_daily:,} daily rows...", flush=True)
            daily = load_daily(conn)

    print("  Running PIT as-of join...", flush=True)
    result = compute_pit_mktcap(daily, shares)

    # §5 check 1: cartesian product guard
    validate_row_count(result, daily)

    missing_rate = result["mktcap_missing"].mean()
    print(f"  mktcap_missing: {missing_rate:.1%} of rows", flush=True)
    print(f"  liqa_missing:   {result['liqa_missing'].mean():.1%} of rows", flush=True)

    # Write output
    print("  Writing daily_mktcap table...", flush=True)
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("DELETE FROM daily_mktcap")
        result.to_sql(
            "daily_mktcap", conn,
            if_exists="append",
            index=False,
            chunksize=50_000,
        )

    # §5 checks 2 and 3
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        bench_ok = validate_benchmarks(conn)
        validate_pit_jump(conn)

    if not bench_ok:
        print("  [WARN] Benchmark mismatch — check close field or total_share units", flush=True)

    print(f"\n  Written: {len(result):,} rows to daily_mktcap", flush=True)
    print("[OK] 16_pit_marketcap.py completed", flush=True)


if __name__ == "__main__":
    main()
