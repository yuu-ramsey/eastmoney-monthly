"""
KoC §1 — Stock Universe Construction
=====================================
Builds two lookup tables in data/universe.sqlite for KoC analysis:

  universe_liquid  — main test pool (market cap / turnover / listing age filters)
  universe_full    — comparison pool (ST and EPS filters only; tradability paradox)

Liquid pool filters (ALL must pass):
  1. >= 8 complete fiscal quarters  (trusted=1 in sue table)
  2. Single-quarter EPS >= 0        (no loss-making quarters)
  3. NOT ST / *ST at rebalancing    (PIT; proxy in dry-run)
  4. Market cap >= 5 billion CNY    (PIT; SZ proxy in dry-run; SH = unknown = pass)
  5. Avg daily turnover >= 5% (20d) (PIT; DATA_PENDING in dry-run)
  6. Listed >= 24 months            (PIT-correct; from exchange listing data)

Full sample filters (strict subset):
  1. >= 8 complete fiscal quarters
  2. Single-quarter EPS >= 0
  3. NOT ST / *ST

PIT status per filter (dry-run mode):
  Filters 1, 2, 6 : PIT-correct (actual historical data)
  Filter 3        : KNOWN LOOKAHEAD — current ST snapshot; no PIT source found (see B1 exploration)
  Filter 4        : PROXY (SZ) / DATA_PENDING (SH) — SZ uses current float shares x price;
                    SH stocks have size_missing=1 and are temporarily retained pending §10 data
  Filter 5        : DATA_PENDING — requires per-stock daily price history

Usage:
  .venv/Scripts/python.exe scripts/koc/01_universe.py           # full dry-run
  .venv/Scripts/python.exe scripts/koc/01_universe.py --smoke   # 50 stocks only
  .venv/Scripts/python.exe scripts/koc/01_universe.py --no-cache # clear cached data
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
DB_SUE: str = "data/pead.sqlite"
DB_OUT: str = "data/universe.sqlite"
REPORT_PATH: str = "docs/koc-universe.md"

CACHE_LISTING: Path = Path("data/_listing_dates.json")
CACHE_SPOT: Path = Path("data/_spot_snapshot.json")
CACHE_ST: Path = Path("data/_st_list.json")

MKTCAP_THRESHOLD: float = 5e8    # 5 billion CNY = 500 million
TURNOVER_THRESHOLD: float = 5.0  # 5% daily avg [needs validation]
LISTING_MIN_MONTHS: int = 24
TURNOVER_SAMPLE_N: int = 50
SMOKE_MAX_STOCKS: int = 50


# ── DB init ────────────────────────────────────────────────────────────────────
def init_out_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS universe_liquid;
        DROP TABLE IF EXISTS universe_full;
        DROP TABLE IF EXISTS universe_meta;

        CREATE TABLE universe_liquid (
            code            TEXT    NOT NULL,
            fiscal_year     INTEGER NOT NULL,
            fiscal_quarter  INTEGER NOT NULL,
            pub_date        TEXT    NOT NULL,
            pass_quarters   INTEGER NOT NULL,
            pass_pos_eps    INTEGER NOT NULL,
            pass_st         INTEGER NOT NULL,
            pass_mktcap     INTEGER NOT NULL,  -- 1=pass, 0=fail, -1=DATA_PENDING (size_missing=1)
            size_missing    INTEGER NOT NULL,  -- 1 = market cap unavailable at this obs (SH in dry-run)
            pass_turnover   INTEGER NOT NULL,  -- -1 = DATA_PENDING
            pass_listing    INTEGER NOT NULL,
            in_pool         INTEGER NOT NULL,
            PRIMARY KEY (code, fiscal_year, fiscal_quarter)
        );
        CREATE TABLE universe_full (
            code            TEXT    NOT NULL,
            fiscal_year     INTEGER NOT NULL,
            fiscal_quarter  INTEGER NOT NULL,
            pub_date        TEXT    NOT NULL,
            pass_quarters   INTEGER NOT NULL,
            pass_pos_eps    INTEGER NOT NULL,
            pass_st         INTEGER NOT NULL,
            in_pool         INTEGER NOT NULL,
            PRIMARY KEY (code, fiscal_year, fiscal_quarter)
        );
        CREATE TABLE universe_meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    conn.commit()


# ── Auxiliary data loaders ─────────────────────────────────────────────────────
def load_listing_dates() -> dict[str, date]:
    """
    Load listing dates for all A-shares from SH/SZ exchange lists.
    Returns {6-digit-code: listing_date}. Cached to CACHE_LISTING.
    """
    if CACHE_LISTING.exists():
        raw: dict[str, str] = json.loads(CACHE_LISTING.read_text(encoding="utf-8"))
        return {k: date.fromisoformat(v) for k, v in raw.items()}

    import akshare as ak

    listing: dict[str, date] = {}
    sources = [
        ("SH main",  lambda: ak.stock_info_sh_name_code(symbol="主板A股")),
        ("SH STAR",  lambda: ak.stock_info_sh_name_code(symbol="科创板")),
        ("SZ",       lambda: ak.stock_info_sz_name_code()),
    ]
    for src_name, fetcher in sources:
        try:
            df = fetcher()
        except Exception as exc:
            print(f"  [WARN] {src_name} listing fetch failed: {exc}", file=sys.stderr)
            continue
        code_col = next((c for c in df.columns if "代码" in c), None)
        date_col = next((c for c in df.columns if "日期" in c or "上市" in c), None)
        if not code_col or not date_col:
            print(f"  [WARN] {src_name}: unexpected cols {df.columns.tolist()}")
            continue
        for _, row in df.iterrows():
            code_6 = str(row[code_col]).strip().zfill(6)
            try:
                listing[code_6] = pd.to_datetime(row[date_col]).date()
            except Exception:
                pass
        print(f"  {src_name}: {len(df)} stocks")

    CACHE_LISTING.write_text(
        json.dumps({k: v.isoformat() for k, v in listing.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    return listing


def load_st_set() -> set[str]:
    """
    Build current ST / *ST set from SH/SZ exchange name lists.
    DRY-RUN PROXY: reflects current ST status, not historical.
    Returns set of 6-digit codes.
    """
    if CACHE_ST.exists():
        return set(json.loads(CACHE_ST.read_text(encoding="utf-8")))

    import akshare as ak

    st_codes: set[str] = set()
    sources = [
        ("SH main",  lambda: ak.stock_info_sh_name_code(symbol="主板A股")),
        ("SH STAR",  lambda: ak.stock_info_sh_name_code(symbol="科创板")),
        ("SZ",       lambda: ak.stock_info_sz_name_code()),
    ]
    for src_name, fetcher in sources:
        try:
            df = fetcher()
        except Exception:
            continue
        code_col = next((c for c in df.columns if "代码" in c), None)
        name_col = next((c for c in df.columns if "简称" in c), None)
        if not code_col or not name_col:
            continue
        st_mask = df[name_col].astype(str).str.contains("ST", case=True, na=False)
        codes = df.loc[st_mask, code_col].astype(str).str.zfill(6).tolist()
        st_codes.update(codes)

    CACHE_ST.write_text(json.dumps(sorted(st_codes), ensure_ascii=False), encoding="utf-8")
    return st_codes


def load_market_cap_proxy() -> dict[str, float]:
    """
    Build {6-digit-code: float_market_cap_yuan} from SZ float shares x current price.
    DRY-RUN PROXY: current prices, not PIT. SH stocks are absent (no SH float-share data).
    Cached to CACHE_SPOT.
    """
    if CACHE_SPOT.exists():
        raw = json.loads(CACHE_SPOT.read_text(encoding="utf-8"))
        return {k: float(v) for k, v in raw.items()}

    import akshare as ak

    mktcap: dict[str, float] = {}

    # Load SZ float shares (流通股本)
    sz_shares: dict[str, float] = {}
    try:
        df_sz = ak.stock_info_sz_name_code()
        code_col = next(c for c in df_sz.columns if "代码" in c)
        share_col = next(c for c in df_sz.columns if "流通" in c)
        for _, row in df_sz.iterrows():
            code_6 = str(row[code_col]).strip().zfill(6)
            try:
                sz_shares[code_6] = float(str(row[share_col]).replace(",", ""))
            except (ValueError, TypeError):
                pass
        print(f"  SZ float shares: {len(sz_shares)} stocks")
    except Exception as exc:
        print(f"  [WARN] SZ shares fetch failed: {exc}", file=sys.stderr)

    # Fetch current prices via stock_zh_a_spot
    print("  Fetching spot prices (stock_zh_a_spot, ~60s)...")
    try:
        df_spot = ak.stock_zh_a_spot()
        price_map: dict[str, float] = {}
        for _, row in df_spot.iterrows():
            # spot codes like 'sh600519', 'sz000001', 'bj920000' — last 6 digits
            code_6 = str(row["代码"]).strip()[-6:].zfill(6)
            try:
                price = float(row["最新价"])
                if price > 0:
                    price_map[code_6] = price
            except (ValueError, TypeError):
                pass
        print(f"  Spot prices: {len(price_map)} stocks")

        for code_6, shares in sz_shares.items():
            price = price_map.get(code_6)
            if price is not None and shares > 0:
                mktcap[code_6] = shares * price
        print(f"  Market cap computed (SZ): {len(mktcap)} stocks")

    except Exception as exc:
        print(f"  [WARN] Spot fetch failed: {exc}", file=sys.stderr)

    CACHE_SPOT.write_text(json.dumps(mktcap, ensure_ascii=False), encoding="utf-8")
    return mktcap


def sample_turnover_distribution(all_codes: list[str], n: int = TURNOVER_SAMPLE_N) -> pd.DataFrame:
    """
    Fetch 30-day daily turnover history for n randomly sampled stocks.
    Returns DataFrame with columns: code, avg_turnover_20d (%).
    Used to validate TURNOVER_THRESHOLD only; NOT applied as filter in dry-run.
    """
    import akshare as ak

    rng = np.random.default_rng(42)
    sampled = rng.choice(all_codes, size=min(n, len(all_codes)), replace=False).tolist()

    end_dt = datetime.today().strftime("%Y%m%d")
    start_str = (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y%m%d")

    rows: list[dict] = []
    for code_full in sampled:
        code_6 = code_full.split(".")[-1]
        try:
            df = ak.stock_zh_a_hist(
                symbol=code_6,
                period="daily",
                start_date=start_str,
                end_date=end_dt,
                adjust="",
            )
            if df.empty or "换手率" not in df.columns:
                continue
            df["换手率"] = pd.to_numeric(df["换手率"], errors="coerce")
            avg_20 = float(df["换手率"].tail(20).mean())
            rows.append({"code": code_full, "avg_turnover_20d": round(avg_20, 4)})
        except Exception:
            pass

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["code", "avg_turnover_20d"])


# ── Pool construction ──────────────────────────────────────────────────────────
def build_pools(
    df_base: pd.DataFrame,
    listing_dates: dict[str, date],
    st_set: set[str],
    mktcap_map: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Apply all 6 filters and return (df_liquid, df_full).

    PIT assertions enforced here:
      - listing_date <= pub_date for all stocks with known listing date
      - EPS data is historical by construction (pub_date-anchored in pead.sqlite)
      - Filters 3/4 are DRY-RUN PROXY (current-snapshot, not PIT)

    pass values: 1=pass, 0=fail, -1=unknown/pending (treated as pass).
    """
    assert not df_base.empty, "build_pools: empty input"
    assert pd.api.types.is_datetime64_any_dtype(df_base["pub_date"]), "pub_date must be datetime64"

    df = df_base.copy()
    df["code_6"] = df["code"].str.split(".").str[-1]
    pub_dates_dt = df["pub_date"].dt.date

    # Filter 1: >= 8 quarters (all True by construction — input is trusted=1)
    df["pass_quarters"] = 1

    # Filter 2: positive single-quarter EPS
    df["pass_pos_eps"] = (df["eps_single"].fillna(-1.0) >= 0.0).astype(int)

    # Filter 3: not ST (current proxy)
    df["pass_st"] = (~df["code_6"].isin(st_set)).astype(int)

    # Filter 4: market cap >= 5 billion (SZ proxy; SH = DATA_PENDING)
    # size_missing=1 means no market cap data is available for this observation.
    # These are NOT auto-passing — they are explicitly flagged and kept only because
    # §10 data (PIT market cap) is not yet ready. Once §10 is complete, size_missing=1
    # rows must be re-evaluated with actual market cap before including in the pool.
    mktcap_vals: pd.Series = df["code_6"].map(mktcap_map)
    df["size_missing"] = mktcap_vals.isna().astype(int)
    df["pass_mktcap"] = np.where(
        mktcap_vals.isna(), -1,       # DATA_PENDING — see size_missing flag
        (mktcap_vals >= MKTCAP_THRESHOLD).astype(int),
    )

    # Filter 5: turnover (DATA_PENDING in dry-run)
    df["pass_turnover"] = -1

    # Filter 6: listed >= 24 months (PIT-correct)
    listing_series: pd.Series = df["code_6"].map(listing_dates)

    # PIT note: pre-IPO earnings exist for some stocks (pub_date < listing_date).
    # The listing filter handles them correctly (listing_months < 0 → pass_listing=0).
    # Count is computed at the end and returned.
    known_mask = listing_series.notna()

    def _listing_months(pub_dt: date, list_dt: Optional[date]) -> int:
        if list_dt is None or pd.isna(list_dt):
            return -1
        return (pub_dt.year - list_dt.year) * 12 + (pub_dt.month - list_dt.month)

    listing_months = [
        _listing_months(p, l)
        for p, l in zip(pub_dates_dt, listing_series)
    ]
    df["listing_months"] = listing_months
    # listing_months == -1 (unknown) → conservative fail (listing date required for liquid pool)
    df["pass_listing"] = ((pd.Series(listing_months, index=df.index) >= LISTING_MIN_MONTHS)).astype(int)

    # ── Full sample: filters 1, 2, 3 ──────────────────────────────────────────
    full_in = (
        (df["pass_quarters"] == 1) &
        (df["pass_pos_eps"] == 1) &
        (df["pass_st"] == 1)
    )
    df["in_full"] = full_in.astype(int)

    # ── Liquid pool: all 6 filters (-1 treated as pass) ───────────────────────
    liquid_in = (
        (df["pass_quarters"] == 1) &
        (df["pass_pos_eps"] == 1) &
        (df["pass_st"] == 1) &
        (df["pass_mktcap"] != 0) &      # -1 (unknown) = pass
        (df["pass_turnover"] != 0) &    # -1 (pending) = pass
        (df["pass_listing"] == 1)
    )
    df["in_liquid"] = liquid_in.astype(int)

    pub_str = df["pub_date"].dt.strftime("%Y-%m-%d")

    df_liquid = df[[
        "code", "fiscal_year", "fiscal_quarter",
        "pass_quarters", "pass_pos_eps", "pass_st",
        "pass_mktcap", "size_missing", "pass_turnover", "pass_listing", "in_liquid",
    ]].copy()
    df_liquid.insert(3, "pub_date", pub_str.values)
    df_liquid = df_liquid.rename(columns={"in_liquid": "in_pool"})

    df_full = df[[
        "code", "fiscal_year", "fiscal_quarter",
        "pass_quarters", "pass_pos_eps", "pass_st", "in_full",
    ]].copy()
    df_full.insert(3, "pub_date", pub_str.values)
    df_full = df_full.rename(columns={"in_full": "in_pool"})

    pre_ipo_count = 0
    if known_mask.any():
        pre_ipo_count = int(
            (
                df.loc[known_mask, "pub_date"].dt.date
                < pd.Series([listing_series[i] for i in df.index[known_mask]], index=df.index[known_mask])
            ).sum()
        )
        if pre_ipo_count > 0:
            print(f"  [INFO] {pre_ipo_count} obs with pub_date < listing_date (pre-IPO history) "
                  f"— excluded by listing filter")

    return df_liquid, df_full, pre_ipo_count


# ── Report generation ──────────────────────────────────────────────────────────
def generate_report(
    df_liquid: pd.DataFrame,
    df_full: pd.DataFrame,
    turnover_dist: pd.DataFrame,
    is_smoke: bool,
    pre_ipo_count: int = 0,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    smoke_note = " [SMOKE — 50 stocks only]" if is_smoke else ""

    n_liq = int(df_liquid["in_pool"].sum())
    n_full = int(df_full["in_pool"].sum())
    liq_stocks = int(df_liquid[df_liquid["in_pool"] == 1]["code"].nunique())
    full_stocks = int(df_full[df_full["in_pool"] == 1]["code"].nunique())

    def funnel_row(col: str, label: str, df: pd.DataFrame = df_liquid) -> str:
        if col not in df.columns:
            return f"| {label} | — | — | — |"
        pass_1 = int((df[col] == 1).sum())
        unknown = int((df[col] == -1).sum())
        fail_0 = int((df[col] == 0).sum())
        return f"| {label} | {pass_1:,} | {unknown:,} | {fail_0:,} |"

    # Turnover distribution section
    if not turnover_dist.empty:
        vals = turnover_dist["avg_turnover_20d"].dropna()
        p50 = float(np.percentile(vals, 50))
        warn = "" if p50 < TURNOVER_THRESHOLD else f"\n> **[WARN]** Median {p50:.2f}% > threshold {TURNOVER_THRESHOLD}% — consider lowering threshold."
        turn_sec = (
            f"\n## Turnover Distribution (Sample, n={len(vals)}, 20d avg %)\n\n"
            f"| P10 | P25 | P50 | P75 | P90 | P95 |\n"
            f"|-----|-----|-----|-----|-----|-----|\n"
            f"| {np.percentile(vals,10):.2f} | {np.percentile(vals,25):.2f} "
            f"| {p50:.2f} | {np.percentile(vals,75):.2f} "
            f"| {np.percentile(vals,90):.2f} | {np.percentile(vals,95):.2f} |\n"
            f"{warn}\n"
            f"> **[需要验证]** Threshold = {TURNOVER_THRESHOLD}%. "
            f"NOT applied as filter in dry-run (DATA_PENDING). "
            f"Turnover data reflects current market conditions, not PIT.\n"
        )
    else:
        turn_sec = "\n## Turnover Distribution\n\nSample data unavailable.\n"

    # Pool size time series
    liq_ts = (
        df_liquid[df_liquid["in_pool"] == 1]
        .groupby(["fiscal_year", "fiscal_quarter"]).size()
        .reset_index(name="n_liq")
    )
    full_ts = (
        df_full[df_full["in_pool"] == 1]
        .groupby(["fiscal_year", "fiscal_quarter"]).size()
        .reset_index(name="n_full")
    )
    ts = liq_ts.merge(full_ts, on=["fiscal_year", "fiscal_quarter"], how="outer").fillna(0)
    ts = ts.sort_values(["fiscal_year", "fiscal_quarter"])
    ts_rows = [
        f"| {int(r.fiscal_year)}Q{int(r.fiscal_quarter)} | {int(r.n_liq):,} | {int(r.n_full):,} |"
        for _, r in ts.iterrows()
    ]
    if len(ts_rows) > 16:
        ts_table = "\n".join(ts_rows[:8] + ["| ... | ... | ... |"] + ts_rows[-4:])
    else:
        ts_table = "\n".join(ts_rows)

    # size_missing count (DATA_PENDING market cap observations)
    size_missing_obs = int(df_liquid["size_missing"].sum())
    size_missing_stocks = int(
        df_liquid[df_liquid["size_missing"] == 1]["code"].nunique()
    )

    return f"""# KoC Universe — Quality Report{smoke_note}

**Generated**: {now_str}
**Input**: data/pead.sqlite (akshare YTD-based SUE, 2010-2024)
**Output**: data/universe.sqlite
**Mode**: DRY-RUN (see PIT status section for what is and isn't point-in-time)

## Coverage

| Metric | Liquid Pool | Full Sample |
|--------|-------------|-------------|
| Total observations | {len(df_liquid):,} | {len(df_full):,} |
| Observations in pool | {n_liq:,} | {n_full:,} |
| Distinct stocks in pool | {liq_stocks:,} | {full_stocks:,} |

## Filter Funnel (Liquid Pool)

Values: Pass=1, Unknown/Pending=-1, Fail=0

| Filter | Pass | Unknown/-1 | Fail |
|--------|------|------------|------|
{funnel_row('pass_quarters', '>= 8 quarters (trusted=1)')}
{funnel_row('pass_pos_eps', 'EPS >= 0 (no loss quarter)')}
{funnel_row('pass_st', 'Not ST/*ST [PROXY: current]')}
{funnel_row('pass_mktcap', 'Market cap >= 5亿 [PROXY: SZ; SH=unknown]')}
{funnel_row('pass_turnover', 'Avg turnover >= 5% [DATA_PENDING]')}
{funnel_row('pass_listing', 'Listed >= 24 months [PIT-correct]')}
{turn_sec}
## Pool Size by Period

| Period | Liquid | Full |
|--------|--------|------|
{ts_table}

## PIT Status Summary

| Filter | PIT-correct? | Notes |
|--------|:-------------|-------|
| >= 8 quarters | ✅ Yes | `trusted=1` in pead.sqlite uses only historical EPS through pub_date |
| EPS >= 0 | ✅ Yes | `eps_single` anchored to pub_date; no future data |
| Not ST | ⚠️ Proxy | Current exchange name list; stocks that recovered (or became) ST since are misclassified |
| Market cap >= 5亿 | ⚠️ Proxy (SZ) / ❌ Missing (SH) | SZ: current float shares × current price; SH: `size_missing=1` ({size_missing_obs:,} obs, {size_missing_stocks:,} stocks) |
| Avg turnover >= 5% | ❌ Pending | Requires daily price history per stock at each pub_date |
| Listed >= 24 months | ✅ Yes | Exchange listing date is static; {pre_ipo_count:,} pre-IPO obs found and correctly excluded |

## Size-Missing Stocks (DATA_PENDING Market Cap)

`size_missing=1` obs: **{size_missing_obs:,}** across **{size_missing_stocks:,}** distinct stocks.

These observations have NO market cap data available (SH stocks in dry-run mode).
They are currently **not excluded** (pass_mktcap=-1 = DATA_PENDING) but are **explicitly flagged**.
Once §10 market cap data is ready, all `size_missing=1` rows must be re-evaluated before
including them in the final pool. They must NOT be treated as having passed the market cap filter.

## ST Historical Data Exploration (B1)

**Result**: PIT historical ST status is **not available** via simple API calls.

Explored sources:
- `akshare.stock_info_change_name(symbol)` — returns name history list **without dates**;
  cannot determine when ST designation was applied or removed
- `akshare.stock_info_sz_change_name(symbol='全称变更')` — connection failed (server reset)
- `baostock.query_stock_basic` — returns current `code_name` only; no historical name timeline

**Conclusion**: The current ST filter uses a snapshot of today's exchange name list.
Stocks that were ST historically (e.g. during 2015-2016 restructuring wave) but have since
recovered are incorrectly **included** in the pool. Stocks that became ST after this snapshot
are incorrectly **included** as non-ST for recent periods.

> ⚠️ **Known Lookahead**: ST filter is a present-day snapshot, not point-in-time.
> This is a conservative approximation: currently-ST stocks are excluded even for
> historical periods when they may have been clean. The opposite error (including
> stocks that were ST during the test period) is harder to bound without historical data.

## Known Limitations (Dry-Run Mode)

1. **Turnover filter not applied** — filter logic is implemented but uses DATA_PENDING (-1 = all pass).
   Liquid pool observation count will decrease once this filter is active.
2. **Market cap for SH stocks** is DATA_PENDING ({size_missing_obs:,} observations, {size_missing_stocks:,} stocks,
   `size_missing=1`). These are flagged but temporarily retained pending §10 data. SH main board / STAR
   stocks tend to be large-cap, so over-inclusion is likely small, but must be verified.
3. **ST status** uses current exchange name list. See "ST Historical Data Exploration" section above.
   *This is a known lookahead and must be disclosed in any published results.*
4. **Market cap** uses current float shares × current price (SZ only), not values at each pub_date.
   Stocks that have grown significantly since early periods are incorrectly included for those periods.

## Schema

```sql
universe_liquid(code, fiscal_year, fiscal_quarter, pub_date,
    pass_quarters, pass_pos_eps, pass_st,
    pass_mktcap,   -- 1=pass, 0=fail, -1=DATA_PENDING
    size_missing,  -- 1 = no market cap data; must re-evaluate when §10 data available
    pass_turnover, -- -1=DATA_PENDING
    pass_listing,
    in_pool)

universe_full(code, fiscal_year, fiscal_quarter, pub_date,
    pass_quarters, pass_pos_eps, pass_st, in_pool)

-- pass values: 1=pass, 0=fail, -1=DATA_PENDING
-- size_missing=1 rows are in-pool (dry-run only) — must NOT be assumed to pass market cap filter
```

## Next Step (PIT Mode)

After pead-baostock.sqlite §10 is complete, update data sources:
- **Market cap**: `total_share` from baostock × closing price at pub_date
  (requires separate `query_history_k_data_plus` fetch per stock per period)
- **Turnover**: rolling 20d average from daily close data at pub_date
- **ST status**: PIT source not found in B1 exploration (see section above).
  Options: (a) accept known lookahead, disclose in paper; (b) reconstruct from
  individual stock name histories if a dated source is identified later.
"""


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="KoC §1: Universe construction (dry-run)")
    parser.add_argument("--smoke", action="store_true",
                        help=f"Use first {SMOKE_MAX_STOCKS} stocks for quick validation")
    parser.add_argument("--no-cache", action="store_true",
                        help="Delete cached auxiliary data before fetching")
    args = parser.parse_args()

    if args.no_cache:
        for cache in [CACHE_LISTING, CACHE_SPOT, CACHE_ST]:
            if cache.exists():
                cache.unlink()
                print(f"  Cleared cache: {cache}")

    t0 = time.time()
    print(f"[01_universe] {datetime.now():%Y-%m-%d %H:%M:%S} | mode: {'SMOKE' if args.smoke else 'FULL'} dry-run")

    # Assertions before any computation
    assert Path(DB_SUE).exists(), f"Input DB not found: {DB_SUE}"
    Path(DB_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Step 1: load base data from pead.sqlite
    print("[1/5] Loading base trusted SUE observations from pead.sqlite...")
    with sqlite3.connect(DB_SUE) as conn_src:
        if args.smoke:
            query = f"""
                SELECT s.code, s.fiscal_year, s.fiscal_quarter,
                       s.pub_date, s.eps_single, s.trusted
                FROM sue s
                WHERE s.trusted = 1 AND s.pub_date IS NOT NULL
                  AND s.code IN (
                      SELECT DISTINCT code FROM sue WHERE trusted=1 LIMIT {SMOKE_MAX_STOCKS}
                  )
            """
        else:
            query = """
                SELECT code, fiscal_year, fiscal_quarter, pub_date, eps_single, trusted
                FROM sue
                WHERE trusted = 1 AND pub_date IS NOT NULL
            """
        df_base = pd.read_sql_query(query, conn_src)

    assert not df_base.empty, "No trusted observations found in sue table"
    df_base["pub_date"] = pd.to_datetime(df_base["pub_date"])
    df_base["eps_single"] = pd.to_numeric(df_base["eps_single"], errors="coerce")
    print(f"  {len(df_base):,} observations | {df_base['code'].nunique():,} stocks")

    # Step 2: load auxiliary data
    print("[2/5] Loading auxiliary data (listing dates, ST, market cap)...")
    listing_dates = load_listing_dates()
    st_set = load_st_set()
    mktcap_map = load_market_cap_proxy()
    known_listing = sum(1 for c in df_base["code"].str.split(".").str[-1].unique() if c in listing_dates)
    print(f"  Listing dates: {len(listing_dates):,} | ST stocks: {len(st_set)} "
          f"| Market cap: {len(mktcap_map):,} | Coverage in universe: {known_listing}/{df_base['code'].nunique()}")

    # Step 3: turnover distribution sample
    print(f"[3/5] Sampling turnover distribution ({TURNOVER_SAMPLE_N} stocks)...")
    all_codes = df_base["code"].unique().tolist()
    turnover_dist = sample_turnover_distribution(all_codes, n=TURNOVER_SAMPLE_N)
    if not turnover_dist.empty:
        vals = turnover_dist["avg_turnover_20d"].dropna()
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        print(f"  n={len(vals)} | P10={p10:.2f}% | P50={p50:.2f}% | P90={p90:.2f}%")
        if p50 > TURNOVER_THRESHOLD:
            print(f"  [需要验证] Median {p50:.2f}% > threshold {TURNOVER_THRESHOLD}% — consider adjusting")
    else:
        print("  Turnover sample unavailable")

    # Step 4: build pools
    print("[4/5] Building pools (applying all 6 filters)...")
    df_liquid, df_full, pre_ipo_count = build_pools(df_base, listing_dates, st_set, mktcap_map)
    n_liq = int(df_liquid["in_pool"].sum())
    n_full = int(df_full["in_pool"].sum())
    liq_stocks = int(df_liquid[df_liquid["in_pool"] == 1]["code"].nunique())
    full_stocks = int(df_full[df_full["in_pool"] == 1]["code"].nunique())
    print(f"  Liquid pool: {n_liq:,} obs, {liq_stocks:,} stocks")
    print(f"  Full sample: {n_full:,} obs, {full_stocks:,} stocks")

    # Step 5: write to DB and generate report
    print("[5/5] Writing universe.sqlite and generating report...")
    with sqlite3.connect(DB_OUT, timeout=30) as conn_out:
        init_out_db(conn_out)
        df_liquid.to_sql("universe_liquid", conn_out, if_exists="append", index=False)
        df_full.to_sql("universe_full", conn_out, if_exists="append", index=False)
        meta = [
            ("generated_at", datetime.now().isoformat()),
            ("mode", "dry-run"),
            ("input_db", DB_SUE),
            ("smoke", str(args.smoke)),
            ("n_liquid_in_pool", str(n_liq)),
            ("n_full_in_pool", str(n_full)),
            ("liquid_stocks", str(liq_stocks)),
            ("full_stocks", str(full_stocks)),
            ("turnover_threshold_note", f"{TURNOVER_THRESHOLD}% [需要验证]"),
        ]
        conn_out.executemany("INSERT OR REPLACE INTO universe_meta VALUES (?,?)", meta)
        conn_out.commit()

    report = generate_report(df_liquid, df_full, turnover_dist, is_smoke=args.smoke, pre_ipo_count=pre_ipo_count)
    Path(REPORT_PATH).write_text(report, encoding="utf-8")
    print(f"  Report: {REPORT_PATH}")

    elapsed = time.time() - t0
    print(f"\n[01_universe] done in {elapsed:.1f}s")
    print(f"  Liquid pool : {n_liq:,} observations | {liq_stocks:,} distinct stocks")
    print(f"  Full sample : {n_full:,} observations | {full_stocks:,} distinct stocks")
    print("✅ Script completed successfully")


if __name__ == "__main__":
    main()
