"""
Build clean SUE on Baostock single-quarter EPS with KoC §2 full spec:
  - Expected EPS: SRW+AR(1), phi point-in-time rolling (no lookahead)
  - Denominator: mean(|surprise|, past 8 quarters)  — NOT std()
  - trusted: >= 8 quarters of single EPS history  AND  >= 8 surprise history
  - phi falls back to 0 when rolling window < 16 qtrs or |phi| < 0.2

Output: data/pead-baostock.sqlite  (table: sue_baostock)
Usage:  python scripts/koc/14_sue_baostock.py

Distribution check at end (per pead-data.md quality gates):
  SUE mean ~ 0, SUE std in [0.5, 10], min stocks/quarter >= 50.
"""
import sqlite3
import sys
from collections import defaultdict
from typing import Optional

import numpy as np

# ── Guards ─────────────────────────────────────────────────────────────────────
DB_PATH: str = "data/pead-baostock.sqlite"
MIN_HISTORY_Q: int = 8         # quarters of single EPS before event is trusted
MIN_PHI_WINDOW: int = 16       # rolling window for AR(1) phi estimation
PHI_MIN_ABS: float = 0.2       # phi below this threshold -> set phi = 0 (pure SRW)
MIN_SURPRISE_DENOM: float = 1e-6  # floor for denominator to avoid divide-by-zero

assert MIN_HISTORY_Q >= 8, "MIN_HISTORY_Q must be at least 8 to match pead.sqlite trusted flag"
assert MIN_PHI_WINDOW > MIN_HISTORY_Q, "phi window must exceed history requirement"


# ── DB helpers ─────────────────────────────────────────────────────────────────
def init_output_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sue_baostock (
            code            TEXT    NOT NULL,
            fiscal_year     INTEGER NOT NULL,
            fiscal_quarter  INTEGER NOT NULL,
            pub_date        TEXT,
            stat_date       TEXT,
            sue             REAL,
            eps_single      REAL,
            expected_eps    REAL,
            surprise        REAL,
            denom           REAL,
            phi             REAL,
            trusted         INTEGER,  -- 1 if >=8q history, 0 otherwise
            PRIMARY KEY (code, fiscal_year, fiscal_quarter)
        )
    """)
    conn.commit()


def load_single_eps(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Return {code: [sorted records]} ordered by (fiscal_year, fiscal_quarter)."""
    rows = conn.execute(
        "SELECT code, fiscal_year, fiscal_quarter, stat_date, pub_date, eps_single "
        "FROM eps_baostock_single "
        "ORDER BY code, fiscal_year, fiscal_quarter"
    ).fetchall()
    by_code: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_code[r[0]].append({
            "code": r[0], "fy": r[1], "fq": r[2],
            "stat_date": r[3], "pub_date": r[4], "eps_single": r[5],
        })
    return dict(by_code)


# ── AR(1) phi estimation ───────────────────────────────────────────────────────
def estimate_phi_rolling(eps_history: list[float],
                         mu_history: list[float]) -> float:
    """Estimate AR(1) phi from rolling window: (eps_{t} - mu_{t}) vs (eps_{t-1} - mu_{t-1}).

    Returns phi in [-1, 1]. Falls back to 0 if < MIN_PHI_WINDOW pairs or weak fit.
    Pure OLS on residuals: phi = Cov(y_t, y_{t-1}) / Var(y_{t-1})
    """
    if len(eps_history) < MIN_PHI_WINDOW:
        return 0.0

    # Use last MIN_PHI_WINDOW observations
    eps_w = np.array(eps_history[-MIN_PHI_WINDOW:])
    mu_w = np.array(mu_history[-MIN_PHI_WINDOW:])
    resid = eps_w - mu_w

    y = resid[1:]
    x = resid[:-1]
    if len(x) < 4:
        return 0.0

    var_x = float(np.var(x))
    if var_x < 1e-12:
        return 0.0
    phi = float(np.cov(x, y)[0, 1] / var_x)
    phi = max(-1.0, min(1.0, phi))  # clip to stationary range
    if abs(phi) < PHI_MIN_ABS:
        return 0.0
    return phi


# ── Seasonal mean (μ_s) helper ─────────────────────────────────────────────────
def seasonal_mean(history: list[dict], target_fq: int,
                  n_seasons: int = 8) -> Optional[float]:
    """Mean of same fiscal-quarter EPS over up to n_seasons prior years."""
    same_q = [r["eps_single"] for r in history
               if r["fq"] == target_fq and r["eps_single"] is not None]
    if not same_q:
        return None
    return float(np.mean(same_q[-n_seasons:]))


# ── Per-stock SUE computation ──────────────────────────────────────────────────
def compute_sue_for_stock(records: list[dict]) -> list[dict]:
    """Compute SUE for one stock's time series.

    Algorithm (point-in-time at each record):
      1. Look back at all prior quarters (strictly before current pub_date)
      2. Compute mu_s = seasonal mean for this quarter (last 8 same-quarter values)
      3. Compute phi via rolling AR(1) on seasonal residuals (last 16 quarters)
      4. E[eps_t] = mu_s + phi * (eps_{t-1} - mu_{s,prev})
      5. surprise = eps_t - E[eps_t]
      6. denom = mean(|surprise_k|) for k in last 8 known surprises
      7. SUE = surprise / denom
    """
    results: list[dict] = []
    n = len(records)

    # Accumulate history up to (but not including) current record
    eps_seen: list[dict] = []          # all prior single EPS records
    surprise_seen: list[float] = []    # prior surprise values

    for i, rec in enumerate(records):
        fy, fq = rec["fy"], rec["fq"]
        eps_curr = rec["eps_single"]

        if eps_curr is None:
            eps_seen.append(rec)
            continue

        # History = strictly prior quarters (PIT)
        prior = eps_seen[:]  # snapshot before adding current
        trusted = len(prior) >= MIN_HISTORY_Q

        # --- Seasonal mean (mu_s) ---
        mu_s = seasonal_mean(prior, fq)
        if mu_s is None:
            # Not enough same-quarter history
            eps_seen.append(rec)
            continue

        # --- phi (AR(1) coefficient, rolling, PIT) ---
        # Build parallel lists of historical eps and their seasonal means
        eps_hist_vals: list[float] = []
        mu_hist_vals: list[float] = []
        for h in prior:
            if h["eps_single"] is None:
                continue
            mu_h = seasonal_mean(prior[:prior.index(h)], h["fq"])
            if mu_h is not None:
                eps_hist_vals.append(h["eps_single"])
                mu_hist_vals.append(mu_h)

        phi = estimate_phi_rolling(eps_hist_vals, mu_hist_vals)

        # --- E[eps_t] = mu_s + phi * (eps_{t-1} - mu_{s,t-1}) ---
        prev_recs = [h for h in prior if h["eps_single"] is not None]
        expected_eps = mu_s
        if prev_recs and abs(phi) >= PHI_MIN_ABS:
            prev = prev_recs[-1]
            mu_prev = seasonal_mean(prior[:prior.index(prev)], prev["fq"])
            if mu_prev is not None:
                expected_eps = mu_s + phi * (prev["eps_single"] - mu_prev)

        # --- Surprise ---
        surprise = eps_curr - expected_eps

        # Update surprise history BEFORE computing current SUE denom
        # (denom uses strictly prior surprises only)
        denom_surprises = surprise_seen[-MIN_HISTORY_Q:]
        if len(denom_surprises) >= 1:
            denom = float(np.mean([abs(s) for s in denom_surprises]))
            denom = max(denom, MIN_SURPRISE_DENOM)
        else:
            denom = None  # not enough surprise history yet

        if denom is not None and denom > MIN_SURPRISE_DENOM:
            sue = surprise / denom
        else:
            sue = None

        results.append({
            "code": rec["code"],
            "fy": fy, "fq": fq,
            "pub_date": rec["pub_date"],
            "stat_date": rec["stat_date"],
            "sue": sue,
            "eps_single": eps_curr,
            "expected_eps": round(expected_eps, 6),
            "surprise": round(surprise, 6),
            "denom": round(denom, 6) if denom else None,
            "phi": round(phi, 4),
            "trusted": 1 if (trusted and sue is not None) else 0,
        })

        # Accumulate for next iteration
        eps_seen.append(rec)
        surprise_seen.append(surprise)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        n_single = conn.execute(
            "SELECT COUNT(*) FROM eps_baostock_single"
        ).fetchone()[0]
    assert n_single > 0, (
        "eps_baostock_single is empty — run 11_baostock_single.py first"
    )
    print(f"  eps_baostock_single: {n_single:,} rows")

    with sqlite3.connect(DB_PATH) as conn:
        init_output_table(conn)
        by_code = load_single_eps(conn)

    print(f"  Stocks: {len(by_code)}")
    print("  Computing SUE (SRW+AR(1), mean|surprise| denom, point-in-time phi)...")

    all_results: list[dict] = []
    for code, records in by_code.items():
        stock_results = compute_sue_for_stock(records)
        all_results.extend(stock_results)

    buffer = [
        (r["code"], r["fy"], r["fq"], r["pub_date"], r["stat_date"],
         r["sue"], r["eps_single"], r["expected_eps"], r["surprise"],
         r["denom"], r["phi"], r["trusted"])
        for r in all_results
    ]

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM sue_baostock")
        conn.executemany(
            "INSERT INTO sue_baostock "
            "(code, fiscal_year, fiscal_quarter, pub_date, stat_date, "
            "sue, eps_single, expected_eps, surprise, denom, phi, trusted) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            buffer,
        )
        conn.commit()
        n_total = conn.execute("SELECT COUNT(*) FROM sue_baostock").fetchone()[0]
        n_trusted = conn.execute(
            "SELECT COUNT(*) FROM sue_baostock WHERE trusted=1"
        ).fetchone()[0]
        stats = conn.execute(
            "SELECT AVG(sue), MIN(sue), MAX(sue), "
            "COUNT(DISTINCT fiscal_year || 'Q' || fiscal_quarter) "
            "FROM sue_baostock WHERE trusted=1"
        ).fetchone()

    print(f"\n  Total SUE records:   {n_total:,}")
    print(f"  Trusted SUE records: {n_trusted:,}")
    if stats[0] is not None:
        print(f"  SUE mean:  {stats[0]:.4f}  (target: ~0)")
        print(f"  SUE range: [{stats[1]:.2f}, {stats[2]:.2f}]")
        print(f"  Quarters with data: {stats[3]}")

    # Quality gates matching pead-data.md checks
    print("\n  Quality checks:")
    mean_ok = abs(stats[0] or 999) < 0.5
    print(f"  SUE mean ~ 0:          {'PASS' if mean_ok else 'WARN'}")

    with sqlite3.connect(DB_PATH) as conn:
        min_q = conn.execute(
            "SELECT MIN(cnt) FROM ("
            "  SELECT fiscal_year || 'Q' || fiscal_quarter as q, COUNT(*) as cnt "
            "  FROM sue_baostock WHERE trusted=1 GROUP BY q"
            ")"
        ).fetchone()[0]
    min_ok = (min_q is not None) and (min_q >= 50)
    print(f"  Min stocks/quarter >= 50: {'PASS' if min_ok else 'FAIL'} (min={min_q})")

    if not mean_ok:
        print("  [WARN] Large mean bias — check expected_eps model or unit mismatch")
    if not min_ok:
        print("  [FAIL] Insufficient cross-section — check coverage from fetch step")

    print("[OK] 14_sue_baostock.py completed")


if __name__ == "__main__":
    main()
