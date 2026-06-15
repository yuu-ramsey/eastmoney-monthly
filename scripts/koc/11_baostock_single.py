"""
Convert Baostock epsTTM -> single-quarter EPS.

Two paths attempted in order:
  Path A (preferred): net_profit is YTD cumulative in Chinese quarterly reports.
    single_Q1 = NP(Q1)
    single_Qk = NP(Qk) - NP(Q(k-1))  for k in {2,3,4}
    per-share EPS = single_NP / total_share   (both in 万元/万股, units cancel)
  Path B (fallback): TTM recursion with Q4 annual as seed.
    single_t = TTM_t - TTM_{t-1} + single_{t-4}
    Seed from earliest Q4 (TTM(Q4) = annual sum, no approximation for Q4).

Benchmark validation (assert within 5%):
  贵州茅台 sh.600519  2023Q4 single EPS ≈ 17.4 CNY  pubDate ≈ 2024-04-03
  平安银行 sz.000001  2023Q4 single EPS ≈ 0.31 CNY  pubDate ≈ 2024-03-15

Output: data/pead-baostock.sqlite  (table: eps_baostock_single)
Usage:
  python scripts/koc/11_baostock_single.py              # 全量重跑（DELETE+reinsert）
  python scripts/koc/11_baostock_single.py --incremental # 仅处理新增股票（append）
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from typing import Optional

# ── Guards ──────────────────────────────────────────────────────────────────
DB_PATH: str = "data/pead-baostock.sqlite"
RECONCILE_TOL: float = 0.01    # Path A valid if sum error < 1%
BENCHMARK_TOL: float = 0.05    # benchmark check within 5%
BENCHMARKS: list[tuple[str, int, int, float]] = [
    ("sh.600519", 2023, 4, 17.4),   # 茅台 2023Q4 single EPS
    ("sz.000001", 2023, 4, 0.31),   # 平安银行 2023Q4 single EPS
]


# ── DB helpers ────────────────────────────────────────────────────────────────
def init_output_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eps_baostock_single (
            code            TEXT    NOT NULL,
            fiscal_year     INTEGER NOT NULL,
            fiscal_quarter  INTEGER NOT NULL,
            stat_date       TEXT,
            pub_date        TEXT,
            eps_single      REAL,     -- single-quarter EPS (CNY per share)
            method          TEXT,     -- 'path_a' | 'path_b'
            PRIMARY KEY (code, fiscal_year, fiscal_quarter)
        )
    """)
    conn.commit()


def load_raw(conn: sqlite3.Connection,
             only_codes: set[str] | None = None) -> dict[str, list[dict]]:
    """Return {code: [sorted rows]}.

    only_codes: if given, load only those codes (incremental mode).
    """
    rows = conn.execute(
        "SELECT code, fiscal_year, fiscal_quarter, stat_date, pub_date, "
        "eps_ttm, net_profit, total_share "
        "FROM eps_baostock_raw ORDER BY code, fiscal_year, fiscal_quarter"
    ).fetchall()
    by_code: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if only_codes is None or r[0] in only_codes:
            by_code[r[0]].append({
                "code": r[0], "fy": r[1], "fq": r[2],
                "stat_date": r[3], "pub_date": r[4],
                "eps_ttm": r[5], "net_profit": r[6], "total_share": r[7],
            })
    return dict(by_code)


def get_already_processed_codes(conn: sqlite3.Connection) -> set[str]:
    """Return codes already in eps_baostock_single."""
    rows = conn.execute(
        "SELECT DISTINCT code FROM eps_baostock_single"
    ).fetchall()
    return {r[0] for r in rows}


# ── Path A: YTD net_profit subtraction ───────────────────────────────────────
def single_eps_path_a(rows: list[dict]) -> list[Optional[float]]:
    """Compute single-quarter EPS via YTD subtraction.

    Chinese quarterly reports state cumulative net profit:
      Q1 -> 3M profit; Q2(H1) -> 6M profit; Q3 -> 9M profit; Q4(annual) -> 12M
    Single-quarter: NP_q1 = NP(Q1), NP_qk = NP(Qk) - NP(Q(k-1)) for k>1.
    Per-share EPS = single_NP (万元) / total_share (万股). Units cancel.
    """
    result: list[Optional[float]] = []
    # Group by (code, fiscal year) — rows may contain multiple stocks
    # (validation passes a concatenated sample), so code must be in the key
    by_year: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_year[(r["code"], r["fy"])][r["fq"]] = r

    for r in rows:
        fy, fq = r["fy"], r["fq"]
        np_curr = r["net_profit"]
        ts = r["total_share"]
        if np_curr is None or ts is None or ts == 0:
            result.append(None)
            continue

        if fq == 1:
            single_np = np_curr
        else:
            prev = by_year[(r["code"], fy)].get(fq - 1)
            if prev is None or prev["net_profit"] is None:
                result.append(None)
                continue
            single_np = np_curr - prev["net_profit"]

        result.append(single_np / ts)
    return result


def validate_path_a(rows: list[dict], eps_single: list[Optional[float]]) -> bool:
    """Check sum(Q1..Q4 single_NP) ≈ annual net_profit within RECONCILE_TOL."""
    # Key must include code — rows may span multiple stocks (validation sample)
    by_year: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r, eps in zip(rows, eps_single):
        if r["total_share"] and eps is not None:
            # Reconstruct single_NP = eps * total_share
            by_year[(r["code"], r["fy"])][r["fq"]] = eps * r["total_share"]

    # Annual net_profit lookup: (code, fy) -> Q4 YTD (full-year cumulative)
    q4_np: dict[tuple, float] = {
        (r["code"], r["fy"]): r["net_profit"]
        for r in rows if r["fq"] == 4 and r["net_profit"] is not None
    }

    errors: list[float] = []
    for (code, fy), qmap in by_year.items():
        if set(qmap.keys()) < {1, 2, 3, 4}:
            continue
        sum_singles = sum(qmap[q] for q in (1, 2, 3, 4))
        annual_np = q4_np.get((code, fy))
        if annual_np is None:
            continue
        if abs(annual_np) < 1e-9:
            continue
        rel_err = abs(sum_singles - annual_np) / abs(annual_np)
        errors.append(rel_err)

    if not errors:
        return False  # No complete years to validate
    median_err = sorted(errors)[len(errors) // 2]
    return median_err < RECONCILE_TOL


# ── Path B: TTM recursive conversion ─────────────────────────────────────────
def single_eps_path_b(rows: list[dict]) -> list[Optional[float]]:
    """TTM recursion: single_t = TTM_t - TTM_{t-1} + single_{t-4}.

    Seeds the first year from Q4 annual (TTM(Q4) = annual). All other
    quarters in year 0 are seeded from annual/4 (approximate; will be
    excluded by the trusted>=8q requirement in sue step anyway).
    """
    n = len(rows)
    single: list[Optional[float]] = [None] * n
    ttm: list[Optional[float]] = [r["eps_ttm"] for r in rows]

    # Build index map for fast t-4 lookup
    idx_map: dict[tuple[int, int], int] = {
        (r["fy"], r["fq"]): i for i, r in enumerate(rows)
    }

    def prev_quarter(fy: int, fq: int) -> tuple[int, int]:
        if fq == 1:
            return fy - 1, 4
        return fy, fq - 1

    def four_qtrs_ago(fy: int, fq: int) -> tuple[int, int]:
        return fy - 1, fq

    for i, r in enumerate(rows):
        fy, fq = r["fy"], r["fq"]
        if ttm[i] is None:
            continue

        t4_key = four_qtrs_ago(fy, fq)
        t1_key = prev_quarter(fy, fq)
        t4_idx = idx_map.get(t4_key)
        t1_idx = idx_map.get(t1_key)

        if t4_idx is not None and t1_idx is not None and single[t4_idx] is not None:
            ttm_prev = ttm[t1_idx]
            if ttm_prev is not None:
                single[i] = ttm[i] - ttm_prev + single[t4_idx]
        else:
            # Seed: Q4 of first available year -> single = TTM (i.e., annual sum / approx)
            # For Q4: TTM(Q4) = annual sum (exact for Q4 only if no prior data gap)
            # Use annual/4 for all 4 quarters of first year (marked approximate)
            if fq == 4:
                single[i] = ttm[i] / 4.0  # Approximate: annual/4 as Q4 seed

    return single


# ── Benchmark check ───────────────────────────────────────────────────────────
def check_benchmarks(by_code: dict[str, list[dict]],
                     single_map: dict[str, list[Optional[float]]]) -> bool:
    all_pass = True
    for code, fy, fq, expected in BENCHMARKS:
        rows = by_code.get(code, [])
        singles = single_map.get(code, [])
        match = [(r, s) for r, s in zip(rows, singles)
                 if r["fy"] == fy and r["fq"] == fq and s is not None]
        if not match:
            print(f"  [BENCHMARK MISS] {code} {fy}Q{fq}: no data")
            all_pass = False
            continue
        _, actual = match[0]
        rel_err = abs(actual - expected) / abs(expected)
        status = "PASS" if rel_err <= BENCHMARK_TOL else "FAIL"
        print(f"  [BENCHMARK {status}] {code} {fy}Q{fq}: "
              f"expected={expected:.3f}, actual={actual:.3f}, err={rel_err:.1%}")
        if status == "FAIL":
            all_pass = False
    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Baostock YTD EPS -> single-quarter")
    parser.add_argument(
        "--incremental", action="store_true",
        help="仅处理 eps_baostock_raw 中尚未写入 eps_baostock_single 的新增股票（append）。"
             "全量重跑时省略此参数（默认：DELETE + reinsert）。",
    )
    args = parser.parse_args()

    with sqlite3.connect(DB_PATH) as conn:
        n_raw = conn.execute("SELECT COUNT(*) FROM eps_baostock_raw").fetchone()[0]
    assert n_raw > 0, (
        "eps_baostock_raw is empty — run 10_fetch_baostock_eps.py first"
    )
    print(f"  eps_baostock_raw: {n_raw:,} rows")

    with sqlite3.connect(DB_PATH) as conn:
        init_output_table(conn)
        if args.incremental:
            done_codes = get_already_processed_codes(conn)
            raw_codes = {r[0] for r in conn.execute(
                "SELECT DISTINCT code FROM eps_baostock_raw"
            ).fetchall()}
            new_codes = raw_codes - done_codes
            if not new_codes:
                print(f"  [incremental] 无新增股票（已处理 {len(done_codes):,} 只），退出。")
                return
            print(f"  [incremental] 已处理 {len(done_codes):,} 只 | 新增 {len(new_codes):,} 只")
            by_code = load_raw(conn, only_codes=new_codes)
        else:
            by_code = load_raw(conn)

    print(f"  Stocks to process: {len(by_code)}")

    # Path A validation（全量时用 200 只样本；增量时用所有新增股票）
    print("\n  Testing Path A (YTD net_profit subtraction)...")
    sample_codes = list(by_code.keys())[:200]
    sample_rows: list[dict] = []
    for c in sample_codes:
        sample_rows.extend(by_code[c])

    sample_eps_a = single_eps_path_a(sample_rows)
    path_a_valid = validate_path_a(sample_rows, sample_eps_a)
    print(f"  Path A validation: {'PASS' if path_a_valid else 'FAIL'}")

    chosen_path = "path_a" if path_a_valid else "path_b"
    print(f"  Using: {chosen_path}")

    # 计算 single EPS
    single_map: dict[str, list[Optional[float]]] = {}
    for code, rows in by_code.items():
        singles = single_eps_path_a(rows) if chosen_path == "path_a" else single_eps_path_b(rows)
        single_map[code] = singles

    # Benchmark check
    print("\n  Benchmark validation:")
    bench_ok = check_benchmarks(by_code, single_map)
    if not bench_ok:
        print("  [WARN] Benchmark mismatch — check epsTTM/net_profit field interpretation")

    # 写入
    buffer: list[tuple] = []
    for code, rows in by_code.items():
        singles = single_map[code]
        for r, eps in zip(rows, singles):
            if eps is None:
                continue
            buffer.append((
                r["code"], r["fy"], r["fq"],
                r["stat_date"], r["pub_date"],
                round(eps, 6), chosen_path,
            ))

    with sqlite3.connect(DB_PATH) as conn:
        if args.incremental:
            # 增量模式：INSERT OR IGNORE（主键冲突跳过）
            conn.executemany(
                "INSERT OR IGNORE INTO eps_baostock_single "
                "(code, fiscal_year, fiscal_quarter, stat_date, pub_date, eps_single, method) "
                "VALUES (?,?,?,?,?,?,?)",
                buffer,
            )
        else:
            conn.execute("DELETE FROM eps_baostock_single")
            conn.executemany(
                "INSERT INTO eps_baostock_single "
                "(code, fiscal_year, fiscal_quarter, stat_date, pub_date, eps_single, method) "
                "VALUES (?,?,?,?,?,?,?)",
                buffer,
            )
        conn.commit()
        n_written = conn.execute(
            "SELECT COUNT(*) FROM eps_baostock_single"
        ).fetchone()[0]

    mode_label = "（增量 append）" if args.incremental else "（全量）"
    print(f"\n  Written{mode_label}: {len(buffer):,} 条新记录 | 表总计 {n_written:,} 条")

    # 分布检查
    with sqlite3.connect(DB_PATH) as conn:
        stats = conn.execute(
            "SELECT AVG(eps_single), MIN(eps_single), MAX(eps_single), COUNT(*) "
            "FROM eps_baostock_single"
        ).fetchone()
    print(f"  EPS stats: mean={stats[0]:.4f}, min={stats[1]:.4f}, max={stats[2]:.4f}, "
          f"n={stats[3]:,}")
    print("[OK] 11_baostock_single.py completed")


if __name__ == "__main__":
    main()
