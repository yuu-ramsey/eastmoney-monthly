"""
轴 B 敏感性网格（7 runs）：验证 Method C 对 clip 参数与 NETPROFIT_FLOOR 的稳健性。

流程：
  1. 对 parent DB 的 eps_single_np（净利润线备份）重新用不同 frac 参数计算归母 EPS
  2. 覆写 eps_single 列
  3. 运行 §14 重建 SUE（monkeypatching）
  4. 运行 §03 FM 回归并捕获 γ1 t-stat
  5. 所有 run 完成后还原基准 frac（clip=0,1; floor=1e7）

敏感性网格：
  Run | clip_min | clip_max | floor
  A   |   0.0    |   1.0    | 1e7   ← 基准（Axis B 已有）
  B   |   0.0    |   1.05   | 1e7
  C   |   0.0    |   1.10   | 1e7
  D   |   0.90   |   1.0    | 1e7
  E   |   0.95   |   1.0    | 1e7
  F   |   0.0    |   1.0    | 5e6
  G   |   0.0    |   1.0    | 2e7

Usage: python scripts/koc/axisb_sensitivity.py
"""
import importlib.util
import io
import re
import sqlite3
import sys
from typing import Optional

import numpy as np
import pandas as pd

PARENT_DB: str = "data/pead-baostock-parent.sqlite"

GRID = [
    ("A-baseline", 0.0, 1.0,  1e7),
    ("B-clip1.05", 0.0, 1.05, 1e7),
    ("C-clip1.10", 0.0, 1.10, 1e7),
    ("D-lo0.90",  0.90, 1.0,  1e7),
    ("E-lo0.95",  0.95, 1.0,  1e7),
    ("F-floor5e6", 0.0, 1.0,  5e6),
    ("G-floor2e7", 0.0, 1.0,  2e7),
]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_frac(conn: sqlite3.Connection,
                 clip_min: float, clip_max: float,
                 floor: float) -> pd.DataFrame:
    """归母占比 frac（不同 clip/floor 参数）。"""
    q4 = pd.read_sql_query(
        "SELECT code, fiscal_year, eps_ttm, net_profit, total_share "
        "FROM eps_baostock_raw "
        "WHERE fiscal_quarter = 4 "
        "  AND eps_ttm IS NOT NULL AND net_profit IS NOT NULL AND total_share IS NOT NULL "
        "ORDER BY code, fiscal_year",
        conn,
    )
    near_zero = q4["net_profit"].abs() < floor
    frac = (q4["eps_ttm"] * q4["total_share"]) / q4["net_profit"]
    frac = frac.where(~near_zero, np.nan)
    n_clip_hi = int((frac > clip_max).sum())
    n_clip_lo = int((frac < clip_min).sum())
    frac = frac.clip(lower=clip_min, upper=clip_max)
    q4 = q4.assign(frac=frac).sort_values(["code", "fiscal_year"])
    q4["frac"] = q4.groupby("code")["frac"].ffill()
    return q4[["code", "fiscal_year", "frac"]], n_clip_hi, n_clip_lo


def apply_frac(conn: sqlite3.Connection, frac_df: pd.DataFrame,
               clip_min: float, clip_max: float) -> int:
    """用新 frac 覆写 eps_single（从 eps_single_np 出发）。返回裁剪行数。"""
    single = pd.read_sql_query(
        "SELECT code, fiscal_year, fiscal_quarter, eps_single_np FROM eps_baostock_single",
        conn,
    )
    merged = single.merge(frac_df, on=["code", "fiscal_year"], how="left")
    merged["frac"] = merged["frac"].fillna(1.0)
    merged["new_eps"] = merged["eps_single_np"] * merged["frac"]
    rows = list(merged[["new_eps", "code", "fiscal_year", "fiscal_quarter"]]
                .itertuples(index=False, name=None))
    conn.executemany(
        "UPDATE eps_baostock_single SET eps_single = ? "
        "WHERE code = ? AND fiscal_year = ? AND fiscal_quarter = ?",
        rows,
    )
    conn.commit()
    clipped = int(((merged["frac"] > clip_max) | (merged["frac"] < clip_min)).sum())
    return clipped


def run_sue14(conn=None) -> None:
    """运行 §14 重建 sue_baostock（monkeypatching DB_PATH）。"""
    sue = _load("sue14_sens", "scripts/koc/14_sue_baostock.py")
    sue.DB_PATH = PARENT_DB
    sue.main()


def run_precheck03() -> Optional[float]:
    """运行 §03 并捕获 γ1 t-stat。返回 float 或 None。"""
    pc = _load("pc03_sens", "scripts/koc/03_precheck.py")
    pc.DB_SUE_BAOSTOCK = PARENT_DB
    pc.REPORT_PATH = "data/_sens_tmp_report.md"  # 临时文件，不覆盖正式报告

    buf = io.StringIO()
    sys.argv = ["03_precheck.py", "--real", "--window", "60", "--sue-spec", "rank"]

    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        pc.main()
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    m = re.search(r"t\s*=\s*([-+]?\d+\.\d+)", output)
    if m:
        return float(m.group(1))
    # 如果没匹配到（可能 /dev/null 失败），搜 γ1 行
    m2 = re.search(r"γ1.*?t\s*=\s*([-+]?\d+\.\d+)", output)
    return float(m2.group(1)) if m2 else None


def main() -> None:
    results = []

    for run_id, clip_min, clip_max, floor in GRID:
        print(f"\n{'='*60}")
        print(f"Run {run_id}: clip=[{clip_min},{clip_max}] floor={floor:.0e}")
        print(f"{'='*60}")

        with sqlite3.connect(PARENT_DB, timeout=60) as conn:
            frac_df, n_hi, n_lo = compute_frac(conn, clip_min, clip_max, floor)
            clipped = apply_frac(conn, frac_df, clip_min, clip_max)
        print(f"  裁剪 >clip_max: {n_hi} | <clip_min: {n_lo} | 有效 frac 行: {len(frac_df)}")

        print("  §14 重建 SUE...")
        run_sue14()

        print("  §03 FM 回归...")
        t_stat = run_precheck03()
        print(f"  → γ1 t = {t_stat}")

        results.append({
            "run": run_id, "clip_min": clip_min, "clip_max": clip_max,
            "floor": floor, "n_clip_hi": n_hi, "n_clip_lo": n_lo,
            "t_stat": t_stat,
        })

    # ── 汇总表 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("敏感性网格汇总")
    print(f"{'='*60}")
    header = f"{'Run':<15} {'clip':^12} {'floor':>8} {'t-stat':>8} {'PASS?':>6}"
    print(header)
    print("-" * 55)
    for r in results:
        t = r["t_stat"]
        ok = "✅" if (t is not None and t > 3.0) else "❌"
        print(f"{r['run']:<15} [{r['clip_min']:.2f},{r['clip_max']:.2f}] "
              f"{r['floor']:>8.0e} {t or 'N/A':>8} {ok:>6}")

    # 还原基准（Run A baseline）
    print("\n还原基准 frac（clip=0,1; floor=1e7）...")
    with sqlite3.connect(PARENT_DB, timeout=60) as conn:
        frac_df, _, _ = compute_frac(conn, 0.0, 1.0, 1e7)
        apply_frac(conn, frac_df, 0.0, 1.0)
    run_sue14()
    print("[OK] 基准已还原")

    print("\n[OK] axisb_sensitivity.py 完成")


if __name__ == "__main__":
    main()
