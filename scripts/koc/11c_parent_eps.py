"""
§11c — 归母（attributable-to-parent）单季 EPS：Method C 归母-占比缩放。

背景：baostock net_profit 是 YTD 含少数股东，§11 Path A 的 eps_single 因此偏高 ~6%
（见 reference-baostock-eps-line）。本脚本把 Path A 的 net_profit 单季 EPS 缩放到归母线：

    归母_eps_single[row] = eps_single[row] × frac[code, year]
    frac[code, year] = epsTTM(Q4) × total_share(Q4) / net_profit(Q4)   # 归母年报 / 总额年报

外审（gpt-oss-120b）裁定 Method C 优于 TTM 递归（Path B）：复用已验证的 Path A 干净 YTD 差分，
只乘每股每年稳定的标量（少数股东占比，年内波动仅 0.6%），无递归 seed 误差传播。

护栏（外审建议）：
  1. frac 裁剪到 [0, 1]（少数股东为负→frac>1，joint-venture 亏损→frac<0，均裁剪并记录）
  2. net_profit(Q4) ≈ 0（亏损股，frac 爆炸）→ 该年 frac 置 NaN
  3. 缺 Q4 的年份 → 用该股上一年 frac 前向填充（ffill）
  4. 仍无 frac → 该年保持 net_profit 线原值（不缩放，标记）

输入/输出：在 --db 指定的 DB 上原地改 eps_baostock_single：
  - 新增列 eps_single_np（备份 net_profit 线原值）
  - eps_single 列改写为归母线

Usage: python scripts/koc/11c_parent_eps.py --db data/pead-baostock-parent.sqlite
"""
import argparse
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

NETPROFIT_FLOOR: float = 1e7   # |net_profit(Q4)| < 此值视为亏损/近零，frac 不可信


def compute_frac(conn: sqlite3.Connection) -> pd.DataFrame:
    """返回每 (code, fiscal_year) 的归母占比 frac，含护栏与 ffill。"""
    q4 = pd.read_sql_query(
        "SELECT code, fiscal_year, eps_ttm, net_profit, total_share "
        "FROM eps_baostock_raw "
        "WHERE fiscal_quarter = 4 "
        "  AND eps_ttm IS NOT NULL AND net_profit IS NOT NULL AND total_share IS NOT NULL "
        "ORDER BY code, fiscal_year",
        conn,
    )
    # 归母年报净利润 = epsTTM(Q4) × total_share(Q4)；frac = 归母 / 总额
    parent_annual = q4["eps_ttm"] * q4["total_share"]
    # 护栏 2：net_profit(Q4) 近零 → frac 不可信
    near_zero = q4["net_profit"].abs() < NETPROFIT_FLOOR
    frac = parent_annual / q4["net_profit"]
    frac = frac.where(~near_zero, np.nan)
    # 护栏 1：裁剪到 [0, 1]
    n_clip_hi = int((frac > 1).sum())
    n_clip_lo = int((frac < 0).sum())
    frac = frac.clip(lower=0.0, upper=1.0)
    q4 = q4.assign(frac=frac)

    # 护栏 3：缺 Q4 frac 的年份用该股上一年 ffill
    q4 = q4.sort_values(["code", "fiscal_year"])
    q4["frac"] = q4.groupby("code")["frac"].ffill()

    n_near_zero = int(near_zero.sum())
    print(f"  frac 计算：{len(q4):,} 个 (code,year)")
    print(f"    护栏命中：裁剪>1 {n_clip_hi} | 裁剪<0 {n_clip_lo} | net_profit≈0 {n_near_zero}")
    valid = q4["frac"].notna()
    print(f"    有效 frac：{int(valid.sum()):,} | 均值 {q4.loc[valid,'frac'].mean():.4f} "
          f"| 中位 {q4.loc[valid,'frac'].median():.4f} | P5 {q4.loc[valid,'frac'].quantile(0.05):.4f}")
    return q4[["code", "fiscal_year", "frac"]]


def apply_parent_line(conn: sqlite3.Connection, frac_df: pd.DataFrame) -> None:
    """把 eps_single 改写为归母线，备份原值到 eps_single_np。"""
    # 备份列
    cols = {r[1] for r in conn.execute("PRAGMA table_info(eps_baostock_single)")}
    if "eps_single_np" not in cols:
        conn.execute("ALTER TABLE eps_baostock_single ADD COLUMN eps_single_np REAL")
        conn.execute("UPDATE eps_baostock_single SET eps_single_np = eps_single")
        conn.commit()
        print("  已备份 net_profit 线到 eps_single_np 列")

    single = pd.read_sql_query(
        "SELECT code, fiscal_year, fiscal_quarter, eps_single_np FROM eps_baostock_single",
        conn,
    )
    merged = single.merge(frac_df, on=["code", "fiscal_year"], how="left")
    # 护栏 4：无 frac → 保持原值（frac=1）
    n_no_frac = int(merged["frac"].isna().sum())
    merged["frac"] = merged["frac"].fillna(1.0)
    merged["eps_single_parent"] = merged["eps_single_np"] * merged["frac"]

    # 写回
    rows = list(merged[["eps_single_parent", "code", "fiscal_year", "fiscal_quarter"]]
                .itertuples(index=False, name=None))
    conn.executemany(
        "UPDATE eps_baostock_single SET eps_single = ? "
        "WHERE code = ? AND fiscal_year = ? AND fiscal_quarter = ?",
        rows,
    )
    conn.commit()

    np_mean = merged["eps_single_np"].mean()
    par_mean = merged["eps_single_parent"].mean()
    print(f"  改写 {len(rows):,} 行 | 无 frac 保持原值 {n_no_frac:,}")
    print(f"  EPS 均值：net_profit 线 {np_mean:.4f} → 归母线 {par_mean:.4f} "
          f"（缩放 {par_mean/np_mean:.4f}）")


def main() -> None:
    parser = argparse.ArgumentParser(description="§11c 归母单季 EPS（Method C）")
    parser.add_argument("--db", required=True, help="目标 DB（应为副本，原地改 eps_baostock_single）")
    args = parser.parse_args()

    with sqlite3.connect(args.db, timeout=60) as conn:
        print(f"[11c] 目标 DB: {args.db}")
        frac_df = compute_frac(conn)
        apply_parent_line(conn, frac_df)

    print("[OK] 11c_parent_eps.py 完成")


if __name__ == "__main__":
    main()
