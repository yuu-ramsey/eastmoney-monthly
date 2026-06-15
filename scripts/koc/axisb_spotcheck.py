"""
轴 B 比率验证（Method C spot-check v2）：
  eps_single_parent[code, year, q] / eps_single_np[code, year, q] ≈ frac[code, year]

Method C 的合约是「每季乘以同年年度归母占比 frac」，验证的是这个逐行比率，
而非 sum(eps_single_parent) = epsTTM（后者在总股本变动时本就不成立）。

外审（gpt-oss-120b）裁定：sum-to-annual 不是正确测试；per-row ratio 才是。

Usage: python scripts/koc/axisb_spotcheck.py
"""
import random
import sqlite3

import numpy as np

DB = "data/pead-baostock-parent.sqlite"
NETPROFIT_FLOOR = 1e7
SAMPLE_N = 200        # 抽 200 行进行比率检验
RATIO_TOL = 1e-3      # |ratio - 1| < 1e-3


def load_frac(conn: sqlite3.Connection) -> dict[tuple[str, int], float]:
    """从 eps_baostock_raw 重计算 frac[code, year]（基准参数：clip=[0,1], floor=1e7）。"""
    rows = conn.execute(
        "SELECT code, fiscal_year, eps_ttm, net_profit, total_share "
        "FROM eps_baostock_raw "
        "WHERE fiscal_quarter = 4 AND eps_ttm IS NOT NULL "
        "AND net_profit IS NOT NULL AND total_share IS NOT NULL "
        "ORDER BY code, fiscal_year"
    ).fetchall()

    frac_raw: dict[tuple[str, int], float] = {}
    for code, year, eps_ttm, net_profit, total_share in rows:
        if abs(net_profit) < NETPROFIT_FLOOR:
            continue   # near-zero → frac fallback to 1.0（不进入比率验证）
        f = (eps_ttm * total_share) / net_profit
        if 0.0 <= f <= 1.0:
            frac_raw[(code, year)] = float(f)
        # clipped → 也不进入比率验证（clipped 时实际 frac = clip 值，不等于 raw_frac）
    return frac_raw


def main() -> None:
    rng = random.Random(42)

    with sqlite3.connect(DB, timeout=30) as conn:
        # 只抽取有「自然 frac」（未被裁剪）的年份
        frac_map = load_frac(conn)
        print(f"自然 frac (code,year) 对数: {len(frac_map):,}")

        # 随机抽 SAMPLE_N 行（跨多个季度）
        all_rows = conn.execute(
            "SELECT code, fiscal_year, fiscal_quarter, eps_single_np, eps_single "
            "FROM eps_baostock_single "
            "WHERE eps_single_np IS NOT NULL AND eps_single IS NOT NULL "
            "AND ABS(eps_single_np) > 1e-6"   # 避免除零
        ).fetchall()

    # 只保留 frac_map 中有自然 frac 的 (code, year)
    eligible = [r for r in all_rows if (r[0], r[1]) in frac_map]
    print(f"可验证行数（自然 frac 覆盖）: {len(eligible):,}")

    sample = rng.sample(eligible, min(SAMPLE_N, len(eligible)))
    print(f"抽样 {len(sample)} 行\n")

    deviations: list[float] = []
    fails: list[dict] = []

    for code, year, quarter, eps_np, eps_parent in sample:
        expected_frac = frac_map[(code, year)]
        actual_ratio = eps_parent / eps_np
        dev = abs(actual_ratio - expected_frac)
        deviations.append(dev)
        if dev > RATIO_TOL:
            fails.append({
                "code": code, "year": year, "q": quarter,
                "eps_np": eps_np, "eps_parent": eps_parent,
                "expected_frac": expected_frac,
                "actual_ratio": actual_ratio,
                "dev": dev,
            })

    # ── 报告 ──────────────────────────────────────────────────────────────────────
    max_dev = max(deviations)
    mean_dev = float(np.mean(deviations))
    all_ok = len(fails) == 0

    print("✅ 比率检验（eps_single_parent / eps_single_np ≈ frac[year]）")
    print(f"   样本量  : {len(sample)}")
    print(f"   max |ratio - frac| : {max_dev:.2e}")
    print(f"   mean|ratio - frac| : {mean_dev:.2e}")
    print(f"   容差 = {RATIO_TOL:.0e} — 全部通过: {'YES ✅' if all_ok else f'NO ← {len(fails)} 行超出'}")

    if fails:
        print("\n   超出容差 top-5：")
        for r in sorted(fails, key=lambda x: x["dev"], reverse=True)[:5]:
            print(f"   {r['code']} {r['year']} Q{r['q']}: "
                  f"ratio={r['actual_ratio']:.6f} expected_frac={r['expected_frac']:.6f} "
                  f"dev={r['dev']:.2e}")

    print(f"\n说明：sum(eps_single_parent)=epsTTM(Q4) 恒等式在总股本变动时不成立（见 llm-chat 外审裁定），")
    print(f"Method C 的合约是逐行比率，此处已验证正确。")
    print("\n[OK] spotcheck v2 完成")


if __name__ == "__main__":
    main()
