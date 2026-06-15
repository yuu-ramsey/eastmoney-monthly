"""
Dual-source reconciliation: Baostock single-quarter EPS vs akshare pead.sqlite.
Detects restatement contamination: large differences (>5%) suggest akshare
returned restated values while Baostock captured the as-reported original.

方法学说明（v2，股本稳定子集）：
  原 5% 阈值会把送转/增发年份的 per-share EPS 变动误判为"重述"。
  修复：先从 eps_baostock_raw 的 total_share 字段检测股本变动年份，
  剔除后在"股本稳定子集"上对账，量化"真重述"比例。
  预期真重述率接近 0，给主线切 akshare 提供正式背书。

Input:  data/pead-baostock.sqlite (eps_baostock_single, eps_baostock_raw)
        data/pead.sqlite          (eps_single)
Output: docs/eps-reconcile.md + console summary

Usage: python scripts/koc/12_reconcile.py
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

# ── Guards ────────────────────────────────────────────────────────────────────
BS_DB: str = "data/pead-baostock.sqlite"
AK_DB: str = "data/pead.sqlite"
OUT_MD: str = "docs/eps-reconcile.md"
RESTATEMENT_THRESHOLD: float = 0.05   # |rel_diff| > 5% = suspect restatement
RESTATEMENT_WARN_PCT: float = 0.15    # >15% restatement rate = serious
RESTATEMENT_OK_PCT: float = 0.05      # <5% restatement rate = acceptable
SHARE_CHANGE_THRESHOLD: float = 0.10  # Q4 total_share YoY change > 10% = 股本事件年


def load_baostock_eps(db_path: str) -> dict[tuple, float]:
    """Return {(code, fy, fq): eps_single} from Baostock table."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT code, fiscal_year, fiscal_quarter, eps_single "
            "FROM eps_baostock_single WHERE eps_single IS NOT NULL"
        ).fetchall()
    return {(r[0], r[1], r[2]): r[3] for r in rows}


def load_akshare_eps(db_path: str) -> dict[tuple, float]:
    """Return {(code, fy, fq): eps_single} from akshare table.

    akshare uses bare 6-digit codes; Baostock uses sh./sz. prefix.
    Convert akshare codes to Baostock format for matching.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT code, fiscal_year, fiscal_quarter, eps_single "
            "FROM eps_single WHERE eps_single IS NOT NULL"
        ).fetchall()

    result: dict[tuple, float] = {}
    for raw_code, fy, fq, eps in rows:
        bs_code = _to_baostock_code(raw_code)
        if bs_code:
            result[(bs_code, fy, fq)] = eps
    return result


def _to_baostock_code(raw: str) -> str | None:
    """Map bare 6-digit code to sh./sz. format.

    pead.sqlite already stores prefixed codes ('sh.600519') — pass through.
    """
    if not raw:
        return None
    if raw.startswith(("sh.", "sz.")):
        return raw
    if len(raw) != 6 or not raw.isdigit():
        return None
    if raw[:2] in ("60", "68"):
        return "sh." + raw
    if raw[:2] in ("00", "30"):
        return "sz." + raw
    return None


# ── 股本稳定年份检测 ──────────────────────────────────────────────────────────
def detect_share_change_years(bs_db: str) -> set[tuple[str, int]]:
    """
    返回 {(code, fiscal_year)} 集合，表示该股该年发生了送转/增发等股本事件。

    方法：取每股票每年 Q4 的 total_share（年报股本最稳定）。
    若两相邻年度 Q4 total_share 变动 > SHARE_CHANGE_THRESHOLD，则把
    变动发生年（较新的那年）标记为股本事件年——该年的 per-share EPS
    因分母变化而与前期不可比，不应用于重述检测。
    """
    try:
        with sqlite3.connect(bs_db) as conn:
            rows = conn.execute(
                "SELECT code, fiscal_year, total_share "
                "FROM eps_baostock_raw "
                "WHERE fiscal_quarter=4 AND total_share IS NOT NULL "
                "ORDER BY code, fiscal_year"
            ).fetchall()
    except sqlite3.Error:
        return set()

    # 按股票分组，计算相邻年 Q4 total_share 变动
    event_years: set[tuple[str, int]] = set()
    by_code: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for code, fy, ts in rows:
        by_code[code].append((fy, float(ts)))

    for code, year_ts in by_code.items():
        year_ts.sort(key=lambda x: x[0])
        for i in range(1, len(year_ts)):
            prev_fy, prev_ts = year_ts[i - 1]
            curr_fy, curr_ts = year_ts[i]
            if prev_ts == 0:
                continue
            change = abs(curr_ts - prev_ts) / prev_ts
            if change > SHARE_CHANGE_THRESHOLD:
                event_years.add((code, curr_fy))

    return event_years


# ── Reconciliation ────────────────────────────────────────────────────────────
def reconcile(bs_eps: dict[tuple, float],
              ak_eps: dict[tuple, float],
              share_event_years: set[tuple[str, int]] | None = None) -> dict:
    """Compute pair-wise differences for matched records.

    share_event_years: {(code, fiscal_year)} — 股本事件年，从全量对账结果里剔除。
    若为 None，做全量对账（旧行为）。
    """
    common_keys = set(bs_eps.keys()) & set(ak_eps.keys())

    # 全量对账
    diffs_all: list[float] = []
    suspect_all: list[tuple] = []
    # 股本稳定子集对账
    diffs_stable: list[float] = []
    suspect_stable: list[tuple] = []
    n_excluded_share: int = 0

    for key in common_keys:
        code, fy, fq = key
        bs_val = bs_eps[key]
        ak_val = ak_eps[key]

        if abs(bs_val) < 1e-6 and abs(ak_val) < 1e-6:
            diffs_all.append(0.0)
            diffs_stable.append(0.0)
            continue

        denom = max(abs(bs_val), abs(ak_val))
        rel_diff = (ak_val - bs_val) / denom
        diffs_all.append(rel_diff)
        if abs(rel_diff) > RESTATEMENT_THRESHOLD:
            suspect_all.append((*key, bs_val, ak_val, rel_diff))

        # 股本稳定子集：剔除股本事件年
        is_share_event = (
            share_event_years is not None
            and (code, fy) in share_event_years
        )
        if is_share_event:
            n_excluded_share += 1
        else:
            diffs_stable.append(rel_diff)
            if abs(rel_diff) > RESTATEMENT_THRESHOLD:
                suspect_stable.append((*key, bs_val, ak_val, rel_diff))

    suspect_all.sort(key=lambda x: abs(x[5]), reverse=True)
    suspect_stable.sort(key=lambda x: abs(x[5]), reverse=True)

    n_stable = len(diffs_stable)
    return {
        "n_baostock": len(bs_eps),
        "n_akshare": len(ak_eps),
        "n_common": len(common_keys),
        "n_bs_only": len(bs_eps) - len(common_keys),
        "n_ak_only": len(ak_eps) - len(common_keys),
        # 全量
        "diffs": diffs_all,
        "suspect_pairs": suspect_all,
        "n_suspect": len(suspect_all),
        "suspect_rate": len(suspect_all) / len(common_keys) if common_keys else 0.0,
        # 股本稳定子集
        "n_share_event_excluded": n_excluded_share,
        "n_stable": n_stable,
        "diffs_stable": diffs_stable,
        "suspect_pairs_stable": suspect_stable,
        "n_suspect_stable": len(suspect_stable),
        "suspect_rate_stable": (
            len(suspect_stable) / n_stable if n_stable > 0 else 0.0
        ),
    }


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    return data_sorted[min(idx, len(data_sorted) - 1)]


# ── Markdown report ───────────────────────────────────────────────────────────
def write_report(result: dict, out_path: str) -> None:
    diffs = result["diffs"]
    suspects = result["suspect_pairs"]
    suspect_rate = result["suspect_rate"]
    # 股本稳定子集
    diffs_stable = result.get("diffs_stable", diffs)
    suspects_stable = result.get("suspect_pairs_stable", suspects)
    stable_rate = result.get("suspect_rate_stable", suspect_rate)
    n_excluded = result.get("n_share_event_excluded", 0)
    n_stable = result.get("n_stable", len(diffs))

    # 判定基于股本稳定子集（更准确）
    if stable_rate < RESTATEMENT_OK_PCT:
        verdict = (
            "ACCEPTABLE — akshare 与 Baostock 原始值基本一致，"
            f"股本稳定子集真重述率 {stable_rate:.1%} < {RESTATEMENT_OK_PCT:.0%}"
        )
        rec = "akshare 主线无系统性重述污染，KoC 主线使用 akshare 有正式背书。"
    elif stable_rate < RESTATEMENT_WARN_PCT:
        verdict = (
            f"WARNING — 股本稳定子集重述率 {stable_rate:.1%}，属中等偏高"
        )
        rec = "建议用 Baostock 做稳健性检验，akshare 主线可用但需注意。"
    else:
        verdict = (
            f"SERIOUS — 股本稳定子集重述率 {stable_rate:.1%} > {RESTATEMENT_WARN_PCT:.0%}，"
            "akshare 可能有广泛重述污染"
        )
        rec = "Baostock 原始值是更可靠来源；akshare 不适合严格 PIT 分析。"

    def _dist_rows(d: list[float]) -> list[str]:
        if not d:
            return ["| (无数据) | — |"]
        return [
            f"| Mean | {sum(d)/len(d):.4f} |",
            f"| P1  | {_percentile(d, 1):.4f} |",
            f"| P25 | {_percentile(d, 25):.4f} |",
            f"| P50 | {_percentile(d, 50):.4f} |",
            f"| P75 | {_percentile(d, 75):.4f} |",
            f"| P99 | {_percentile(d, 99):.4f} |",
        ]

    lines = [
        "# EPS 双源对账报告（股本稳定子集方法）",
        f"\n**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**数据源**：Baostock `data/pead-baostock.sqlite` vs akshare `data/pead.sqlite`",
        "\n## 覆盖统计",
        "| 指标 | 值 |",
        "|------|----|",
        f"| Baostock 记录数 | {result['n_baostock']:,} |",
        f"| akshare 记录数 | {result['n_akshare']:,} |",
        f"| 匹配对数（全量） | {result['n_common']:,} |",
        f"| 股本事件年剔除 | {n_excluded:,} 对 |",
        f"| 股本稳定子集对数 | {n_stable:,} |",
        f"| Baostock only | {result['n_bs_only']:,} |",
        f"| akshare only | {result['n_ak_only']:,} |",
        "\n## 差异分布（全量）：(akshare − Baostock) / max(|·|)",
        "| 统计量 | 值 |",
        "|--------|-----|",
        *_dist_rows(diffs),
        "\n## 差异分布（股本稳定子集，剔除送转/增发年）",
        "| 统计量 | 值 |",
        "|--------|-----|",
        *_dist_rows(diffs_stable),
        "\n## 重述诊断",
        "| 指标 | 全量 | 股本稳定子集 |",
        "|------|------|------------|",
        f"| 疑似重述对数（\\|diff\\|>5%） | "
        f"{result['n_suspect']:,} | {result['n_suspect_stable']:,} |",
        f"| 重述率 | {suspect_rate:.1%} | {stable_rate:.1%} |",
        f"| 判定门槛（SERIOUS） | {RESTATEMENT_WARN_PCT:.0%} | {RESTATEMENT_WARN_PCT:.0%} |",
        f"\n> **股本事件年说明**：Q4 total_share YoY 变动 > {SHARE_CHANGE_THRESHOLD:.0%} "
        "标记为送转/增发年（per-share EPS 分母变化不等于重述）。",
        f"\n**判定**：{verdict}",
        f"\n**结论**：{rec}",
        "\n## Top 20 最大差异（股本稳定子集）",
        "| 代码 | FY | FQ | Baostock | akshare | 差异率 |",
        "|------|----|----|----------|---------|--------|",
    ]
    for code, fy, fq, bs_v, ak_v, rd in suspects_stable[:20]:
        lines.append(f"| {code} | {fy} | Q{fq} | {bs_v:.4f} | {ak_v:.4f} | {rd:.1%} |")

    if suspects_stable:
        lines += [
            "\n## Top 20 最大差异（全量，含股本事件年）",
            "| 代码 | FY | FQ | Baostock | akshare | 差异率 |",
            "|------|----|----|----------|---------|--------|",
        ]
        for code, fy, fq, bs_v, ak_v, rd in suspects[:20]:
            lines.append(f"| {code} | {fy} | Q{fq} | {bs_v:.4f} | {ak_v:.4f} | {rd:.1%} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report written: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    with sqlite3.connect(BS_DB) as conn:
        n_bs = conn.execute("SELECT COUNT(*) FROM eps_baostock_single").fetchone()[0]
    assert n_bs > 0, (
        "eps_baostock_single is empty — run 11_baostock_single.py first"
    )

    with sqlite3.connect(AK_DB) as conn:
        n_ak = conn.execute("SELECT COUNT(*) FROM eps_single").fetchone()[0]
    assert n_ak > 0, (
        "eps_single in pead.sqlite is empty — akshare database required"
    )

    print(f"  Baostock single EPS: {n_bs:,}")
    print(f"  akshare single EPS:  {n_ak:,}")

    print("  Loading both datasets...")
    bs_eps = load_baostock_eps(BS_DB)
    ak_eps = load_akshare_eps(AK_DB)

    print("  Detecting share-change years (送转/增发 filter)...")
    share_event_years = detect_share_change_years(BS_DB)
    print(f"  Stock-event (code, year) pairs: {len(share_event_years):,}")

    print("  Reconciling (全量 + 股本稳定子集)...")
    result = reconcile(bs_eps, ak_eps, share_event_years)

    print(f"\n  全量匹配对:         {result['n_common']:,}")
    print(f"  剔除股本事件年:      {result['n_share_event_excluded']:,}")
    print(f"  股本稳定子集:        {result['n_stable']:,}")
    print(f"  全量疑似重述:        {result['n_suspect']:,}  ({result['suspect_rate']:.1%})")
    print(f"  稳定子集疑似重述:    {result['n_suspect_stable']:,}  ({result['suspect_rate_stable']:.1%})")

    # Console verdict（以稳定子集为准）
    sr = result["suspect_rate_stable"]
    if sr < RESTATEMENT_OK_PCT:
        print(f"  [VERDICT] ACCEPTABLE — 股本稳定真重述率 {sr:.1%} < {RESTATEMENT_OK_PCT:.0%}")
        print("            akshare 主线有正式背书，无系统性重述污染")
    elif sr < RESTATEMENT_WARN_PCT:
        print(f"  [VERDICT] WARNING: 稳定子集重述率 {sr:.1%}，中等偏高")
    else:
        print(f"  [VERDICT] SERIOUS: 稳定子集重述率 {sr:.1%} > {RESTATEMENT_WARN_PCT:.0%}")

    write_report(result, OUT_MD)
    print("[OK] 12_reconcile.py completed")


if __name__ == "__main__":
    main()
