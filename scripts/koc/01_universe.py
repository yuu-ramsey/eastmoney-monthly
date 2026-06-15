"""
KoC §3 Universe Builder — PIT Full Version
==========================================
基于 §14 SUE + §16 PIT 市值 + §15 换手率 + §17 ST 历史，
构建两个可比池：

  流动池（liquid_pool）：
    - ST filter:   事件公告日前最近 snapshot 中 is_st=0
    - Size filter: PIT 市值 ≥ 30亿（mktcap_missing=1 视为不通过）
    - Turn filter: turn_20d ≥ 0.5% 且 turn_insufficient=0

  全样本（full_sample）：
    - 所有 trusted=1 的 SUE 事件（不施任何 filter）

两池分别写入 koc_universe 表的 in_liquid_pool / in_full_sample 列，
供 §4/§5/§6 各读取，实现悖论对比。

报告输出 docs/koc-universe.md，含：
  [1] 各 filter 单独剔除量 + 最终池规模
  [2] 板块分布（沪主板/深主板/中小板/创业板）
  [3] 市值分布（10/25/50/75/90 分位）
  [4] 入池时间序列（每季度，2010-2024）

Usage: python scripts/koc/01_universe.py [--real]
"""
import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DB_PATH: str = "data/pead-baostock.sqlite"
REPORT_PATH: str = "docs/koc-universe.md"

# 流动池过滤阈值
MKTCAP_MIN_YI: float = 30.0       # 市值下限（亿元）
TURN_20D_MIN: float = 0.5         # 20日均换手率下限（%）


# ── 板块分类 ──────────────────────────────────────────────────────────────────
def classify_board(code: str) -> str:
    """根据股票代码前缀判断板块。"""
    code_num = code.split(".")[-1]
    if code.startswith("sh."):
        if code_num.startswith("688"):
            return "科创板"
        return "沪主板"
    elif code.startswith("sz."):
        if code_num.startswith("300") or code_num.startswith("301"):
            return "创业板"
        if code_num.startswith("002") or code_num.startswith("003"):
            return "中小板"
        return "深主板"
    return "其他"


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_sue_events(conn: sqlite3.Connection) -> pd.DataFrame:
    """加载全部 trusted=1 的 SUE 事件。"""
    df = pd.read_sql_query(
        "SELECT code, fiscal_year, fiscal_quarter, pub_date, sue "
        "FROM sue_baostock WHERE trusted=1 AND pub_date IS NOT NULL",
        conn,
    )
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    return df


def build_st_filter(conn: sqlite3.Connection, events: pd.DataFrame) -> pd.Series:
    """
    返回 bool Series（长度=len(events)），True=该事件股票在 pub_date 时是 ST。

    实现：pandas merge_asof(direction='backward')，以 pub_date 对应最近 snapshot_date 的 is_st。
    """
    st = pd.read_sql_query(
        "SELECT code, snapshot_date, is_st FROM st_history ORDER BY code, snapshot_date",
        conn,
    )
    st["snapshot_date"] = pd.to_datetime(st["snapshot_date"])

    # merge_asof 要求 on 列全局有序
    ev_sorted = events[["code", "pub_date"]].copy().sort_values("pub_date")
    st_sorted = st.sort_values("snapshot_date")

    merged = pd.merge_asof(
        ev_sorted.reset_index(),         # 保留原始索引
        st_sorted,
        left_on="pub_date",
        right_on="snapshot_date",
        by="code",
        direction="backward",
    )
    # is_st NaN → 无历史记录 → 视为非 ST（is_st=0）
    merged["is_st"] = merged["is_st"].fillna(0).astype(int)

    # 还原到原始顺序
    result = merged.set_index("index")["is_st"].reindex(events.index)
    return result.fillna(0).astype(bool)


def build_mktcap_filter(conn: sqlite3.Connection, events: pd.DataFrame) -> pd.DataFrame:
    """
    返回 DataFrame，含 market_cap_yi（事件公告日对应的 PIT 市值，亿）和 mktcap_missing 列。

    as-of join：pit_mktcap WHERE trade_date ≤ pub_date，取最近一条。
    注意：pit_mktcap.trade_date 是交易日，pub_date 通常是公历日；
    merge_asof backward 会找 pub_date 之前最近的交易日市值。
    """
    pit = pd.read_sql_query(
        "SELECT code, trade_date, market_cap_yi, mktcap_missing "
        "FROM pit_mktcap ORDER BY trade_date",   # merge_asof 要求 on 列全局有序
        conn,
    )
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])

    ev_sorted = events[["code", "pub_date"]].copy().sort_values("pub_date")

    merged = pd.merge_asof(
        ev_sorted.reset_index(),
        pit,
        left_on="pub_date",
        right_on="trade_date",
        by="code",
        direction="backward",
    )
    merged["mktcap_missing"] = merged["mktcap_missing"].fillna(1).astype(int)
    merged["market_cap_yi"] = merged["market_cap_yi"].where(
        merged["mktcap_missing"] == 0
    )

    result = merged.set_index("index")[["market_cap_yi", "mktcap_missing"]].reindex(events.index)
    result["mktcap_missing"] = result["mktcap_missing"].fillna(1).astype(int)
    return result


def build_turn_filter(conn: sqlite3.Connection, events: pd.DataFrame) -> pd.DataFrame:
    """
    返回 DataFrame，含 turn_20d 和 turn_insufficient 列。
    kline_event_features 按 (code, pub_date) 直接 JOIN，不需要 as-of。
    """
    kef = pd.read_sql_query(
        "SELECT code, pub_date, turn_20d, turn_insufficient "
        "FROM kline_event_features",
        conn,
    )
    kef["pub_date"] = pd.to_datetime(kef["pub_date"])

    result = events[["code", "pub_date"]].merge(
        kef, on=["code", "pub_date"], how="left"
    )
    result.index = events.index
    result["turn_insufficient"] = result["turn_insufficient"].fillna(1).astype(int)
    return result[["turn_20d", "turn_insufficient"]]


# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def build_universe(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    返回完整 universe DataFrame，含过滤标志列。
    """
    print("[1/5] 加载 SUE 事件...")
    events = load_sue_events(conn)
    n_total = len(events)
    print(f"  全样本: {n_total:,} 事件, {events['code'].nunique()} 只股票")

    print("[2/5] 板块分类...")
    events["board"] = events["code"].apply(classify_board)

    print("[3/5] ST 过滤（as-of join st_history）...")
    events["is_st"] = build_st_filter(conn, events)
    n_st = int(events["is_st"].sum())
    print(f"  ST 事件: {n_st} ({100*n_st/n_total:.1f}%)")

    print("[4/5] 市值过滤（as-of join pit_mktcap）...")
    mktcap_df = build_mktcap_filter(conn, events)
    events["market_cap_yi"] = mktcap_df["market_cap_yi"]
    events["mktcap_missing"] = mktcap_df["mktcap_missing"]
    events["excluded_size"] = (
        (events["mktcap_missing"] == 1) |
        (events["market_cap_yi"] < MKTCAP_MIN_YI)
    )
    n_miss = int(events["mktcap_missing"].sum())
    n_small = int(((events["mktcap_missing"] == 0) & (events["market_cap_yi"] < MKTCAP_MIN_YI)).sum())
    print(f"  mktcap_missing: {n_miss} ({100*n_miss/n_total:.1f}%)")
    print(f"  市值 < {MKTCAP_MIN_YI}亿: {n_small}")
    print(f"  总 size 排除: {int(events['excluded_size'].sum())}")

    print("[5/5] 换手率过滤（join kline_event_features）...")
    turn_df = build_turn_filter(conn, events)
    events["turn_20d"] = turn_df["turn_20d"]
    events["turn_insufficient"] = turn_df["turn_insufficient"]
    events["excluded_turn"] = (
        (events["turn_insufficient"] == 1) |
        (events["turn_20d"] < TURN_20D_MIN)
    )
    n_turn_insuf = int((events["turn_insufficient"] == 1).sum())
    n_turn_low = int(((events["turn_insufficient"] == 0) & (events["turn_20d"] < TURN_20D_MIN)).sum())
    print(f"  turn_insufficient: {n_turn_insuf}")
    print(f"  turn_20d < {TURN_20D_MIN}%: {n_turn_low}")
    print(f"  总换手率排除: {int(events['excluded_turn'].sum())}")

    # 汇总
    events["in_full_sample"] = True
    events["in_liquid_pool"] = (
        ~events["is_st"] &
        ~events["excluded_size"] &
        ~events["excluded_turn"]
    )

    n_liquid = int(events["in_liquid_pool"].sum())
    print(f"\n  流动池事件: {n_liquid:,} ({100*n_liquid/n_total:.1f}% of full sample)")
    print(f"  流动池股票: {events.loc[events['in_liquid_pool'], 'code'].nunique()}")

    return events


def write_universe_table(conn: sqlite3.Connection, events: pd.DataFrame) -> None:
    """写入 koc_universe 表。"""
    conn.execute("DROP TABLE IF EXISTS koc_universe")
    conn.execute("""
        CREATE TABLE koc_universe (
            code            TEXT    NOT NULL,
            fiscal_year     INTEGER NOT NULL,
            fiscal_quarter  INTEGER NOT NULL,
            pub_date        TEXT    NOT NULL,
            board           TEXT,
            is_st           INTEGER DEFAULT 0,
            mktcap_missing  INTEGER DEFAULT 0,
            market_cap_yi   REAL,
            excluded_size   INTEGER DEFAULT 0,
            turn_20d        REAL,
            turn_insufficient INTEGER DEFAULT 0,
            excluded_turn   INTEGER DEFAULT 0,
            in_liquid_pool  INTEGER DEFAULT 0,
            in_full_sample  INTEGER DEFAULT 1,
            PRIMARY KEY (code, fiscal_year, fiscal_quarter)
        )
    """)

    out = events[[
        "code", "fiscal_year", "fiscal_quarter", "pub_date",
        "board", "is_st", "mktcap_missing", "market_cap_yi",
        "excluded_size", "turn_20d", "turn_insufficient", "excluded_turn",
        "in_liquid_pool", "in_full_sample",
    ]].copy()
    out["pub_date"] = out["pub_date"].dt.strftime("%Y-%m-%d")
    out["is_st"] = out["is_st"].astype(int)
    out["excluded_size"] = out["excluded_size"].astype(int)
    out["excluded_turn"] = out["excluded_turn"].astype(int)
    out["in_liquid_pool"] = out["in_liquid_pool"].astype(int)
    out["in_full_sample"] = out["in_full_sample"].astype(int)

    out.to_sql("koc_universe", conn, if_exists="append", index=False, chunksize=5_000)
    conn.commit()


def build_report(events: pd.DataFrame) -> str:
    """生成 docs/koc-universe.md 报告。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_total = len(events)
    n_stocks = int(events["code"].nunique())
    n_liquid = int(events["in_liquid_pool"].sum())
    n_st = int(events["is_st"].sum())
    n_size = int(events["excluded_size"].sum())
    n_turn = int(events["excluded_turn"].sum())

    # [1] Filter summary
    filter_rows = f"""
| Filter | 排除事件数 | 占全样本比 |
|--------|-----------|----------|
| ST (is_st=1) | {n_st:,} | {100*n_st/n_total:.1f}% |
| 市值缺失或 < {MKTCAP_MIN_YI}亿 | {n_size:,} | {100*n_size/n_total:.1f}% |
| 换手率不足 (turn_20d < {TURN_20D_MIN}% 或不足) | {n_turn:,} | {100*n_turn/n_total:.1f}% |
| **全样本** | **{n_total:,}** | 100% |
| **流动池** | **{n_liquid:,}** | **{100*n_liquid/n_total:.1f}%** |
"""

    # [2] Board distribution
    board_full = events["board"].value_counts().to_dict()
    board_liq = events.loc[events["in_liquid_pool"], "board"].value_counts().to_dict()
    boards = ["沪主板", "深主板", "中小板", "创业板", "科创板", "其他"]
    board_rows = "\n".join(
        f"| {b} | {board_full.get(b, 0):,} | {100*board_full.get(b,0)/n_total:.1f}% "
        f"| {board_liq.get(b, 0):,} | "
        f"{100*board_liq.get(b,0)/n_liquid:.1f}% |"
        for b in boards if board_full.get(b, 0) > 0 or board_liq.get(b, 0) > 0
    )
    board_section = f"""
| 板块 | 全样本事件 | 占比 | 流动池事件 | 占比 |
|------|-----------|-----|-----------|-----|
{board_rows}
"""

    # [3] Market cap distribution (liquid pool only)
    mc = events.loc[events["in_liquid_pool"] & events["market_cap_yi"].notna(), "market_cap_yi"]
    if len(mc) > 0:
        pcts = np.percentile(mc, [10, 25, 50, 75, 90])
        mc_section = f"""
| 分位 | 市值（亿元） |
|------|------------|
| P10 | {pcts[0]:.1f} |
| P25 | {pcts[1]:.1f} |
| P50（中位数） | {pcts[2]:.1f} |
| P75 | {pcts[3]:.1f} |
| P90 | {pcts[4]:.1f} |

> 注：流动池市值分布中位数 {pcts[2]:.1f}亿，大市值股票占比高反映可交易性筛选效果。
"""
    else:
        mc_section = "\n> 无有效市值数据。\n"

    # [4] Time series (quarterly count)
    events["fy_fq"] = (
        events["fiscal_year"].astype(str) + "Q" + events["fiscal_quarter"].astype(str)
    )
    ts_full = events.groupby("fy_fq")["in_full_sample"].sum()
    ts_liq = events.groupby("fy_fq")["in_liquid_pool"].sum()
    ts_df = pd.DataFrame({"full": ts_full, "liquid": ts_liq}).dropna()
    ts_rows = "\n".join(
        f"| {q} | {int(ts_df.loc[q,'full'])} | {int(ts_df.loc[q,'liquid'])} |"
        for q in sorted(ts_df.index)
        if ts_df.loc[q, "full"] >= 10
    )
    ts_section = f"""
| 季度 | 全样本 | 流动池 |
|------|-------|-------|
{ts_rows}
"""

    return f"""# KoC §3 Universe 报告

**生成时间**：{now}
**数据来源**：§14 baostock SUE + §16 PIT 市值 + §15 换手率 + §17 ST 历史

## [1] Filter 剔除汇总
{filter_rows}

## [2] 板块分布
{board_section}

## [3] 流动池市值分布
{mc_section}

## [4] 入池时间序列（每季度）
{ts_section}

## 过滤参数

| 参数 | 值 |
|------|---|
| 市值下限 | {MKTCAP_MIN_YI}亿元 |
| 20日均换手率下限 | {TURN_20D_MIN}% |
| ST 判断 | st_history PIT as-of join（snapshot_date ≤ pub_date） |
| 市值判断 | pit_mktcap as-of join（trade_date ≤ pub_date） |

## 注意事项

- 本 universe 覆盖 {n_stocks:,} 只股票（trusted SUE 事件）。§10 EPS 已全量重建
  （4734/5225 complete，491 只次新股经 akshare 双重确认真无数据，标 no_data_confirmed）。
- mktcap_missing=1 统一排除出流动池（不允许"未知=通过"）。
- turn_insufficient=1 同样排除（无法判断流动性）。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="KoC §3 Universe Builder")
    parser.add_argument("--real", action="store_true", default=True,
                        help="真实数据模式（当前唯一模式）")
    args = parser.parse_args()

    t0 = datetime.now()
    print(f"[01_universe] {t0:%Y-%m-%d %H:%M:%S}")

    # 读操作用独立连接，读完显式关闭再写
    read_conn = sqlite3.connect(DB_PATH)
    try:
        events = build_universe(read_conn)
    finally:
        read_conn.close()

    print("\n写入 koc_universe 表...")
    write_conn = sqlite3.connect(DB_PATH)
    try:
        write_universe_table(write_conn, events)
        n_written = write_conn.execute("SELECT COUNT(*) FROM koc_universe").fetchone()[0]
        print(f"  koc_universe: {n_written:,} 行")
    finally:
        write_conn.close()

    print("\n生成报告...")
    report = build_report(events)
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text(report, encoding="utf-8")
    print(f"  报告: {REPORT_PATH}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n[01_universe] 完成，耗时 {elapsed:.1f}s")
    print("[OK] Script completed successfully")


if __name__ == "__main__":
    main()
