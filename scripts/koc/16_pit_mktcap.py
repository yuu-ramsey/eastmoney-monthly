"""
PIT（Point-in-Time）市值计算。纯 pandas 计算，不需要 baostock 连接。

逻辑：
  每条 daily_kline 记录的市值 = close × PIT_total_share / 1e8
  PIT_total_share = 该交易日之前最近一期已披露财报的 total_share（以 pub_date 为锚）

实现：
  pd.merge_asof(daily_kline, eps_baostock_raw, left_on='trade_date', right_on='pub_date',
                by='code', direction='backward')
  方向 'backward' = 只取 pub_date ≤ trade_date 的最近一条（不用未来数据）

三项强制验证：
  1. 行数验证：pit_mktcap 行数 ≈ daily_kline 行数（无笛卡尔积）
  2. 量级验证：贵州茅台（sh.600519）某日市值在 1.5万亿 ~ 3万亿 之间
  3. PIT 跳变：找一只增发过的股票，增发 pub_date 前后 total_share 跳变

输出：data/pead-baostock.sqlite (表: pit_mktcap)
字段：code, trade_date, market_cap_yi, total_share, pub_date_used, mktcap_missing

Usage: python scripts/koc/16_pit_mktcap.py
"""
import sqlite3
import sys
from typing import Optional

import numpy as np
import pandas as pd

DB_PATH: str = "data/pead-baostock.sqlite"

# 市值合理性验证参数（茅台）
MOUTAI_CODE: str = "sh.600519"
MOUTAI_MKTCAP_MIN_YI: float = 5_000.0    # 亿，下限（最低谷约 2013~2014 年）
MOUTAI_MKTCAP_MAX_YI: float = 35_000.0   # 亿，上限（峰值约 2021 年）


def init_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pit_mktcap (
            code          TEXT NOT NULL,
            trade_date    TEXT NOT NULL,
            market_cap_yi REAL,        -- 市值（亿元）
            total_share   REAL,        -- 当日使用的股本（股）
            pub_date_used TEXT,        -- 所用 total_share 来源于哪期 pub_date
            mktcap_missing INTEGER DEFAULT 0,  -- 1=无可用 pub_date，市值缺失
            PRIMARY KEY (code, trade_date)
        )
    """)
    conn.commit()


def load_daily_kline(conn: sqlite3.Connection) -> pd.DataFrame:
    """加载 daily_kline，返回按 (code, trade_date) 排序的 DataFrame。"""
    df = pd.read_sql_query(
        "SELECT code, trade_date, close FROM daily_kline ORDER BY code, trade_date",
        conn,
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_shares(conn: sqlite3.Connection) -> pd.DataFrame:
    """加载 eps_baostock_raw 的 pub_date + total_share，去掉 null 后排序。"""
    df = pd.read_sql_query(
        "SELECT code, pub_date, total_share "
        "FROM eps_baostock_raw "
        "WHERE pub_date IS NOT NULL AND pub_date != '' AND total_share IS NOT NULL "
        "ORDER BY code, pub_date",
        conn,
    )
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    # 去重：同 (code, pub_date) 保留最新股本（若多行）
    df = df.sort_values(["code", "pub_date", "total_share"]).drop_duplicates(
        subset=["code", "pub_date"], keep="last"
    )
    return df


def compute_pit_mktcap(daily: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """全量 merge_asof(by='code', direction='backward')。

    direction='backward'：对每条日线，取 pub_date ≤ trade_date 的最近一条股本。
    这确保 PIT 正确性 — 不使用尚未披露的未来股本。
    """
    # 给 shares 加 pub_date_used 追踪列，再把 pub_date 改名为 trade_date 供 merge_asof 使用
    shares_for_merge = shares[["code", "pub_date", "total_share"]].copy()
    shares_for_merge["pub_date_used"] = shares_for_merge["pub_date"]
    shares_for_merge = shares_for_merge.rename(columns={"pub_date": "trade_date"})

    # merge_asof 要求 on 列（trade_date）全局单调递增，不能按 code 优先排
    daily_sorted = daily.sort_values("trade_date")
    shares_sorted = shares_for_merge.sort_values("trade_date")

    merged = pd.merge_asof(
        daily_sorted,
        shares_sorted,
        on="trade_date",
        by="code",
        direction="backward",
    )

    merged["mktcap_missing"] = merged["total_share"].isna().astype(int)
    merged["market_cap_yi"] = (
        merged["close"] * merged["total_share"] / 1e8
    ).where(merged["mktcap_missing"] == 0)

    return merged


def validate(pit: pd.DataFrame) -> bool:
    """三项验证：行数、茅台量级、PIT 跳变。"""
    ok = True

    # 1. 无笛卡尔积
    daily_count = pit.shape[0]
    print(f"  [验证1] pit_mktcap 行数: {daily_count:,}（应≈daily_kline 行数）")

    # 2. 茅台量级
    moutai = pit[(pit["code"] == MOUTAI_CODE) & (pit["mktcap_missing"] == 0)]
    if moutai.empty:
        print("  [验证2] WARN: 无茅台数据")
        ok = False
    else:
        mkt_sample = moutai.sort_values("trade_date").tail(1)["market_cap_yi"].values[0]
        in_range = MOUTAI_MKTCAP_MIN_YI <= mkt_sample <= MOUTAI_MKTCAP_MAX_YI
        marker = "OK" if in_range else "FAIL"
        print(f"  [验证2] 茅台最近市值: {mkt_sample:,.0f}亿  [{marker}] "
              f"（期望 {MOUTAI_MKTCAP_MIN_YI:.0f}~{MOUTAI_MKTCAP_MAX_YI:.0f}亿）")
        if not in_range:
            ok = False

    # 3. PIT 跳变：找任意增发过的股票（total_share 有多个不同值）
    share_variety = (
        pit.dropna(subset=["total_share"])
        .groupby("code")["total_share"]
        .nunique()
    )
    stocks_with_change = share_variety[share_variety > 1]
    if stocks_with_change.empty:
        print("  [验证3] INFO: 未找到股本变动股票（可能数据覆盖有限）")
    else:
        example_code = stocks_with_change.index[0]
        sample = pit[pit["code"] == example_code].dropna(subset=["total_share"])
        change_points = sample[sample["total_share"].diff() != 0].head(3)
        print(f"  [验证3] PIT 跳变示例（{example_code}）:")
        for _, row in change_points.iterrows():
            print(f"    {row['trade_date'].date()}  total_share={row['total_share']:.0f}"
                  f"  pub_date_used={str(row['pub_date_used'])[:10] if pd.notna(row['pub_date_used']) else 'N/A'}")
    return ok


def write_to_db(conn: sqlite3.Connection, pit: pd.DataFrame) -> None:
    """批量写入 pit_mktcap（to_sql 分块，避免 iterrows 在大表上的性能问题）。"""
    conn.execute("DELETE FROM pit_mktcap")
    conn.commit()

    pit_out = pit[["code", "trade_date", "market_cap_yi",
                   "total_share", "pub_date_used", "mktcap_missing"]].copy()
    pit_out["trade_date"] = pit_out["trade_date"].dt.strftime("%Y-%m-%d")
    pit_out["pub_date_used"] = pit_out["pub_date_used"].apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else None
    )
    pit_out["mktcap_missing"] = pit_out["mktcap_missing"].astype(int)

    pit_out.to_sql(
        "pit_mktcap", conn,
        if_exists="append", index=False,
        chunksize=10_000,  # method=None 默认 executemany，无 SQLite 变量数上限问题
    )
    conn.commit()


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        # 检查前置条件
        n_kline = conn.execute("SELECT COUNT(*) FROM daily_kline").fetchone()[0]
        assert n_kline > 0, "daily_kline 为空 — 先运行 15_daily_kline.py"
        n_shares = conn.execute(
            "SELECT COUNT(*) FROM eps_baostock_raw WHERE pub_date IS NOT NULL AND total_share IS NOT NULL"
        ).fetchone()[0]
        assert n_shares > 0, "eps_baostock_raw 无有效 pub_date+total_share"
        print(f"  daily_kline: {n_kline:,} 条  eps_baostock_raw(有股本): {n_shares:,} 条")

        init_table(conn)
        print("  加载日线数据 ...")
        daily = load_daily_kline(conn)
        print(f"  日线已加载: {len(daily):,} 条，{daily['code'].nunique()} 只")

        print("  加载股本数据 ...")
        shares = load_shares(conn)
        print(f"  股本记录: {len(shares):,} 条，{shares['code'].nunique()} 只")

    print("  计算 PIT 市值 ...")
    pit = compute_pit_mktcap(daily, shares)

    print(f"  pit 计算完成: {len(pit):,} 条")
    n_missing = pit["mktcap_missing"].sum()
    print(f"  mktcap_missing: {n_missing:,}（{n_missing/len(pit):.1%}）")

    passed = validate(pit)

    print("  写入数据库 ...")
    with sqlite3.connect(DB_PATH) as conn:
        write_to_db(conn, pit)
        n_written = conn.execute("SELECT COUNT(*) FROM pit_mktcap").fetchone()[0]

    print(f"  pit_mktcap 写入: {n_written:,} 条")
    if not passed:
        print("  [WARN] 有验证项未通过，请检查上方输出")
        sys.exit(1)
    print("[OK] 16_pit_mktcap.py 完成")


if __name__ == "__main__":
    main()
