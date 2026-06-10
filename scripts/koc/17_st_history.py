"""
月度 ST 状态快照（2008-01 ~ 2024-12，共 204 个月）。

通过 baostock query_all_stock 逐月拉取所有 A 股的 ST 状态，
用于 PEAD 回测时剔除 ST/退市预警股票。
检查点续传：已完成月份跳过。

ST 判定规则：
  - 代码名称大写后以 'ST' 或 '*ST' 开头 → is_st=1
  - 以 '退' 开头（退市预警）→ is_st=1

输出:  data/pead-baostock.sqlite (表: st_history)
验证:  sh.600070 (*ST富润) 在摘帽前后 is_st 变化

Usage: python scripts/koc/17_st_history.py
"""
import contextlib
import io
import sqlite3
import time
from datetime import date, timedelta

import baostock as bs

DB_PATH: str = "data/pead-baostock.sqlite"
START_YEAR: int = 2008
END_YEAR: int = 2024
SLEEP_PER_QUERY: float = 0.05


def init_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS st_history (
            code          TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            is_st         INTEGER NOT NULL DEFAULT 0,
            trade_status  TEXT,
            code_name     TEXT,
            PRIMARY KEY (code, snapshot_date)
        )
    """)
    conn.commit()


def get_done_months(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT DISTINCT snapshot_date FROM st_history"
    ).fetchall()
    return {r[0] for r in rows}


def _is_a_share(code: str) -> bool:
    return (
        code.startswith("sh.60") or code.startswith("sh.68")
        or code.startswith("sz.00") or code.startswith("sz.30")
    )


def _detect_st(code_name: str) -> int:
    """1 = ST/∗ST/退市预警，0 = 正常。"""
    if not code_name:
        return 0
    name = code_name.strip()
    upper = name.upper()
    if upper.startswith("ST") or upper.startswith("*ST"):
        return 1
    if name.startswith("退"):
        return 1
    return 0


def _query_snapshot(query_date: str) -> list[tuple]:
    """baostock 原始查询，snapshot_date 字段用 query_date。"""
    records: list[tuple] = []
    silence = io.StringIO()
    try:
        with contextlib.redirect_stdout(silence):
            rs = bs.query_all_stock(day=query_date)
        if rs.error_code != "0":
            print(f"  [WARN] {query_date}: {rs.error_msg}")
            return records
        while rs.next():
            row = rs.get_row_data()
            code = row[0]
            if not _is_a_share(code):
                continue
            trade_status = row[1] if len(row) > 1 else ""
            code_name = row[2] if len(row) > 2 else ""
            records.append((code, query_date, _detect_st(code_name), trade_status, code_name))
    except Exception as exc:
        print(f"  [ERROR] {query_date}: {exc}")
    return records


def fetch_month(nominal_date: str, max_back: int = 10) -> list[tuple]:
    """拉取单月快照，非交易日（空结果）向 15 号之前回溯最多 max_back 天。

    回溯方向 = 严格 PIT（as-of 语义）：快照只含 nominal_date 当天及之前的信息，
    即"截至 15 号的最新已知 ST 状态"。春节月取节前最后交易日。
    max_back=10：春节可吞掉 15 号前整周——2013-02 与 2024-02 需 -7（节前最后
    交易日均为 02-08），-5 实测漏掉这两个月；-10 全覆盖，多余尝试只是空查询。
    不向后顺延：顺延会把节后状态（最多 +7 天未来信息）标成 15 号，
    与 §16 as-of join 等全库 backward 语义不一致。

    存库时 snapshot_date 始终用 nominal_date（YYYY-MM-15），
    不论实际查询哪天，保持主键语义一致。
    """
    base = date.fromisoformat(nominal_date)
    for delta in range(max_back + 1):
        actual = (base - timedelta(days=delta)).isoformat()
        raw = _query_snapshot(actual)
        if raw:
            if delta > 0:
                print(f"  [ROLL] {nominal_date} 非交易日 → 回溯 {delta}d 至 {actual}，{len(raw)} A股")
            # 用 nominal_date 替换 snapshot_date，保证主键为月份 15 号
            return [(r[0], nominal_date, r[2], r[3], r[4]) for r in raw]
    return []


def main() -> None:
    # 每月取 15 号为 nominal date；非交易日回溯至 15 号前最近交易日（最多 -5d，PIT as-of 语义）
    snapshot_dates: list[str] = [
        f"{year}-{month:02d}-15"
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    ]

    with sqlite3.connect(DB_PATH) as conn:
        init_table(conn)
        done = get_done_months(conn)

    pending = [d for d in snapshot_dates if d not in done]
    print(f"  总快照: {len(snapshot_dates)}，已完成: {len(done)}，待抓: {len(pending)}")
    if not pending:
        print("  所有快照已完成。")
        return

    silence = io.StringIO()
    with contextlib.redirect_stdout(silence):
        login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login_result.error_msg}")
    print("  baostock 登录成功")

    # 单次持久连接写入，避免每次迭代重新竞争锁；timeout=60 等待 Monitor 释放
    write_conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        for i, snapshot_date in enumerate(pending):
            records = fetch_month(snapshot_date)
            if records:
                write_conn.executemany(
                    "INSERT OR REPLACE INTO st_history "
                    "(code, snapshot_date, is_st, trade_status, code_name) "
                    "VALUES (?,?,?,?,?)",
                    records,
                )
                write_conn.commit()
            # 每 24 个月（2年）打印一次进度
            if (i + 1) % 24 == 0 or i == len(pending) - 1:
                print(f"  [{i+1}/{len(pending)}] {snapshot_date}: {len(records)} A股")
            time.sleep(SLEEP_PER_QUERY)
    finally:
        write_conn.close()
        with contextlib.redirect_stdout(silence):
            bs.logout()
        print("  baostock 退出")

    # ── 验证 ──────────────────────────────────────────────────────────────────
    with sqlite3.connect(DB_PATH) as conn:
        total_rows = conn.execute("SELECT COUNT(*) FROM st_history").fetchone()[0]
        st_rows = conn.execute(
            "SELECT COUNT(*) FROM st_history WHERE is_st=1"
        ).fetchone()[0]
        sample = conn.execute(
            "SELECT snapshot_date, is_st, code_name FROM st_history "
            "WHERE code='sh.600070' AND snapshot_date LIKE '2020-%' "
            "ORDER BY snapshot_date LIMIT 6"
        ).fetchall()
        # 误伤扫描：is_st=1 但名称不含 ST/退（理论上应为 0）
        false_positives = conn.execute("""
            SELECT DISTINCT code_name FROM st_history
            WHERE is_st=1
              AND UPPER(code_name) NOT LIKE 'ST%'
              AND UPPER(code_name) NOT LIKE '*ST%'
              AND code_name NOT LIKE '退%'
            LIMIT 10
        """).fetchall()

    print(f"\n  总记录: {total_rows:,}，ST 记录: {st_rows:,}（{st_rows/total_rows:.1%}）")
    if sample:
        print("  sh.600070 验证（2020年）：")
        for row in sample:
            print(f"    {row[0]}  is_st={row[1]}  name={row[2]}")
    else:
        print("  [WARN] sh.600070 2020年无数据")
    if false_positives:
        print("  [WARN] 疑似误伤（is_st=1 但名称无 ST/退）：")
        for row in false_positives:
            print(f"    {row[0]}")
    else:
        print("  误伤检查: 0 条")
    print("[OK] 17_st_history.py 完成")


if __name__ == "__main__":
    main()
