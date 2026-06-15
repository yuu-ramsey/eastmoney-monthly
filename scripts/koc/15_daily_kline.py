"""
拉取 A 股日线数据（adjustflag='3' 不复权，市值用）并计算事件窗口特征。

流程：
  1. 对每只股票计算合并窗口 [min(pub_date) - 60d, max(pub_date) + 90d]
  2. baostock query_history_k_data_plus 拉取日线（一只股票一次查询）
  3. 写入 daily_kline 表；检查点续传（kline_fetch_log）
  4. 顺带：ALTER TABLE eps_baostock_raw ADD COLUMN liqa_share，
           对每只股票查询 2024 年最新一季报提取 liqaShare 补存
  5. 所有日线入库后，按事件（code, pub_date）计算：
       turn_20d   = 公告前 20 交易日平均换手率
       amihud_20d = 公告前 20 交易日平均 Amihud 非流动性（|pctChg|/100 / amount*1e6）
       turn_insufficient = 1 若不足 20 个交易日

输出：
  data/pead-baostock.sqlite
    daily_kline        (code, trade_date, close, turn, amount, pct_chg, tradestatus, is_st)
    kline_fetch_log    (code, status, n_rows, fetched_at)
    kline_event_features (code, pub_date, fiscal_year, fiscal_quarter,
                          turn_20d, amihud_20d, turn_insufficient)
  data/pead-baostock.sqlite  eps_baostock_raw.liqa_share（新增列）

Usage: python scripts/koc/15_daily_kline.py
"""
import contextlib
import io
import os
import socket
import sqlite3
import sys
import time
from datetime import date, timedelta
from typing import Optional

import baostock as bs
import numpy as np
import pandas as pd

DB_PATH: str = "data/pead-baostock.sqlite"
PEAD_DB_PATH: str = "data/pead.sqlite"
LOCKFILE: str = "data/15_kline.pid"
SLEEP_PER_QUERY: float = 0.15
# 每处理 N 只股票重新登录一次，避免触发 baostock 每 session ~400 次查询限制（实测上限 ~400）
RELOGIN_EVERY: int = 350
# 窗口两端缓冲（自然日）
PRE_WINDOW_DAYS: int = 60
POST_WINDOW_DAYS: int = 90
# 事件特征计算参数
PRE_EVENT_TRADING_DAYS: int = 20    # turn_20d 使用的前事件交易日数
# 日线最早/最晚可查日期（baostock 限制）
KLINE_START_DATE: str = "2009-01-01"
KLINE_END_DATE: str = "2025-06-30"


# ── DB 初始化 ─────────────────────────────────────────────────────────────────
def init_tables(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_kline (
            code        TEXT NOT NULL,
            trade_date  TEXT NOT NULL,
            close       REAL,
            turn        REAL,
            amount      REAL,
            pct_chg     REAL,
            tradestatus INTEGER,
            is_st       INTEGER,
            PRIMARY KEY (code, trade_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kline_fetch_log (
            code       TEXT PRIMARY KEY,
            status     TEXT,       -- 'complete' | 'no_data' | 'error'
            n_rows     INTEGER,
            fetched_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kline_event_features (
            code               TEXT    NOT NULL,
            pub_date           TEXT    NOT NULL,
            fiscal_year        INTEGER,
            fiscal_quarter     INTEGER,
            turn_20d           REAL,
            amihud_20d         REAL,
            turn_insufficient  INTEGER DEFAULT 0,
            PRIMARY KEY (code, pub_date)
        )
    """)
    conn.commit()


def ensure_liqa_share_column(conn: sqlite3.Connection) -> None:
    """若 eps_baostock_raw 尚无 liqa_share 列则 ALTER TABLE 添加。"""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(eps_baostock_raw)")}
    if "liqa_share" not in columns:
        conn.execute("ALTER TABLE eps_baostock_raw ADD COLUMN liqa_share REAL")
        conn.commit()
        print("  [DDL] eps_baostock_raw 新增 liqa_share 列")


def get_done_codes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT code FROM kline_fetch_log WHERE status IN ('complete', 'no_data')"
    ).fetchall()
    return {r[0] for r in rows}


# ── 股票 universe ─────────────────────────────────────────────────────────────
def _board_priority(code: str) -> int:
    """缺失板块优先：sz.* → 0，sh.688 → 1，上交主板 → 2（已基本完成）。"""
    if code.startswith("sz."):
        return 0
    if code.startswith("sh.688"):
        return 1
    return 2


def load_stock_windows_from_pead(pead_db: str) -> dict[str, tuple[str, str]]:
    """从 pead.sqlite eps_ytd 读取全量 5220 只股票的 pub_date 窗口。
    按缺失板块优先排序（sz.000/002/300 > sh.688 > sh.60x），
    checkpoint 会自动跳过已完成的上交主板股票。
    """
    with sqlite3.connect(pead_db) as pconn:
        rows = pconn.execute("""
            SELECT code, MIN(pub_date), MAX(pub_date)
            FROM eps_ytd
            WHERE pub_date IS NOT NULL AND pub_date != ''
            GROUP BY code
        """).fetchall()

    rows_sorted = sorted(rows, key=lambda r: (_board_priority(r[0]), r[0]))

    windows: dict[str, tuple[str, str]] = {}
    for code, min_pub, max_pub in rows_sorted:
        try:
            start_d = date.fromisoformat(min_pub) - timedelta(days=PRE_WINDOW_DAYS)
            end_d = date.fromisoformat(max_pub) + timedelta(days=POST_WINDOW_DAYS)
            window_start = max(start_d, date.fromisoformat(KLINE_START_DATE)).isoformat()
            window_end = min(end_d, date.fromisoformat(KLINE_END_DATE)).isoformat()
            windows[code] = (window_start, window_end)
        except (ValueError, TypeError):
            pass
    return windows


def load_stock_windows(conn: sqlite3.Connection) -> dict[str, tuple[str, str]]:
    """返回 {code: (window_start, window_end)} 只包含有 pub_date 的股票。"""
    rows = conn.execute("""
        SELECT code, MIN(pub_date), MAX(pub_date)
        FROM eps_baostock_raw
        WHERE pub_date IS NOT NULL AND pub_date != ''
        GROUP BY code
    """).fetchall()

    windows: dict[str, tuple[str, str]] = {}
    for code, min_pub, max_pub in rows:
        try:
            start_d = date.fromisoformat(min_pub) - timedelta(days=PRE_WINDOW_DAYS)
            end_d = date.fromisoformat(max_pub) + timedelta(days=POST_WINDOW_DAYS)
            window_start = max(start_d, date.fromisoformat(KLINE_START_DATE)).isoformat()
            window_end = min(end_d, date.fromisoformat(KLINE_END_DATE)).isoformat()
            windows[code] = (window_start, window_end)
        except (ValueError, TypeError):
            pass
    return windows


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _safe_float(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: str) -> Optional[int]:
    if not value:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


# ── 日线拉取 ──────────────────────────────────────────────────────────────────
def fetch_kline(
    code: str,
    window_start: str,
    window_end: str,
) -> tuple[list[tuple], bool]:
    """拉取单只股票日线数据。返回 (rows, had_error)。
    had_error=True 表示网络/API 异常（可重试），False 表示正常（含合法空集）。
    """
    rows: list[tuple] = []
    silence = io.StringIO()
    try:
        with contextlib.redirect_stdout(silence):
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,close,turn,amount,pctChg,tradestatus,isST",
                start_date=window_start,
                end_date=window_end,
                frequency="d",
                adjustflag="3",    # 不复权（收盘价用于计算市值）
            )
        if rs.error_code != "0":
            print(f"  [API_ERR] kline {code}: error_code={rs.error_code}")
            return rows, True
        while rs.next():
            row = rs.get_row_data()
            if len(row) < 8:
                continue
            rows.append((
                code,
                row[0],                     # date
                _safe_float(row[2]),        # close
                _safe_float(row[3]),        # turn
                _safe_float(row[4]),        # amount
                _safe_float(row[5]),        # pctChg
                _safe_int(row[6]),          # tradestatus
                _safe_int(row[7]),          # isST
            ))
    except Exception as exc:
        print(f"  [ERROR] kline {code}: {exc}")
        return rows, True
    return rows, False


# ── liqaShare 补存 ────────────────────────────────────────────────────────────
def fetch_liqa_share(code: str) -> Optional[float]:
    """查询 2024 年最新季报的 liqaShare（字段索引 10）。
    按 Q4→Q3→Q2→Q1 顺序尝试，取第一个有效值。
    """
    silence = io.StringIO()
    for quarter in ("4", "3", "2", "1"):
        try:
            with contextlib.redirect_stdout(silence):
                rs = bs.query_profit_data(code=code, year="2024", quarter=quarter)
            if rs.error_code != "0":
                continue
            while rs.next():
                row = rs.get_row_data()
                if len(row) > 10:
                    value = _safe_float(row[10])
                    if value is not None:
                        return value
        except Exception:
            pass
        time.sleep(SLEEP_PER_QUERY)
    return None


def update_liqa_share(conn: sqlite3.Connection, code: str, liqa_share: float) -> None:
    """将 liqaShare 写入 eps_baostock_raw 中该股票所有行（同一股票股本变化相对缓慢）。"""
    conn.execute(
        "UPDATE eps_baostock_raw SET liqa_share=? WHERE code=?",
        (liqa_share, code),
    )


# ── 主抓取循环 ────────────────────────────────────────────────────────────────
def run_fetch(
    conn: sqlite3.Connection,
    windows: dict[str, tuple[str, str]],
    done: set[str],
    baostock_codes: set[str],
) -> None:
    pending = [(code, ws, we) for code, (ws, we) in windows.items() if code not in done]
    total = len(pending)
    print(f"  待抓股票: {total}（已完成: {len(done)}）")

    # 防止 baostock 查询挂死（TCP 半开连接无响应）
    socket.setdefaulttimeout(60)

    silence = io.StringIO()
    with contextlib.redirect_stdout(silence):
        login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {login_result.error_msg}")
    print("  baostock 登录成功")

    try:
        for i, (code, window_start, window_end) in enumerate(pending):
            # 每 RELOGIN_EVERY 只重新登录，防止 baostock session 查询计数超限（error_code=10002007）
            if i > 0 and i % RELOGIN_EVERY == 0:
                print(f"  [RELOGIN] 已处理 {i} 只，重新登录...")
                with contextlib.redirect_stdout(silence):
                    bs.logout()
                time.sleep(15)
                with contextlib.redirect_stdout(silence):
                    relogin = bs.login()
                if relogin.error_code != "0":
                    print(f"  [WARN] 重新登录失败: {relogin.error_msg}，继续...")
                else:
                    print(f"  [RELOGIN] 重新登录成功")

            # 拉日线
            kline_rows, had_error = fetch_kline(code, window_start, window_end)
            time.sleep(SLEEP_PER_QUERY)

            # 写日线
            if kline_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO daily_kline "
                    "(code, trade_date, close, turn, amount, pct_chg, tradestatus, is_st) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    kline_rows,
                )

            # 拉 liqaShare 并补存（仅对 eps_baostock_raw 中已有记录的股票，避免无效查询）
            liqa_share = fetch_liqa_share(code) if code in baostock_codes else None
            if liqa_share is not None:
                update_liqa_share(conn, code, liqa_share)

            # 记录检查点：error 状态下次重跑会重试；no_data 是合法空集不重试
            if had_error:
                status = "error"
            elif kline_rows:
                status = "complete"
            else:
                status = "no_data"
            conn.execute(
                "INSERT OR REPLACE INTO kline_fetch_log (code, status, n_rows, fetched_at) "
                "VALUES (?, ?, ?, datetime('now', 'localtime'))",
                (code, status, len(kline_rows)),
            )
            conn.commit()

            if (i + 1) % 200 == 0 or i == total - 1:
                print(f"  [{i+1}/{total}] {code}  kline={len(kline_rows)}行  liqa={'ok' if liqa_share else 'null'}")

    finally:
        with contextlib.redirect_stdout(silence):
            bs.logout()
        print("  baostock 退出")


# ── 事件特征计算（纯 Python/pandas，不占 baostock 连接）─────────────────────
def compute_event_features(conn: sqlite3.Connection) -> None:
    print("  计算 kline_event_features ...")

    # 载入所有事件（code, pub_date, fiscal_year, fiscal_quarter）
    events = pd.read_sql_query(
        "SELECT code, pub_date, fiscal_year, fiscal_quarter "
        "FROM eps_baostock_raw "
        "WHERE pub_date IS NOT NULL AND pub_date != '' "
        "ORDER BY code, pub_date",
        conn,
    )
    if events.empty:
        print("  [WARN] eps_baostock_raw 无 pub_date 数据，跳过特征计算")
        return

    events["pub_date"] = pd.to_datetime(events["pub_date"])
    # eps_baostock_raw 可能有重复 (code, pub_date)，去重后才能满足 UNIQUE 约束
    events = events.drop_duplicates(subset=["code", "pub_date"], keep="first")
    unique_codes = events["code"].unique()
    print(f"  事件数: {len(events)}，股票数: {len(unique_codes)}")

    feature_rows: list[tuple] = []
    for code in unique_codes:
        # 载入该股票全部日线
        klines = pd.read_sql_query(
            "SELECT trade_date, turn, amount, pct_chg FROM daily_kline "
            "WHERE code=? ORDER BY trade_date",
            conn, params=(code,),
        )
        if klines.empty:
            continue
        klines["trade_date"] = pd.to_datetime(klines["trade_date"])
        klines = klines.sort_values("trade_date").reset_index(drop=True)

        code_events = events[events["code"] == code].copy()
        for _, evt in code_events.iterrows():
            pub_dt = evt["pub_date"]
            # 公告前 trading days
            pre_klines = klines[klines["trade_date"] < pub_dt]
            n_pre = len(pre_klines)

            turn_insufficient = 0
            turn_20d: Optional[float] = None
            amihud_20d: Optional[float] = None

            if n_pre < PRE_EVENT_TRADING_DAYS:
                turn_insufficient = 1
            # 取最后 20 个（不足时取全部）
            window_rows = pre_klines.tail(PRE_EVENT_TRADING_DAYS)

            if not window_rows.empty:
                turns = window_rows["turn"].dropna()
                if not turns.empty:
                    turn_20d = float(turns.mean())

                # Amihud = |pctChg/100| / (amount/1e6)，跳过成交额为 0 的行
                valid = window_rows[
                    window_rows["amount"].notna() & (window_rows["amount"] > 0)
                    & window_rows["pct_chg"].notna()
                ].copy()
                if not valid.empty:
                    illiq = (valid["pct_chg"].abs() / 100.0) / (valid["amount"] / 1e6)
                    amihud_20d = float(illiq.mean())

            feature_rows.append((
                code,
                pub_dt.strftime("%Y-%m-%d"),
                int(evt["fiscal_year"]) if pd.notna(evt["fiscal_year"]) else None,
                int(evt["fiscal_quarter"]) if pd.notna(evt["fiscal_quarter"]) else None,
                turn_20d,
                amihud_20d,
                turn_insufficient,
            ))

    if feature_rows:
        conn.execute("DELETE FROM kline_event_features")
        conn.executemany(
            "INSERT OR REPLACE INTO kline_event_features "
            "(code, pub_date, fiscal_year, fiscal_quarter, turn_20d, amihud_20d, turn_insufficient) "
            "VALUES (?,?,?,?,?,?,?)",
            feature_rows,
        )
        conn.commit()
        n_insuff = sum(1 for r in feature_rows if r[6] == 1)
        print(f"  kline_event_features: {len(feature_rows):,} 条，"
              f"turn_insufficient: {n_insuff}（{n_insuff/len(feature_rows):.1%}）")
    else:
        print("  [WARN] 无事件特征可写")


# ── PID 锁（防止多实例同时跑 baostock，会互踢 session）────────────────────────
def _acquire_lock() -> None:
    if os.path.exists(LOCKFILE):
        try:
            pid = int(open(LOCKFILE).read().strip())
            import psutil  # type: ignore
            if psutil.pid_exists(pid):
                print(f"[ABORT] 另一个实例 (PID {pid}) 正在运行，退出。")
                sys.exit(1)
        except Exception:
            pass  # 读不到或 psutil 不可用时忽略，允许继续
    with open(LOCKFILE, "w") as f:
        f.write(str(os.getpid()))


def _release_lock() -> None:
    try:
        os.remove(LOCKFILE)
    except OSError:
        pass


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main() -> None:
    _acquire_lock()
    try:
        _main()
    finally:
        _release_lock()


def _main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        init_tables(conn)
        ensure_liqa_share_column(conn)
        done = get_done_codes(conn)
        baostock_codes: set[str] = {
            r[0] for r in conn.execute("SELECT DISTINCT code FROM eps_baostock_raw")
        }

    # 清单来自 pead.sqlite 全量（5220只），不依赖 §10 爬取进度
    windows = load_stock_windows_from_pead(PEAD_DB_PATH)
    print(f"  pead.sqlite 股票窗口总数: {len(windows)}（eps_baostock_raw 覆盖: {len(baostock_codes)}）")

    if len(done) < len(windows):
        with sqlite3.connect(DB_PATH, timeout=60) as conn:
            run_fetch(conn, windows, done, baostock_codes)
    else:
        print("  所有股票日线已完成，跳过拉取阶段。")

    # 事件特征（无论是否刚拉取，都重新计算以保持最新）
    with sqlite3.connect(DB_PATH) as conn:
        compute_event_features(conn)

    # ── 汇总统计 ──────────────────────────────────────────────────────────────
    with sqlite3.connect(DB_PATH) as conn:
        n_kline = conn.execute("SELECT COUNT(*) FROM daily_kline").fetchone()[0]
        n_codes = conn.execute("SELECT COUNT(DISTINCT code) FROM daily_kline").fetchone()[0]
        n_complete = conn.execute(
            "SELECT COUNT(*) FROM kline_fetch_log WHERE status='complete'"
        ).fetchone()[0]
        n_nodata = conn.execute(
            "SELECT COUNT(*) FROM kline_fetch_log WHERE status='no_data'"
        ).fetchone()[0]
        n_liqa = conn.execute(
            "SELECT COUNT(DISTINCT code) FROM eps_baostock_raw WHERE liqa_share IS NOT NULL"
        ).fetchone()[0]

    print(f"\n  daily_kline: {n_kline:,} 条，{n_codes:,} 只")
    print(f"  kline_fetch_log: complete={n_complete}  no_data={n_nodata}")
    print(f"  liqa_share 已填: {n_liqa} 只")
    print("[OK] 15_daily_kline.py 完成")


if __name__ == "__main__":
    main()
