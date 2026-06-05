"""
Data adapter — reads monthly kline data from project SQLite, converts to Kronos Predictor input format

Input: Monthly kline records for a stock from SQLite
输出: pandas DataFrame
  columns = ['open','high','low','close','volume','amount']
  index 为 DatetimeIndex（月线日期）
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# 项目根目录（kronos/ 的父目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认数据库路径
DEFAULT_DB_PATH = _PROJECT_ROOT / ".eastmoney-ai" / "db" / "klines-v2.sqlite"

# 字段映射：SQLite 列名 → DataFrame 列名（均为同名，amount 单独处理）
_KLINE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def load_monthly_klines(
    code: str,
    db_path: Optional[str | Path] = None,
    min_records: int = 12,
) -> pd.DataFrame:
    """
    从 SQLite 月线数据库加载单只股票的 OHLCV 数据。

    Args:
        code: 股票代码，如 "600519"
        db_path: SQLite 数据库路径，默认使用项目 klines-v2.sqlite
        min_records: 最低记录数要求，不足则报错

    Returns:
        DataFrame，columns 包含 OHLCV + timestamps，按时间升序排列

    Raises:
        FileNotFoundError: 数据库文件不存在
        ValueError: 股票代码无数据或记录数不足
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"数据库不存在: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        query = """
            SELECT date, open, high, low, close, volume, amount
            FROM monthly_klines
            WHERE code = ?
            ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(code,))
    finally:
        conn.close()

    if df.empty:
        raise ValueError(f"股票 {code} 无月线数据")

    if len(df) < min_records:
        raise ValueError(
            f"股票 {code} 月线数据不足: 需要 ≥{min_records} 条，实际 {len(df)} 条"
        )

    # amount 缺失则填 0
    if "amount" not in df.columns or df["amount"].isna().all():
        df["amount"] = 0.0
    else:
        df["amount"] = df["amount"].fillna(0.0)

    # 解析日期 → pandas datetime
    # 月线格式 "YYYY-MM"，统一补 "-01" 转为月初日期
    date_str = df["date"].astype(str).str.strip()
    df["timestamps"] = pd.to_datetime(
        date_str.apply(_normalize_date), format="%Y-%m-%d"
    )

    # 只保留 Kronos 需要的 7 列，按时间升序
    result = df[["open", "high", "low", "close", "volume", "amount", "timestamps"]].copy()

    # 确保数值列为 float
    for col in _KLINE_COLUMNS:
        result[col] = result[col].astype(float)

    # 按时间升序，设置 timestamps 为 DatetimeIndex（predictor 用 df.index 做时间过滤）
    result = result.sort_values("timestamps")
    result = result.set_index("timestamps")

    # 摘要输出到 stderr（避免污染 stdout 的 JSON 输出）
    import sys
    print(f"股票: {code}", file=sys.stderr)
    print(f"数据条数: {len(result)}", file=sys.stderr)
    print(f"时间范围: {result.index.min().strftime('%Y-%m')} → "
          f"{result.index.max().strftime('%Y-%m')}", file=sys.stderr)

    return result


def _normalize_date(date_str: str) -> str:
    """将 "YYYY-MM" 补全为 "YYYY-MM-DD"；已是完整日期则原样返回"""
    date_str = date_str.strip()
    if len(date_str) == 7:  # "YYYY-MM"
        return date_str + "-01"
    return date_str


# ============================================================
# 验证入口
# ============================================================

if __name__ == "__main__":
    TEST_CODE = "600519"

    print("=" * 60)
    print("Kronos 数据适配层验证")
    print(f"数据库: {DEFAULT_DB_PATH}")
    print("=" * 60)

    try:
        df = load_monthly_klines(TEST_CODE)
        print(f"\n字段列表: {list(df.columns)}")
        print(f"数据类型:\n{df.dtypes}")
        print(f"\n前 5 条:")
        print(df.head())
        print(f"\n后 5 条:")
        print(df.tail())
        print(f"\n空值检查:")
        print(df.isna().sum())
        print(f"\n基本统计:")
        print(df[["open", "high", "low", "close", "volume", "amount"]].describe())
        print("\n验证通过.")
    except Exception as e:
        print(f"\n验证失败: {e}")
        raise
