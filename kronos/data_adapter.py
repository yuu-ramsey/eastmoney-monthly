"""
Data adapter - reads monthly kline data from project SQLite, converts to Kronos Predictor input format

Input: Monthly kline records for a stock from SQLite
Output: pandas DataFrame
  columns = ['open','high','low','close','volume','amount']
  index is DatetimeIndex (monthly candle dates)
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# Project root directory (parent of kronos/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default database path
DEFAULT_DB_PATH = _PROJECT_ROOT / ".eastmoney-ai" / "db" / "klines-v2.sqlite"

# Column mapping: SQLite column names -> DataFrame column names (all same-name, amount handled separately)
_KLINE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def load_monthly_klines(
    code: str,
    db_path: Optional[str | Path] = None,
    min_records: int = 12,
) -> pd.DataFrame:
    """
    Load OHLCV data for a single stock from the SQLite monthly kline database.

    Args:
        code: stock code, e.g. "600519"
        db_path: SQLite database path, defaults to project klines-v2.sqlite
        min_records: minimum record count required, raises error if insufficient

    Returns:
        DataFrame with OHLCV columns + timestamps, sorted by time in ascending order

    Raises:
        FileNotFoundError: database file not found
        ValueError: stock code has no data or insufficient record count
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

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
        raise ValueError(f"Stock {code} has no monthly kline data")

    if len(df) < min_records:
        raise ValueError(
            f"Stock {code} has insufficient monthly kline data: need >= {min_records} records, got {len(df)}"
        )

    # Fill missing amount with 0
    if "amount" not in df.columns or df["amount"].isna().all():
        df["amount"] = 0.0
    else:
        df["amount"] = df["amount"].fillna(0.0)

    # Parse date -> pandas datetime
    # Monthly format "YYYY-MM", uniformly append "-01" to convert to month-start date
    date_str = df["date"].astype(str).str.strip()
    df["timestamps"] = pd.to_datetime(
        date_str.apply(_normalize_date), format="%Y-%m-%d"
    )

    # Keep only the 7 columns Kronos needs, sorted by time ascending
    result = df[["open", "high", "low", "close", "volume", "amount", "timestamps"]].copy()

    # Ensure numeric columns are float
    for col in _KLINE_COLUMNS:
        result[col] = result[col].astype(float)

    # Sort by time ascending, set timestamps as DatetimeIndex (predictor uses df.index for time filtering)
    result = result.sort_values("timestamps")
    result = result.set_index("timestamps")

    # Summary output to stderr (avoid polluting stdout JSON output)
    import sys
    print(f"Stock: {code}", file=sys.stderr)
    print(f"Records: {len(result)}", file=sys.stderr)
    print(f"Date range: {result.index.min().strftime('%Y-%m')} -> "
          f"{result.index.max().strftime('%Y-%m')}", file=sys.stderr)

    return result


def _normalize_date(date_str: str) -> str:
    """Convert "YYYY-MM" to "YYYY-MM-DD"; return as-is if already a full date"""
    date_str = date_str.strip()
    if len(date_str) == 7:  # "YYYY-MM"
        return date_str + "-01"
    return date_str


# ============================================================
# Verification entry point
# ============================================================

if __name__ == "__main__":
    TEST_CODE = "600519"

    print("=" * 60)
    print("Kronos Data Adapter Verification")
    print(f"Database: {DEFAULT_DB_PATH}")
    print("=" * 60)

    try:
        df = load_monthly_klines(TEST_CODE)
        print(f"\nColumns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"\nFirst 5:")
        print(df.head())
        print(f"\nLast 5:")
        print(df.tail())
        print(f"\nNull check:")
        print(df.isna().sum())
        print(f"\nBasic statistics:")
        print(df[["open", "high", "low", "close", "volume", "amount"]].describe())
        print("\nVerification passed.")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        raise
