"""
§15 事件特征离线重算（不联网）。

背景：§10 EPS 全量抓取后，Baostock 对本机 IP 触发黑名单（error 10001011），
§15 主脚本开头的 run_fetch() 会 bs.login() 失败。但日线数据 daily_kline
已抓全（5131 只 / 12.19M 行），kline_event_features 只需依据最新事件集
（eps_baostock_raw 的全部 pub_date，§10 后已达 4734 只）重新计算。

本驱动复用 §15 的 compute_event_features()（纯 pandas，零逻辑漂移），
跳过被封的联网抓取阶段。缺日线的少数次新股（约 24 只）在该函数内被
自然跳过（klines.empty → continue），其事件不写入特征表，§01 universe
join 时按 turn_insufficient=1 处理，符合既有口径。

Usage: python scripts/koc/15b_event_features_only.py
"""
import importlib.util
import sqlite3
from pathlib import Path

DB_PATH: str = "data/pead-baostock.sqlite"
SOURCE_SCRIPT: str = "scripts/koc/15_daily_kline.py"


def _load_module():
    """以文件路径加载 §15 模块（文件名以数字开头，无法常规 import）。

    exec_module 仅执行顶层 import（baostock 等库导入，不触发登录/抓取），
    main() 受 __name__ guard 保护不会运行。
    """
    spec = importlib.util.spec_from_file_location("kline15", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not Path(SOURCE_SCRIPT).exists():
        raise FileNotFoundError(f"未找到源脚本: {SOURCE_SCRIPT}")

    module = _load_module()

    with sqlite3.connect(DB_PATH, timeout=60) as conn:
        module.init_tables(conn)          # 确保 kline_event_features 表存在
        module.compute_event_features(conn)   # 纯 pandas 重算，DELETE+全量重插

        n = conn.execute("SELECT COUNT(*) FROM kline_event_features").fetchone()[0]
        d = conn.execute(
            "SELECT COUNT(DISTINCT code) FROM kline_event_features"
        ).fetchone()[0]
        print(f"  kline_event_features 最终: {n:,} 条，{d} 只")

    print("[OK] 15b_event_features_only.py 完成")


if __name__ == "__main__":
    main()
