"""
轴 B driver：在归母线副本 DB 上重建 §14 SUE + §01 universe。

复用 §14/§01 原模块逻辑（零漂移），仅把 DB_PATH 监patch 到副本 DB
（data/pead-baostock-parent.sqlite，eps_single 已是归母线，见 11c）。
报告路径也重定向，避免覆盖 net_profit 线的 v2 产物。

Usage: python scripts/koc/axisb_rebuild.py
"""
import importlib.util
import sys

PARENT_DB: str = "data/pead-baostock-parent.sqlite"


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    # ── §14 SUE（归母线）──
    print("=" * 60)
    print("§14 SUE 重建（归母线副本）")
    print("=" * 60)
    sue = _load("sue14", "scripts/koc/14_sue_baostock.py")
    sue.DB_PATH = PARENT_DB
    sue.main()

    # ── §01 universe（归母线）──
    print("\n" + "=" * 60)
    print("§01 universe 重建（归母线副本）")
    print("=" * 60)
    uni = _load("uni01", "scripts/koc/01_universe.py")
    uni.DB_PATH = PARENT_DB
    uni.REPORT_PATH = "docs/koc-universe-parent.md"
    # §01 main 用 argparse；给最小 argv（默认 --real=True）
    sys.argv = ["01_universe.py"]
    uni.main()

    print("\n[OK] axisb_rebuild.py 完成")


if __name__ == "__main__":
    main()
