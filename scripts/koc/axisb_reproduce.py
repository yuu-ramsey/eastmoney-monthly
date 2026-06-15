"""
轴 B driver：在归母线副本 DB 上跑三复现点（§03 / §07 / §09）。

复用原模块逻辑，仅 DB 常量监patch 到副本，报告路径重定向避免覆盖 v2 net_profit 产物。

Usage: python scripts/koc/axisb_reproduce.py
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
    # ── §03 复现点 1：全期 γ1 t ──
    print("=" * 60)
    print("§03 precheck（归母线，--real --window 60 --sue-spec rank）")
    print("=" * 60)
    pc = _load("pc03", "scripts/koc/03_precheck.py")
    pc.DB_SUE_BAOSTOCK = PARENT_DB
    pc.REPORT_PATH = "docs/koc-precheck-parent.md"
    sys.argv = ["03_precheck.py", "--real", "--window", "60", "--sue-spec", "rank"]
    pc.main()

    # ── §07 复现点 2：时代分裂 ──
    print("\n" + "=" * 60)
    print("§07 era_mechanism（归母线）")
    print("=" * 60)
    era = _load("era07", "scripts/koc/07_era_mechanism.py")
    era.DB = PARENT_DB
    era.main()

    # ── §09 复现点 3：amihud 套利消除 ──
    print("\n" + "=" * 60)
    print("§09 paradox_quant（归母线）")
    print("=" * 60)
    px = _load("px09", "scripts/koc/09_paradox_quant.py")
    px.DB = PARENT_DB
    px.main()

    print("\n[OK] axisb_reproduce.py 完成")


if __name__ == "__main__":
    main()
