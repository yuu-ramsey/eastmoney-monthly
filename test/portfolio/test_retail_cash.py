"""
散户纯现金组合策略核心函数测试(自包含断言,无需 pytest)
============================================================================
用法:.venv/Scripts/python.exe test/portfolio/test_retail_cash.py
覆盖:配置拆分、双低过滤、退债/强赎排除、集中度、破发止损、底仓筛选、业绩披露。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# —— 导入路径:项目根 ——
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.portfolio.retail_cash import (                       # noqa: E402
    allocation, filter_double_low, exclude_delisting, strong_redemption_filter,
    concentration_weights, cap_feasible, ipo_breakeven_action, base_position_screen,
    expected_performance, disclaimer, CB_SINGLE_CAP,
)

PASSED = 0
FAILED = 0


def check(name: str, cond: bool) -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}  <<< 失败")


def test_allocation() -> None:
    print("test_allocation")
    a = allocation(1_000_000)
    check("三 sleeve 之和 = 总现金", abs(a["base"] + a["convertible"] + a["repo"] - 1_000_000) < 1.0)
    check("底仓 33%", abs(a["base"] - 330_000) < 1.0)
    check("可转债 37%", abs(a["convertible"] - 370_000) < 1.0)
    check("逆回购为正", a["repo"] > 0)
    # 非法输入显式报错
    raised = False
    try:
        allocation(-1)
    except ValueError:
        raised = True
    check("负现金抛 ValueError", raised)
    raised = False
    try:
        allocation(1_000_000, base_pct=0.6, conv_pct=0.5)   # 和 >=1
    except ValueError:
        raised = True
    check("占比和>=1 抛 ValueError", raised)


def test_filter_double_low() -> None:
    print("test_filter_double_low")
    df = pd.DataFrame({
        "code": ["a", "b", "c", "d", "e", "f"],
        "close": [100, 95, 200, 110, 105, 124],   # c=200 超价格上限
        "premium": [10, 5, 8, 50, 15, -5],        # d=50 超溢价;f=-5 负溢价(双低分会很低易被选)
        "pure_value": [90, 96, 150, 100, 103, 95],  # b: close95<pure96 → distress 排除
        "conv_value": [91, 90, 185, 105, 91, 131],  # f=131 ≥125 强赎触发区 → 必须排除
    })
    out = filter_double_low(df, hold_count=10)
    codes = set(out["code"])
    check("排除超价格上限(c)", "c" not in codes)
    check("排除超溢价(d)", "d" not in codes)
    check("排除 distress close<pure(b)", "b" not in codes)
    check("排除强赎 conv_value≥125(f,负溢价雷)", "f" not in codes)
    check("保留合格(a,e)", {"a", "e"}.issubset(codes))
    check("含双低分列 dl", "dl" in out.columns)
    # 缺列报错(含 conv_value)
    raised = False
    try:
        filter_double_low(pd.DataFrame({"close": [100], "premium": [5], "pure_value": [90]}))
    except KeyError:
        raised = True
    check("缺 conv_value 列抛 KeyError", raised)


def test_exclude_delisting() -> None:
    print("test_exclude_delisting")
    df = pd.DataFrame({"code": ["x", "y"], "name": ["正常转债", "搜特退债"]})
    out = exclude_delisting(df)
    check("排除含退字", set(out["code"]) == {"x"})


def test_strong_redemption() -> None:
    print("test_strong_redemption")
    df = pd.DataFrame({"code": ["111", "222", "333"]})
    out = strong_redemption_filter(df, {"222"})
    check("排除强赎代码", set(out["code"]) == {"111", "333"})
    check("空强赎集合不动", len(strong_redemption_filter(df, set())) == 3)


def test_concentration() -> None:
    print("test_concentration")
    w25 = concentration_weights(25)
    check("25只权重和=1(满仓部署)", abs(sum(w25) - 1.0) < 1e-9)
    check("25只每只=0.04", abs(w25[0] - 0.04) < 1e-9)
    # 关键回归:旧bug是 min+归一让 n<20 时上限失效;现 min 不归一 → 上限由构造保证
    w10 = concentration_weights(10)
    check("10只每只封顶=0.05(=上限,非0.1)", abs(w10[0] - 0.05) < 1e-9)
    check("10只权重和=0.5(余额留现金,不强行满仓)", abs(sum(w10) - 0.5) < 1e-9)
    # 硬约束:任意只数下 max(weight) ≤ cap(llm-chat 审核要求)
    for n in (3, 10, 19, 20, 25, 50):
        check(f"{n}只 max(weight)≤{CB_SINGLE_CAP}", max(concentration_weights(n)) <= CB_SINGLE_CAP + 1e-12)
    check("cap_feasible(25)=True / (10)=False / (20)=True",
          cap_feasible(25) and not cap_feasible(10) and cap_feasible(20))
    raised = False
    try:
        concentration_weights(0)
    except ValueError:
        raised = True
    check("0只抛 ValueError", raised)


def test_ipo_breakeven() -> None:
    print("test_ipo_breakeven")
    check("跌破>15% 止损卖", ipo_breakeven_action(20.0, 16.5) == "sell")   # -17.5%
    check("跌幅<15% 持有", ipo_breakeven_action(20.0, 18.0) == "hold")     # -10%
    check("次日价未知则持有", ipo_breakeven_action(20.0, None) == "hold")


def test_base_position() -> None:
    print("test_base_position")
    uni = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "div_yield": [5.0, 1.0, 1.0, 1.0],     # A 红利达标
        "vol60": [50, 10, 50, 50],             # B 低波
        "roe": [5, 5, 20, 5],                  # C 质量
        "is_st": [False, False, False, True],  # D 被 ST 排除
    })
    out = base_position_screen(uni)
    sel = set(out["ticker"])
    check("红利/低波/质量任一入选(A,B,C)", {"A", "B", "C"}.issubset(sel))
    check("ST 个股被排除(D)", "D" not in sel)


def test_expected_performance() -> None:
    print("test_expected_performance")
    p = expected_performance()
    check("诚实 Sharpe 下限=2.0", p.honest_sharpe_low == 2.0)
    check("诚实 Sharpe 上限=2.2(非夸大)", p.honest_sharpe_high == 2.2)
    check("披露不可达3", "3" in p.note)


def test_disclaimer() -> None:
    print("test_disclaimer")
    d = disclaimer()
    check("声明含'不要碰杠杆'", "杠杆" in d)
    check("声明含'不要碰期权'", "期权" in d)
    check("声明含'不是赚钱机器'类表述", "赚钱" in d)
    check("声明含'实验性/研究小工具'", "实验性" in d or "研究小工具" in d)
    check("声明含'后果自负'", "后果自负" in d)


def main() -> int:
    for fn in [test_allocation, test_filter_double_low, test_exclude_delisting,
               test_strong_redemption, test_concentration, test_ipo_breakeven,
               test_base_position, test_expected_performance, test_disclaimer]:
        fn()
    print(f"\n结果:{PASSED} 通过 / {FAILED} 失败")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
