"""
散户纯现金组合策略 — 核心纯函数(可测试,不含实时数据抓取)
============================================================================
诚实业绩:纯现金版非重叠月度 deflated Sharpe ~2.0-2.2(经1300+回测+三方复审确立)。
仅三条纯现金 sleeve:可转债双低 + 打新 + 低波红利底仓 + 逆回购闲置现金。
禁用融券/股指期货/期权(普通散户接触不到)。

设计依据见 docs/retail-strategy-product.md。本模块只做选择/过滤/配置的纯计算,
实时数据抓取与报告输出在 scripts/koc/retail_signal.py。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# —— 配置常量(llm-chat 设计评审确定的甜点区,改动会影响诚实 Sharpe)——
BASE_POSITION_PCT: float = 0.33        # 底仓占比(>0.40 会把 Sharpe 拖到 2.0 以下)
CONVERTIBLE_PCT: float = 0.37          # 可转债双低占比
# 逆回购占剩余(约 0.30),提供打新缴款流动性 + 闲置现金收益

CB_PRICE_LOW: float = 88.0             # 可转债价格下限(避免深度折价 distress)
CB_PRICE_HIGH: float = 128.0           # 价格上限(双低带;同时覆盖债券价格维度的赎回风险)
CB_PREMIUM_MAX: float = 40.0           # 转股溢价率上限(%)
CB_CONV_VALUE_MAX: float = 125.0       # 转股价值上限:强赎硬门(经 llm-chat 辩论确认)
#   A股有条件赎回标准触发=正股≥130%转股价=conv_value≥130;用125保守缓冲(catch少数125%触发+月内上行)
CB_HOLD_COUNT: int = 25                # 持有只数(等权)
CB_SINGLE_CAP: float = 0.05            # 单只可转债占可转债 sleeve 上限(集中度风控)

IPO_BREAK_STOP_PCT: float = -0.15      # 已申购新股次日跌破发行价超此比例 → 止损卖出

# —— 强制免责声明(必须在所有面向用户的输出中显著展示)——
DISCLAIMER: str = (
    "⚠️ 重要声明 —— 请务必读完:\n"
    "  1. 这只是一个【实验性研究小工具】,不是投资系统,更不是赚钱机器。\n"
    "  2. 【不要碰杠杆】(融券/配资/股指期货)、【不要碰期权】——普通散户碰这些大概率亏得更多。\n"
    "  3. 历史回测 Sharpe ~2.0-2.2 只是研究数字,不代表未来,不是任何收益承诺。\n"
    "  4. 【不要以为靠这个能从股市稳定赚钱】——股市有风险,绝大多数散户长期是亏损的。\n"
    "  5. 仅供个人学习研究,非投资建议,一切操作后果自负。"
)


def disclaimer() -> str:
    """返回必须展示给用户的免责声明全文。"""
    return DISCLAIMER


@dataclass(frozen=True)
class ExpectedPerformance:
    """诚实业绩披露(非保证,回测中心值);仅陈述本产品自身,不做对标承诺"""
    honest_sharpe_low: float = 2.0
    honest_sharpe_high: float = 2.2
    method: str = "非重叠月度 √12,已对多重检验后缩减(deflated)"
    note: str = "本策略为纯现金组合,历史回测净 Sharpe 约 2.0-2.2(非保证);纯现金口径不可达 Sharpe 3。"


def expected_performance() -> ExpectedPerformance:
    """返回诚实业绩区间(用于报告披露,严禁夸大)"""
    return ExpectedPerformance()


def allocation(total_cash: float, base_pct: float = BASE_POSITION_PCT,
               conv_pct: float = CONVERTIBLE_PCT) -> dict[str, float]:
    """
    按甜点区拆分现金到三条 sleeve。

    参数:
        total_cash: 账户总现金(元),必须 > 0
        base_pct/conv_pct: 底仓/可转债占比,二者之和必须 < 1(余下给逆回购)
    返回:
        {"base": 底仓金额, "convertible": 可转债金额, "repo": 逆回购金额}
    """
    if total_cash <= 0:
        raise ValueError(f"total_cash 必须为正,收到 {total_cash}")
    if base_pct < 0 or conv_pct < 0 or base_pct + conv_pct >= 1.0:
        raise ValueError(f"占比非法:base={base_pct} conv={conv_pct}(二者和须 <1)")
    base = round(total_cash * base_pct, 2)
    convertible = round(total_cash * conv_pct, 2)
    repo = round(total_cash - base - convertible, 2)
    return {"base": base, "convertible": convertible, "repo": repo}


def exclude_delisting(bonds: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
    """排除含"退"字的退市/将退市可转债(信用风险)。"""
    if name_col not in bonds.columns:
        return bonds
    mask = ~bonds[name_col].astype(str).str.contains("退", na=False)
    return bonds[mask].copy()


def strong_redemption_filter(bonds: pd.DataFrame, redemption_codes: set[str],
                             code_col: str = "code") -> pd.DataFrame:
    """
    排除已公告强制赎回的可转债(继续持有会被按赎回价强制赎回 → 损失)。

    参数:
        redemption_codes: 已触发/公告强赎的债券代码集合(由实时公告数据提供)
    """
    if code_col not in bonds.columns or not redemption_codes:
        return bonds.copy()
    mask = ~bonds[code_col].astype(str).isin({str(c) for c in redemption_codes})
    return bonds[mask].copy()


def filter_double_low(bonds: pd.DataFrame, price_lo: float = CB_PRICE_LOW,
                      price_hi: float = CB_PRICE_HIGH, premium_max: float = CB_PREMIUM_MAX,
                      conv_value_max: float = CB_CONV_VALUE_MAX,
                      hold_count: int = CB_HOLD_COUNT) -> pd.DataFrame:
    """
    可转债双低选债:价格区间 + 溢价上限 + 高于纯债价值(避 distress)
    + 转股价值上限(强赎硬门)→ 双低排名取前 N。

    强赎门用 conv_value(转股价值)而非价格:A股强制赎回触发=正股≥130%转股价=conv_value≥130;
    双低选低溢价会"专挑"负溢价的高 conv_value 强赎雷,故必须用 conv_value 拦截(经辩论确认机制)。

    必需列:close(现价)、premium(转股溢价率%)、pure_value(纯债价值)、conv_value(转股价值)。
    返回:按双低分(价格排名+溢价排名)升序的前 hold_count 只,附 dl 列。
    """
    required = {"close", "premium", "pure_value", "conv_value"}
    missing = required - set(bonds.columns)
    if missing:
        raise KeyError(f"filter_double_low 缺少必需列:{missing}")
    d = bonds.dropna(subset=["close", "premium", "pure_value", "conv_value"]).copy()
    d = d[(d["close"] >= price_lo) & (d["close"] <= price_hi)
          & (d["premium"] <= premium_max) & (d["close"] > d["pure_value"])
          & (d["conv_value"] < conv_value_max)]              # 强赎硬门
    if d.empty:
        return d
    d["dl"] = d["close"].rank() + d["premium"].rank()      # 双低分:越小越好
    return d.nsmallest(hold_count, "dl").reset_index(drop=True)


def concentration_weights(n_bonds: int, single_cap: float = CB_SINGLE_CAP) -> list[float]:
    """
    单只权重 = min(1/N, single_cap),**不归一化**。
    - N ≥ 20(=1/0.05):每只 1/N ≤ 5%,权重和 = 1(满仓部署)。
    - N < 20:每只封顶 5%,权重和 = N×5% < 1,**未部署部分留现金/逆回购**(而非强行满仓)。
    这样单只权重恒 ≤ single_cap(由构造保证,非靠归一),是 llm-chat 审核要求的硬约束。
    返回:长度 N 的权重列表(占可转债 sleeve 的比例,和可能 <1)。
    """
    if n_bonds <= 0:
        raise ValueError("n_bonds 必须为正")
    weight = min(1.0 / n_bonds, single_cap)
    return [weight] * n_bonds


def cap_feasible(n_bonds: int, single_cap: float = CB_SINGLE_CAP) -> bool:
    """
    单只上限在 n_bonds 只等权下是否可行:需 n_bonds ≥ round(1/single_cap)。
    例:single_cap=0.05 需 ≥20 只;持有 25 只时 1/25=0.04 ≤ 0.05,可行。
    """
    if single_cap <= 0:
        return False
    return n_bonds >= round(1.0 / single_cap)


def ipo_breakeven_action(issue_price: float, day2_price: float | None) -> str:
    """
    打新破发处理:
      - 若已申购且次日价已知且跌破发行价超阈值 → "sell"(止损)
      - 否则 → "hold"
    申购前的破发判断(是否申购)在 CLI 用首日表现/市场情绪,这里只管已持有的止损。
    """
    if day2_price is None or issue_price <= 0:
        return "hold"
    ret = day2_price / issue_price - 1.0
    return "sell" if ret <= IPO_BREAK_STOP_PCT else "hold"


def base_position_screen(universe: pd.DataFrame) -> pd.DataFrame:
    """
    底仓筛选:低波 + 红利 + 质量(为打新提供个股市值,且压低 2022 式回撤)。
    ETF/基金/可转债不计入打新市值,故底仓必须是普通 A 股。

    必需列:div_yield(股息率%)、vol60(60日年化波动)、roe(%)、is_st(bool)。
    返回:通过三筛之一且非 ST 的个股(等权候选池)。
    """
    required = {"div_yield", "vol60", "roe", "is_st"}
    missing = required - set(universe.columns)
    if missing:
        raise KeyError(f"base_position_screen 缺少必需列:{missing}")
    u = universe[~universe["is_st"].astype(bool)].copy()
    if u.empty:
        return u
    vol_threshold = u["vol60"].quantile(0.30)              # 低波:波动最低 30%
    div_ok = u["div_yield"] >= 4.0
    lowvol_ok = u["vol60"] <= vol_threshold
    quality_ok = u["roe"] >= 12.0
    return u[div_ok | lowvol_ok | quality_ok].reset_index(drop=True)
