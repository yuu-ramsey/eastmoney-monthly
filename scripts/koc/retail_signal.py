"""
散户纯现金组合策略 — 每周实盘信号生成器(产品 CLI)
============================================================================
输出当期可操作清单 + 配置建议 + 诚实业绩披露,生成 markdown 报告 + JSON。
三条纯现金 sleeve:① 可转债双低 ② 打新本期 ③ 低波红利底仓;闲置现金逆回购。
无融券/期货/期权(普通散户接触不到)。诚实 Sharpe ~2.0-2.2(非保证)。

用法:.venv/Scripts/python.exe scripts/koc/retail_signal.py [总现金,默认1000000]
依赖:data/cb_value.parquet(本地可转债)+ akshare(实时打新)。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.portfolio.retail_cash import (                       # noqa: E402
    allocation, filter_double_low, exclude_delisting, strong_redemption_filter,
    concentration_weights, cap_feasible, expected_performance, disclaimer,
)

CB_REQUIRED_COLS = {"code", "close", "premium", "pure_value", "conv_value", "date"}

CB_VALUE = ROOT / "data" / "cb_value.parquet"
OUT_JSON = ROOT / "data" / "retail_signal_latest.json"
OUT_MD = ROOT / "docs" / "retail-signal-latest.md"


def fetch_cb_names() -> dict[str, str]:
    """从 akshare 取可转债代码→简称(用于排退债);失败则空。"""
    try:
        meta = ak.bond_zh_cov().rename(columns={"债券代码": "code", "债券简称": "name"})
        return dict(zip(meta["code"].astype(str), meta["name"].astype(str)))
    except Exception as exc:
        print(f"  [可转债名称获取失败,跳过退债名称过滤: {str(exc)[:60]}]")
        return {}


def build_cb_signal() -> pd.DataFrame:
    """可转债双低买入清单(本地最新数据 + 退债排除 + 双低选债 + 等权)。"""
    if not CB_VALUE.exists():
        raise FileNotFoundError(f"缺少 {CB_VALUE},请先生成可转债数据")
    cv = pd.read_parquet(CB_VALUE, engine="pyarrow")
    missing = CB_REQUIRED_COLS - set(cv.columns)          # schema 校验(防数据源漂移)
    if missing:
        raise ValueError(f"cb_value.parquet 缺少必需列:{missing}")
    for col in ("close", "premium", "pure_value", "conv_value"):  # 强制数值类型(防注入/脏数据)
        cv[col] = pd.to_numeric(cv[col], errors="coerce")
    latest = cv["date"].max()
    d = cv[cv["date"] == latest].copy()
    names = fetch_cb_names()
    d["name"] = d["code"].astype(str).map(names).fillna("")
    d = exclude_delisting(d, name_col="name")            # 排退债(信用风险)
    # 强赎:cb_value 无实时强赎 flag → 留空集合;但价格上限128已排除>130的强赎触发区债,
    # 提供大部分保护;实盘仍需人工核对强赎公告(见报告风控提示)
    d = strong_redemption_filter(d, redemption_codes=set(), code_col="code")
    picks = filter_double_low(d)                          # 价格/溢价/distress 过滤 + 双低前25
    picks = picks.copy()
    picks["weight"] = concentration_weights(len(picks))   # 每只 ≤5%(min不归一)
    if not cap_feasible(len(picks)):                      # <20只:每只封5%,sleeve未满,余额留现金
        deployed = sum(picks["weight"])
        print(f"  [提示] 仅 {len(picks)} 只可转债通过筛选(<20),每只封顶 5%,"
              f"可转债 sleeve 仅部署 {deployed:.0%},剩余 {1-deployed:.0%} 留逆回购")
    picks["signal_date"] = pd.Timestamp(latest).strftime("%Y-%m-%d")
    return picks[["code", "name", "close", "premium", "weight", "signal_date"]]


def build_dazin_signal() -> pd.DataFrame:
    """打新本期可申购清单(实时;需个股市值底仓)。"""
    df = ak.stock_xgsglb_em(symbol="全部股票").rename(columns={
        "股票简称": "name", "申购代码": "scode", "申购日期": "date",
        "顶格申购需配市值": "mktcap", "发行价格": "price"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp(datetime.now().date())
    upcoming = df[(df["date"] >= today - pd.Timedelta(days=2))
                  & (df["date"] <= today + pd.Timedelta(days=7))].copy()
    note = ""
    if upcoming.empty:
        upcoming = df.dropna(subset=["date"]).sort_values("date").tail(5).copy()
        note = "(近期无新申购,显示最近5只供参考)"
    upcoming["note"] = note
    cols = [c for c in ["name", "scode", "date", "mktcap", "price", "note"] if c in upcoming.columns]
    return upcoming[cols]


def render_report(total_cash: float, cb: pd.DataFrame, dz: pd.DataFrame) -> tuple[str, dict]:
    """生成 markdown 报告 + JSON 结构。"""
    alloc = allocation(total_cash)
    perf = expected_performance()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 散户纯现金组合策略信号 — {now}",
        "",
        "```",
        disclaimer(),                                    # 免责声明置顶,必须最先看到
        "```",
        "",
        f"> 诚实业绩:净 Sharpe **{perf.honest_sharpe_low}-{perf.honest_sharpe_high}**"
        f"({perf.method});{perf.note}",
        "",
        f"## 配置建议(总现金 {total_cash:,.0f} 元)",
        "",
        f"- 底仓(低波红利个股,打新市值):**{alloc['base']:,.0f}** ({alloc['base']/total_cash:.0%})",
        f"- 可转债双低:**{alloc['convertible']:,.0f}** ({alloc['convertible']/total_cash:.0%})",
        f"- 逆回购(闲置现金 + 打新缴款流动性):**{alloc['repo']:,.0f}** ({alloc['repo']/total_cash:.0%})",
        "",
        "## ① 可转债双低买入清单(等权)",
        "",
        "| 代码 | 简称 | 现价 | 溢价率 | 权重 |",
        "|---|---|---|---|---|",
    ]
    for _, r in cb.iterrows():
        lines.append(f"| {r['code']} | {r['name']} | {r['close']:.1f} | {r['premium']:.1f}% | {r['weight']:.1%} |")
    lines += ["", "## ② 打新本期可申购(需个股市值底仓)", ""]
    if not dz.empty and dz.iloc[0].get("note"):
        lines.append(f"> {dz.iloc[0]['note']}")
    lines += ["| 简称 | 申购代码 | 申购日 | 需配市值 |", "|---|---|---|---|"]
    for _, r in dz.iterrows():
        d = r["date"].strftime("%Y-%m-%d") if pd.notna(r.get("date")) else "?"
        lines.append(f"| {r.get('name','?')} | {r.get('scode','?')} | {d} | {r.get('mktcap','?')} |")
    lines += [
        "", "## ③ 底仓构建原则(打新市值)",
        "",
        "- **必须个股**(ETF/基金/可转债不计入打新市值)",
        "- 低波(60日波动最低30%) + 红利(股息率≥4%) + 质量(ROE≥12%)三筛任一",
        "- 排除 ST / 停牌 / 退市;压低 2022 式回撤",
        "",
        "## 风控提示(实盘必查)",
        "- **可转债强赎**:本工具未接强赎实时源。价格上限 128 已排除 >130 的强赎触发区债(提供大部分保护),"
        "但仍需**人工核对每只持有债的强赎公告**,出现强赎在赎回登记日前清仓。",
        "- 可转债信用:已排退债;留意评级下调,1 交易日内卖出",
        "- 打新破发:科创/创业板可能破发,首日破发的不申购,已申购次日跌破发行价 >15% 止损",
        "- 单只可转债 ≤ 5%:持有 25 只等权(每只 4%)自动满足;若通过筛选不足 20 只,报告会警告集中度偏高",
    ]
    md = "\n".join(lines)
    payload = {
        "generated": now,
        "total_cash": total_cash,
        "allocation": alloc,
        "expected_sharpe": [perf.honest_sharpe_low, perf.honest_sharpe_high],
        "convertibles": cb.to_dict(orient="records"),
        "ipo": dz.assign(date=dz["date"].astype(str) if "date" in dz else None).to_dict(orient="records") if not dz.empty else [],
        "disclaimer": disclaimer(),
    }
    return md, payload


def main() -> int:
    total_cash = float(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000.0
    print(disclaimer())                                  # 控制台先打免责声明
    print(f"\n生成散户纯现金组合信号(总现金 {total_cash:,.0f})...")
    try:
        cb = build_cb_signal()
    except Exception as exc:
        print(f"  [可转债信号失败: {str(exc)[:100]}]")
        cb = pd.DataFrame(columns=["code", "name", "close", "premium", "weight", "signal_date"])
    try:
        dz = build_dazin_signal()
    except Exception as exc:
        print(f"  [打新信号失败: {str(exc)[:100]}]")
        dz = pd.DataFrame(columns=["name", "scode", "date", "mktcap"])
    md, payload = render_report(total_cash, cb, dz)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"  可转债 {len(cb)} 只 / 打新 {len(dz)} 只")
    print(f"  报告 → {OUT_MD}")
    print(f"  JSON → {OUT_JSON}")
    print("[OK] 信号生成完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
