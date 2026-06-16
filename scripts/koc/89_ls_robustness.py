"""
夜间自主研究①:融券版股票LS流鲁棒性压力测试套件
============================================================================
判定 LS 的诚实 Sharpe(~2.5)是真实还是过拟合/脆弱。全自动,缓存数据,无GPU。
压力测试(llm-chat 建议 + 标准 LS 诊断):
  A 基线:周/非重叠月/季 Sharpe
  B 块bootstrap(20周块,3000抽):月Sharpe分布+CI+P(<1.5)
  C 逐年子期稳定性
  D 制度分段:高/低市场波动 regime
  E 成本敏感:换手成本1x/2x/3x × 融券费0/4/8/12%
  F 十分位单调性(是否真单调,非仅Q10-Q1侥幸)
  G 流动性稳健:空腿取不同流动性分位
  H 去尾月:剔除最好5%月份后是否仍成立(集中度)
输出 docs/koc-ls-robustness.md。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREDC = "data/pred_ensemble.parquet"
PANEL = "data/mf_panel_weekly_v3.parquet"
OUT = "docs/koc-ls-robustness.md"


def sr(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")


def agg(weekly, freq):
    s = weekly.dropna().copy()
    try:
        s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    except Exception:
        s.index = pd.to_datetime(s.index)
    return (1 + s).groupby(s.index.to_period(freq)).prod() - 1


def load_panel():
    p = pd.read_parquet(PREDC).merge(
        pd.read_parquet(PANEL)[["code", "wk", "amt20", "wret"]], on=["code", "wk"], how="left"
    ).sort_values(["code", "wk"])
    p["sm"] = p.groupby("code")["ens"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    return p


def build_ls(p, cost_mult=1.0, borrow=0.08, liq_floor=0.5, n_dec=10):
    """构造 LS 周收益序列(可参数化成本/融券费/空腿流动性下限)。"""
    rows, ptop, pbot = [], {}, {}
    for wk, g in p.groupby("wk"):
        d = g.dropna(subset=["sm", "fwd", "amt20"]); n = len(d)
        if n < 100:
            continue
        d = d.copy(); d["lq"] = d["amt20"].rank(pct=True)
        rk = d["sm"].rank(method="first"); grp = np.ceil(rk / n * n_dec).clip(1, n_dec)
        top = d[grp == n_dec]; bot = d[(grp == 1) & (d["lq"] >= liq_floor)]
        if len(bot) < 3:
            bot = d[grp == 1]
        rt, rb = float(top["fwd"].mean()), float(bot["fwd"].mean())
        tw_ = {c: 1 / len(top) for c in top["code"]}; bw = {c: 1 / len(bot) for c in bot["code"]}
        tot = sum(abs(tw_.get(c, 0) - ptop.get(c, 0)) for c in set(tw_) | set(ptop))
        tob = sum(abs(bw.get(c, 0) - pbot.get(c, 0)) for c in set(bw) | set(pbot))
        ptop, pbot = tw_, bw
        rows.append({"wk": str(wk), "ls": (rt - rb) - (tot + tob) * 0.0008 * cost_mult - borrow / 52})
    return pd.DataFrame(rows).set_index("wk")["ls"]


def decile_profile(p):
    """各十分位平均 fwd(检验单调性)。"""
    accum = {i: [] for i in range(1, 11)}
    for wk, g in p.groupby("wk"):
        d = g.dropna(subset=["sm", "fwd"]); n = len(d)
        if n < 100:
            continue
        rk = d["sm"].rank(method="first"); grp = np.ceil(rk / n * 10).clip(1, 10)
        for i in range(1, 11):
            sub = d[grp == i]
            if len(sub):
                accum[i].append(float(sub["fwd"].mean()))
    return {i: float(np.mean(v)) * 52 for i, v in accum.items() if v}


def main():
    p = load_panel()
    lines = ["# 融券版 LS 流鲁棒性压力测试", "",
             "**夜间自主研究** | 全自动缓存数据 | 判定 LS Sharpe 真实性 vs 过拟合/脆弱", ""]

    # A 基线
    ls = build_ls(p); ls.index = ls.index.astype(str)
    yr = pd.Series(ls.index.str.slice(0, 4).astype(int), index=ls.index)
    oos = ls[yr >= 2021]
    base_w, base_m, base_q = sr(oos, 52), sr(agg(oos, "M"), 12), sr(agg(oos, "Q"), 4)
    lines += ["## A 基线 Sharpe(OOS 2021+,扣8%融券)",
              f"- 周√52 = **{base_w:.2f}** | 非重叠月√12 = **{base_m:.2f}** | 非重叠季√4 = **{base_q:.2f}**",
              f"- 全期(含IS)周√52 = {sr(ls, 52):.2f}", ""]

    # B 块bootstrap(月度)
    m = agg(oos, "M").dropna().values; rng = np.random.default_rng(42); sb = []
    for _ in range(3000):
        idx = rng.integers(0, max(1, len(m) - 3), max(1, len(m) // 3))
        x = np.concatenate([m[i:i + 3] for i in idx])
        sb.append(x.mean() * 12 / (x.std(ddof=1) * np.sqrt(12) + 1e-12))
    sb = np.array(sb)
    lines += ["## B 块bootstrap(月度,20周≈3月块,3000抽)",
              f"- 均值 {sb.mean():.2f} | CI[{np.percentile(sb,2.5):.2f}, {np.percentile(sb,97.5):.2f}]",
              f"- P(Sharpe<1.5) = **{np.mean(sb<1.5):.0%}** | P(<1.0) = {np.mean(sb<1.0):.0%} | P(>2.0) = {np.mean(sb>2.0):.0%}", ""]

    # C 逐年
    lines += ["## C 逐年子期稳定性(周√52)"]
    yearly = {y: sr(ls[yr == y], 52) for y in range(2021, 2026)}
    lines += ["- " + " | ".join(f"{y}={v:.2f}" for y, v in yearly.items()),
              f"- 最差年 = {min(yearly.values()):.2f}(>0 则无单年崩盘)", ""]

    # D 制度分段(高/低市场波动)
    mkt = p.groupby("wk")["wret"].mean(); mkt.index = mkt.index.astype(str)
    mvol = mkt.rolling(13, min_periods=6).std()
    med = mvol[yr.index.isin(mvol.index)].median()
    hi = ls[ls.index.isin(mvol[mvol >= med].index) & (yr >= 2021)]
    lo = ls[ls.index.isin(mvol[mvol < med].index) & (yr >= 2021)]
    lines += ["## D 制度分段(市场波动 高/低)",
              f"- 高波动周: Sharpe(周√52) = {sr(hi,52):.2f}(n={len(hi)})",
              f"- 低波动周: Sharpe(周√52) = {sr(lo,52):.2f}(n={len(lo)})",
              "- (两 regime 都为正 = 非靠单一市场状态)", ""]

    # E 成本敏感
    lines += ["## E 成本/融券费敏感(OOS 月√12)", "", "| 换手成本 | 融券0% | 4% | 8% | 12% |", "|---|---|---|---|---|"]
    for cm in (1.0, 2.0, 3.0):
        cells = []
        for br in (0.0, 0.04, 0.08, 0.12):
            lsx = build_ls(p, cost_mult=cm, borrow=br); lsx.index = lsx.index.astype(str)
            yx = pd.Series(lsx.index.str.slice(0, 4).astype(int), index=lsx.index)
            cells.append(f"{sr(agg(lsx[yx>=2021],'M'),12):.2f}")
        lines.append(f"| {cm:.0f}x | " + " | ".join(cells) + " |")
    lines.append("")

    # F 十分位单调性
    prof = decile_profile(p)
    mono = all(prof[i] <= prof[i + 1] + 0.05 for i in range(1, 10))  # 容忍小噪声
    lines += ["## F 十分位单调性(年化平均fwd,Q1→Q10)",
              "- " + " ".join(f"Q{i}={prof[i]*100:.1f}%" for i in range(1, 11)),
              f"- 近似单调递增 = **{mono}**(真单调 = 信号非仅Q10-Q1端点侥幸)", ""]

    # G 流动性稳健(空腿流动性下限)
    lines += ["## G 空腿流动性稳健(OOS 月√12)"]
    gcells = []
    for lf in (0.0, 0.3, 0.5, 0.7):
        lsx = build_ls(p, liq_floor=lf); lsx.index = lsx.index.astype(str)
        yx = pd.Series(lsx.index.str.slice(0, 4).astype(int), index=lsx.index)
        gcells.append(f"流动性下限{lf:.0%}: {sr(agg(lsx[yx>=2021],'M'),12):.2f}")
    lines += ["- " + " | ".join(gcells),
              "- (下限越高=空腿越可融券;若高下限仍稳=可执行性强)", ""]

    # H 去尾月(集中度)
    mser = agg(oos, "M").dropna()
    thr = mser.quantile(0.95)
    trimmed = mser[mser < thr]
    lines += ["## H 去尾月集中度检验",
              f"- 全月√12 = {sr(mser,12):.2f} | 剔除最好5%月后 = **{sr(trimmed,12):.2f}**",
              "- (剔除后仍高 = 非少数月份驱动)", ""]

    # 裁定
    robust = (base_q >= 1.5 and np.mean(sb < 1.5) < 0.3 and min(yearly.values()) > 0
              and sr(hi, 52) > 0 and sr(lo, 52) > 0 and mono)
    lines += ["## 裁定",
              f"- **LS 鲁棒性: {'稳健 ✓' if robust else '存疑 ⚠️'}**",
              f"- 关键:季度{base_q:.2f}(≥1.5) · P(<1.5)={np.mean(sb<1.5):.0%}(<30%) · 最差年{min(yearly.values()):.2f}(>0) · 双regime正 · 单调{mono} · 去尾后{sr(trimmed,12):.2f}",
              ""]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-6:]))
    print(f"\n[OK] LS鲁棒性报告 → {OUT}")


if __name__ == "__main__":
    main()
