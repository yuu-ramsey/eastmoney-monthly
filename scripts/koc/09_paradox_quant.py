"""
可交易性悖论量化（链 B：B2 三层分组交互项 + B4 成本瀑布表，2026-06-10）

B2 三层分组：全样本按 turn_20d（截面三分位）分流动性三层，
  每层内 rank-SUE 五分组 Q5−Q1 的 60d BHAR 价差 → 检验"alpha 是否藏在不可交易层"。
  辅以 FM 交互项：ret ~ sue_z + illiq_z + sue_z×illiq_z + 全控制，
  交互项 γ>0 = 流动性越差 SUE 漂移越强（悖论实锤）。
  市值三层版同跑（size 维度悖论）。

B4 成本瀑布：流动池 top 五分组 long-only（A股可执行对象），
  gross BHAR → 佣金 0.05% → 印花 0.10% → 冲击 0.10%（注册参数，需验证）→ net。
  L/S 纸面版单列（空头腿 A 股不可执行，仅披露）。
  全部按正典切分 2010-2021 / 2022-2024 分列。

Usage: python scripts/koc/09_paradox_quant.py
"""
import json
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

DB = "data/pead-baostock.sqlite"
INDUSTRY_MAP = "data/industry-map.json"
NW = 4
HOLD_START = 5
MIN_N_LAYER = 150        # B2 三层×五分组需足够截面
ERA_SPLIT = 2021         # ≤2021 旧时代

# B4 成本参数（注册于 docs/koc-backtest-design.md，标"需验证"）
COST_COMMISSION = 0.0005     # 双边佣金合计
COST_STAMP = 0.0010          # 印花税（卖出）
COST_IMPACT = 0.0010         # 冲击成本（估）


def nwt(series: np.ndarray) -> tuple[float, float]:
    s = np.asarray(series, dtype=float)
    m = float(s.mean())
    e = s - m
    var = float(np.dot(e, e))
    for lag in range(1, min(NW, len(s) - 1) + 1):
        var += 2.0 * (1.0 - lag / (NW + 1)) * float(np.dot(e[lag:], e[:-lag]))
    se = float(np.sqrt(max(var, 0.0) / (len(s) ** 2)))
    return m, m / (se + 1e-12)


def era_stat(d: dict[tuple[int, int], float], lo: int, hi: int) -> tuple[float, float, int]:
    vals = [v for (fy, _), v in sorted(d.items()) if lo <= fy <= hi]
    if len(vals) < 4:
        return float("nan"), float("nan"), len(vals)
    m, t = nwt(np.array(vals))
    return m, t, len(vals)


def main() -> None:
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT sb.code, sb.fiscal_year, sb.fiscal_quarter, sb.pub_date, sb.sue, "
            "ku.in_liquid_pool, ku.market_cap_yi, ku.mktcap_missing, ku.turn_20d "
            "FROM sue_baostock sb "
            "LEFT JOIN koc_universe ku USING (code, fiscal_year, fiscal_quarter) "
            "WHERE sb.trusted=1 AND sb.pub_date IS NOT NULL",
            conn,
        )
        kline = pd.read_sql_query(
            "SELECT code, trade_date, pct_chg FROM daily_kline "
            "WHERE tradestatus='1' AND pct_chg IS NOT NULL "
            "ORDER BY code, trade_date",
            conn,
        )
    finally:
        conn.close()

    df["pub_date"] = pd.to_datetime(df["pub_date"])
    kline["trade_date"] = pd.to_datetime(kline["trade_date"])

    mkt = kline.groupby("trade_date")["pct_chg"].mean().sort_index()
    mkt_dates = mkt.index.values
    mkt_cs0 = np.concatenate([[0.0], np.cumsum(np.log1p(mkt.values / 100.0))])
    dates_by: dict[str, np.ndarray] = {}
    cs0_by: dict[str, np.ndarray] = {}
    for code, grp in kline.groupby("code", sort=False):
        grp = grp.sort_values("trade_date")
        dates_by[code] = grp["trade_date"].values
        cs0_by[code] = np.concatenate(
            [[0.0], np.cumsum(np.log1p(grp["pct_chg"].values / 100.0))]
        )

    def ev_ret(code: str, pub: pd.Timestamp) -> Optional[float]:
        dates = dates_by.get(code)
        if dates is None or len(dates) == 0:
            return None
        idx = int(np.searchsorted(dates, pub.to_datetime64(), side="right"))
        a, b = idx + HOLD_START, idx + HOLD_START + 60 - 1
        cs0 = cs0_by[code]
        if b + 1 >= len(cs0) or a < 1:
            return None
        ret = float(np.expm1(cs0[b + 1] - cs0[a]))
        i_a = int(np.searchsorted(mkt_dates, dates[a - 1], side="right"))
        i_b = int(np.searchsorted(mkt_dates, dates[b], side="right"))
        if i_b > len(mkt_cs0) - 1:
            return None
        return ret - float(np.expm1(mkt_cs0[i_b] - mkt_cs0[i_a]))

    print("计算 60d BHAR...")
    df["ret60"] = [ev_ret(r.code, r.pub_date) for _, r in df.iterrows()]

    # 控制变量（交互项 FM 用）
    with open(INDUSTRY_MAP, encoding="utf-8") as f:
        im = json.load(f)
    s2l2 = im["stockToIndustry"]
    l2l1 = {i["name"]: i["l1Name"] for i in im["industries"]}
    df["l1"] = df["code"].map(
        lambda c: l2l1.get(s2l2.get(c.split(".")[-1], "UNK"), "UNK")
    )
    df["log_mc"] = np.log(
        df["market_cap_yi"].clip(lower=1e-6) * 1e8
    ).where(df["mktcap_missing"].fillna(1) == 0)
    df = df.sort_values(["code", "fiscal_year", "fiscal_quarter"])
    ac_list: list[float] = []
    for code, grp in df.groupby("code", sort=False):
        sues = grp["sue"].tolist()
        for i in range(len(sues)):
            h = sues[:i]
            if len(h) < 4:
                ac_list.append(float("nan"))
                continue
            y_a = np.array(h[1:]); x_a = np.array(h[:-1]); v = np.var(x_a)
            ac_list.append(
                0.0 if v < 1e-12
                else max(-1.0, min(1.0, float(np.cov(x_a, y_a)[0, 1] / v)))
            )
    df["sue_autocorr"] = ac_list

    # ══ B2：三层分组（流动性 turn_20d 与市值两维各跑）═══════════════════════════
    def layered_spreads(layer_col: str, label: str) -> None:
        print()
        print(f"█ B2 — {label} 三层分组 × SUE 五分组（全样本，60d BHAR 价差）")
        spread_by_layer: dict[int, dict[tuple[int, int], float]] = {1: {}, 2: {}, 3: {}}
        for (fy, fq), g in df.groupby(["fiscal_year", "fiscal_quarter"]):
            d = g.dropna(subset=["ret60", "sue", layer_col])
            n = len(d)
            if n < MIN_N_LAYER:
                continue
            lr = d[layer_col].rank(method="first")
            layer = np.ceil(lr / n * 3).clip(1, 3)      # 1=最低（最不流动/最小）
            for lay in (1, 2, 3):
                sub = d[layer == lay]
                ns = len(sub)
                if ns < 30:
                    continue
                rk = sub["sue"].rank(method="first")
                qq = np.ceil(rk / ns * 5).clip(1, 5)
                m5 = sub.loc[qq == 5, "ret60"].mean()
                m1 = sub.loc[qq == 1, "ret60"].mean()
                spread_by_layer[lay][(int(fy), int(fq))] = float(m5 - m1)

        names = {1: f"{label}最低层", 2: f"{label}中层", 3: f"{label}最高层"}
        print(f"{'层':<12}{'2010-2021':>26}{'2022-2024':>26}{'全期':>26}")
        for lay in (1, 2, 3):
            m1_, t1_, n1_ = era_stat(spread_by_layer[lay], 2010, ERA_SPLIT)
            m2_, t2_, n2_ = era_stat(spread_by_layer[lay], ERA_SPLIT + 1, 2024)
            ma_, ta_, na_ = era_stat(spread_by_layer[lay], 2010, 2024)
            print(f"{names[lay]:<12}"
                  f"{m1_:+8.4f} (t={t1_:+5.2f},{n1_:>2}期)"
                  f"{m2_:+8.4f} (t={t2_:+5.2f},{n2_:>2}期)"
                  f"{ma_:+8.4f} (t={ta_:+5.2f},{na_:>2}期)")

    layered_spreads("turn_20d", "流动性")
    layered_spreads("market_cap_yi", "市值")

    # ── B2 交互项 FM（全样本，全控制 + sue×illiq + sue×small）──────────────────
    print()
    print("█ B2 — FM 交互项（全样本，全控制）")
    print("  模型: ret60 ~ sue_z + illiq_z + small_z + sue×illiq + sue×small + ac + mc + 行业")
    g_inter_illiq: dict[tuple[int, int], float] = {}
    g_inter_small: dict[tuple[int, int], float] = {}
    for (fy, fq), g in df.groupby(["fiscal_year", "fiscal_quarter"]):
        d = g.dropna(subset=["ret60", "sue", "turn_20d", "market_cap_yi",
                             "sue_autocorr", "log_mc"])
        n = len(d)
        if n < MIN_N_LAYER:
            continue
        sr = d["sue"].rank() / (n + 1) - 0.5
        sue_z = ((sr - sr.mean()) / (sr.std() + 1e-8)).values
        ir = (-d["turn_20d"]).rank() / (n + 1) - 0.5          # 越不流动越大
        illiq_z = ((ir - ir.mean()) / (ir.std() + 1e-8)).values
        sm = (-d["market_cap_yi"]).rank() / (n + 1) - 0.5     # 越小越大
        small_z = ((sm - sm.mean()) / (sm.std() + 1e-8)).values
        ac = d["sue_autocorr"]
        ac_z = ((ac - ac.mean()) / (ac.std() + 1e-8)).values
        mc = d["log_mc"]
        mc_z = ((mc - mc.mean()) / (mc.std() + 1e-8)).values
        inds = pd.get_dummies(d["l1"], prefix="i", drop_first=True)
        x_mat = np.hstack([
            np.ones((n, 1)),
            sue_z.reshape(-1, 1),                    # [1]
            illiq_z.reshape(-1, 1),                  # [2]
            small_z.reshape(-1, 1),                  # [3]
            (sue_z * illiq_z).reshape(-1, 1),        # [4] 交互：流动性
            (sue_z * small_z).reshape(-1, 1),        # [5] 交互：市值
            ac_z.reshape(-1, 1),
            mc_z.reshape(-1, 1),
            inds.values.astype(float),
        ])
        try:
            coef, _, rk_, _ = np.linalg.lstsq(x_mat, d["ret60"].values, rcond=None)
            if rk_ < x_mat.shape[1]:
                continue
            g_inter_illiq[(int(fy), int(fq))] = float(coef[4])
            g_inter_small[(int(fy), int(fq))] = float(coef[5])
        except np.linalg.LinAlgError:
            continue

    for d_, lbl in [(g_inter_illiq, "sue×illiq（流动性悖论）"),
                    (g_inter_small, "sue×small（市值悖论）")]:
        m1_, t1_, n1_ = era_stat(d_, 2010, ERA_SPLIT)
        m2_, t2_, n2_ = era_stat(d_, ERA_SPLIT + 1, 2024)
        ma_, ta_, na_ = era_stat(d_, 2010, 2024)
        print(f"  {lbl:<26} 2010-2021: {m1_:+.5f}(t={t1_:+.2f})"
              f"  2022-2024: {m2_:+.5f}(t={t2_:+.2f})"
              f"  全期: {ma_:+.5f}(t={ta_:+.2f},{na_}期)")

    # ══ B4：成本瀑布（流动池 long-only top 五分组）═════════════════════════════
    print()
    print("█ B4 — 成本瀑布（流动池 top 五分组 long-only，每季一次 round-trip）")
    # ⚠ 基准口径：BHAR 的等权市场基准含全部 5131 只（小盘溢价），流动池偏大盘，
    #   直接看 long 腿 BHAR 会得到"基准失配为负"的假象。
    #   可执行对象的正确度量 = top 五分组对池内同期均值的主动超额（cohort 内自基准）。
    liq = df[df["in_liquid_pool"] == 1]
    long_gross: dict[tuple[int, int], float] = {}     # 主动超额：m5 − 池均值
    ls_gross: dict[tuple[int, int], float] = {}
    for (fy, fq), g in liq.groupby(["fiscal_year", "fiscal_quarter"]):
        d = g.dropna(subset=["ret60", "sue"])
        n = len(d)
        if n < 50:
            continue
        rk = d["sue"].rank(method="first")
        qq = np.ceil(rk / n * 5).clip(1, 5)
        m5 = float(d.loc[qq == 5, "ret60"].mean())
        m1 = float(d.loc[qq == 1, "ret60"].mean())
        pool_mean = float(d["ret60"].mean())
        long_gross[(int(fy), int(fq))] = m5 - pool_mean
        ls_gross[(int(fy), int(fq))] = m5 - m1

    total_cost = COST_COMMISSION + COST_STAMP + COST_IMPACT
    print(f"  成本参数（注册值，标'需验证'）: 佣金 {COST_COMMISSION:.2%} + "
          f"印花 {COST_STAMP:.2%} + 冲击 {COST_IMPACT:.2%} = {total_cost:.2%}/round-trip")
    print()
    print(f"{'瀑布层':<26}{'2010-2021':>22}{'2022-2024':>22}")
    rows = [
        ("Long 主动超额(vs池均)/季", 0.0),
        ("− 佣金", COST_COMMISSION),
        ("− 印花税", COST_COMMISSION + COST_STAMP),
        ("− 冲击成本 = Net", total_cost),
    ]
    for lbl, cum_cost in rows:
        d_net = {k: v - cum_cost for k, v in long_gross.items()}
        m1_, t1_, _ = era_stat(d_net, 2010, ERA_SPLIT)
        m2_, t2_, _ = era_stat(d_net, ERA_SPLIT + 1, 2024)
        print(f"{lbl:<26}{m1_:+10.4f} (t={t1_:+5.2f}){m2_:+10.4f} (t={t2_:+5.2f})")

    # L/S 纸面版（披露用，空头腿不可执行）
    print()
    print("  [纸面披露] L/S 五分组价差（空头腿 A 股不可执行，双腿成本 2×）：")
    d_ls_net = {k: v - 2 * total_cost for k, v in ls_gross.items()}
    for d_, lbl in [(ls_gross, "L/S gross"), (d_ls_net, "L/S net(纸面)")]:
        m1_, t1_, _ = era_stat(d_, 2010, ERA_SPLIT)
        m2_, t2_, _ = era_stat(d_, ERA_SPLIT + 1, 2024)
        print(f"    {lbl:<16} 2010-2021: {m1_:+.4f}(t={t1_:+.2f})"
              f"  2022-2024: {m2_:+.4f}(t={t2_:+.2f})")

    print()
    print("[OK] 09_paradox_quant.py 完成")


if __name__ == "__main__":
    main()
