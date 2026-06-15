"""
时代切分机制诊断（2026-06-10，预先指定的机制检验，非信号挖掘）

任务（对应用户指令 1-4）：
  1. 正典切分 2010-2021 / 2022-2024 的全套统计（FM γ1 端 + 价差端）
  2. 机制诊断：公告窗（第1-2交易日）即时反应强度分时段对比
     预测：若漂移死亡伴随即时反应增强 → "消化提速"机制闭环
     若即时反应同步走弱 → 机制故事不完整，须如实记录
  3. 稳健性 a：剔除 fiscal 2024Q1 cohort 重算 2022-2024
     稳健性 b：按日历重叠界定 2024-02 量化危机污染 cohort（中位窗口与
       [2024-01-15, 2024-03-31] 重叠者）并剔除重算——fiscal 2024Q1 的窗口
       在 2024-05~08，未必罩住 2 月危机，须实证核对
  4. 稳健性 c：2010-2021 剔除 2016（熔断年）——脚注支持

Usage: python scripts/koc/07_era_mechanism.py
"""
import json
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

DB = "data/pead-baostock.sqlite"
INDUSTRY_MAP = "data/industry-map.json"
NW = 4
MIN_N_FM = 30
MIN_N_SPREAD = 50
HOLD_START = 5

ERA_OLD_END = 2021      # 正典切分：≤2021 旧时代，≥2022 新时代
CRASH_LO = np.datetime64("2024-01-15")
CRASH_HI = np.datetime64("2024-03-31")


def nwt(series: np.ndarray) -> tuple[float, float]:
    s = np.asarray(series, dtype=float)
    m = float(s.mean())
    e = s - m
    var = float(np.dot(e, e))
    for lag in range(1, min(NW, len(s) - 1) + 1):
        var += 2.0 * (1.0 - lag / (NW + 1)) * float(np.dot(e[lag:], e[:-lag]))
    se = float(np.sqrt(max(var, 0.0) / (len(s) ** 2)))
    return m, m / (se + 1e-12)


def main() -> None:
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT sb.code, sb.fiscal_year, sb.fiscal_quarter, sb.pub_date, sb.sue, "
            "ku.in_liquid_pool, ku.market_cap_yi, ku.mktcap_missing "
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

    def win(code: str, pub: pd.Timestamp, d_from: int, d_to: int,
            adj: bool) -> Optional[tuple[float, np.datetime64, np.datetime64]]:
        """返回 (收益, 窗首日历日, 窗末日历日)。adj=True 减同窗市场收益。"""
        dates = dates_by.get(code)
        if dates is None or len(dates) == 0:
            return None
        idx = int(np.searchsorted(dates, pub.to_datetime64(), side="right"))
        a, b = idx + d_from - 1, idx + d_to - 1
        cs0 = cs0_by[code]
        if b + 1 >= len(cs0) or a < 1:
            return None
        ret = float(np.expm1(cs0[b + 1] - cs0[a]))
        if adj:
            i_a = int(np.searchsorted(mkt_dates, dates[a - 1], side="right"))
            i_b = int(np.searchsorted(mkt_dates, dates[b], side="right"))
            if i_b > len(mkt_cs0) - 1:
                return None
            ret -= float(np.expm1(mkt_cs0[i_b] - mkt_cs0[i_a]))
        return ret, dates[a], dates[b]

    print("计算公告窗（1-2日）与漂移窗（6-65日）收益...")
    ann_raw: list[Optional[float]] = []
    ann_adj: list[Optional[float]] = []
    drift_adj: list[Optional[float]] = []
    w_start: list[Optional[np.datetime64]] = []
    w_end: list[Optional[np.datetime64]] = []
    for _, r in df.iterrows():
        a_raw = win(r.code, r.pub_date, 1, 2, adj=False)
        a_adj = win(r.code, r.pub_date, 1, 2, adj=True)
        d_adj = win(r.code, r.pub_date, HOLD_START + 1, HOLD_START + 60, adj=True)
        ann_raw.append(a_raw[0] if a_raw else None)
        ann_adj.append(a_adj[0] if a_adj else None)
        if d_adj:
            drift_adj.append(d_adj[0]); w_start.append(d_adj[1]); w_end.append(d_adj[2])
        else:
            drift_adj.append(None); w_start.append(None); w_end.append(None)
    df["ann_raw"] = ann_raw
    df["ann_adj"] = ann_adj
    df["drift_adj"] = drift_adj
    df["w_start"] = w_start
    df["w_end"] = w_end

    # 控制变量（FM 用）
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

    liq = df[df["in_liquid_pool"] == 1]

    # ── 每财季统计 ────────────────────────────────────────────────────────────
    per_q: dict[tuple[int, int], dict] = {}
    for (fy, fq), g in liq.groupby(["fiscal_year", "fiscal_quarter"]):
        key = (int(fy), int(fq))
        rec: dict = {}
        # 公告窗 IC + 价差
        d_ann = g.dropna(subset=["ann_raw", "sue"])
        if len(d_ann) >= MIN_N_SPREAD:
            rho, _ = stats.spearmanr(d_ann["sue"], d_ann["ann_raw"])
            rec["ann_ic"] = float(rho)
            d2 = g.dropna(subset=["ann_adj", "sue"])
            n = len(d2)
            rk = d2["sue"].rank(method="first")
            qq = np.ceil(rk / n * 5).clip(1, 5)
            rec["ann_spread"] = float(
                d2.loc[qq == 5, "ann_adj"].mean() - d2.loc[qq == 1, "ann_adj"].mean()
            )
        # 漂移价差 + 窗口日历
        d_dr = g.dropna(subset=["drift_adj", "sue"])
        if len(d_dr) >= MIN_N_SPREAD:
            n = len(d_dr)
            rk = d_dr["sue"].rank(method="first")
            qq = np.ceil(rk / n * 5).clip(1, 5)
            rec["drift_spread"] = float(
                d_dr.loc[qq == 5, "drift_adj"].mean() - d_dr.loc[qq == 1, "drift_adj"].mean()
            )
            rec["w_start_med"] = pd.Series(d_dr["w_start"].values).median()
            rec["w_end_med"] = pd.Series(d_dr["w_end"].values).median()
        # FM γ1（rank 全控制）
        d_fm = g.dropna(subset=["drift_adj", "sue", "sue_autocorr", "log_mc"])
        n = len(d_fm)
        if n >= MIN_N_FM:
            sr = d_fm["sue"].rank() / (n + 1) - 0.5
            sue_z = ((sr - sr.mean()) / (sr.std() + 1e-8)).values
            ac = d_fm["sue_autocorr"]
            ac_z = ((ac - ac.mean()) / (ac.std() + 1e-8)).values
            mc = d_fm["log_mc"]
            mc_z = ((mc - mc.mean()) / (mc.std() + 1e-8)).values
            inds = pd.get_dummies(d_fm["l1"], prefix="i", drop_first=True)
            x_mat = np.hstack([
                np.ones((n, 1)), sue_z.reshape(-1, 1), ac_z.reshape(-1, 1),
                mc_z.reshape(-1, 1), inds.values.astype(float),
            ])
            try:
                coef, _, rk_, _ = np.linalg.lstsq(x_mat, d_fm["drift_adj"].values, rcond=None)
                if rk_ == x_mat.shape[1]:
                    rec["fm_gamma"] = float(coef[1])
            except np.linalg.LinAlgError:
                pass
        if rec:
            per_q[key] = rec

    def era_stat(metric: str, lo: int, hi: int,
                 exclude: set[tuple[int, int]] | None = None,
                 exclude_years: set[int] | None = None) -> tuple[float, float, int]:
        vals = [
            rec[metric] for (fy, fq), rec in sorted(per_q.items())
            if lo <= fy <= hi and metric in rec
            and (exclude is None or (fy, fq) not in exclude)
            and (exclude_years is None or fy not in exclude_years)
        ]
        if len(vals) < 4:
            return float("nan"), float("nan"), len(vals)
        m, t = nwt(np.array(vals))
        return m, t, len(vals)

    # ══ 1. 正典切分全套 ═══════════════════════════════════════════════════════
    print()
    print("█ 1. 正典切分 2010-2021 / 2022-2024（流动池）")
    print(f"{'指标':<22}{'2010-2021':>26}{'2022-2024':>26}")
    for metric, label in [
        ("fm_gamma", "FM γ1（rank全控制）"),
        ("drift_spread", "漂移价差 Q5-Q1/季"),
    ]:
        m1, t1, n1 = era_stat(metric, 2010, ERA_OLD_END)
        m2, t2, n2 = era_stat(metric, ERA_OLD_END + 1, 2024)
        print(f"{label:<22}{m1:+10.4f} (t={t1:+5.2f},{n1}期)"
              f"{m2:+10.4f} (t={t2:+5.2f},{n2}期)")

    # ══ 2. 机制诊断：公告窗即时反应分时段 ═════════════════════════════════════
    print()
    print("█ 2. 机制诊断：公告窗（第1-2交易日）即时反应强度")
    for metric, label in [
        ("ann_ic", "公告窗 Spearman IC"),
        ("ann_spread", "公告窗价差 Q5-Q1"),
    ]:
        m1, t1, n1 = era_stat(metric, 2010, ERA_OLD_END)
        m2, t2, n2 = era_stat(metric, ERA_OLD_END + 1, 2024)
        print(f"{label:<22}{m1:+10.4f} (t={t1:+5.2f},{n1}期)"
              f"{m2:+10.4f} (t={t2:+5.2f},{n2}期)")
    print("  判读：新时代即时反应若 ≥ 旧时代且漂移死亡 → '消化提速'闭环；")
    print("        若同步走弱 → 机制故事不完整，如实记录。")

    # ══ 3. 危机重叠 cohort 实证核对 ═══════════════════════════════════════════
    print()
    print("█ 3. 2024-02 量化危机日历重叠核对（cohort 中位漂移窗）")
    crash_cohorts: set[tuple[int, int]] = set()
    for (fy, fq), rec in sorted(per_q.items()):
        if "w_start_med" not in rec or fy < 2023:
            continue
        ws, we = rec["w_start_med"], rec["w_end_med"]
        overlap = (np.datetime64(ws) <= CRASH_HI) and (np.datetime64(we) >= CRASH_LO)
        if fy >= 2023:
            print(f"  {fy}Q{fq}: 中位窗 {pd.Timestamp(ws).date()} ~ {pd.Timestamp(we).date()}"
                  f"  {'⚠ 与危机窗重叠' if overlap else ''}")
        if overlap:
            crash_cohorts.add((fy, fq))
    print(f"  危机重叠 cohort: {sorted(crash_cohorts)}")

    # ══ 4. 稳健性 ═════════════════════════════════════════════════════════════
    print()
    print("█ 4. 稳健性重算（漂移价差，2022-2024）")
    m0, t0, n0 = era_stat("drift_spread", 2022, 2024)
    print(f"  基准:                {m0:+.4f} (t={t0:+.2f}, {n0}期)")
    m_a, t_a, n_a = era_stat("drift_spread", 2022, 2024, exclude={(2024, 1)})
    print(f"  剔除 fiscal 2024Q1:  {m_a:+.4f} (t={t_a:+.2f}, {n_a}期)（用户指令字面口径）")
    if crash_cohorts:
        m_b, t_b, n_b = era_stat("drift_spread", 2022, 2024, exclude=crash_cohorts)
        print(f"  剔除危机重叠 cohort: {m_b:+.4f} (t={t_b:+.2f}, {n_b}期)"
              f"（日历口径 {sorted(crash_cohorts)}）")
    print()
    print("█ 4c. 稳健性：2010-2021 剔除 2016 熔断年（漂移价差）")
    m_c0, t_c0, n_c0 = era_stat("drift_spread", 2010, 2021)
    m_c, t_c, n_c = era_stat("drift_spread", 2010, 2021, exclude_years={2016})
    print(f"  含 2016:  {m_c0:+.4f} (t={t_c0:+.2f}, {n_c0}期)")
    print(f"  剔 2016:  {m_c:+.4f} (t={t_c:+.2f}, {n_c}期)")


if __name__ == "__main__":
    main()
