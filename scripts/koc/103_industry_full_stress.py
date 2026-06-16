"""
行业中性 LS 组合 → 完整预注册压测(同 94 阈值,不移球门)确认是否稳过2.8
============================================================================
102 发现行业中性 LS 把 2×成本组合推到 2.89。本脚本跑 94 的全套预注册压测验证:
A成本2×≥2.60&3×≥2.40 | B 0.5×打新+2×≥2.30 | C 2025留出≥2.00 | 尾部 | moving-block bootstrap
kill线: min(2×,0.5打新,2025)<2.30 → 否。不调参,实验记录。
"""
from __future__ import annotations

import akshare as ak
import numpy as np
import pandas as pd

PE = "data/pred_ensemble.parquet"; PURGED = "data/pred_purged.parquet"; LGBP = "data/pred_v3_lgb_purged.parquet"
PANEL = "data/mf_panel_weekly_v3.parquet"; CBR = "data/cb_v3_returns.parquet"; REF = 1_000_000


def ann(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")
def agg_m(w):
    s = w.dropna().copy(); s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1
def cs_z(df, c): return df.groupby("wk")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def ls_in(cost_mult=1.0, borrow=0.08):
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]; pl = pd.read_parquet(PANEL)[["code", "wk", "l1"]]
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").merge(pl, on=["code", "wk"], how="left").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")
    m["si"] = m["score"] - m.groupby(["wk", "l1"])["score"].transform("mean")
    m["sm"] = m.groupby("code")["si"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    res, pt, pb = [], {}, {}
    for wk, g in m.groupby("wk"):
        d = g.dropna(subset=["sm", "fwd"]); n = len(d)
        if n < 50: continue
        rk = d["sm"].rank(method="first"); grp = np.ceil(rk / n * 10).clip(1, 10)
        top, bot = d[grp == 10], d[grp == 1]
        if len(top) < 3 or len(bot) < 3: continue
        rt, rb = float(top["fwd"].mean()), float(bot["fwd"].mean())
        tw = {c: 1/len(top) for c in top["code"]}; bw = {c: 1/len(bot) for c in bot["code"]}
        tot = sum(abs(tw.get(c, 0) - pt.get(c, 0)) for c in set(tw) | set(pt))
        tob = sum(abs(bw.get(c, 0) - pb.get(c, 0)) for c in set(bw) | set(pb)); pt, pb = tw, bw
        res.append({"wk": str(wk), "ls": (rt - rb) - (tot + tob) * 0.0008 * cost_mult - borrow / 52})
    return pd.DataFrame(res).set_index("wk")["ls"]


def dazin():
    dz = ak.stock_xgsglb_em(symbol="全部股票").rename(columns={"申购日期": "date", "中签率": "rate", "每中一签获利": "profit", "申购上限": "cap"})
    dz["date"] = pd.to_datetime(dz["date"], errors="coerce")
    for c in ("rate", "profit", "cap"): dz[c] = pd.to_numeric(dz[c], errors="coerce")
    dz = dz.dropna(subset=["date", "rate", "profit"]); dz = dz[(dz["rate"] > 0) & (dz["profit"] > 0)]
    dz["units"] = (dz["cap"].fillna(dz["cap"].median()) / 500).clip(upper=2000); dz["ep"] = dz["rate"] / 100 * dz["profit"] * dz["units"]
    dz["wk"] = dz["date"].dt.to_period("W-FRI").astype(str)
    return (dz.groupby("wk")["ep"].sum() / REF)


def main():
    cb = pd.read_parquet(CBR).pipe(lambda d: d.set_index(d.index.astype(str))["ret_net"]); cb.index = cb.index.astype(str)
    dz = dazin(); dz.index = dz.index.astype(str)

    def combo(cost_mult, cb_extra=0.0, dz_scale=1.0, dz_slip=1.0, holdout=False):
        ls = ls_in(cost_mult); ls.index = ls.index.astype(str)
        cbx = cb - cb_extra; dzx = dz * dz_scale * dz_slip
        df = pd.concat({"LS": ls, "可转债": cbx, "打新": dzx}, axis=1); df["打新"] = df["打新"].fillna(0.0)
        df = df.dropna(subset=["LS", "可转债"]); yr = pd.Series(df.index.str.slice(0,4).astype(int), index=df.index)
        po = (df[["LS", "可转债", "打新"]] / 3).sum(axis=1)
        return po[yr == 2025] if holdout else po[yr >= 2021]

    base = ann(agg_m(combo(1.0)), 12)
    a2 = ann(agg_m(combo(2.0, cb_extra=0.0005, dz_slip=0.98)), 12)
    a3 = ann(agg_m(combo(3.0, cb_extra=0.0010, dz_slip=0.98)), 12)
    b = ann(agg_m(combo(2.0, cb_extra=0.0005, dz_scale=0.5, dz_slip=0.98)), 12)
    c = ann(agg_m(combo(1.0, holdout=True)), 12)
    print("基线=" + str(round(base, 2)), flush=True)
    print("A成本 2×=" + str(round(a2, 2)) + "(≥2.60) 3×=" + str(round(a3, 2)) + "(≥2.40) → " + ("PASS" if a2 >= 2.6 and a3 >= 2.4 else "FAIL"), flush=True)
    print("B 0.5打新+2×=" + str(round(b, 2)) + "(≥2.30) → " + ("PASS" if b >= 2.3 else "FAIL"), flush=True)
    print("C 2025留出=" + str(round(c, 2)) + "(≥2.00) → " + ("PASS" if c >= 2.0 else "FAIL"), flush=True)
    dmin = min(a2, b, c)
    print("kill线 min(2×,0.5打新,2025)=" + str(round(dmin, 2)) + " → " + ("稳过 ✓" if dmin >= 2.3 else "否"), flush=True)
    # 尾部 + moving-block bootstrap
    mo = agg_m(combo(2.0, cb_extra=0.0005, dz_slip=0.98))
    cum = (1 + mo).cumprod(); dd = (cum / cum.cummax() - 1).min()
    r = mo.dropna().values; rng = np.random.default_rng(7); sb = []
    for _ in range(3000):
        idx = rng.integers(0, max(1, len(r) - 6), max(1, len(r) // 6)); x = np.concatenate([r[i:i+6] for i in idx])
        sb.append(x.mean() * 12 / (x.std(ddof=1) * np.sqrt(12) + 1e-12))
    sb = np.array(sb)
    print("尾部最大回撤(2×)=" + str(round(dd * 100, 1)) + "%  | bootstrap(6月)=" + str(round(sb.mean(), 2)) + " P(>2.8)=" + str(round(np.mean(sb > 2.8) * 100)) + "%", flush=True)
    print("[OK] 103 行业中性全压测 完成", flush=True)


if __name__ == "__main__":
    main()
