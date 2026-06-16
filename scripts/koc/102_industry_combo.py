"""
实验:行业中性 LS(+0.30)→ 组合 + 预注册2×成本压测,看 Version B 现在到哪
============================================================================
101 发现行业中性化把 LS 月 2.27→2.58。本脚本把这个更强 LS 灌进组合,跑同一预注册压测:
等权(LS+可转债+打新),1×/2×成本,2025留出。之前旧LS的2×成本=2.53,看现在。实验记录。
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


def ls_industry_neutral(cost_mult=1.0, borrow=0.08):
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]
    pl = pd.read_parquet(PANEL)[["code", "wk", "l1"]]
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").merge(pl, on=["code", "wk"], how="left").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")
    m["score_in"] = m["score"] - m.groupby(["wk", "l1"])["score"].transform("mean")   # 行业中性
    m["sm"] = m.groupby("code")["score_in"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
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

    def combo(cost_mult, holdout=False):
        ls = ls_industry_neutral(cost_mult); ls.index = ls.index.astype(str)
        df = pd.concat({"LS": ls, "可转债": cb, "打新": dz}, axis=1); df["打新"] = df["打新"].fillna(0.0)
        df = df.dropna(subset=["LS", "可转债"]); yr = pd.Series(df.index.str.slice(0,4).astype(int), index=df.index)
        po = (df[["LS", "可转债", "打新"]] / 3).sum(axis=1)
        sub = po[yr == 2025] if holdout else po[yr >= 2021]
        return ann(agg_m(sub), 12)

    print("=== 行业中性 LS 组合(等权)预注册压测 ===", flush=True)
    c1 = combo(1.0); c2 = combo(2.0); ho = combo(1.0, holdout=True)
    print("基线成本1×: 月√12=" + str(round(c1, 2)), flush=True)
    print("2×成本(真实): 月√12=" + str(round(c2, 2)) + "  (旧LS=2.53; 线≥2.6)", flush=True)
    print("2025留出: 月√12=" + str(round(ho, 2)) + "  (线≥2.0)", flush=True)
    print("裁定: 2×成本 " + ("≥2.8 突破! ✓✓" if c2 >= 2.8 else ("≥2.6 过压测线" if c2 >= 2.6 else "<2.6")), flush=True)
    print("[OK] 102 行业中性组合 完成", flush=True)


if __name__ == "__main__":
    main()
