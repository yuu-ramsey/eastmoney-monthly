"""
实验:行业+市值双中性 LS(标准 style-neutral,a-priori)→ 在 +0.30 基础上能否再推
============================================================================
行业中性已 +0.30。标准 quant 做法是同时中性化已知风险因子(行业+市值)。
信号先按行业去均值,再每周对 size 回归取残差(剔除市值暴露)。a-priori、不搜索哪些因子。
对比:基线 / 仅行业 / 行业+size。leak-free、非重叠月度。实验记录。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PE = "data/pred_ensemble.parquet"; PURGED = "data/pred_purged.parquet"; LGBP = "data/pred_v3_lgb_purged.parquet"
PANEL = "data/mf_panel_weekly_v3.parquet"


def ann(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")
def aggf(w, f):
    s = w.dropna().copy(); s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period(f)).prod() - 1
def cs_z(df, c): return df.groupby("wk")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def resid_on_size(g):
    x = g["size"].astype(float).values; y = g["score"].values
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return y
    A = np.vstack([x[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    out = y.copy(); out[ok] = y[ok] - A @ coef
    return out


def build(mode):
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]; pl = pd.read_parquet(PANEL)[["code", "wk", "l1", "size"]]
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").merge(pl, on=["code", "wk"], how="left").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")
    if mode in ("ind", "ind_size"):
        m["score"] = m["score"] - m.groupby(["wk", "l1"])["score"].transform("mean")
    if mode == "ind_size":
        m["score"] = m.groupby("wk", group_keys=False).apply(lambda g: pd.Series(resid_on_size(g), index=g.index))
    m["sm"] = m.groupby("code")["score"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
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
        res.append({"wk": str(wk), "ls": (rt - rb) - (tot + tob) * 0.0008 - 0.08 / 52})
    ls = pd.DataFrame(res).set_index("wk")["ls"]
    yr = pd.Series(ls.index.str.slice(0, 4).astype(int), index=ls.index); o = ls[yr >= 2021]
    return ann(o, 52), ann(aggf(o, "M"), 12), ann(aggf(o, "Q"), 4)


def main():
    for name, mode in [("基线", "base"), ("行业中性", "ind"), ("行业+市值中性", "ind_size")]:
        w, mn, q = build(mode)
        print(name + ": 周" + str(round(w, 2)) + " 月" + str(round(mn, 2)) + " 季" + str(round(q, 2)), flush=True)
    print("[OK] 105 双中性 完成", flush=True)


if __name__ == "__main__":
    main()
