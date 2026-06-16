"""
实验:行业中性化 LS(信号按行业去均值)→ LS 是否升(Version B,CPU快,非OHLCV)
============================================================================
假设:LS 现跨全市场排名,残留行业暴露=额外波动。按 l1 行业去均值化信号后排名,
去掉行业 bet → 更纯市场中性 → 同收益低波 → Sharpe 或升。leak-free,非重叠月度。实验记录。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PE = "data/pred_ensemble.parquet"; PURGED = "data/pred_purged.parquet"; LGBP = "data/pred_v3_lgb_purged.parquet"
PANEL = "data/mf_panel_weekly_v3.parquet"


def ann(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")
def agg_m(w):
    s = w.dropna().copy(); s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1
def cs_z(df, c): return df.groupby("wk")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def build_ls(m, score):
    f = m.copy(); f["sm"] = f.groupby("code")[score].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    res, pt, pb = [], {}, {}
    for wk, g in f.groupby("wk"):
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
    yr = pd.Series(ls.index.str.slice(0, 4).astype(int), index=ls.index)
    return ann(agg_m(ls[yr >= 2021]), 12), ann(ls[yr >= 2021], 52)


def main():
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]
    pl = pd.read_parquet(PANEL)[["code", "wk", "l1"]]
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").merge(pl, on=["code", "wk"], how="left").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")
    # 行业中性:按 (wk,l1) 去均值
    m["score_in"] = m["score"] - m.groupby(["wk", "l1"])["score"].transform("mean")
    base_m, base_w = build_ls(m, "score")
    neu_m, neu_w = build_ls(m, "score_in")
    print("LS 基线(跨市场): 月SR=" + str(round(base_m, 3)) + " 周SR=" + str(round(base_w, 3)), flush=True)
    print("LS 行业中性:     月SR=" + str(round(neu_m, 3)) + " 周SR=" + str(round(neu_w, 3)), flush=True)
    print("DELTA(中性-基线) 月=" + str(round(neu_m - base_m, 3)), flush=True)
    print("[OK] 101 行业中性 完成", flush=True)


if __name__ == "__main__":
    main()
