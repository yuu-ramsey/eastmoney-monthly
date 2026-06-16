"""
实验:LGB 用全部面板因子(现仅15)→ LS gross 能否升(Version B alpha,非OHLCV,CPU快)
============================================================================
假设:面板有 ~20 因子,LGB 现只用 15(F16)。加全部 → LS 信号或更强。
leak-free purged 重训,与 tcn_purged 组 ensemble,LS gross(0成本)对比。实验记录非天花板。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

PANEL = "data/mf_panel_weekly_v3.parquet"; PE = "data/pred_ensemble.parquet"; PURGED = "data/pred_purged.parquet"
EMBARGO_WK = 16


def xrank(s): r = s.rank(method="average"); return r / (r.notna().sum() + 1) - 0.5
def ann(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")
def agg_m(w):
    s = w.dropna().copy(); s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1
def cs_z(df, c): return df.groupby("wk")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def lgb_purged(panel, rfeat, years):
    lp = dict(objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=300,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=200, verbose=-1)
    preds = np.full(len(panel), np.nan, dtype=np.float32)
    for Y in range(2019, 2026):
        cut = pd.Timestamp(str(Y) + "-01-01") - pd.Timedelta(weeks=EMBARGO_WK)
        tr = (years < Y) & (panel["_wkend"].values < np.datetime64(cut)); te = years == Y
        if tr.sum() < 5000 or te.sum() < 50: continue
        m = lgb.LGBMRegressor(**lp).fit(panel.loc[tr, rfeat], panel.loc[tr, "ylab"])
        preds[te] = m.predict(panel.loc[te, rfeat])
    return preds


def build_ls_sr(m, score):
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
        res.append({"wk": str(wk), "ls": (rt - rb) - (tot + tob) * 0.0008})   # 0融券费,看 gross 趋势
    ls = pd.DataFrame(res).set_index("wk")["ls"]
    yr = pd.Series(ls.index.str.slice(0, 4).astype(int), index=ls.index)
    return ann(agg_m(ls[yr >= 2021]), 12)


def main():
    w = pd.read_parquet(PANEL); w["year"] = w["year"].astype(int)
    F16 = ["reversal", "lowvol", "illiq", "size", "sue", "maxret", "turn", "f_idio", "f_amihud",
           "f_turnmom", "f_skew", "f_downvol", "f_sdv", "f_volskew", "f_turnac"]
    ALL = [c for c in w.columns if c not in ["code", "wk", "year", "wret", "fwd", "l1", "stflag", "hist_n", "amt20"]]
    wk_end = pd.PeriodIndex(w["wk"].astype(str), freq="W-FRI").to_timestamp(how="end"); w = w.assign(_wkend=wk_end)
    print("F16=" + str(len(F16)) + " ALL=" + str(len(ALL)), flush=True)
    out = {}
    for name, feats in [("F16", F16), ("ALL", ALL)]:
        cols = [c for c in feats if c in w.columns]
        rows = []
        for wk, g in w.groupby("wk"):
            g = g[(g["stflag"] == 0) & (g["hist_n"] >= 52) & g["amt20"].notna()]
            if len(g) < 100: continue
            g = g[g["amt20"] >= g["amt20"].quantile(0.30)].copy()
            if len(g) < 100: continue
            for f in cols: g["r_" + f] = xrank(g[f]).fillna(0.0)
            g["ylab"] = g["fwd"] - g["fwd"].mean(); rows.append(g)
        panel = pd.concat(rows, ignore_index=True); rfeat = ["r_" + f for f in cols]
        panel["lgbx"] = lgb_purged(panel, rfeat, panel["year"].values)
        pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
        lx = panel[["code", "wk", "lgbx"]].dropna(subset=["lgbx"])
        m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lx, on=["code", "wk"], how="inner").dropna(subset=["tcn_purged", "lgbx", "fwd"])
        m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgbx")
        out[name] = build_ls_sr(m, "score")
        print(name + " LS gross(0融券) 月SR = " + str(round(out[name], 3)), flush=True)
    print("DELTA(ALL-F16) = " + str(round(out["ALL"] - out["F16"], 3)), flush=True)
    print("[OK] 100 LGB allfeat 完成", flush=True)


if __name__ == "__main__":
    main()
