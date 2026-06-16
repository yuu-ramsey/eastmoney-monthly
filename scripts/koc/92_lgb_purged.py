"""
LGB leak-free 重训(同 embargo)+ 全 leak-free ensemble 定论
============================================================================
llm-chat 签核要求:TCN 已 leak-free,但部署 ensemble 的 LGB(pred_v3_lgb)未重训,
残留泄漏不可靠地靠算术推(~0.145)。本脚本直接重训 LGB(同 16周 embargo 断年边界),
得 pred_v3_lgb_purged,再与 tcn_purged 组全 leak-free ensemble,定论真实泄漏与诚实 Sharpe。
纯 CPU(LightGBM tabular),几分钟。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

PANEL = "data/mf_panel_weekly_v3.parquet"
PE = "data/pred_ensemble.parquet"
PURGED = "data/pred_purged.parquet"
EMBARGO_WK = 16
F16 = ["reversal", "lowvol", "illiq", "size", "sue", "maxret", "turn",
       "f_idio", "f_amihud", "f_turnmom", "f_skew", "f_downvol", "f_sdv", "f_volskew", "f_turnac"]


def xrank(s):
    r = s.rank(method="average"); return r / (r.notna().sum() + 1) - 0.5


def ann(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")


def agg_m(weekly):
    s = weekly.dropna().copy()
    s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1


def build_ls(frame, score):
    f = frame.copy()
    f["sm"] = f.groupby("code")[score].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    res, ptop, pbot = [], {}, {}
    for wk, g in f.groupby("wk"):
        d = g.dropna(subset=["sm", "fwd"]); n = len(d)
        if n < 50:
            continue
        rk = d["sm"].rank(method="first"); grp = np.ceil(rk / n * 10).clip(1, 10)
        top, bot = d[grp == 10], d[grp == 1]
        if len(top) < 3 or len(bot) < 3:
            continue
        rt, rb = float(top["fwd"].mean()), float(bot["fwd"].mean())
        tw = {c: 1/len(top) for c in top["code"]}; bw = {c: 1/len(bot) for c in bot["code"]}
        tot = sum(abs(tw.get(c, 0) - ptop.get(c, 0)) for c in set(tw) | set(ptop))
        tob = sum(abs(bw.get(c, 0) - pbot.get(c, 0)) for c in set(bw) | set(pbot))
        ptop, pbot = tw, bw
        res.append({"wk": str(wk), "ls": (rt - rb) - (tot + tob) * 0.0008 - 0.08 / 52})
    return pd.DataFrame(res).set_index("wk")["ls"] if res else pd.Series(dtype=float)


def oos(ls):
    yr = pd.Series(ls.index.str.slice(0, 4).astype(int), index=ls.index)
    return ls[yr >= 2021]


def cs_z(df, col):
    return df.groupby("wk")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def main():
    w = pd.read_parquet(PANEL)
    w["year"] = w["year"].astype(int)
    cols = [c for c in F16 if c in w.columns]
    wk_end = pd.PeriodIndex(w["wk"].astype(str), freq="W-FRI").to_timestamp(how="end")
    w = w.assign(_wkend=wk_end)
    # 宇宙筛选 + 截面rank特征 + 去均值标签(与原 LGB 同)
    rows = []
    for wk, g in w.groupby("wk"):
        g = g[(g["stflag"] == 0) & (g["hist_n"] >= 52) & g["amt20"].notna()]
        if len(g) < 100:
            continue
        g = g[g["amt20"] >= g["amt20"].quantile(0.30)].copy()
        if len(g) < 100:
            continue
        for f in cols:
            g[f"r_{f}"] = xrank(g[f]).fillna(0.0)
        g["ylab"] = g["fwd"] - g["fwd"].mean()
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True)
    rfeat = [f"r_{f}" for f in cols]
    years = panel["year"].values
    lp = dict(objective="regression", num_leaves=31, learning_rate=0.05, n_estimators=300,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=200, verbose=-1)

    preds = np.full(len(panel), np.nan, dtype=np.float32)
    for Y in range(2019, 2026):
        y_start = pd.Timestamp(f"{Y}-01-01")
        embargo_cut = y_start - pd.Timedelta(weeks=EMBARGO_WK)
        tr = (years < Y) & (panel["_wkend"].values < np.datetime64(embargo_cut))   # 同 embargo
        te = years == Y
        if tr.sum() < 5000 or te.sum() < 50:
            continue
        m = lgb.LGBMRegressor(**lp).fit(panel.loc[tr, rfeat], panel.loc[tr, "ylab"])
        preds[te] = m.predict(panel.loc[te, rfeat])
        print(f"  LGB-purged {Y}: train={tr.sum()} test={te.sum()} (embargo剔{((years<Y)&(panel['_wkend'].values>=np.datetime64(embargo_cut))).sum()})", flush=True)
    panel["lgb_purged"] = preds
    panel[["code", "wk", "year", "lgb_purged"]].dropna(subset=["lgb_purged"]).to_parquet(
        "data/pred_v3_lgb_purged.parquet", index=False)

    # 合并 leaky(tcn,pred,ens)+ leak-free(tcn_purged, lgb_purged)
    pe = pd.read_parquet(PE)
    pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lf = panel[["code", "wk", "lgb_purged"]].dropna(subset=["lgb_purged"])
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lf, on=["code", "wk"], how="inner")
    m = m.dropna(subset=["tcn", "pred", "ens", "tcn_purged", "lgb_purged", "fwd"])
    print(f"\n全口径对齐样本 {len(m)}", flush=True)
    # 全 leak-free ensemble:0.3*z(tcn_purged)+0.7*z(lgb_purged)
    m["full_lf"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")

    out = []
    for lab, col in [("leaky ensemble(部署)", "ens"),
                     ("半leak-free(仅TCN purged)", None),  # 占位,下面单算
                     ("全leak-free ensemble", "full_lf")]:
        if col is None:
            m["half_lf"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "pred")
            col = "half_lf"
        ls = build_ls(m, col); o = oos(ls)
        out.append((lab, ann(o, 52), ann(agg_m(o), 12)))

    print("\n=== 全口径泄漏定论(OOS2021+,扣8%融券)===", flush=True)
    print(f"{'信号':<26}{'周√52':>9}{'月√12':>9}")
    for lab, ww, mm in out:
        print(f"{lab:<26}{ww:>9.2f}{mm:>9.2f}")
    leaky_w = out[0][1]; full_w = out[2][1]; full_m = out[2][2]
    total_leak = leaky_w - full_w
    print(f"\n全泄漏(leaky→全leak-free 周vs周) = {total_leak:+.2f}  ({leaky_w:.2f}→{full_w:.2f})", flush=True)
    print(f"诚实 LS(全leak-free)月√12 = **{full_m:.2f}**", flush=True)
    verdict = ("LS基本坐实,泄漏温和" if total_leak <= 0.5 else
               "LS含显著泄漏" if total_leak <= 1.0 else "LS主要靠泄漏")
    print(f"裁定:{verdict}(全泄漏{total_leak:+.2f})", flush=True)
    print("[OK] 92 LGB-purged 全口径定论完成", flush=True)


if __name__ == "__main__":
    main()
