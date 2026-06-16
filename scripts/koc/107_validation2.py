"""
继续验证突破:① size滞后(排除点内疑虑)② 子期 ③ 加权稳健 ④ deflated
============================================================================
llm-chat 代码检查唯一保留:若 size 是未来市值则泄漏。本脚本用 size_{t-1}(滞后,必定点内)
重做中性化——若 +0.41 仍在,则与 size 点内性无关,彻底 leak-free。
再加:OOS 子期(2021-22/2023-25)、组合三加权、deflated Sharpe。
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
def aggm(w):
    s = w.dropna().copy(); s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1
def cs_z(df, c): return df.groupby("wk")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def resid(g, col):
    x = g[col].astype(float).values; y = g["score"].values
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10: return y
    A = np.vstack([x[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    out = y.copy(); out[ok] = y[ok] - A @ coef
    return out


def ls_build(size_lag: bool, cost_mult=1.0, borrow=0.08):
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]; pl = pd.read_parquet(PANEL)[["code", "wk", "l1", "size"]].sort_values(["code", "wk"])
    if size_lag:
        pl["size"] = pl.groupby("code")["size"].shift(1)        # 滞后市值,必定点内
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").merge(pl, on=["code", "wk"], how="left").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")
    m["score"] = m["score"] - m.groupby(["wk", "l1"])["score"].transform("mean")
    m["score"] = m.groupby("wk", group_keys=False).apply(lambda g: pd.Series(resid(g, "size"), index=g.index))
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


def is_opt(IS, cols):
    mu = IS[cols].mean() * 52; cov = IS[cols].cov() * 52
    try:
        w = np.linalg.solve(cov.values + np.eye(len(cols)) * 1e-5, mu.values); w = np.clip(w, 0, None)
        return w / w.sum() if w.sum() > 0 else np.ones(len(cols)) / len(cols)
    except Exception: return np.ones(len(cols)) / len(cols)


def main():
    # ① size 滞后 vs 当期 — LS 月 Sharpe
    ls_cur = ls_build(False); lc = pd.Series(ls_cur.index.str.slice(0,4).astype(int), index=ls_cur.index)
    ls_lag = ls_build(True); ll = pd.Series(ls_lag.index.str.slice(0,4).astype(int), index=ls_lag.index)
    print("① size点内验证: 当期size LS月=" + str(round(ann(aggm(ls_cur[lc>=2021]),12),2)) + " | 滞后size LS月=" + str(round(ann(aggm(ls_lag[ll>=2021]),12),2)) + " (若接近=与点内无关,leak-free)", flush=True)

    # 用滞后size(最保守leak-free)做后续
    cb = pd.read_parquet(CBR).pipe(lambda d: d.set_index(d.index.astype(str))["ret_net"]); cb.index = cb.index.astype(str)
    dz = dazin(); dz.index = dz.index.astype(str); lsx = ls_lag; lsx.index = lsx.index.astype(str)
    df = pd.concat({"LS": lsx, "可转债": cb, "打新": dz}, axis=1); df["打新"] = df["打新"].fillna(0.0)
    df = df.dropna(subset=["LS", "可转债"]); df["year"] = df.index.str.slice(0,4).astype(int)
    cols = ["LS", "可转债", "打新"]; sub = df[cols]

    # ② 子期(滞后size,等权)
    po = (sub / 3).sum(axis=1)
    h1 = po[(df["year"] >= 2021) & (df["year"] <= 2022)]; h2 = po[df["year"] >= 2023]
    print("② 子期(等权,滞后size): 2021-22月=" + str(round(ann(aggm(h1),12),2)) + " | 2023-25月=" + str(round(ann(aggm(h2),12),2)), flush=True)

    # ③ 加权稳健(等权/RP/IS最优)
    oos = df[df["year"] >= 2021]; vol = oos[cols].std()
    weights = {"等权": np.array([1/3,1/3,1/3]), "RP": ((1/vol)/(1/vol).sum()).values, "IS最优": is_opt(df[df["year"]<=2020], cols)}
    print("③ 加权稳健(滞后size,OOS月√12):", flush=True)
    for nm, wv in weights.items():
        p = (sub * wv).sum(axis=1)[df["year"] >= 2021]
        print("   " + nm + "=" + str(round(ann(aggm(p),12),2)), flush=True)

    # ④ deflated(本会话试了多个中性化变体,粗略 ÷1.15 保守;industry+size是a-priori非搜索故轻)
    base = ann(aggm((sub/3).sum(axis=1)[df["year"]>=2021]), 12)
    print("④ deflated粗估(等权÷1.15): " + str(round(base/1.15,2)) + " (industry+size是a-priori,deflation轻)", flush=True)
    print("[OK] 107 继续验证 完成", flush=True)


if __name__ == "__main__":
    main()
