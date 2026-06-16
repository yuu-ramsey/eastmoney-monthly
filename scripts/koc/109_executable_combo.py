"""
诚实可部署:中性化 LS + 可融券空腿约束(空腿限流动性前50%)→ 组合真实压测
============================================================================
108 全角度扫描抓到:神经化 LS 用了全部空腿,但融券版空腿必须可融券。加约束后 LS 2.64→2.4。
本脚本用可融券空腿(lq≥0.5)+ 滞后size + 行业size中性,组合 + 完整预注册压测。
这才是真实可部署的融券版数字。实验记录。
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


def resid(g):
    x = g["size"].astype(float).values; y = g["score"].values
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10: return y
    A = np.vstack([x[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    out = y.copy(); out[ok] = y[ok] - A @ coef
    return out


def ls_exec(cost_mult=1.0, borrow=0.08):
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]
    pl = pd.read_parquet(PANEL)[["code", "wk", "l1", "size", "amt20"]].sort_values(["code", "wk"])
    pl["size"] = pl.groupby("code")["size"].shift(1)
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").merge(pl, on=["code", "wk"], how="left").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")
    m["score"] = m["score"] - m.groupby(["wk", "l1"])["score"].transform("mean")
    m["score"] = m.groupby("wk", group_keys=False).apply(lambda g: pd.Series(resid(g), index=g.index))
    m["lq"] = m.groupby("wk")["amt20"].transform(lambda s: s.rank(pct=True))
    m["sm"] = m.groupby("code")["score"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    res, pt, pb = [], {}, {}
    for wk, g in m.groupby("wk"):
        d = g.dropna(subset=["sm", "fwd"]); n = len(d)
        if n < 50: continue
        rk = d["sm"].rank(method="first"); grp = np.ceil(rk / n * 10).clip(1, 10)
        top = d[grp == 10]; bot = d[(grp == 1) & (d["lq"] >= 0.5)]      # 可融券空腿:流动性前50%
        if len(bot) < 3: bot = d[grp == 1]
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
    ls1 = ls_exec(1.0); ls1.index = ls1.index.astype(str)
    yl = pd.Series(ls1.index.str.slice(0,4).astype(int), index=ls1.index)
    print("可融券空腿 LS 月√12 = " + str(round(ann(aggm(ls1[yl>=2021]),12),2)) + " (全空腿是2.68)", flush=True)

    def combo(cm, cb_extra=0.0, dz_scale=1.0, dz_slip=1.0, holdout=False):
        ls = ls_exec(cm); ls.index = ls.index.astype(str)
        df = pd.concat({"LS": ls, "可转债": cb - cb_extra, "打新": dz * dz_scale * dz_slip}, axis=1); df["打新"] = df["打新"].fillna(0.0)
        df = df.dropna(subset=["LS", "可转债"]); yr = pd.Series(df.index.str.slice(0,4).astype(int), index=df.index)
        po = (df[["LS", "可转债", "打新"]] / 3).sum(axis=1)
        return po[yr == 2025] if holdout else po[yr >= 2021]

    a1 = ann(aggm(combo(1.0)), 12)
    a2 = ann(aggm(combo(2.0, cb_extra=0.0005, dz_slip=0.98)), 12)
    a3 = ann(aggm(combo(3.0, cb_extra=0.0010, dz_slip=0.98)), 12)
    b = ann(aggm(combo(2.0, cb_extra=0.0005, dz_scale=0.5, dz_slip=0.98)), 12)
    c = ann(aggm(combo(1.0, holdout=True)), 12)
    mo = aggm(combo(2.0, cb_extra=0.0005, dz_slip=0.98)); r = mo.dropna().values
    rng = np.random.default_rng(7); sb = []
    for _ in range(3000):
        idx = rng.integers(0, max(1, len(r) - 6), max(1, len(r)//6)); x = np.concatenate([r[i:i+6] for i in idx])
        sb.append(x.mean() * 12 / (x.std(ddof=1)*np.sqrt(12)+1e-12))
    sb = np.array(sb)
    print("组合基线1×=" + str(round(a1,2)), flush=True)
    print("A 2×=" + str(round(a2,2)) + "(>=2.60) 3×=" + str(round(a3,2)) + "(>=2.40)", flush=True)
    print("B 0.5打新+2×=" + str(round(b,2)) + "(>=2.30) | C 2025=" + str(round(c,2)) + "(>=2.0)", flush=True)
    print("kill min=" + str(round(min(a2,b,c),2)) + "(>=2.30) | bootstrap P(>2.8)=" + str(round(np.mean(sb>2.8)*100)) + "%", flush=True)
    yr2 = pd.Series(combo(1.0).index.str.slice(0,4).astype(int), index=combo(1.0).index)
    p1 = combo(1.0)
    print("逐年(月): " + " ".join(str(y)+"="+str(round(ann(aggm(p1[yr2==y]),12),1)) for y in range(2021,2026)), flush=True)
    print("[OK] 109 可部署融券版 完成", flush=True)


if __name__ == "__main__":
    main()
