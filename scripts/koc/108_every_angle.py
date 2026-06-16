"""
全方位稳健性扫描:中性化突破是否跨所有 arbitrary 参数都稳(每个角度)
============================================================================
若只在某参数成立=过拟合;若跨所有维度都稳=真实。用 size滞后(最保守leak-free)。
扫:① ensemble权重(TCN占比0-1) ② 分位数(5/10/20) ③ EWM span(2/4/8)
    ④ 宇宙流动性筛(0/30/50%) ⑤ 空腿可融券(全/流动前50%)
报每维 LS 月√12 范围。默认: tcn0.3, 十分位, span4, liq30%, 空腿全。
输出写文件(避开 harness 渲染 bug)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PE = "data/pred_ensemble.parquet"; PURGED = "data/pred_purged.parquet"; LGBP = "data/pred_v3_lgb_purged.parquet"
PANEL = "data/mf_panel_weekly_v3.parquet"


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


def load():
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]
    pl = pd.read_parquet(PANEL)[["code", "wk", "l1", "size", "amt20"]].sort_values(["code", "wk"])
    pl["size"] = pl.groupby("code")["size"].shift(1)        # 滞后size,最保守leak-free
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").merge(pl, on=["code", "wk"], how="left").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    return m


def ls_sr(m0, tcn_w=0.3, ndec=10, span=4, liq=0.30, short_liq=0.0):
    m = m0.copy()
    if liq > 0:
        m = m[m["amt20"].notna()].copy()
        thr = m.groupby("wk")["amt20"].transform(lambda s: s.quantile(liq))
        m = m[m["amt20"] >= thr].copy()
    m["score"] = tcn_w * cs_z(m, "tcn_purged") + (1 - tcn_w) * cs_z(m, "lgb_purged")
    m["score"] = m["score"] - m.groupby(["wk", "l1"])["score"].transform("mean")
    m["score"] = m.groupby("wk", group_keys=False).apply(lambda g: pd.Series(resid(g), index=g.index))
    m["lq"] = m.groupby("wk")["amt20"].transform(lambda s: s.rank(pct=True))
    m["sm"] = m.groupby("code")["score"].transform(lambda s: s.ewm(span=span, adjust=False).mean())
    res, pt, pb = [], {}, {}
    for wk, g in m.groupby("wk"):
        d = g.dropna(subset=["sm", "fwd"]); n = len(d)
        if n < 50: continue
        rk = d["sm"].rank(method="first"); grp = np.ceil(rk / n * ndec).clip(1, ndec)
        top = d[grp == ndec]; bot = d[grp == 1]
        if short_liq > 0:
            bot = bot[bot["lq"] >= short_liq]
            if len(bot) < 3: bot = d[grp == 1]
        if len(top) < 3 or len(bot) < 3: continue
        rt, rb = float(top["fwd"].mean()), float(bot["fwd"].mean())
        tw = {c: 1/len(top) for c in top["code"]}; bw = {c: 1/len(bot) for c in bot["code"]}
        tot = sum(abs(tw.get(c, 0) - pt.get(c, 0)) for c in set(tw) | set(pt))
        tob = sum(abs(bw.get(c, 0) - pb.get(c, 0)) for c in set(bw) | set(pb)); pt, pb = tw, bw
        res.append({"wk": str(wk), "ls": (rt - rb) - (tot + tob) * 0.0008 - 0.08 / 52})
    ls = pd.DataFrame(res).set_index("wk")["ls"]
    yr = pd.Series(ls.index.str.slice(0, 4).astype(int), index=ls.index)
    return ann(aggm(ls[yr >= 2021]), 12)


def main():
    m0 = load()
    lines = ["全方位稳健性扫描(滞后size,LS月SR,OOS2021+,默认tcn0.3/十分位/span4/liq30/空腿全)"]
    lines.append("[default] = " + str(round(ls_sr(m0), 2)))
    lines.append("ensemble权重(TCN占比): " + " ".join("tcn" + str(w) + "=" + str(round(ls_sr(m0, tcn_w=w), 2)) for w in (0.0, 0.3, 0.5, 0.7, 1.0)))
    lines.append("分位数: " + " ".join("q" + str(nd) + "=" + str(round(ls_sr(m0, ndec=nd), 2)) for nd in (5, 10, 20)))
    lines.append("EWM span: " + " ".join("s" + str(sp) + "=" + str(round(ls_sr(m0, span=sp), 2)) for sp in (2, 4, 8)))
    lines.append("宇宙流动性筛: " + " ".join("liq" + str(int(lq*100)) + "=" + str(round(ls_sr(m0, liq=lq), 2)) for lq in (0.0, 0.30, 0.50)))
    lines.append("空腿可融券约束: " + " ".join("sl" + str(int(s*100)) + "=" + str(round(ls_sr(m0, short_liq=s), 2)) for s in (0.0, 0.5, 0.7)))
    with open("data/koc-overnight/every_angle.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    for ln in lines:
        print(ln, flush=True)
    print("[OK] 108 every-angle 完成", flush=True)


if __name__ == "__main__":
    main()
