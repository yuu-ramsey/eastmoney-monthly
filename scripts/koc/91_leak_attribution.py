"""
泄漏归因(修正 confound):同模型对比 leaky-TCN vs leak-free-TCN,隔离纯泄漏
============================================================================
90 的 ref 用了 ensemble、purged 用 TCN-only,混淆泄漏与模型差异。本脚本同模型同口径:
  - leaky TCN-only(pred_ensemble 的 'tcn' 列,旧无embargo walk-forward)
  - leak-free TCN-only(pred_purged 的 'tcn_purged',embargo+丢末日)
  - 同时给 leaky-ensemble 与 leak-free-ensemble(0.3*tcn+0.7*lgb)作部署口径对照
全部 OOS2021+ 周√52(隔离泄漏)+ 非重叠月√12(诚实)。纯CPU缓存数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PE = "data/pred_ensemble.parquet"
PURGED = "data/pred_purged.parquet"


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
    pe = pd.read_parquet(PE)
    pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    m = pe.merge(pg, on=["code", "wk"], how="inner").dropna(subset=["tcn", "tcn_purged", "pred", "fwd"])
    print(f"对齐样本 {len(m)}(同 code/wk 上 leaky-tcn 与 leak-free-tcn 并存)")
    # leak-free ensemble:截面标准化后 0.3*tcn_purged + 0.7*lgb
    m["tcnp_z"] = cs_z(m, "tcn_purged"); m["lgb_z"] = cs_z(m, "pred")
    m["ens_leakfree"] = 0.3 * m["tcnp_z"] + 0.7 * m["lgb_z"]

    rows = []
    for label, col in [("leaky TCN-only", "tcn"), ("leak-free TCN-only", "tcn_purged"),
                       ("leaky ensemble(部署)", "ens"), ("leak-free ensemble", "ens_leakfree")]:
        ls = build_ls(m, col); o = oos(ls)
        rows.append((label, ann(o, 52), ann(agg_m(o), 12)))

    print("\n=== 同口径泄漏归因(OOS2021+,扣8%融券)===")
    print(f"{'信号':<22}{'周√52':>9}{'月√12':>9}")
    for lab, w, mo in rows:
        print(f"{lab:<22}{w:>9.2f}{mo:>9.2f}")

    d = dict((r[0], r[1]) for r in rows)
    tcn_leak = d["leaky TCN-only"] - d["leak-free TCN-only"]
    ens_leak = d["leaky ensemble(部署)"] - d["leak-free ensemble"]
    print(f"\nTCN 纯泄漏(同模型 周vs周) = {tcn_leak:+.2f}  ({d['leaky TCN-only']:.2f}→{d['leak-free TCN-only']:.2f})")
    print(f"部署ensemble 泄漏           = {ens_leak:+.2f}  ({d['leaky ensemble(部署)']:.2f}→{d['leak-free ensemble']:.2f})")
    lf_ens_m = [r[2] for r in rows if r[0] == "leak-free ensemble"][0]
    print(f"\n诚实结论:leak-free 部署ensemble 月√12 = {lf_ens_m:.2f}")
    print(f"裁定:{'原LS含显著泄漏(同模型口径证实)⚠️' if ens_leak > 0.4 else 'LS泄漏不显著'}")
    print("[OK] 91 泄漏归因完成")


if __name__ == "__main__":
    main()
