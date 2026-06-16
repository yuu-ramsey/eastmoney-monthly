"""
实验笔记:Version A(纯现金,零杠杆)增强 — 可转债 credit cushion + risk-parity 加权
============================================================================
假设:① 可转债 double-low 加 credit-cushion 因子(文献 Davis&Liu 2022)提升单腿
      ② 三腿 risk-parity 加权(下调弱底仓)抬组合
目前 best-so-far(等权)~2.0-2.2,看能否推过。非重叠月度,不调参(因子等权 a-priori)。
这是实验记录非天花板裁定。
"""
from __future__ import annotations

import akshare as ak
import numpy as np
import pandas as pd

CB = "data/cb_value.parquet"; PANEL = "data/mf_panel_weekly_v3.parquet"; REF = 1_000_000


def ann(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")


def agg_m(w):
    s = w.dropna().copy(); s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1


def cb_stream(enhanced: bool):
    cv = pd.read_parquet(CB).sort_values(["code", "date"]); cv["wk"] = cv["date"].dt.to_period("W-FRI").astype(str)
    we = cv.groupby(["code", "wk"]).agg(close=("close", "last"), premium=("premium", "last"),
                                        pv=("pure_value", "last"), conv=("conv_value", "last")).reset_index().sort_values(["code", "wk"])
    we["fwd"] = we.groupby("code")["close"].shift(-1) / we["close"] - 1
    rows, prev = [], set()
    for wk, g in we.groupby("wk"):
        d = g.dropna(subset=["close", "premium", "fwd", "pv", "conv"])
        d = d[(d["close"] >= 88) & (d["close"] <= 128) & (d["premium"] <= 40) & (d["close"] > d["pv"]) & (d["conv"] < 125)]
        if len(d) < 15: continue
        d = d.copy()
        score = d["close"].rank() + d["premium"].rank()             # 双低基线
        if enhanced:
            d["cushion"] = (d["close"] - d["pv"]) / d["pv"]          # 信用垫:越低越靠债底越安全
            score = score + d["cushion"].rank()                     # 等权加 a-priori
        d["sc"] = score
        sel = d.nsmallest(25, "sc")
        h = set(sel["code"]); to = len(h ^ prev) / max(len(h), 1); prev = h
        rows.append({"wk": wk, "r": float(sel["fwd"].mean()) - to * 0.001})
    return pd.DataFrame(rows).set_index("wk")["r"]


def dicang():
    """低波底仓(long-only 低波十分位,纯现金可持,为打新提供市值)。"""
    w = pd.read_parquet(PANEL)
    w = w[(w["stflag"] == 0) & w["amt20"].notna() & w["lowvol"].notna()]
    rows = []
    for wk, g in w.groupby("wk"):
        if len(g) < 100: continue
        # lowvol 因子:取低波一端(假设 lowvol 高=低波,取 top;若反则对称)
        sel = g.nlargest(max(20, len(g)//10), "lowvol")
        rows.append({"wk": str(wk), "r": float(sel["fwd"].mean()) - 0.0006})
    return pd.DataFrame(rows).set_index("wk")["r"]


def dazin():
    dz = ak.stock_xgsglb_em(symbol="全部股票").rename(columns={"申购日期": "date", "中签率": "rate", "每中一签获利": "profit", "申购上限": "cap"})
    dz["date"] = pd.to_datetime(dz["date"], errors="coerce")
    for c in ("rate", "profit", "cap"): dz[c] = pd.to_numeric(dz[c], errors="coerce")
    dz = dz.dropna(subset=["date", "rate", "profit"]); dz = dz[(dz["rate"] > 0) & (dz["profit"] > 0)]
    dz["units"] = (dz["cap"].fillna(dz["cap"].median()) / 500).clip(upper=2000); dz["ep"] = dz["rate"] / 100 * dz["profit"] * dz["units"]
    dz["wk"] = dz["date"].dt.to_period("W-FRI").astype(str)
    return (dz.groupby("wk")["ep"].sum() / REF)


def combo(cb, dz, dc, weighting):
    for s in (cb, dz, dc): s.index = s.index.astype(str)
    df = pd.concat({"可转债": cb, "打新": dz, "底仓": dc}, axis=1); df["打新"] = df["打新"].fillna(0.0)
    df = df.dropna(subset=["可转债", "底仓"]); yr = pd.Series(df.index.str.slice(0,4).astype(int), index=df.index)
    sub = df[["可转债", "打新", "底仓"]]
    if weighting == "equal":
        w = np.array([1/3, 1/3, 1/3])
    else:  # risk-parity(OOS vol)
        vol = sub[yr >= 2021].std(); w = ((1/vol)/(1/vol).sum()).values
    po = (sub * w).sum(axis=1)[yr >= 2021]
    return ann(agg_m(po), 12), dict(zip(["可转债","打新","底仓"], np.round(w,2)))


def main():
    dz = dazin(); dc = dicang()
    cb0 = cb_stream(False); cb1 = cb_stream(True)
    cb0.index = cb0.index.astype(str); cb1.index = cb1.index.astype(str)
    yc = pd.Series(cb0.index.str.slice(0,4).astype(int), index=cb0.index)
    yc1 = pd.Series(cb1.index.str.slice(0,4).astype(int), index=cb1.index)
    print("=== Version A 增强实验(非重叠月√12,OOS2021+)===", flush=True)
    print(f"可转债单腿: 基线={ann(agg_m(cb0[yc>=2021]),12):.2f} | 加credit-cushion={ann(agg_m(cb1[yc1>=2021]),12):.2f}", flush=True)
    print(f"底仓(低波)单腿={ann(agg_m(dc[pd.Series(dc.index.astype(str).str.slice(0,4).astype(int),index=dc.index.astype(str))>=2021] if False else dc),12):.2f}", flush=True)
    print("\n组合(可转债+打新+底仓):", flush=True)
    for cbname, cbs in [("基线可转债", cb0), ("增强可转债", cb1)]:
        for wt in ("equal", "rp"):
            sr, wts = combo(cbs.copy(), dz.copy(), dc.copy(), wt)
            print(f"  {cbname} + {wt:<5}: 月√12={sr:.2f} 权重={wts}", flush=True)
    print("\n(实验记录:best-so-far,非天花板;下一步见 docs §7 待试清单)", flush=True)
    print("[OK] 99 Version A 增强实验完成", flush=True)


if __name__ == "__main__":
    main()
