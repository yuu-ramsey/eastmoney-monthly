"""
预注册压力测试 battery(阈值看结果前已锁,见 docs/koc-ls-leakfree-verdict 与 llm-chat 签核)
============================================================================
被测:等权(1/3)leak-free LS + 可转债双低 + 打新,OOS2021+ 非重叠月√12。
预注册 kill 线:min(2×成本Sharpe, 0.5×打新Sharpe, 2025留出Sharpe) < 2.30 → "稳过2.8"被否。
  A 成本:2×成本≥2.60 且 3×≥2.40
  B 打新容量:0.5×打新(带2×成本)≥2.30
  C 2025留出(等权基线成本)≥2.00
  + 尾部:最大回撤≤25% / 月VaR99
  + moving-block bootstrap(L=6月)
  + 打新中签率 2024-2025 衰减(直接读数据)
成本近似:LS 精确(turnover×0.0008×mult);可转债加 (mult-1)×0.0005/周;打新 30bp滑点≈×0.98。
"""
from __future__ import annotations

import akshare as ak
import numpy as np
import pandas as pd

PE = "data/pred_ensemble.parquet"; PURGED = "data/pred_purged.parquet"; LGBP = "data/pred_v3_lgb_purged.parquet"
CBR = "data/cb_v3_returns.parquet"; REF = 1_000_000


def ann(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")


def agg_m(weekly):
    s = weekly.dropna().copy(); s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1


def cs_z(df, col):
    return df.groupby("wk")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def leakfree_ls(cost_mult=1.0, borrow=0.08):
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")
    m["sm"] = m.groupby("code")["score"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    res, ptop, pbot = [], {}, {}
    for wk, g in m.groupby("wk"):
        d = g.dropna(subset=["sm", "fwd"]); n = len(d)
        if n < 50: continue
        rk = d["sm"].rank(method="first"); grp = np.ceil(rk / n * 10).clip(1, 10)
        top, bot = d[grp == 10], d[grp == 1]
        if len(top) < 3 or len(bot) < 3: continue
        rt, rb = float(top["fwd"].mean()), float(bot["fwd"].mean())
        tw = {c: 1/len(top) for c in top["code"]}; bw = {c: 1/len(bot) for c in bot["code"]}
        tot = sum(abs(tw.get(c, 0) - ptop.get(c, 0)) for c in set(tw) | set(ptop))
        tob = sum(abs(bw.get(c, 0) - pbot.get(c, 0)) for c in set(bw) | set(pbot))
        ptop, pbot = tw, bw
        res.append({"wk": str(wk), "ls": (rt - rb) - (tot + tob) * 0.0008 * cost_mult - borrow / 52})
    return pd.DataFrame(res).set_index("wk")["ls"]


def dazin():
    dz = ak.stock_xgsglb_em(symbol="全部股票").rename(columns={"申购日期": "date", "中签率": "rate", "每中一签获利": "profit", "申购上限": "cap"})
    dz["date"] = pd.to_datetime(dz["date"], errors="coerce")
    for c in ("rate", "profit", "cap"): dz[c] = pd.to_numeric(dz[c], errors="coerce")
    dz = dz.dropna(subset=["date", "rate", "profit"]); dz = dz[(dz["rate"] > 0) & (dz["profit"] > 0)]
    dz["units"] = (dz["cap"].fillna(dz["cap"].median()) / 500).clip(upper=2000); dz["ep"] = dz["rate"] / 100 * dz["profit"] * dz["units"]
    dz["wk"] = dz["date"].dt.to_period("W-FRI").astype(str)
    return (dz.groupby("wk")["ep"].sum() / REF), dz


def combo(ls, cb, dz, cb_extra=0.0, dz_scale=1.0, dz_slip=1.0):
    cbx = cb - cb_extra                              # 可转债额外成本
    dzx = dz * dz_scale * dz_slip                    # 打新容量缩放 + 滑点
    df = pd.concat({"LS": ls, "可转债": cbx, "打新": dzx}, axis=1)
    df["打新"] = df["打新"].fillna(0.0); df = df.dropna(subset=["LS", "可转债"])
    df["year"] = df.index.str.slice(0, 4).astype(int)
    po = (df[["LS", "可转债", "打新"]] / 3).sum(axis=1)
    return po, df["year"]


def main():
    cb = pd.read_parquet(CBR).pipe(lambda d: d.set_index(d.index.astype(str))["ret_net"]); cb.index = cb.index.astype(str)
    dzs, dzraw = dazin(); dzs.index = dzs.index.astype(str)
    ls1 = leakfree_ls(1.0); ls1.index = ls1.index.astype(str)
    ls2 = leakfree_ls(2.0); ls2.index = ls2.index.astype(str)
    ls3 = leakfree_ls(3.0); ls3.index = ls3.index.astype(str)

    def oos_sr(po, yr, hold=False):
        sub = po[(yr == 2025)] if hold else po[yr >= 2021]
        return ann(agg_m(sub), 12)

    print("=== 预注册压测(阈值已锁)===", flush=True)
    # 基线
    po, yr = combo(ls1, cb, dzs); base = oos_sr(po, yr)
    print(f"基线 等权OOS月√12 = {base:.2f}", flush=True)
    # A 成本
    poA2, yrA2 = combo(ls2, cb, dzs, cb_extra=0.0005, dz_slip=0.98)
    poA3, yrA3 = combo(ls3, cb, dzs, cb_extra=0.0010, dz_slip=0.98)
    a2, a3 = oos_sr(poA2, yrA2), oos_sr(poA3, yrA3)
    print(f"A 成本: 2×={a2:.2f}(线2.60) | 3×={a3:.2f}(线2.40) → {'PASS' if a2>=2.6 and a3>=2.4 else 'FAIL'}", flush=True)
    # B 打新容量(0.5×打新 + 2×成本)
    poB, yrB = combo(ls2, cb, dzs, cb_extra=0.0005, dz_scale=0.5, dz_slip=0.98)
    b = oos_sr(poB, yrB)
    print(f"B 打新0.5×(带2×成本)={b:.2f}(线2.30) → {'PASS' if b>=2.3 else 'FAIL'}", flush=True)
    # C 2025留出(基线成本,等权)
    c = oos_sr(po, yr, hold=True)
    print(f"C 2025留出={c:.2f}(线2.00) → {'PASS' if c>=2.0 else 'FAIL'}", flush=True)
    # kill 线
    decisive_min = min(a2, b, c)
    print(f"\n决定性最小(2×成本,0.5打新,2025留出)={decisive_min:.2f}(kill线2.30) → "
          f"{'稳过2.8 ✓' if decisive_min>=2.3 else '未稳过2.8 ⚠️'}", flush=True)

    # 尾部:最大回撤 + 月VaR99
    mo = agg_m(po[yr >= 2021]); cum = (1 + mo).cumprod(); dd = (cum / cum.cummax() - 1).min()
    var99 = np.percentile(mo, 1)
    print(f"\n尾部: 最大回撤={dd:.1%}(线≤-25%) {'PASS' if dd>=-0.25 else 'FAIL'} | 月VaR99={var99:.1%}", flush=True)
    # moving-block bootstrap L=6月
    r = mo.dropna().values; rng = np.random.default_rng(7); sb = []
    for _ in range(3000):
        idx = rng.integers(0, max(1, len(r) - 6), max(1, len(r) // 6)); x = np.concatenate([r[i:i+6] for i in idx])
        sb.append(x.mean() * 12 / (x.std(ddof=1) * np.sqrt(12) + 1e-12))
    sb = np.array(sb)
    print(f"moving-block(6月)bootstrap={sb.mean():.2f} CI[{np.percentile(sb,2.5):.2f},{np.percentile(sb,97.5):.2f}] P(>2.8)={np.mean(sb>2.8):.0%}", flush=True)
    # 打新中签率衰减
    dzraw["year"] = dzraw["date"].dt.year
    wr = dzraw.groupby("year")["rate"].mean()
    print(f"打新中签率(均%): " + " ".join(f"{y}={wr.get(y,float('nan')):.3f}" for y in range(2021, 2026)), flush=True)
    print("[OK] 94 预注册压测完成", flush=True)


if __name__ == "__main__":
    main()
