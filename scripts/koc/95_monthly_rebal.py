"""
预注册降换手试验:月频再平衡 LS+可转债(冻结信号一个月)→ 2×成本下能否升向2.8
============================================================================
llm-chat 预注册spec:月频(每月首周)再平衡,信号冻结持有整月,等权十分位,2×成本,
非重叠月度评估。成功线:真实成本净 Sharpe 比周频2×(2.53)升 ≥0.05。
诚实先验(llm-chat):可能净中性/微负,2.5 或是真天花板。但LS周频换手极高,月频节省或更大。
对照:周频 vs 月频,各报 gross + 2×成本净。
"""
from __future__ import annotations

import akshare as ak
import numpy as np
import pandas as pd

PE = "data/pred_ensemble.parquet"; PURGED = "data/pred_purged.parquet"; LGBP = "data/pred_v3_lgb_purged.parquet"
CB = "data/cb_value.parquet"; REF = 1_000_000


def ann(m, ppy):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")


def agg_m(weekly):
    s = weekly.dropna().copy(); s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1


def cs_z(df, col):
    return df.groupby("wk")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def wk_month(wk):
    return pd.Period(wk, freq="W-FRI").to_timestamp(how="end").to_period("M")


def ls_stream(monthly: bool, cost_mult: float, borrow=0.08):
    """LS 周收益流。monthly=True 则每月首周再平衡、持有整月(换手只发生在再平衡周)。"""
    pe = pd.read_parquet(PE); pg = pd.read_parquet(PURGED)[["code", "wk", "tcn_purged"]]
    lp = pd.read_parquet(LGBP)[["code", "wk", "lgb_purged"]]
    m = pe.merge(pg, on=["code", "wk"], how="inner").merge(lp, on=["code", "wk"], how="inner").dropna(subset=["tcn_purged", "lgb_purged", "fwd"])
    m["score"] = 0.3 * cs_z(m, "tcn_purged") + 0.7 * cs_z(m, "lgb_purged")
    m["sm"] = m.groupby("code")["score"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    rows, prev_top, prev_bot = [], {}, {}
    cur_top, cur_bot, cur_mon = {}, {}, None
    for wk in sorted(m["wk"].unique()):
        d = m[m["wk"] == wk].dropna(subset=["sm", "fwd"]); n = len(d)
        if n < 50:
            continue
        mon = wk_month(wk)
        rebal = (not monthly) or (mon != cur_mon)
        if rebal:
            rk = d["sm"].rank(method="first"); grp = np.ceil(rk / n * 10).clip(1, 10)
            top, bot = d[grp == 10], d[grp == 1]
            if len(top) < 3 or len(bot) < 3:
                continue
            cur_top = {c: 1/len(top) for c in top["code"]}; cur_bot = {c: 1/len(bot) for c in bot["code"]}
            cur_mon = mon
            tot = sum(abs(cur_top.get(c, 0) - prev_top.get(c, 0)) for c in set(cur_top) | set(prev_top))
            tob = sum(abs(cur_bot.get(c, 0) - prev_bot.get(c, 0)) for c in set(cur_bot) | set(prev_bot))
            prev_top, prev_bot = cur_top, cur_bot
            cost = (tot + tob) * 0.0008 * cost_mult
        else:
            cost = 0.0
        fwdmap = dict(zip(d["code"], d["fwd"]))
        rt = np.mean([fwdmap[c] for c in cur_top if c in fwdmap]) if any(c in fwdmap for c in cur_top) else 0.0
        rb = np.mean([fwdmap[c] for c in cur_bot if c in fwdmap]) if any(c in fwdmap for c in cur_bot) else 0.0
        rows.append({"wk": str(wk), "ls": (rt - rb) - cost - borrow / 52})
    return pd.DataFrame(rows).set_index("wk")["ls"]


def cb_stream(monthly: bool, cost_mult: float):
    """可转债双低 周收益流(月频则持有整月)。"""
    cv = pd.read_parquet(CB).sort_values(["code", "date"]); cv["wk"] = cv["date"].dt.to_period("W-FRI").astype(str)
    we = cv.groupby(["code", "wk"]).agg(close=("close", "last"), premium=("premium", "last"),
                                        pv=("pure_value", "last"), conv=("conv_value", "last")).reset_index().sort_values(["code", "wk"])
    we["fwd"] = we.groupby("code")["close"].shift(-1) / we["close"] - 1
    rows, prev, cur, cur_mon = [], set(), set(), None
    for wk in sorted(we["wk"].unique()):
        d = we[we["wk"] == wk].dropna(subset=["close", "premium", "fwd", "pv", "conv"])
        d = d[(d["close"] >= 88) & (d["close"] <= 128) & (d["premium"] <= 40) & (d["close"] > d["pv"]) & (d["conv"] < 125)]
        if len(d) < 15:
            continue
        mon = wk_month(wk); rebal = (not monthly) or (mon != cur_mon)
        if rebal:
            d2 = d.copy(); d2["dl"] = d2["close"].rank() + d2["premium"].rank()
            cur = set(d2.nsmallest(25, "dl")["code"]); cur_mon = mon
            to = len(cur ^ prev) / max(len(cur), 1); prev = cur
            cost = to * 0.001 * cost_mult
        else:
            cost = 0.0
        fwdmap = dict(zip(d["code"], d["fwd"]))
        r = np.mean([fwdmap[c] for c in cur if c in fwdmap]) if any(c in fwdmap for c in cur) else 0.0
        rows.append({"wk": str(wk), "r": r - cost})
    return pd.DataFrame(rows).set_index("wk")["r"]


def dazin():
    dz = ak.stock_xgsglb_em(symbol="全部股票").rename(columns={"申购日期": "date", "中签率": "rate", "每中一签获利": "profit", "申购上限": "cap"})
    dz["date"] = pd.to_datetime(dz["date"], errors="coerce")
    for c in ("rate", "profit", "cap"): dz[c] = pd.to_numeric(dz[c], errors="coerce")
    dz = dz.dropna(subset=["date", "rate", "profit"]); dz = dz[(dz["rate"] > 0) & (dz["profit"] > 0)]
    dz["units"] = (dz["cap"].fillna(dz["cap"].median()) / 500).clip(upper=2000); dz["ep"] = dz["rate"] / 100 * dz["profit"] * dz["units"]
    dz["wk"] = dz["date"].dt.to_period("W-FRI").astype(str)
    return (dz.groupby("wk")["ep"].sum() / REF)


def combo_sr(ls, cb, dz):
    for s in (ls, cb, dz): s.index = s.index.astype(str)
    df = pd.concat({"LS": ls, "可转债": cb, "打新": dz}, axis=1)
    df["打新"] = df["打新"].fillna(0.0); df = df.dropna(subset=["LS", "可转债"])
    yr = pd.Series(df.index.str.slice(0, 4).astype(int), index=df.index)
    po = (df[["LS", "可转债", "打新"]] / 3).sum(axis=1)[yr >= 2021]
    return ann(agg_m(po), 12)


def main():
    dz = dazin()
    print("=== 周频 vs 月频再平衡(等权组合,OOS月√12)===", flush=True)
    for mode, mon in [("周频", False), ("月频", True)]:
        for cm, tag in [(1.0, "成本1×"), (2.0, "成本2×")]:
            sr = combo_sr(ls_stream(mon, cm), cb_stream(mon, cm), dz.copy())
            print(f"  {mode} {tag}: 月√12 = {sr:.2f}", flush=True)
    # 关键对照:周频2×(=2.53基准) vs 月频2×
    w2 = combo_sr(ls_stream(False, 2.0), cb_stream(False, 2.0), dz.copy())
    m2 = combo_sr(ls_stream(True, 2.0), cb_stream(True, 2.0), dz.copy())
    print(f"\n降换手效果(2×成本): 周频={w2:.2f} → 月频={m2:.2f} | Δ={m2-w2:+.2f}(成功线≥+0.05)", flush=True)
    print(f"裁定:{'月频降换手有效' if m2-w2>=0.05 else '降换手无效/中性,~2.5是真天花板'} | 月频2×{'≥2.8 ✓达标' if m2>=2.8 else f'={m2:.2f}<2.8'}", flush=True)
    print("[OK] 95 月频降换手试验完成", flush=True)


if __name__ == "__main__":
    main()
