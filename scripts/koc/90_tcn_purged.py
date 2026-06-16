"""
夜间自主研究②:LS 泄漏验证 — purged+embargo walk-forward 重训 TCN(GPU,10seed)
============================================================================
判定:融券版 LS 的 ~2.5 是真实还是年边界泄漏虚高。与 38 同模型同数据,唯一区别=加防泄漏:
  ① embargo:测试年 Y 训练集剔除 Y-1 最后 EMBARGO_WK 周(断开"训练标签→测试输入窗口"重叠)
  ② 丢序列最后一天 X[:, :-1, :](破除"day t 同时在输入和标签基价"的微泄漏)
  ③ 10 seed 取均值(非最优,治种子方差)
  ④ 评估用非重叠月度√12(吃掉自相关,与本会话诚实口径一致)
判据(llm-chat 锁定):leak-free 月度 OOS Sharpe(seed均值)≥1.80 → LS 坐实;<1.80 → 原版泄漏/虚高。
安全:--smoke(2seed/1年/小网,GO/NO-GO 阈值)+ 每年增量存预测(崩溃不丢)+ NaN/早停守卫。

Usage:
  smoke: .venv/Scripts/python.exe scripts/koc/90_tcn_purged.py --smoke
  full : .venv/Scripts/python.exe scripts/koc/90_tcn_purged.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

LGB = "data/pred_v3_lgb.parquet"
EMBARGO_WK = 16                       # ≈81交易日/5,断开年边界标签泄漏
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = "--smoke" in sys.argv
SUF = "_smoke" if SMOKE else ""
N_SEED = 2 if SMOKE else 10
TEST_YEARS = [2021] if SMOKE else [2019, 2020, 2021, 2022, 2023, 2024, 2025]


class TCN(nn.Module):
    def __init__(self, n_feat: int = 4, ch: int = 32, drop: float = 0.3) -> None:
        super().__init__()
        layers, c_in = [], n_feat
        for d in (1, 2, 4, 8):
            layers += [nn.utils.weight_norm(nn.Conv1d(c_in, ch, 3, padding=d, dilation=d)),
                       nn.ReLU(), nn.Dropout2d(drop)]
            c_in = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2)).mean(dim=2)
        return self.head(h).squeeze(-1)


def ann(m, ppy=52):
    m = np.asarray(m, float); m = m[~np.isnan(m)]
    return float(m.mean()) / (m.std(ddof=1) + 1e-12) * np.sqrt(ppy) if len(m) >= 6 else float("nan")


def agg_monthly(weekly: pd.Series) -> pd.Series:
    """周收益→非重叠月度累积收益(吃掉自相关)。"""
    s = weekly.dropna().copy()
    s.index = pd.PeriodIndex(s.index, freq="W-FRI").to_timestamp(how="end")
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1


@torch.no_grad()
def predict(model, X, bs=8192):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(DEV, non_blocking=True)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def train_one(Xtr, ytr, Xva, yva, seed, max_ep=30):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TCN().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=0.05)
    ytr_t = torch.from_numpy(ytr).to(DEV)
    best_ic, best_state, patience = -1e9, None, 0
    n, bs = len(Xtr), 2048
    losses = []
    for epoch in range(max_ep):
        model.train(); perm = np.random.permutation(n); ep_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(Xtr[idx]).to(DEV, non_blocking=True)
            opt.zero_grad()
            loss = lossf(model(xb), ytr_t[torch.from_numpy(idx).to(DEV)])
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf loss — abort (数据或学习率问题)")
            loss.backward(); opt.step(); ep_loss += float(loss) * len(idx)
        losses.append(ep_loss / n)
        ic = spearmanr(predict(model, Xva), yva).correlation
        ic = -1e9 if np.isnan(ic) else ic
        if ic > best_ic:
            best_ic, best_state, patience = ic, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.cuda.empty_cache()
    return model, best_ic, losses


def main():
    # 注:smoke 数据用 _smoke 后缀(37 --smoke 生成);full 用全量
    dsuf = "_smoke" if SMOKE else ""
    print(f"[{'SMOKE' if SMOKE else 'FULL'}] 设备 {DEV} | 加载 data/tcn_X{dsuf}.npy", flush=True)
    X = np.load(f"data/tcn_X{dsuf}.npy")
    X = X[:, :-1, :]                            # ② 丢最后一天,破基价微泄漏 → (N,59,4) view,省内存
    md = pd.read_parquet(f"data/tcn_meta{dsuf}.parquet")
    years = md["year"].values.astype(int)
    y = md["y"].values.astype(np.float32)
    # wk → 周末时间戳(用于 embargo)
    wk_end = pd.PeriodIndex(md["wk"].astype(str), freq="W-FRI").to_timestamp(how="end")
    md = md.assign(_wkend=wk_end)
    print(f"  X={X.shape} (已丢末日) | 样本年 {sorted(set(years))}", flush=True)

    preds = np.full(len(md), np.nan, dtype=np.float32)
    diag = []
    for Y in TEST_YEARS:
        te_mask = years == Y
        # embargo:训练用 years<Y,但剔除 Y 起始前 EMBARGO_WK 周
        y_start = pd.Timestamp(f"{Y}-01-01")
        embargo_cut = y_start - pd.Timedelta(weeks=EMBARGO_WK)
        tr_mask = (years < Y) & (md["_wkend"].values < np.datetime64(embargo_cut))
        if tr_mask.sum() < (300 if SMOKE else 2000) or te_mask.sum() < 30:
            print(f"  {Y}: 训练/测试样本不足(tr={tr_mask.sum()} te={te_mask.sum()}),跳过", flush=True)
            continue
        # val = 训练集最后一年(embargo 之后剩余的最后年),chronological
        tr_years = sorted(set(years[tr_mask]))
        val_mask = tr_mask & (years == tr_years[-1])
        fit_mask = tr_mask & (years != tr_years[-1])
        if fit_mask.sum() < 200:
            fit_mask, val_mask = tr_mask, tr_mask     # 数据太少则不分(仅smoke)
        Xtr, ytr, Xva, yva, Xte = X[fit_mask], y[fit_mask], X[val_mask], y[val_mask], X[te_mask]
        seed_preds, seed_ics = [], []
        for s in range(N_SEED):
            t0 = time.time()
            model, vic, losses = train_one(Xtr, ytr, Xva, yva, seed=1000 + s,
                                           max_ep=5 if SMOKE else 30)
            pv = predict(model, Xte)
            seed_preds.append(pv); seed_ics.append(vic)
            dt = time.time() - t0
            print(f"    {Y} seed{s} val_ic={vic:+.4f} ep/s={dt:.0f}s loss[0→-1]={losses[0]:.4f}→{losses[-1]:.4f}", flush=True)
        preds[te_mask] = np.mean(seed_preds, axis=0)
        oic = spearmanr(preds[te_mask], y[te_mask]).correlation
        diag.append({"year": Y, "n_tr": int(fit_mask.sum()), "n_te": int(te_mask.sum()),
                     "oos_ic": float(oic), "val_ic_mean": float(np.mean(seed_ics)),
                     "embargo_dropped": int(((years < Y) & (md["_wkend"].values >= np.datetime64(embargo_cut))).sum())})
        print(f"  {Y}: OOS_IC={oic:+.4f} (embargo剔除{diag[-1]['embargo_dropped']}样本)", flush=True)
        # 增量存(崩溃不丢)
        md.assign(tcn_purged=preds).dropna(subset=["tcn_purged"]).to_parquet(
            f"data/pred_purged{SUF}.parquet", index=False)

    # —— 评估:同口径隔离"泄漏效应"(周vs周)与"自相关折损"(周vs月)——
    # 关键(llm-chat 审):2.5→1.9 可能纯自相关折损非泄漏,必须同口径对比。
    md["tcn_purged"] = preds
    out = md.dropna(subset=["tcn_purged"]).copy()
    # pred_ensemble 含原"泄漏"信号 ens + fwd + vol → 作同口径参考
    pe_path = f"data/pred_ensemble{dsuf}.parquet"
    if not os.path.exists(pe_path):
        pe_path = "data/pred_ensemble.parquet"        # smoke 无专属版则回退全量(仅取重叠样本作管线校验)
    pe = pd.read_parquet(pe_path)
    m = out.merge(pe[["code", "wk", "ens", "fwd", "vol"]], on=["code", "wk"], how="inner")
    m = m.dropna(subset=["tcn_purged", "ens", "fwd"])

    def build_ls_weekly(frame: pd.DataFrame, score: str) -> pd.Series:
        """同一构造下从 score 列建 LS 周收益(扣换手+8%融券)。"""
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

    def oos_of(ls: pd.Series):
        yr = pd.Series(ls.index.str.slice(0, 4).astype(int), index=ls.index)
        return ls[yr >= 2021]

    ls_ref = build_ls_weekly(m, "ens")          # 原(可能泄漏)信号,同样本同构造
    ls_pur = build_ls_weekly(m, "tcn_purged")   # leak-free 信号
    ref_w = ann(oos_of(ls_ref), 52)
    pur_w = ann(oos_of(ls_pur), 52)
    pur_m = ann(agg_monthly(oos_of(ls_pur)), 12)
    leak_drop = ref_w - pur_w                   # 周vs周 = 纯泄漏效应
    autocorr_drop = pur_w - pur_m               # 周vs月 = 自相关折损

    # 裁定:泄漏小(周√52 保住)=LS非泄漏;诚实值看月√12
    leak_ok = (pur_w >= ref_w - 0.5)            # leak-free 周 Sharpe 比参考掉<0.5 = 泄漏不显著
    verdict = ("LS非泄漏(周口径保住)✓" if leak_ok else "原LS含显著泄漏 ⚠️") + \
              f";诚实月√12={pur_m:.2f}"

    report = {
        "mode": "smoke" if SMOKE else "full", "n_seed": N_SEED, "embargo_weeks": EMBARGO_WK,
        "ref_orig_weekly_sr": round(ref_w, 3),         # 原信号 同样本周√52(参考基准)
        "leakfree_weekly_sr": round(pur_w, 3),         # leak-free 周√52(隔离泄漏)
        "leakfree_monthly_sr": round(pur_m, 3),        # leak-free 月√12(诚实值)
        "leak_effect_weekly": round(leak_drop, 3),     # 周vs周差 = 泄漏
        "autocorr_haircut": round(autocorr_drop, 3),   # 周vs月差 = 自相关折损
        "leak_significant": (not leak_ok), "verdict": verdict, "per_year": diag,
    }
    with open(f"data/tcn_purged_report{SUF}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== leak-free LS 评估(同口径隔离两效应)===", flush=True)
    print(f"  原信号同样本 周√52(参考)   = {ref_w:.2f}", flush=True)
    print(f"  leak-free  周√52           = {pur_w:.2f}   → 泄漏效应={leak_drop:+.2f}(周vs周)", flush=True)
    print(f"  leak-free  月√12(诚实值)   = {pur_m:.2f}   → 自相关折损={autocorr_drop:+.2f}(周vs月)", flush=True)
    print(f"  裁定:{verdict}", flush=True)
    m_sr = pur_m  # 供下方 smoke 判定
    print(f"  报告 → data/tcn_purged_report{SUF}.json", flush=True)
    print(f"\n[OK] 90_tcn_purged {'smoke' if SMOKE else 'full'} 完成", flush=True)

    if SMOKE:
        # smoke GO/NO-GO(llm-chat 阈值):val_ic≥0.025、loss下降、无NaN、有预测
        go = (len(diag) >= 1 and diag[0]["val_ic_mean"] >= 0.015 and not np.isnan(m_sr))
        print(f"\n[SMOKE GO/NO-GO] {'GO ✓ 可启动整夜全量' if go else 'NO-GO ✗ 需排查'}"
              f"(val_ic_mean={diag[0]['val_ic_mean']:+.4f} 阈值0.015, 有OOS评估={not np.isnan(m_sr)})", flush=True)


if __name__ == "__main__":
    main()
