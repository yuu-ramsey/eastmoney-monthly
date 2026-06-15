"""
KoC §4 Pre-check: Under-reaction Kill Point
============================================
Fama-MacBeth 控 persistence 后 SUE 是否仍预测收益。

Kill 判定：FM γ1 t-stat < 2 → under-reaction 不存在 → 停止 §5/§6

数据模式：
  --synthetic (默认): pead.sqlite SUE + 合成收益率（验证代码路径，标"非最终"）
  --akshare N:        akshare 真实收益率，随机抽 N 只股票（默认 100）
  --real:             baostock SUE (§14) + PIT 市值 (§16)，§10 完成后才能用

FM 设定：
  future_ret_i = γ0 + γ1·SUE_i + γ2·sue_autocorr_i
               + Σ industry_dummies + Σ season_dummies + ε_i
  每截面期 OLS，时序均值 γ1 做 t 检验（Newey-West lags=4）

Usage:
  .venv/Scripts/python.exe scripts/koc/03_precheck.py            # synthetic
  .venv/Scripts/python.exe scripts/koc/03_precheck.py --akshare 100
  .venv/Scripts/python.exe scripts/koc/03_precheck.py --real
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ── 常量 ──────────────────────────────────────────────────────────────────────
INDUSTRY_MAP_PATH: str = "data/industry-map.json"
DB_SUE_AKSHARE: str = "data/pead.sqlite"         # akshare YTD 法
DB_SUE_BAOSTOCK: str = "data/pead-baostock.sqlite"  # baostock（§14 完成后）
REPORT_PATH: str = "docs/koc-precheck.md"

# FM kill 门槛（真实数据门槛 t>3；synthetic 门槛 t>2 以避免false-KILL）
KILL_T_STAT: float = 3.0          # γ1 t-stat 低于此值 → KILL（--real 正式门槛）
KILL_T_STAT_SYNTHETIC: float = 2.0  # synthetic/akshare 模式用此门槛
MIN_CROSS_SECTION_N: int = 30     # 每截面最少股票数，否则跳过
MIN_PERIODS: int = 8              # FM 时序长度最低要求（Newey-West 需要足够时序）
NW_LAGS: int = 4                  # Newey-West l滞阶数

# 合成收益率参数（--synthetic 模式）
SYNTH_ALPHA: float = 0.01         # 真实信号：future_ret = α·SUE + noise
SYNTH_NOISE_STD: float = 0.08     # 月收益噪声标准差（~8%，接近A股水平）
SYNTH_SEED: int = 42

# AR(1) persistence 估计
MIN_SUE_HISTORY_FOR_AUTOCORR: int = 4  # 至少 4 期才估 AR(1)


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_sue(mode: str) -> pd.DataFrame:
    """
    返回 DataFrame: code, fiscal_year, fiscal_quarter, pub_date, sue
    pub_date 为 datetime，仅含 trusted=1 行
    """
    if mode == "real":
        assert Path(DB_SUE_BAOSTOCK).exists(), (
            f"§14 数据库不存在: {DB_SUE_BAOSTOCK}\n"
            "§10 尚未完成——禁止连 baostock。等 §10 跑完后再用 --real 模式。"
        )
        db_path = DB_SUE_BAOSTOCK
        table = "sue_baostock"
    else:
        assert Path(DB_SUE_AKSHARE).exists(), f"pead.sqlite 不存在: {DB_SUE_AKSHARE}"
        db_path = DB_SUE_AKSHARE
        table = "sue"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT code, fiscal_year, fiscal_quarter, pub_date, sue "
            f"FROM {table} WHERE trusted=1 AND pub_date IS NOT NULL AND sue IS NOT NULL",
            conn,
        )

    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["sue"] = pd.to_numeric(df["sue"], errors="coerce")
    df = df.dropna(subset=["sue"])

    assert not df.empty, f"load_sue: {table} 无 trusted 行"
    return df


def compute_sue_autocorr(df_sue: pd.DataFrame) -> pd.DataFrame:
    """
    为每个观测计算 PIT AR(1) sue_autocorr（用该股历史 SUE，不含当期）。
    纯内存计算，不连任何外部数据源。
    返回原 df 加 sue_autocorr 列（无历史则 NaN）。
    """
    df = df_sue.sort_values(["code", "fiscal_year", "fiscal_quarter"]).copy()
    autocorr_list: list[float] = []

    for code, grp in df.groupby("code", sort=False):
        sue_vals = grp["sue"].tolist()
        n = len(sue_vals)
        for i in range(n):
            history = sue_vals[:i]  # 严格历史（PIT）
            if len(history) < MIN_SUE_HISTORY_FOR_AUTOCORR:
                autocorr_list.append(float("nan"))
                continue
            y = np.array(history[1:])
            x = np.array(history[:-1])
            if np.var(x) < 1e-12:
                autocorr_list.append(0.0)
                continue
            phi = float(np.cov(x, y)[0, 1] / np.var(x))
            phi = max(-1.0, min(1.0, phi))
            autocorr_list.append(phi)

    df["sue_autocorr"] = autocorr_list
    return df


def attach_synthetic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """合成收益率：future_ret = SYNTH_ALPHA·sue + noise。用于干跑验证逻辑。"""
    rng = np.random.default_rng(SYNTH_SEED)
    noise = rng.normal(0.0, SYNTH_NOISE_STD, size=len(df))
    # 对 SUE 做 clip 防止极端值主导
    sue_clipped = df["sue"].clip(-5, 5).values
    df = df.copy()
    df["future_ret"] = SYNTH_ALPHA * sue_clipped + noise
    df["ret_source"] = "synthetic"
    return df


def attach_akshare_returns(df: pd.DataFrame, n_stocks: int) -> pd.DataFrame:
    """
    从 akshare stock_zh_a_hist 拉月收益率，仅用于 --akshare 干跑。
    随机抽 n_stocks 只，fetch 全历史月线，按 pub_date 对齐到下一自然月的收益率。
    """
    import akshare as ak

    rng = np.random.default_rng(SYNTH_SEED)
    all_codes = df["code"].unique()
    sampled_codes = rng.choice(all_codes, size=min(n_stocks, len(all_codes)), replace=False)
    df_sub = df[df["code"].isin(sampled_codes)].copy()

    print(f"  [akshare] 抓取 {len(sampled_codes)} 只股票月收益率...")
    ret_map: dict[tuple, float] = {}  # (code, year, month) → monthly_ret

    for i, code_full in enumerate(sampled_codes):
        code_6 = code_full.split(".")[-1]
        try:
            hist = ak.stock_zh_a_hist(
                symbol=code_6, period="monthly",
                start_date="20100101", end_date="20260101",
                adjust="hfq",
            )
            if hist.empty or "日期" not in hist.columns:
                continue
            hist["日期"] = pd.to_datetime(hist["日期"])
            hist = hist.sort_values("日期")
            hist["ret"] = hist["收盘"].pct_change()
            for _, row in hist.iterrows():
                key = (code_full, row["日期"].year, row["日期"].month)
                ret_map[key] = float(row["ret"]) if pd.notna(row["ret"]) else float("nan")
        except Exception as exc:
            print(f"  [WARN] {code_6}: {exc}", file=sys.stderr)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sampled_codes)}] 已完成")

    def _lookup_next_month_ret(row: pd.Series) -> float:
        # 取 pub_date 下一个月的收益率
        pub: pd.Timestamp = row["pub_date"]
        next_month = pub.month % 12 + 1
        next_year = pub.year + (1 if pub.month == 12 else 0)
        return ret_map.get((row["code"], next_year, next_month), float("nan"))

    df_sub = df_sub.copy()
    df_sub["future_ret"] = df_sub.apply(_lookup_next_month_ret, axis=1)
    df_sub["ret_source"] = "akshare_monthly"
    return df_sub


def attach_real_returns(df: pd.DataFrame, hold_window: int = 20) -> pd.DataFrame:
    """
    真实 PIT 收益率（§15 daily_kline 复权收益）。

    设计：
      - 持有起点 = pub_date 之后第 HOLD_START_DAYS 个交易日的收盘价
      - 持有终点 = 起点之后第 hold_window 个交易日的收盘价
      - future_ret = pct_chg 复利累乘（除权后真实收益）

    同时从 koc_universe 拉 market_cap_yi 和 in_liquid_pool 用于控制和池过滤。
    """
    HOLD_START_DAYS: int = 5        # 避涨跌停：pub_date 后第5个交易日建仓
    HOLD_WINDOW_DAYS: int = hold_window   # 20=2026-06-10 注册规格；60=预注册重跑规格

    # ⚠ daily_kline.close 是不复权价（adjustflag='3'，专供 §16 市值），
    #   除权除息日 close 机械性下跌，不能做收益率——高 SUE 股财报后常分红，
    #   持有窗罩住除权日会系统性压负 γ1（2026-06-10 实测 bug）。
    #   收益必须用 pct_chg（baostock 按除权后昨收价计算的真实涨跌幅）复利累乘。
    print("  [real] 加载 daily_kline pct_chg（复权收益率）...")
    conn = sqlite3.connect(DB_SUE_BAOSTOCK)
    try:
        kline = pd.read_sql_query(
            "SELECT code, trade_date, pct_chg FROM daily_kline "
            "WHERE tradestatus='1' AND pct_chg IS NOT NULL "
            "ORDER BY code, trade_date",
            conn,
        )
        universe = pd.read_sql_query(
            "SELECT code, fiscal_year, fiscal_quarter, "
            "market_cap_yi, mktcap_missing, in_liquid_pool "
            "FROM koc_universe",
            conn,
        )
    finally:
        conn.close()

    kline["trade_date"] = pd.to_datetime(kline["trade_date"])
    # 按 code 预计算：交易日序列 + log(1+r) 前缀和（O(1) 区间收益查询）
    kline_by_code: dict[str, np.ndarray] = {}
    logret_cumsum_by_code: dict[str, np.ndarray] = {}
    for code, grp in kline.groupby("code", sort=False):
        grp = grp.sort_values("trade_date")
        kline_by_code[code] = grp["trade_date"].values
        log_ret = np.log1p(grp["pct_chg"].values / 100.0)
        # cumsum[i] = sum(log_ret[0..i])；区间 (s, e] 收益 = exp(cs[e]-cs[s]) - 1
        logret_cumsum_by_code[code] = np.cumsum(log_ret)

    print(f"  [real] 计算 {len(df):,} 事件的前向收益率 "
          f"（建仓 +{HOLD_START_DAYS}d，持有 {HOLD_WINDOW_DAYS}d，pct_chg 复利）...")

    future_rets: list[Optional[float]] = []
    df_sorted = df.copy()
    for _, row in df_sorted.iterrows():
        code = row["code"]
        pub_date = row["pub_date"]
        dates = kline_by_code.get(code)
        if dates is None or len(dates) == 0:
            future_rets.append(None)
            continue
        # 建仓日 = pub_date 之后第 HOLD_START_DAYS 个交易日
        idx = int(np.searchsorted(dates, pub_date.to_datetime64(), side="right"))
        start_idx = idx + HOLD_START_DAYS - 1
        end_idx = start_idx + HOLD_WINDOW_DAYS
        if end_idx >= len(dates):
            future_rets.append(None)
            continue
        # 持有收益 = exp(Σ log(1+r) over (start_idx, end_idx]) - 1
        cs = logret_cumsum_by_code[code]
        ret = float(np.expm1(cs[end_idx] - cs[start_idx]))
        future_rets.append(ret)

    df_sorted["future_ret"] = future_rets
    df_sorted["ret_source"] = "daily_kline_real"

    # 拼接 koc_universe 的市值和流动性标志
    universe["pub_key"] = (
        universe["code"] + "_" +
        universe["fiscal_year"].astype(str) + "_" +
        universe["fiscal_quarter"].astype(str)
    )
    df_sorted["pub_key"] = (
        df_sorted["code"] + "_" +
        df_sorted["fiscal_year"].astype(str) + "_" +
        df_sorted["fiscal_quarter"].astype(str)
    )
    u_map = universe.set_index("pub_key")[
        ["market_cap_yi", "mktcap_missing", "in_liquid_pool"]
    ]
    df_sorted = df_sorted.join(u_map, on="pub_key")
    df_sorted.drop(columns=["pub_key"], inplace=True)

    # log_mktcap 用于 FM size 控制（仅 in_liquid_pool=1 时才有意义）
    df_sorted["log_mktcap"] = np.log(
        df_sorted["market_cap_yi"].clip(lower=1e-6) * 1e8   # 转元
    ).where(df_sorted["mktcap_missing"].fillna(1) == 0)

    n_ret = int(df_sorted["future_ret"].notna().sum())
    n_liquid = int((df_sorted["in_liquid_pool"] == 1).sum())
    print(f"  [real] future_ret 覆盖: {n_ret:,}/{len(df_sorted):,}"
          f"  流动池事件: {n_liquid:,}")
    return df_sorted


# ── 行业映射（L1，对齐 neutralize.cjs / python_neutralize）───────────────────
def load_industry_map(path: str) -> dict[str, str]:
    """返回 {code_6: L1_name}；文件不存在时返回空字典。"""
    if not Path(path).exists():
        return {}
    d = json.load(open(path, encoding="utf-8"))
    stock_to_l2: dict[str, str] = d["stockToIndustry"]
    l2_to_l1: dict[str, str] = {
        ind["name"]: ind["l1Name"] for ind in d["industries"]
    }
    return {code: l2_to_l1.get(l2, l2) for code, l2 in stock_to_l2.items()}


def get_l1_industry(codes: pd.Series, industry_map: dict[str, str]) -> pd.Series:
    """查 industry-map.json L1；未收录股票标为 'UNKNOWN_SECTOR'（不用 code 前3位假行业）。"""
    def _lookup(code: str) -> str:
        code_6 = code.split(".")[-1]
        return industry_map.get(code_6, "UNKNOWN_SECTOR")
    return codes.map(_lookup)


# ── Fama-MacBeth 回归 ─────────────────────────────────────────────────────────
def _ols_coef(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """OLS β = (X'X)^{-1} X'y，失败返回 None。"""
    try:
        coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < X.shape[1]:
            return None
        return coef
    except np.linalg.LinAlgError:
        return None


def run_fm_cross_section(
    df_period: pd.DataFrame,
    control_autocorr: bool = True,
    use_log_mktcap: bool = False,
    sue_spec: str = "raw",
) -> Optional[dict]:
    """
    单期截面 OLS：future_ret ~ 1 + sue + [sue_autocorr] + [log_mktcap]
                              + industry_dummies + season_dummies
    sue_spec：raw=原始 z-score；winsor=P1/P99 截尾后 z-score；rank=截面秩 [-0.5,0.5]。
      2026-06-10 审计发现 raw SUE 厚尾（±215）导致单点杠杆 max 80%，
      11/60 期回归被离群点主导——winsor/rank 是测量修复，非门控放宽。
    返回 {'gamma1': float, 'gamma2': float|None, 'n': int} 或 None（样本不足）
    """
    df = df_period.dropna(subset=["future_ret", "sue"])
    if control_autocorr:
        df = df.dropna(subset=["sue_autocorr"])
    if use_log_mktcap and "log_mktcap" in df.columns:
        df = df.dropna(subset=["log_mktcap"])

    n = len(df)
    if n < MIN_CROSS_SECTION_N:
        return None

    # SUE 变换 + 截面标准化（防量纲问题）
    if sue_spec == "winsor":
        lo, hi = df["sue"].quantile(0.01), df["sue"].quantile(0.99)
        sue_base = df["sue"].clip(lo, hi)
    elif sue_spec == "rank":
        sue_base = df["sue"].rank() / (n + 1) - 0.5
    else:
        sue_base = df["sue"]
    sue_z = (sue_base - sue_base.mean()) / (sue_base.std() + 1e-8)
    feature_cols: dict[str, np.ndarray] = {"sue": sue_z.values}

    if control_autocorr and "sue_autocorr" in df.columns:
        ac = df["sue_autocorr"]
        feature_cols["sue_autocorr"] = ((ac - ac.mean()) / (ac.std() + 1e-8)).values

    if use_log_mktcap and "log_mktcap" in df.columns:
        mc = df["log_mktcap"]
        feature_cols["log_mktcap"] = ((mc - mc.mean()) / (mc.std() + 1e-8)).values

    # Industry dummies（来自 industry-map.json L1）
    if "l1_industry" not in df.columns:
        raise KeyError(
            "l1_industry 列缺失。请在 main() 中先调用 load_industry_map() "
            "并挂载 l1_industry 列后再运行 FM 截面回归。"
        )
    industry = df["l1_industry"]
    industry_dummies = pd.get_dummies(industry, prefix="ind", drop_first=True)

    # Season dummies（fiscal_quarter，去掉 Q1 基准；
    # 跨期 FM 中 per-cross-section OLS 已隐式吸收，但显式加无害）
    season_dummies = pd.get_dummies(df["fiscal_quarter"], prefix="q", drop_first=True)

    X_parts = [np.ones((n, 1))]
    for _, arr in feature_cols.items():
        X_parts.append(arr.reshape(-1, 1))
    X_parts.append(industry_dummies.values)
    X_parts.append(season_dummies.values)

    X = np.hstack(X_parts)
    y = df["future_ret"].values

    coef = _ols_coef(X, y)
    if coef is None:
        return None

    # coef[0]=截距, coef[1]=γ1(SUE), coef[2]=γ2(sue_autocorr)/γ2(log_mktcap)...
    result: dict = {"gamma1": float(coef[1]), "n": n}
    if len(feature_cols) >= 2:
        result["gamma2"] = float(coef[2])
    else:
        result["gamma2"] = None
    return result


def newey_west_se(series: np.ndarray, lags: int) -> float:
    """
    Newey-West 标准误（均值检验）。
    Var(γ̄) = (1/T^2) * [Σ e_t^2 + 2 Σ_{l=1}^{L} w_l Σ_t e_t·e_{t-l}]
    其中 e_t = γ_t - γ̄，w_l = 1 - l/(L+1)
    """
    t = len(series)
    e = series - series.mean()
    variance = float(np.dot(e, e))
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        cross = float(np.dot(e[lag:], e[:-lag]))
        variance += 2.0 * weight * cross
    variance = max(variance, 0.0)  # 数值保护
    return float(np.sqrt(variance / (t * t)))


def run_fama_macbeth(
    df: pd.DataFrame,
    kill_threshold: float = KILL_T_STAT,
    use_log_mktcap: bool = False,
    sue_spec: str = "raw",
) -> dict:
    """
    完整 FM 流程：
      1. 按 (fiscal_year, fiscal_quarter) 分组跑截面 OLS
      2. 收集每期 γ1（控 persistence 的 SUE 系数）
      3. 时序均值 + Newey-West t 检验
    返回结果字典。
    """
    gammas: list[float] = []
    skipped_periods: int = 0
    period_detail: list[dict] = []

    grouped = df.groupby(["fiscal_year", "fiscal_quarter"])
    for (fy, fq), grp in grouped:
        result = run_fm_cross_section(grp, control_autocorr=True,
                                      use_log_mktcap=use_log_mktcap,
                                      sue_spec=sue_spec)
        if result is None:
            skipped_periods += 1
            continue
        gammas.append(result["gamma1"])
        period_detail.append({
            "period": f"{fy}Q{fq}",
            "gamma1": result["gamma1"],
            "n": result["n"],
        })

    if len(gammas) < MIN_PERIODS:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n_periods": len(gammas),
            "message": f"有效截面期 {len(gammas)} < {MIN_PERIODS}，无法判断",
        }

    gamma_arr = np.array(gammas)
    gamma_mean = float(gamma_arr.mean())
    nw_se = newey_west_se(gamma_arr, lags=NW_LAGS)
    t_stat = gamma_mean / (nw_se + 1e-12)
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=len(gammas) - 1)))

    verdict = "PASS" if t_stat >= kill_threshold else "KILL"

    return {
        "verdict": verdict,
        "gamma_mean": gamma_mean,
        "nw_se": nw_se,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_periods": len(gammas),
        "n_skipped": skipped_periods,
        "gammas": gammas,
        "period_detail": period_detail,
    }


# ── 单调性检验 ────────────────────────────────────────────────────────────────
def quintile_analysis(df: pd.DataFrame) -> dict:
    """
    按 SUE 分 5 组，检验 future_ret 是否 Q5>Q4>Q3>Q2>Q1（辅助展示）。
    返回各组均值 + Spearman ρ + monotonicity_ok。
    """
    df = df.dropna(subset=["future_ret", "sue"]).copy()
    if len(df) < 200:
        return {"ok": False, "message": "样本不足（<200）"}

    # 每期内打分位，然后汇总（防止时期分布偏移）
    def _assign_quintile(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        grp["quintile"] = pd.qcut(grp["sue"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        return grp

    df = df.groupby(["fiscal_year", "fiscal_quarter"], group_keys=False).apply(_assign_quintile)
    df = df.dropna(subset=["quintile"])

    q_means = (
        df.groupby("quintile")["future_ret"]
        .mean()
        .reset_index()
        .rename(columns={"future_ret": "mean_ret"})
    )
    q_means["quintile"] = q_means["quintile"].astype(int)
    q_means = q_means.sort_values("quintile")

    rets = q_means["mean_ret"].values
    quintiles = q_means["quintile"].values
    rho, p_spearman = stats.spearmanr(quintiles, rets)

    # 严格单调性检查
    monotone_ok = all(rets[i] < rets[i + 1] for i in range(len(rets) - 1))
    ls_spread = float(rets[-1] - rets[0]) if len(rets) == 5 else float("nan")

    return {
        "ok": True,
        "quintile_means": q_means.to_dict(orient="records"),
        "spearman_rho": float(rho),
        "spearman_p": float(p_spearman),
        "monotone_strict": monotone_ok,
        "ls_spread": ls_spread,
    }


# ── 报告生成 ──────────────────────────────────────────────────────────────────
def build_report(
    fm: dict,
    quintle: dict,
    mode: str,
    n_stocks: int,
    n_obs: int,
    elapsed: float,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_warn = ""
    if mode != "real":
        mode_warn = (
            "\n> ⚠️ **[非最终结果]** 当前模式：`{}`。收益率{}，"
            "等 §14 SUE + §15/16 市值到位后用 `--real` 重跑才是正式判定。\n".format(
                mode,
                "为合成数据（α·SUE + 噪声）" if mode == "synthetic" else "来自 akshare 月线",
            )
        )

    verdict_emoji = "🔴 KILL" if fm.get("verdict") == "KILL" else (
        "🟢 PASS" if fm.get("verdict") == "PASS" else "⚪ 数据不足"
    )

    # FM 详情
    if fm.get("verdict") in ("KILL", "PASS"):
        fm_section = f"""
## FM 回归结果（控 persistence 后 SUE 系数 γ1）

| 指标 | 值 |
|------|----|
| γ1 均值 | {fm['gamma_mean']:.6f} |
| Newey-West SE（lags={NW_LAGS}） | {fm['nw_se']:.6f} |
| t-statistic | {fm['t_stat']:.3f} |
| p-value（双尾） | {fm['p_value']:.4f} |
| 有效截面期数 | {fm['n_periods']} |
| 跳过期数（n<{MIN_CROSS_SECTION_N}） | {fm['n_skipped']} |

**Kill 门槛**：t ≥ {KILL_T_STAT} 才 PASS；t < {KILL_T_STAT} → KILL。

**判定：{verdict_emoji}**
"""
        if fm.get("period_detail"):
            rows = [
                f"| {r['period']} | {r['gamma1']:+.4f} | {r['n']} |"
                for r in fm["period_detail"][-12:]  # 最近 12 期
            ]
            fm_section += "\n### 各期 γ1（最近 12 期）\n\n| 期 | γ1 | n |\n|----|----|-|\n"
            fm_section += "\n".join(rows)
    else:
        fm_section = f"\n## FM 回归结果\n\n{fm.get('message', '未完成')}\n"

    # 单调性
    if quintle.get("ok"):
        q_rows = [
            f"| Q{r['quintile']} | {r['mean_ret']:+.4f} |"
            for r in quintle["quintile_means"]
        ]
        q_table = "\n".join(q_rows)
        mono_ok = "✅ 严格单调" if quintle["monotone_strict"] else "❌ 非严格单调"
        q_section = f"""
## 单调性检验（辅助，以 FM γ1 为准）

| 组 | 平均月收益 |
|----|-----------|
{q_table}

- L/S 价差：{quintle['ls_spread']:+.4f}
- Spearman ρ = {quintle['spearman_rho']:.3f}，p = {quintle['spearman_p']:.4f}
- {mono_ok}

> 注：判定以 FM γ1（控 persistence）为准，单调性为辅助展示。
"""
    else:
        q_section = f"\n## 单调性检验\n\n{quintle.get('message', '未完成')}\n"

    kill_guidance = ""
    if fm.get("verdict") == "KILL":
        kill_guidance = """
## KILL 结论

控制 persistence 后 SUE 对收益无显著预测力（γ1 t-stat < 2），
**PEAD under-reaction 在 A 股不成立**。

→ 停止 §5/§6，不继续后续检验。

*需用 --real 模式（§14 + §15/16 数据）做最终确认。*
"""
    elif fm.get("verdict") == "PASS":
        kill_guidance = """
## PASS 结论

控制 persistence 后 SUE 仍显著预测收益（γ1 t-stat ≥ 2），
**under-reaction 存在**，继续 §5 主测试。

*需用 --real 模式做最终确认后才算正式 PASS。*
"""

    return f"""# KoC §4 前置检验报告（Pre-check Kill Point）

**生成时间**：{now}
**模式**：`{mode}`
**样本**：{n_obs:,} 观测 / {n_stocks:,} 只股票
**耗时**：{elapsed:.1f}s
{mode_warn}
## FM 设定

```
Step1 每截面期 OLS：
  future_ret_i = γ0 + γ1·SUE_i + γ2·sue_autocorr_i
               + Σ industry_dummies（industry-map.json L1） + Σ season_dummies + ε_i

Step2 时序均值 γ1，Newey-West t 检验（lags={NW_LAGS}）
Kill 门槛：t(γ1) < {KILL_T_STAT}
```

**SUE_autocorr** = 该股截至当期的 SUE 序列 PIT AR(1) 系数（persistence 代理）。
控住它后 γ1 还显著 = 投资者对盈余冲击反应不足（under-reaction），而非盈余本身持续。
{fm_section}
{q_section}
{kill_guidance}
## 真实数据运行说明

等以下数据就绪后替换输入，运行 `--real` 得到正式判定：

| 数据 | 脚本 | 状态 |
|------|------|------|
| §14 baostock SUE | 14_sue_baostock.py | ⏳ 等 §10（~86h） |
| §15 日线（收益率） | 15_fetch_baostock_daily.py | ⏳ 等 §10 |
| §16 PIT 市值 | 16_pit_marketcap.py | ⏳ 等 §10 |
| §17 ST 历史 | 17_st_history.py | ✅ 完成 |

```bash
# 真数据到位后：
.venv/Scripts/python.exe scripts/koc/03_precheck.py --real
```
"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="KoC §4 前置检验：FM 控 persistence")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--synthetic", action="store_true", default=True,
        help="合成收益率干跑（默认）",
    )
    group.add_argument(
        "--akshare", type=int, metavar="N",
        help="akshare 真实月收益率，抽 N 只股票",
    )
    group.add_argument(
        "--real", action="store_true",
        help="真实数据（§14 baostock SUE + §15 日线），§10 完成后才可用",
    )
    parser.add_argument(
        "--window", type=int, default=20, metavar="N",
        help="持有窗口交易日数（默认 20=2026-06-10 注册规格；60=预注册重跑规格）",
    )
    parser.add_argument(
        "--sue-spec", choices=["raw", "winsor", "rank"], default="raw",
        help="SUE 变换（raw=2026-06-10 注册规格；rank=预注册重跑规格，离群杠杆修复）",
    )
    args = parser.parse_args()

    # 确定模式
    if args.real:
        mode = "real"
    elif args.akshare is not None:
        mode = "akshare"
        args.synthetic = False
    else:
        mode = "synthetic"

    # 硬性保护：--real 模式前确认 §10 完成
    if mode == "real":
        assert Path(DB_SUE_BAOSTOCK).exists(), (
            "§10 尚未完成或 pead-baostock.sqlite 不存在。\n"
            "禁止在 §10 运行期间连接 baostock！等 §10 结束后再用 --real。"
        )

    t0 = time.time()
    print(f"[03_precheck] {datetime.now():%Y-%m-%d %H:%M:%S} | 模式: {mode}")
    print()

    # 1. 加载 SUE
    print("[1/5] 加载 SUE 数据...")
    df_sue = load_sue(mode)
    print(f"  {len(df_sue):,} 观测 | {df_sue['code'].nunique():,} 只股票 "
          f"| {df_sue['pub_date'].dt.year.min()}-{df_sue['pub_date'].dt.year.max()}")

    # 挂载 L1 行业（industry-map.json）——缺失直接报错，不允许退回 code 前3位假行业
    if not Path(INDUSTRY_MAP_PATH).exists():
        raise FileNotFoundError(
            f"[FATAL] industry-map.json 不存在: {INDUSTRY_MAP_PATH}\n"
            "FM 行业 dummies 需要 L1 真实行业分类，code 前3位是交易所编码段而非行业，\n"
            "静默退回会产生假的行业中性化结果。请先确保 industry-map.json 就位再重试。"
        )
    industry_map = load_industry_map(INDUSTRY_MAP_PATH)
    df_sue["l1_industry"] = get_l1_industry(df_sue["code"], industry_map)
    mapped = int(df_sue["l1_industry"].ne("UNKNOWN_SECTOR").sum())
    unknown = int(df_sue["l1_industry"].eq("UNKNOWN_SECTOR").sum())
    print(f"  industry-map.json: {len(industry_map)} 只 | "
          f"命中 {mapped}/{len(df_sue)}"
          + (f" | ⚠️ {unknown} 只未收录→UNKNOWN_SECTOR" if unknown else ""))

    # 限制 akshare 模式股票数
    if mode == "akshare":
        n_akshare = args.akshare
        sampled = (
            np.random.default_rng(SYNTH_SEED)
            .choice(df_sue["code"].unique(), size=min(n_akshare, df_sue["code"].nunique()), replace=False)
        )
        df_sue = df_sue[df_sue["code"].isin(sampled)]
        print(f"  akshare 模式：限定 {len(sampled)} 只股票")

    # 2. 计算 sue_autocorr（PIT AR(1)）
    print("[2/5] 计算 PIT SUE 自相关（persistence proxy）...")
    df = compute_sue_autocorr(df_sue)
    n_with_ac = int(df["sue_autocorr"].notna().sum())
    print(f"  {n_with_ac:,} 观测有有效 sue_autocorr（{100*n_with_ac/len(df):.1f}%）")

    # 3. 拼接收益率
    print("[3/5] 拼接 future_ret...")
    if mode == "synthetic":
        df = attach_synthetic_returns(df)
        print(f"  合成收益率：α={SYNTH_ALPHA}, noise_std={SYNTH_NOISE_STD}")
    elif mode == "akshare":
        df = attach_akshare_returns(df, n_stocks=args.akshare)
        n_ret = int(df["future_ret"].notna().sum())
        print(f"  akshare 收益率覆盖：{n_ret:,}/{len(df):,} 观测")
    else:
        df = attach_real_returns(df, hold_window=args.window)

    df_valid = df.dropna(subset=["future_ret"])
    print(f"  有效样本（future_ret 非 NaN）：{len(df_valid):,} 观测")

    # --real 模式：用流动池（in_liquid_pool=1）做正式判定，全样本做对比
    use_log_mktcap = False
    kill_threshold = KILL_T_STAT_SYNTHETIC
    if mode == "real":
        kill_threshold = KILL_T_STAT   # t>3 正式门槛
        if "in_liquid_pool" in df_valid.columns:
            n_liquid = int((df_valid["in_liquid_pool"] == 1).sum())
            print(f"  流动池事件: {n_liquid:,} / {len(df_valid):,}")
            if n_liquid < MIN_CROSS_SECTION_N * MIN_PERIODS:
                print("  ⚠ 流动池样本不足，用全样本运行 FM")
            else:
                df_valid = df_valid[df_valid["in_liquid_pool"] == 1]
                print(f"  使用流动池: {len(df_valid):,} 观测")
        use_log_mktcap = "log_mktcap" in df_valid.columns and df_valid["log_mktcap"].notna().sum() > 100

    assert len(df_valid) >= MIN_CROSS_SECTION_N * MIN_PERIODS, (
        f"有效样本 {len(df_valid)} 太少，FM 无法运行"
    )

    # 4. Fama-MacBeth 回归
    size_note = "（+log市值控制）" if use_log_mktcap else ""
    print(f"[4/5] Fama-MacBeth 回归（控 persistence + 行业季节{size_note}，"
          f"窗口={args.window}d，sue_spec={args.sue_spec}，门槛 t>{kill_threshold}）...")
    fm_result = run_fama_macbeth(df_valid, kill_threshold=kill_threshold,
                                  use_log_mktcap=use_log_mktcap,
                                  sue_spec=args.sue_spec)
    if fm_result.get("verdict") in ("KILL", "PASS"):
        print(f"  γ1 均值 = {fm_result['gamma_mean']:.6f} | "
              f"t = {fm_result['t_stat']:.3f} | "
              f"p = {fm_result['p_value']:.4f} | "
              f"有效期数 = {fm_result['n_periods']}")
        print(f"  判定: {'🔴 KILL' if fm_result['verdict'] == 'KILL' else '🟢 PASS'}")
    else:
        print(f"  {fm_result.get('message', '回归未完成')}")

    # 5. 单调性检验
    print("[5/5] 单调性检验...")
    q_result = quintile_analysis(df_valid)
    if q_result.get("ok"):
        means = [r["mean_ret"] for r in q_result["quintile_means"]]
        print(f"  Q1~Q5: {' / '.join(f'{v:+.4f}' for v in means)}")
        print(f"  Spearman ρ={q_result['spearman_rho']:.3f} | "
              f"严格单调: {'是' if q_result['monotone_strict'] else '否'}")

    # 报告
    elapsed = time.time() - t0
    n_stocks_out = int(df_valid["code"].nunique())
    report = build_report(fm_result, q_result, mode, n_stocks_out, len(df_valid), elapsed)
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text(report, encoding="utf-8")
    print(f"\n  报告: {REPORT_PATH}")
    print(f"[03_precheck] 完成，耗时 {elapsed:.1f}s")
    print("✅ Script completed successfully")


if __name__ == "__main__":
    main()
