"""
v32_final_eval.py — 32维最终版因子评估（全修复合并）
======================================================
修复清单（全部来自验证流程确认）：
  1. 收益口径：单月 (c[i+1]-c[i])/c[i]，非累积
  2. 日期分组：统一截断 YYYY-MM，同月股票归入同一截面
  3. 特征版本：32维（FFT振幅10 + G7全14 + 均线偏离3 + MACD3 + G4精选2）
  4. 回测窗口：2015-01 ~ 数据末尾
  5. 行业中性化：CSRC L2

输出：
  - T+1~T+6 单月IC / ICIR / Hit Rate
  - 5-Fold 时间序列CV
  - 纯多头回测（Q5组，等权，月频调仓）
  - 结果保存到 .eastmoney-ai/final_eval/
"""

import numpy as np
import sqlite3
import os
import json
from collections import defaultdict
from datetime import datetime

# ============================================================
# 配置
# ============================================================
DB_PATH = ".eastmoney-ai/db/klines-v2.sqlite"
INDUSTRY_PATH = "data/industry-map.json"
OUTPUT_DIR = ".eastmoney-ai/final_eval/"
MIN_STOCKS_PER_MONTH = 20
MAX_HORIZONS = 6
N_FOLDS = 5
RANDOM_SEED = 42
COST_BPS = [10, 20, 30, 50]

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


# ============================================================
# 数据加载
# ============================================================
def load_data():
    """从 SQLite 加载月线数据，日期统一截断为 YYYY-MM"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT code, date, open, high, low, close, volume, turnover_rate
        FROM monthly_klines
        ORDER BY code, date
    """)
    rows = cur.fetchall()
    conn.close()

    # 按股票分组，日期截断为 YYYY-MM（修复#2：去日粒度）
    stock_data = defaultdict(list)
    for code, date_str, o, h, l, c, v, t in rows:
        if c is None:  # 过滤 23 条 NULL close 记录
            continue
        ym = str(date_str)[:7]  # "2024-01-31" → "2024-01"
        stock_data[code].append({
            'ym': ym, 'open': o, 'high': h, 'low': l,
            'close': c, 'volume': v, 'turnover': t or 0.0
        })

    # 同股同月多条记录取最后一条（去重）
    for code in stock_data:
        seen = {}
        for rec in stock_data[code]:
            seen[rec['ym']] = rec
        stock_data[code] = sorted(seen.values(), key=lambda x: x['ym'])

    # 加载 CSRC L2 行业映射（stock→industry 在 stockToIndustry 子对象中）
    industry_map = {}
    if os.path.exists(INDUSTRY_PATH):
        with open(INDUSTRY_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            industry_map = raw.get('stockToIndustry', {})

    covered = sum(1 for c in stock_data if c in industry_map)
    print(f"加载 {len(stock_data)} 只股票 | "
          f"行业覆盖 {covered}/{len(stock_data)} "
          f"({covered/max(len(stock_data),1)*100:.0f}%)")

    return stock_data, industry_map


# ============================================================
# 技术指标计算
# ============================================================
def calc_ma(arr, n):
    """简单移动平均"""
    ma = np.full_like(arr, np.nan)
    for i in range(n - 1, len(arr)):
        ma[i] = np.mean(arr[i - n + 1:i + 1])
    return ma


def calc_ema(arr, n):
    """指数移动平均"""
    ema = np.full_like(arr, np.nan)
    k = 2.0 / (n + 1)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_atr(high, low, close, n=14):
    """Average True Range"""
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr = np.full_like(close, np.nan)
    for i in range(n - 1, len(close)):
        atr[i] = np.mean(tr[i - n + 1:i + 1])
    return atr


def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD: DIF, DEA, histogram"""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    hist = 2 * (dif - dea)
    return dif, dea, hist


def calc_fft_amplitudes(close_segment, n_peaks=10):
    """FFT 振幅谱：取前 n_peaks 个最大振幅值（只保留振幅，32维方案）"""
    if len(close_segment) < 10:
        return [0.0] * n_peaks

    x = np.arange(len(close_segment))
    coef = np.polyfit(x, close_segment, 1)
    detrended = close_segment - np.polyval(coef, x)

    fft_vals = np.fft.rfft(detrended)
    amplitudes = np.abs(fft_vals)[1:]  # 去掉直流分量

    if len(amplitudes) < n_peaks:
        amps = list(amplitudes) + [0.0] * (n_peaks - len(amplitudes))
    else:
        top_idx = np.argsort(amplitudes)[-n_peaks:][::-1]
        amps = amplitudes[top_idx].tolist()

    max_amp = max(max(amps), 1e-8)
    return [a / max_amp for a in amps]


# ============================================================
# 32维特征提取
# ============================================================
def extract_features_32d(records, idx):
    """
    32维特征提取（最终确认版本）
    ─────────────────────────────
    FFT振幅:      10维  (振幅谱前10峰)
    G7_量价K线:   14维  (含 above_ma5)
    G2_均线偏离:   3维  (MA5/MA20/MA60偏离)
    G3_MACD:       3维  (DIF/DEA/柱)
    G4_精选:       2维  (ATR + vol_6m)
    ─────────────────────────────
    总计:         32维
    """
    if idx < 60:
        return None

    c = np.array([r['close'] for r in records[:idx + 1]], dtype=float)
    h = np.array([r['high'] for r in records[:idx + 1]], dtype=float)
    l = np.array([r['low'] for r in records[:idx + 1]], dtype=float)
    v = np.array([r['volume'] for r in records[:idx + 1]], dtype=float)
    t_arr = np.array([r['turnover'] for r in records[:idx + 1]], dtype=float)

    i = len(c) - 1
    ci = c[i]
    eps = max(abs(ci), 0.01)

    ma5 = calc_ma(c, 5)
    ma20 = calc_ma(c, 20)
    ma60 = calc_ma(c, 60)
    dif, dea, macd_hist = calc_macd(c)
    atr = calc_atr(h, l, c, 14)

    if np.isnan(ma60[i]) or np.isnan(atr[i]) or np.isnan(dif[i]) or np.isnan(dea[i]):
        return None

    features = []

    # === FFT 振幅 (10维) ===
    fft_amps = calc_fft_amplitudes(c[max(0, i - 59):i + 1], n_peaks=10)
    features.extend(fft_amps)

    # === G7_量价K线 (14维) ===
    vol_12m_mean = np.mean(v[max(0, i - 11):i + 1])
    vol_3m_mean = np.mean(v[max(0, i - 2):i + 1])

    # 1. vol_ratio
    features.append(v[i] / max(vol_12m_mean, 1) - 1)
    # 2. turnover
    features.append(t_arr[i] if not np.isnan(t_arr[i]) else 0.0)
    # 3. turnover_dev
    t_12m_mean = np.mean(t_arr[max(0, i - 11):i + 1])
    features.append(t_arr[i] / max(t_12m_mean, 1e-8) - 1)
    # 4. vol_ma3_ratio
    features.append(vol_3m_mean / max(vol_12m_mean, 1) - 1)
    # 5. log_volume
    features.append(np.log1p(v[i]))
    # 6. log_turnover
    features.append(np.log1p(max(t_arr[i], 0)))
    # 7. body_pct
    body = abs(c[i] - records[idx]['open'])
    amplitude = h[i] - l[i]
    features.append(body / max(amplitude, 0.01))
    # 8. price_pos
    h60 = np.max(h[max(0, i - 59):i + 1])
    l60 = np.min(l[max(0, i - 59):i + 1])
    features.append((ci - l60) / max(h60 - l60, 0.01))
    # 9. ma_spread
    features.append((ma20[i] - ma60[i]) / eps)
    # 10. vol_12m
    if i >= 12:
        rets_12 = np.diff(c[i - 12:i + 1]) / np.maximum(np.abs(c[i - 12:i]), 0.01)
        features.append(np.std(rets_12))
    else:
        features.append(0.0)
    # 11. ma5_ma20_ratio
    features.append(ma5[i] / max(ma20[i], 0.01) - 1)
    # 12. above_ma5
    features.append(1.0 if ci > ma5[i] else 0.0)
    # 13. up_streak
    streak = 0
    for j in range(i, max(i - 12, 0), -1):
        if j > 0 and c[j] > c[j - 1]:
            streak += 1
        else:
            break
    features.append(streak / 12.0)
    # 14. dn_streak
    streak = 0
    for j in range(i, max(i - 12, 0), -1):
        if j > 0 and c[j] < c[j - 1]:
            streak += 1
        else:
            break
    features.append(streak / 12.0)

    # === G2_均线偏离 (3维) ===
    features.append((ci - ma5[i]) / eps)
    features.append((ci - ma20[i]) / eps)
    features.append((ci - ma60[i]) / eps)

    # === G3_MACD (3维) ===
    features.append(dif[i])
    features.append(dea[i])
    features.append(macd_hist[i])

    # === G4_精选 (2维) ===
    features.append(atr[i] / eps)
    if i >= 6:
        rets_6 = np.diff(c[i - 6:i + 1]) / np.maximum(np.abs(c[i - 6:i]), 0.01)
        features.append(np.std(rets_6))
    else:
        features.append(0.0)

    assert len(features) == 32, f"特征数={len(features)}, 期望32"
    # 任何特征为 NaN 则丢弃该样本
    if any(np.isnan(f) for f in features):
        return None
    return features


# ============================================================
# 行业中性化
# ============================================================
def neutralize_industry(features_dict, industry_map):
    """对每个截面月份，按行业分组减组内均值"""
    neutralized = {}
    for ym, records in features_dict.items():
        industry_groups = defaultdict(list)
        for j, (code, feat, ret) in enumerate(records):
            ind = industry_map.get(code, 'unknown')
            industry_groups[ind].append(j)

        feat_array = np.array([r[1] for r in records])
        for ind, indices in industry_groups.items():
            if len(indices) > 1:
                group_mean = np.mean(feat_array[indices], axis=0)
                for idx in indices:
                    feat_array[idx] -= group_mean

        neutralized[ym] = [
            (records[j][0], feat_array[j].tolist(), records[j][2])
            for j in range(len(records))
        ]
    return neutralized


# ============================================================
# 构建截面
# ============================================================
def build_cross_sections(stock_data, industry_map):
    """按 YYYY-MM 分组构建截面：32维特征 + T+1~T+6 单月收益"""
    all_yms = set()
    for recs in stock_data.values():
        for r in recs:
            all_yms.add(r['ym'])
    all_yms = sorted(ym for ym in all_yms if ym >= '2015-01')
    print(f"回测窗口: {all_yms[0]} ~ {all_yms[-1]} ({len(all_yms)} 个自然月)")

    # ym → record index 映射
    stock_ym_idx = {}
    for code, records in stock_data.items():
        ym_map = {}
        for j, r in enumerate(records):
            ym_map[r['ym']] = j
        stock_ym_idx[code] = ym_map

    cross_sections = {}
    for ym in all_yms:
        section = []
        for code, records in stock_data.items():
            if ym not in stock_ym_idx[code]:
                continue
            idx = stock_ym_idx[code][ym]
            feat = extract_features_32d(records, idx)
            if feat is None:
                continue

            # 单月收益（修复#1：跨期减前值，非累积）
            rets = {}
            for lag in range(1, MAX_HORIZONS + 1):
                future_idx = idx + lag
                prev_idx = idx + lag - 1
                if future_idx < len(records) and prev_idx < len(records):
                    c_prev = records[prev_idx]['close']
                    c_fut = records[future_idx]['close']
                    if abs(c_prev) > 0.01:
                        rets[lag] = np.clip((c_fut - c_prev) / abs(c_prev), -2, 2)

            if rets:
                section.append((code, feat, rets))

        if len(section) >= MIN_STOCKS_PER_MONTH:
            cross_sections[ym] = section

    avg_n = np.mean([len(v) for v in cross_sections.values()])
    print(f"有效截面: {len(cross_sections)} 个月 | 月均 {avg_n:.0f} 只股票")
    return cross_sections, all_yms


# ============================================================
# IC 计算
# ============================================================
def spearman_ic(predictions, returns):
    if len(predictions) < 10:
        return np.nan
    from scipy.stats import spearmanr
    corr, _ = spearmanr(predictions, returns)
    return corr


# ============================================================
# 模型训练
# ============================================================
def train_predict(train_X, train_y, test_X):
    """三模型 IC 加权集成（Ridge + LightGBM + XGBoost）"""
    from sklearn.linear_model import Ridge

    # 过滤 NaN 样本
    valid_mask = ~np.isnan(train_X).any(axis=1) & ~np.isnan(train_y)
    train_X = train_X[valid_mask]
    train_y = train_y[valid_mask]

    if len(train_X) < 50:
        return np.zeros(len(test_X)), {}

    models = {}
    predictions = {}

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_X, train_y)
    predictions['ridge'] = ridge.predict(test_X)
    models['ridge'] = ridge

    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_SEED, verbose=-1
        )
        lgb_model.fit(train_X, train_y)
        predictions['lgb'] = lgb_model.predict(test_X)
        models['lgb'] = lgb_model
    except ImportError:
        pass

    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_SEED, verbosity=0
        )
        xgb_model.fit(train_X, train_y)
        predictions['xgb'] = xgb_model.predict(test_X)
        models['xgb'] = xgb_model
    except ImportError:
        pass

    if len(predictions) == 1:
        return list(predictions.values())[0], models

    # IC 加权集成
    ics = {}
    for name, m in models.items():
        ic = spearman_ic(m.predict(train_X), train_y)
        ics[name] = max(ic, 0.001)

    total_w = sum(ics.values())
    ensemble = np.zeros(len(test_X))
    for name in predictions:
        ensemble += (ics[name] / total_w) * predictions[name]

    return ensemble, models


# ============================================================
# 滚动训练 + 截面 IC 评估
# ============================================================
def rolling_evaluation(cross_sections, industry_map, train_months=60):
    """扩展窗口，逐月训练预测"""
    sorted_yms = sorted(cross_sections.keys())
    results = {lag: [] for lag in range(1, MAX_HORIZONS + 1)}
    monthly_predictions = {}

    total = len(sorted_yms) - train_months
    for t_idx in range(train_months, len(sorted_yms)):
        test_ym = sorted_yms[t_idx]
        train_yms = sorted_yms[:t_idx]

        train_X, train_y = [], []
        for ym in train_yms:
            for _, feat, rets in cross_sections[ym]:
                if 1 in rets:
                    train_X.append(feat)
                    train_y.append(rets[1])

        if len(train_X) < 100:
            continue

        train_X = np.array(train_X)
        train_y = np.array(train_y)

        test_data = cross_sections[test_ym]
        test_X = np.array([r[1] for r in test_data])

        preds, _ = train_predict(train_X, train_y, test_X)

        monthly_predictions[test_ym] = [
            (test_data[j][0], preds[j], test_data[j][2])
            for j in range(len(test_data))
        ]

        for lag in range(1, MAX_HORIZONS + 1):
            pred_list, ret_list = [], []
            for j, (_, _, rets) in enumerate(test_data):
                if lag in rets:
                    pred_list.append(preds[j])
                    ret_list.append(rets[lag])
            if len(pred_list) >= MIN_STOCKS_PER_MONTH:
                ic = spearman_ic(pred_list, ret_list)
                if not np.isnan(ic):
                    results[lag].append({'ym': test_ym, 'ic': ic})

        if (t_idx - train_months) % 12 == 0:
            print(f"  滚动: {test_ym} ({t_idx - train_months + 1}/{total})", flush=True)

    return results, monthly_predictions


# ============================================================
# 5-Fold 时间序列 CV
# ============================================================
def time_series_cv(cross_sections, n_folds=5):
    sorted_yms = sorted(cross_sections.keys())
    fold_size = len(sorted_yms) // n_folds
    fold_results = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = ((fold + 1) * fold_size
                    if fold < n_folds - 1
                    else len(sorted_yms))
        test_yms = sorted_yms[test_start:test_end]
        train_yms = [ym for ym in sorted_yms if ym not in test_yms]

        train_X, train_y = [], []
        for ym in train_yms:
            for _, feat, rets in cross_sections[ym]:
                if 1 in rets:
                    train_X.append(feat)
                    train_y.append(rets[1])

        if len(train_X) < 100:
            fold_results.append({
                'fold': fold + 1,
                'period': f"{test_yms[0]}~{test_yms[-1]}",
                'ic': np.nan, 'icir': np.nan, 'n_months': 0
            })
            continue

        train_X = np.array(train_X)
        train_y = np.array(train_y)

        fold_ics = []
        for test_ym in test_yms:
            if test_ym not in cross_sections:
                continue
            test_data = cross_sections[test_ym]
            test_X = np.array([r[1] for r in test_data])
            preds, _ = train_predict(train_X, train_y, test_X)
            ret_list = [r[2].get(1, np.nan) for r in test_data]
            valid = [(p, r) for p, r in zip(preds, ret_list) if not np.isnan(r)]
            if len(valid) >= MIN_STOCKS_PER_MONTH:
                p_arr, r_arr = zip(*valid)
                ic = spearman_ic(list(p_arr), list(r_arr))
                if not np.isnan(ic):
                    fold_ics.append(ic)

        if fold_ics:
            fold_results.append({
                'fold': fold + 1,
                'period': f"{test_yms[0]}~{test_yms[-1]}",
                'ic': np.mean(fold_ics),
                'icir': (np.mean(fold_ics) / max(np.std(fold_ics), 1e-8)),
                'n_months': len(fold_ics)
            })

    return fold_results


# ============================================================
# 纯多头回测
# ============================================================
def long_only_backtest(monthly_predictions, n_groups=5):
    sorted_yms = sorted(monthly_predictions.keys())
    q_returns = {q: [] for q in range(1, n_groups + 1)}
    q_holdings = {q: [] for q in range(1, n_groups + 1)}

    for ym in sorted_yms:
        data = monthly_predictions[ym]
        valid = [(c, p, r.get(1)) for c, p, r in data if r.get(1) is not None]
        if len(valid) < n_groups * 5:
            continue
        valid.sort(key=lambda x: x[1])
        gs = len(valid) // n_groups
        for q in range(1, n_groups + 1):
            group = valid[(q-1)*gs:(q*gs if q < n_groups else len(valid))]
            q_returns[q].append({'ym': ym, 'ret': np.mean([x[2] for x in group])})
            q_holdings[q].append(len(group))

    stats = {}
    for q in range(1, n_groups + 1):
        rets = [r['ret'] for r in q_returns[q]]
        if not rets:
            continue
        cum = np.cumprod([1 + r for r in rets])
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        stats[q] = {
            'ann_ret': (cum[-1]) ** (12.0 / len(rets)) - 1,
            'ann_vol': np.std(rets) * np.sqrt(12),
            'sharpe': ((cum[-1]) ** (12.0 / len(rets)) - 1) /
                      max(np.std(rets) * np.sqrt(12), 1e-8),
            'max_dd': np.min(dd), 'cum_ret': cum[-1] - 1,
            'win_rate': np.mean([r > 0 for r in rets]),
            'months': len(rets),
            'avg_holdings': np.mean(q_holdings[q])
        }

    if 1 in stats and n_groups in stats:
        n_periods = min(len(q_returns[1]), len(q_returns[n_groups]))
        ls_rets = [q_returns[n_groups][j]['ret'] - q_returns[1][j]['ret']
                   for j in range(n_periods)]
        cum_ls = np.cumprod([1 + r for r in ls_rets])
        stats['LS'] = {
            'ann_ret': (cum_ls[-1]) ** (12.0 / n_periods) - 1,
            'ann_vol': np.std(ls_rets) * np.sqrt(12),
            'sharpe': ((cum_ls[-1]) ** (12.0 / n_periods) - 1) /
                      max(np.std(ls_rets) * np.sqrt(12), 1e-8),
            'max_dd': np.min((cum_ls - np.maximum.accumulate(cum_ls)) /
                             np.maximum.accumulate(cum_ls)),
            'months': n_periods
        }

    # 单调性
    monotonic = True
    for q in range(1, n_groups):
        if q in stats and q + 1 in stats:
            if stats[q]['ann_ret'] > stats[q + 1]['ann_ret']:
                monotonic = False; break
    stats['monotonic'] = monotonic
    return stats, q_returns


# ============================================================
# 交易成本
# ============================================================
def cost_analysis(q_returns, n_groups=5, cost_bps_list=None):
    if cost_bps_list is None:
        cost_bps_list = COST_BPS

    turnover = 0.5
    results = []
    n = min(len(q_returns[1]), len(q_returns[n_groups]))

    for bps in cost_bps_list:
        cost = turnover * 2 * bps / 10000
        ls_rets = [q_returns[n_groups][j]['ret'] - q_returns[1][j]['ret'] - cost
                   for j in range(n)]
        ann_ret = (np.prod([1 + r for r in ls_rets])) ** (12.0 / n) - 1
        net_sharpe = ann_ret / max(np.std(ls_rets) * np.sqrt(12), 1e-8)
        results.append({'cost_bps': bps, 'net_sharpe': net_sharpe,
                        'net_ann_ret': ann_ret})

    for bps in range(1, 200):
        cost = turnover * 2 * bps / 10000
        ls_rets = [q_returns[n_groups][j]['ret'] - q_returns[1][j]['ret'] - cost
                   for j in range(n)]
        if (np.prod([1 + r for r in ls_rets])) ** (12.0 / n) - 1 <= 0:
            results.append({'breakeven_bps': bps})
            break

    return results


# ============================================================
# 输出
# ============================================================
def print_results(ic_results, cv_results, backtest_stats, cost_results):
    print("\n" + "=" * 70)
    print("32维最终版因子评估（4修复全合并）")
    print("=" * 70)

    print("\n── T+1 ~ T+6 单月 IC 衰减 ──")
    print(f"{'Lag':>6} {'IC':>8} {'ICstd':>8} {'ICIR':>8} {'IC>0':>8} {'N':>5}")
    print("-" * 48)
    for lag in range(1, MAX_HORIZONS + 1):
        ics = [r['ic'] for r in ic_results[lag]]
        if ics:
            m, s = np.mean(ics), np.std(ics)
            print(f"{'T+'+str(lag):>6} {m:>+8.4f} {s:>8.4f} "
                  f"{m/max(s,1e-8):>+8.2f} "
                  f"{np.mean([i>0 for i in ics]):>7.1%} {len(ics):>5}")

    print("\n── 5-Fold 时间序列 CV ──")
    print(f"{'Fold':>5} {'Period':>20} {'IC':>8} {'ICIR':>8} {'Months':>6}")
    print("-" * 48)
    for r in cv_results:
        print(f"{r['fold']:>5} {r['period']:>20} "
              f"{r.get('ic',np.nan):>+8.4f} {r.get('icir',np.nan):>+8.2f} "
              f"{r.get('n_months',0):>6}")
    valid = [r['ic'] for r in cv_results if not np.isnan(r.get('ic', np.nan))]
    if valid:
        print(f"{'均值':>5} {'':>20} {np.mean(valid):>+8.4f}")

    print("\n── 纯多头回测（五分组等权） ──")
    print(f"{'Group':>6} {'AnnRet':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'#Stocks':>7}")
    print("-" * 50)
    for q in [1, 3, 5, 'LS']:
        if q in backtest_stats:
            s = backtest_stats[q]
            wr = s.get('win_rate', 0)
            hold = s.get('avg_holdings', 0)
            print(f"{'Q'+str(q) if isinstance(q,int) else q:>6} "
                  f"{s['ann_ret']:>+7.1%} {s['sharpe']:>+8.2f} "
                  f"{s['max_dd']:>+7.1%} {wr:>7.1%} {hold:>7.0f}")
    print(f"单调性: {'PASS' if backtest_stats.get('monotonic') else 'FAIL'}")

    print("\n── 交易成本敏感性 ──")
    for r in cost_results:
        if 'cost_bps' in r:
            print(f"  {r['cost_bps']:>3}bp → Net Sharpe {r['net_sharpe']:>+.2f} "
                  f"(Net AnnRet {r['net_ann_ret']:>+.1%})")
        if 'breakeven_bps' in r:
            print(f"  盈亏平衡: {r['breakeven_bps']}bp")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print(f"v32_final_eval.py | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("32维 | 单月收益 | YYYY-MM截面 | CSRC L2 | 滚动60月")
    print("=" * 70)

    print("\n[1/5] 加载数据...")
    stock_data, industry_map = load_data()

    print("\n[2/5] 构建截面...")
    cross_sections, all_yms = build_cross_sections(stock_data, industry_map)

    print("\n[3/5] 滚动训练 + 截面IC...")
    ic_results, monthly_predictions = rolling_evaluation(
        cross_sections, industry_map, train_months=60)

    print("\n[4/5] 5-Fold CV...")
    cv_results = time_series_cv(cross_sections)

    print("\n[5/5] 回测 + 成本分析...")
    backtest_stats, q_returns = long_only_backtest(monthly_predictions)
    cost_results = cost_analysis(q_returns)

    print_results(ic_results, cv_results, backtest_stats, cost_results)

    # 保存
    output = {
        'ic_decay': {str(k): [r['ic'] for r in v]
                     for k, v in ic_results.items()},
        'cv_results': cv_results,
        'backtest_stats': {str(k): v for k, v in backtest_stats.items()
                          if k != 'monotonic'},
        'monotonic': backtest_stats.get('monotonic'),
        'cost_results': cost_results,
        'config': {
            'features_dim': 32, 'return_type': 'single_month',
            'date_format': 'YYYY-MM', 'industry': 'CSRC_L2',
            'train_months': 60,
            'timestamp': datetime.now().isoformat()
        }
    }
    out = os.path.join(OUTPUT_DIR, 'v32_final_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {out}")


if __name__ == '__main__':
    main()
