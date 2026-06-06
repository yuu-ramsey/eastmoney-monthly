"""
v32_freq_eval.py — 32d factor monthly/weekly/daily three-frequency evaluation (based on v32_final_eval.py logic)
============================================================
Unified fix: single-period returns + date grouping + 32-dim + CSRC L2 neutralization

Frequency parameterization:
  monthly: YYYY-MM, MA(5,20,60),  FFT60,  train 60 months
  weekly:  YYYY-Www,  MA(12,48,96), FFT104, train 156 weeks
  daily:   YYYY-MM-DD,MA(20,60,120),FFT252, train 504 days

Output (per frequency):
  - T+1~T+6 single-period IC/IC_std/ICIR/hit rate
  - 5-Fold time-series CV
  - 5-group backtest (Q1/Q3/Q5/LS annualized return/Sharpe/MaxDD/monotonicity)
  - Transaction costs (10/20/30/50bp Net Sharpe + breakeven point)
"""
import numpy as np
import sqlite3
import os
import json
import sys
from collections import defaultdict
from datetime import datetime, date

DB_PATH = ".eastmoney-ai/db/klines-v2.sqlite"
INDUSTRY_PATH = "data/industry-map.json"
OUTPUT_DIR = ".eastmoney-ai/final_eval/"
RANDOM_SEED = 42
COST_BPS = [10, 20, 30, 50]

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

# ============================================================
# Frequency configuration
# ============================================================
def _to_iso_week(date_str):
    """date string -> ISO week (YYYY-Www)"""
    s = str(date_str)[:10]
    d = datetime.strptime(s, '%Y-%m-%d')
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"

FREQ_CONFIG = {
    "monthly": {
        "table": "monthly_klines",
        "label": "Monthly",
        "group_fn": lambda d: str(d)[:7],                    # YYYY-MM
        "ma_windows": (5, 20, 60),
        "fft_window": 60,
        "atr_window": 14,
        "macd_params": (12, 26, 9),
        "vol_long": 12, "vol_short": 3, "vol_mid": 6,
        "min_bars": 60,
        "train_periods": 60,
        "min_stocks": 20,
        "max_horizons": 6,
        "n_folds": 5,
        "start_date": "2015-01",
        "ann_factor": 12,  # annualization factor
        "min_periods_bt": 12,
    },
    "weekly": {
        "table": "weekly_klines",
        "label": "Weekly",
        "group_fn": _to_iso_week,                              # YYYY-Www
        "ma_windows": (12, 48, 96),
        "fft_window": 104,
        "atr_window": 14,
        "macd_params": (12, 26, 9),
        "vol_long": 48, "vol_short": 12, "vol_mid": 24,
        "min_bars": 104,
        "train_periods": 156,
        "min_stocks": 30,
        "max_horizons": 6,
        "n_folds": 5,
        "start_date": "2015-01-01",
        "ann_factor": 52,  # annualization factor (weekly)
        "min_periods_bt": 52,
    },
    "daily": {
        "table": "daily_klines",
        "label": "Daily",
        "group_fn": lambda d: str(d)[:10],                     # YYYY-MM-DD
        "ma_windows": (20, 60, 120),
        "fft_window": 252,
        "atr_window": 14,
        "macd_params": (12, 26, 9),
        "vol_long": 240, "vol_short": 60, "vol_mid": 120,
        "min_bars": 252,
        "train_periods": 504,
        "min_stocks": 50,
        "max_horizons": 6,
        "n_folds": 5,
        "start_date": "2018-01-01",
        "ann_factor": 252,  # annualization factor (daily)
        "min_periods_bt": 252,
    },
}

# ============================================================
# Technical indicators
# ============================================================
def calc_ma(arr, n):
    ma = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        ma[i] = np.mean(arr[i - n + 1:i + 1])
    return ma

def calc_ema(arr, n):
    ema = np.full_like(arr, np.nan, dtype=float)
    ema[0] = arr[0]
    k = 2.0 / (n + 1)
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
    return ema

def calc_atr(high, low, close, n=14):
    tr = np.zeros(len(close), dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr = np.full_like(close, np.nan, dtype=float)
    for i in range(n - 1, len(close)):
        atr[i] = np.mean(tr[i - n + 1:i + 1])
    return atr

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    hist = 2 * (dif - dea)
    return dif, dea, hist

def calc_fft_amplitudes(close_segment, n_peaks=10):
    if len(close_segment) < 10:
        return [0.0] * n_peaks
    x = np.arange(len(close_segment), dtype=float)
    coef = np.polyfit(x, close_segment, 1)
    detrended = close_segment - np.polyval(coef, x)
    fft_vals = np.fft.rfft(detrended)
    amplitudes = np.abs(fft_vals)[1:]
    if len(amplitudes) < n_peaks:
        amps = list(amplitudes) + [0.0] * (n_peaks - len(amplitudes))
    else:
        top_idx = np.argsort(amplitudes)[-n_peaks:][::-1]
        amps = amplitudes[top_idx].tolist()
    max_amp = max(max(amps), 1e-8)
    return [float(a) / max_amp for a in amps]

# ============================================================
# 32-dim feature extraction (frequency parameterized)
# ============================================================
def extract_features_32d(records, idx, cfg):
    if idx < cfg["min_bars"]:
        return None

    c = np.array([r['close'] for r in records[:idx + 1]], dtype=float)
    h = np.array([r['high'] for r in records[:idx + 1]], dtype=float)
    l = np.array([r['low'] for r in records[:idx + 1]], dtype=float)
    v = np.array([r['volume'] for r in records[:idx + 1]], dtype=float)
    t_arr = np.array([r['turnover'] for r in records[:idx + 1]], dtype=float)

    i = len(c) - 1
    ci = c[i]
    eps = max(abs(ci), 0.01)

    maw = cfg["ma_windows"]
    ma1 = calc_ma(c, maw[0])
    ma2 = calc_ma(c, maw[1])
    ma3 = calc_ma(c, maw[2])
    mp = cfg["macd_params"]
    dif, dea, macd_hist = calc_macd(c, mp[0], mp[1], mp[2])
    atr = calc_atr(h, l, c, cfg["atr_window"])

    if np.isnan(ma3[i]) or np.isnan(atr[i]) or np.isnan(dif[i]) or np.isnan(dea[i]):
        return None

    v_long = cfg["vol_long"]
    v_short = cfg["vol_short"]
    v_mid = cfg["vol_mid"]
    fft_w = cfg["fft_window"]

    features = []

    # FFT amplitude (10-dim)
    fft_amps = calc_fft_amplitudes(c[max(0, i - fft_w + 1):i + 1], n_peaks=10)
    features.extend(fft_amps)

    # G7 volume-price (14-dim)
    vol_long_mean = np.mean(v[max(0, i - v_long + 1):i + 1])
    vol_short_mean = np.mean(v[max(0, i - v_short + 1):i + 1])

    features.append(float(v[i] / max(vol_long_mean, 1) - 1))                    # 1. vol_ratio
    features.append(float(t_arr[i] if not np.isnan(t_arr[i]) else 0.0))         # 2. turnover
    t_long_mean = np.mean(t_arr[max(0, i - v_long + 1):i + 1])
    features.append(float(t_arr[i] / max(t_long_mean, 1e-8) - 1))               # 3. turnover_dev
    features.append(float(vol_short_mean / max(vol_long_mean, 1) - 1))          # 4. vol_ma_ratio
    features.append(float(np.log1p(v[i])))                                      # 5. log_volume
    features.append(float(np.log1p(max(t_arr[i], 0))))                          # 6. log_turnover
    body = abs(c[i] - records[idx]['open'])
    amplitude = h[i] - l[i]
    features.append(float(body / max(amplitude, 0.01)))                         # 7. body_pct
    h_w = np.max(h[max(0, i - fft_w + 1):i + 1])
    l_w = np.min(l[max(0, i - fft_w + 1):i + 1])
    features.append(float((ci - l_w) / max(h_w - l_w, 0.01)))                   # 8. price_pos
    features.append(float((ma2[i] - ma3[i]) / eps))                              # 9. ma_spread
    if i >= v_long:
        rets_n = np.diff(c[i - v_long:i + 1]) / np.maximum(np.abs(c[i - v_long:i]), 0.01)
        features.append(float(np.std(rets_n)))                                   # 10. vol_Nm
    else:
        features.append(0.0)
    features.append(float(ma1[i] / max(ma2[i], 0.01) - 1))                      # 11. ma_ratio
    features.append(float(1.0 if ci > ma1[i] else 0.0))                         # 12. above_ma1
    # up_streak
    streak = 0
    for j in range(i, max(i - v_long, 0), -1):
        if j > 0 and c[j] > c[j - 1]: streak += 1
        else: break
    features.append(float(streak / float(v_long)))                               # 13. up_streak
    # dn_streak
    streak = 0
    for j in range(i, max(i - v_long, 0), -1):
        if j > 0 and c[j] < c[j - 1]: streak += 1
        else: break
    features.append(float(streak / float(v_long)))                               # 14. dn_streak

    # G2 MA deviation (3-dim)
    features.append(float((ci - ma1[i]) / eps))   # ma1_dev
    features.append(float((ci - ma2[i]) / eps))   # ma2_dev
    features.append(float((ci - ma3[i]) / eps))   # ma3_dev

    # G3 MACD (3-dim)
    features.append(float(dif[i]))
    features.append(float(dea[i]))
    features.append(float(macd_hist[i]))

    # G4 selected (2-dim)
    features.append(float(atr[i] / eps))                                          # ATR
    if i >= v_mid:
        rets_mid = np.diff(c[i - v_mid:i + 1]) / np.maximum(np.abs(c[i - v_mid:i]), 0.01)
        features.append(float(np.std(rets_mid)))                                  # vol_mid
    else:
        features.append(0.0)

    assert len(features) == 32, f"feature count={len(features)}, expected 32"
    if any(np.isnan(f) for f in features):
        return None
    return features


# ============================================================
# Data Loading
# ============================================================
def load_data(period):
    cfg = FREQ_CONFIG[period]
    table = cfg["table"]
    group_fn = cfg["group_fn"]

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT code, date, open, high, low, close, volume, turnover_rate FROM {table} ORDER BY code, date")
    rows = cur.fetchall()
    conn.close()

    stock_data = defaultdict(list)
    for code, date_str, o, h, l, c, v, t in rows:
        if c is None or o is None or h is None or l is None:
            continue
        stock_data[code].append({
            'date': str(date_str), 'open': o, 'high': h, 'low': l,
            'close': c, 'volume': v, 'turnover': t or 0.0
        })

    # Deduplicate: group by frequency key, keep last record per group
    for code in stock_data:
        seen = {}
        for rec in stock_data[code]:
            key = group_fn(rec['date'])
            seen[key] = rec
        stock_data[code] = sorted(seen.values(), key=lambda r: r['date'])

    # Load CSRC L2 industry mapping
    industry_map = {}
    if os.path.exists(INDUSTRY_PATH):
        with open(INDUSTRY_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            industry_map = raw.get('stockToIndustry', {})

    covered = sum(1 for c in stock_data if c in industry_map)
    print(f"  Loaded {len(stock_data)} stocks, {sum(len(v) for v in stock_data.values()):,} records "
          f"| industry coverage {covered}/{len(stock_data)} ({covered/max(len(stock_data),1)*100:.0f}%)")

    return stock_data, industry_map


# ============================================================
# Build cross-sections
# ============================================================
def build_cross_sections(stock_data, period):
    """
    Build cross-sections by native frequency grouping:
    - monthly: last bar of each month -> next month's return
    - weekly: last bar of each week -> next week's return
    - daily: each bar -> next day's return
    """
    cfg = FREQ_CONFIG[period]
    group_fn = cfg["group_fn"]
    max_h = cfg["max_horizons"]

    # Build group_key -> bar_index mapping for each stock
    stock_groups = {}
    for code, records in stock_data.items():
        groups = {}
        for j, r in enumerate(records):
            key = group_fn(r['date'])
            groups[key] = j  # last entry per group (covering within group)
        stock_groups[code] = groups

    # Collect all valid period keys
    all_keys = set()
    for groups in stock_groups.values():
        all_keys.update(groups.keys())
    all_keys = sorted(k for k in all_keys if k >= cfg["start_date"])
    period_index = {k: i for i, k in enumerate(all_keys)}  # key -> global index
    print(f"  Backtest window: {all_keys[0]} ~ {all_keys[-1]} ({len(all_keys)} cross-sections)")

    cross_sections = {}
    for period_key in all_keys:
        section = []
        pi = period_index[period_key]  # current period's position in global calendar
        for code, records in stock_data.items():
            groups = stock_groups[code]
            if period_key not in groups:
                continue
            idx = groups[period_key]
            feat = extract_features_32d(records, idx, cfg)
            if feat is None:
                continue

            # Compute single-period returns using global period calendar (not accumulated)
            # T+N: take close at all_keys[pi+N] vs close at all_keys[pi+N-1]
            # Key: both stocks must have data to calculate (skip if suspended)
            rets = {}
            for lag in range(1, max_h + 1):
                fwd_pi = pi + lag
                prev_pi = pi + lag - 1
                if fwd_pi >= len(all_keys):
                    break
                fwd_key = all_keys[fwd_pi]
                prev_key = all_keys[prev_pi]
                if prev_key in groups and fwd_key in groups:
                    c_prev = records[groups[prev_key]]['close']
                    c_fut = records[groups[fwd_key]]['close']
                    if abs(c_prev) > 0.01:
                        rets[lag] = float(np.clip((c_fut - c_prev) / abs(c_prev), -2, 2))

            if rets:
                section.append((code, feat, rets))

        if len(section) >= cfg["min_stocks"]:
            cross_sections[period_key] = section

    avg_n = np.mean([len(v) for v in cross_sections.values()])
    print(f"  Valid cross-sections: {len(cross_sections)} | avg {avg_n:.0f} stocks per section")
    return cross_sections


# ============================================================
# Industry neutralization
# ============================================================
def neutralize_industry(cross_sections, industry_map):
    neutralized = {}
    for key, section in cross_sections.items():
        industry_groups = defaultdict(list)
        for j, (code, feat, rets) in enumerate(section):
            ind = industry_map.get(code, 'unknown')
            industry_groups[ind].append(j)

        feat_array = np.array([r[1] for r in section], dtype=float)
        for ind, indices in industry_groups.items():
            if len(indices) > 1:
                group_mean = np.mean(feat_array[indices], axis=0)
                for idx in indices:
                    feat_array[idx] -= group_mean

        neutralized[key] = [
            (section[j][0], feat_array[j].tolist(), section[j][2])
            for j in range(len(section))
        ]
    return neutralized


# ============================================================
# IC calculation
# ============================================================
def spearman_ic(predictions, returns):
    if len(predictions) < 10:
        return np.nan
    from scipy.stats import spearmanr
    corr, _ = spearmanr(predictions, returns)
    return float(corr)


# ============================================================
# Model training (IC-weighted ensemble, same as v32_final_eval.py)
# ============================================================
def train_predict(train_X, train_y, test_X):
    from sklearn.linear_model import Ridge

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
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_SEED, verbosity=0
        )
        xgb_model.fit(train_X, train_y)
        predictions['xgb'] = xgb_model.predict(test_X)
        models['xgb'] = xgb_model
    except ImportError:
        pass

    if len(predictions) == 1:
        return list(predictions.values())[0], models

    # IC-weighted ensemble
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
# Rolling training + cross-sectional IC evaluation
# ============================================================
def rolling_evaluation(cross_sections, period):
    cfg = FREQ_CONFIG[period]
    train_periods = cfg["train_periods"]
    max_h = cfg["max_horizons"]

    sorted_keys = sorted(cross_sections.keys())
    results = {lag: [] for lag in range(1, max_h + 1)}
    period_predictions = {}

    total = max(len(sorted_keys) - train_periods, 1)
    for t_idx in range(train_periods, len(sorted_keys)):
        test_key = sorted_keys[t_idx]
        train_keys = sorted_keys[:t_idx]

        train_X, train_y = [], []
        for k in train_keys:
            for _, feat, rets in cross_sections[k]:
                if 1 in rets:
                    train_X.append(feat)
                    train_y.append(rets[1])

        if len(train_X) < 100:
            continue

        train_X = np.array(train_X, dtype=float)
        train_y = np.array(train_y, dtype=float)
        test_data = cross_sections[test_key]
        test_X = np.array([r[1] for r in test_data], dtype=float)

        preds, _ = train_predict(train_X, train_y, test_X)

        period_predictions[test_key] = [
            (test_data[j][0], float(preds[j]), test_data[j][2])
            for j in range(len(test_data))
        ]

        for lag in range(1, max_h + 1):
            pred_list, ret_list = [], []
            for j, (_, _, rets) in enumerate(test_data):
                if lag in rets:
                    pred_list.append(float(preds[j]))
                    ret_list.append(rets[lag])
            if len(pred_list) >= cfg["min_stocks"]:
                ic = spearman_ic(pred_list, ret_list)
                if not np.isnan(ic):
                    results[lag].append({'key': test_key, 'ic': float(ic)})

        progress = t_idx - train_periods + 1
        if progress % max(1, total // 20) == 0 or progress == total:
            print(f"  Rolling: {test_key} ({progress}/{total})", flush=True)

    return results, period_predictions


# ============================================================
# 5-Fold time-series CV
# ============================================================
def time_series_cv(cross_sections, period):
    cfg = FREQ_CONFIG[period]
    sorted_keys = sorted(cross_sections.keys())
    n_folds = cfg["n_folds"]
    fold_size = len(sorted_keys) // n_folds
    fold_results = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(sorted_keys)
        test_keys = sorted_keys[test_start:test_end]
        train_keys = [k for k in sorted_keys if k not in test_keys]

        train_X, train_y = [], []
        for k in train_keys:
            for _, feat, rets in cross_sections[k]:
                if 1 in rets:
                    train_X.append(feat)
                    train_y.append(rets[1])

        if len(train_X) < 100:
            fold_results.append({
                'fold': fold + 1, 'period': f"{test_keys[0]}~{test_keys[-1]}",
                'ic': float('nan'), 'icir': float('nan'), 'n_periods': 0
            })
            continue

        train_X = np.array(train_X, dtype=float)
        train_y = np.array(train_y, dtype=float)

        # Train once per fold (Ridge sufficient for robustness check, avoid retraining LGB/XGB each period)
        from sklearn.linear_model import Ridge as CVRidge
        valid_mask = ~np.isnan(train_X).any(axis=1) & ~np.isnan(train_y)
        cv_model = CVRidge(alpha=1.0)
        cv_model.fit(train_X[valid_mask], train_y[valid_mask])

        fold_ics = []
        for test_key in test_keys:
            if test_key not in cross_sections:
                continue
            test_data = cross_sections[test_key]
            test_X = np.array([r[1] for r in test_data], dtype=float)
            preds = cv_model.predict(test_X)
            ret_list = [r[2].get(1, float('nan')) for r in test_data]
            valid = [(p, r) for p, r in zip(preds, ret_list) if not np.isnan(r)]
            if len(valid) >= cfg["min_stocks"]:
                p_arr, r_arr = zip(*valid)
                ic = spearman_ic(list(p_arr), list(r_arr))
                if not np.isnan(ic):
                    fold_ics.append(float(ic))

        if fold_ics:
            mean_ic = np.mean(fold_ics)
            ic_std = np.std(fold_ics)
            fold_results.append({
                'fold': fold + 1,
                'period': f"{test_keys[0]}~{test_keys[-1]}",
                'ic': float(mean_ic),
                'icir': float(mean_ic / ic_std) if ic_std > 0 else 0.0,
                'n_periods': len(fold_ics)
            })
        else:
            fold_results.append({
                'fold': fold + 1, 'period': f"{test_keys[0]}~{test_keys[-1]}",
                'ic': float('nan'), 'icir': float('nan'), 'n_periods': 0
            })

    return fold_results


# ============================================================
# Long-only backtest (5-group, equal-weight)
# ============================================================
def long_only_backtest(period_predictions, period):
    cfg = FREQ_CONFIG[period]
    ann_factor = cfg["ann_factor"]
    sorted_keys = sorted(period_predictions.keys())

    if len(sorted_keys) < cfg["min_periods_bt"]:
        return None

    monthly_returns = []
    for key in sorted_keys:
        entries = period_predictions[key]
        preds = np.array([e[1] for e in entries])
        rets_t1 = np.array([e[2].get(1, float('nan')) for e in entries])
        valid = ~np.isnan(rets_t1)
        if valid.sum() < cfg["min_stocks"]:
            continue
        preds = preds[valid]
        rets_t1 = rets_t1[valid]
        quintiles = np.percentile(preds, [20, 40, 60, 80])
        q5_mask = preds >= quintiles[3]
        q1_mask = preds <= quintiles[0]
        q3_mask = (preds >= quintiles[1]) & (preds <= quintiles[2])
        monthly_returns.append({
            'key': key,
            'q1_ret': float(np.mean(rets_t1[q1_mask])) if q1_mask.any() else 0.0,
            'q3_ret': float(np.mean(rets_t1[q3_mask])) if q3_mask.any() else 0.0,
            'q5_ret': float(np.mean(rets_t1[q5_mask])) if q5_mask.any() else 0.0,
            'q1_n': int(q1_mask.sum()),
            'q3_n': int(q3_mask.sum()),
            'q5_n': int(q5_mask.sum()),
        })

    if len(monthly_returns) < cfg["min_periods_bt"]:
        return None

    def calc_stats(ret_list, ann_f):
        rets = np.array(ret_list)
        ann_ret = float(np.mean(rets) * ann_f)
        ann_vol = float(np.std(rets) * np.sqrt(ann_f))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = float(np.min(dd))
        win_rate = float(np.mean(rets > 0))
        return {'ann_ret': ann_ret, 'sharpe': sharpe, 'max_dd': max_dd,
                'win_rate': win_rate, 'n_periods': len(rets)}

    q1_rets = [m['q1_ret'] for m in monthly_returns]
    q3_rets = [m['q3_ret'] for m in monthly_returns]
    q5_rets = [m['q5_ret'] for m in monthly_returns]
    ls_rets = [q5 - q1 for q5, q1 in zip(q5_rets, q1_rets)]

    q1_stats = calc_stats(q1_rets, ann_factor)
    q3_stats = calc_stats(q3_rets, ann_factor)
    q5_stats = calc_stats(q5_rets, ann_factor)
    ls_stats = calc_stats(ls_rets, ann_factor)

    means = [q1_stats['ann_ret'], q3_stats['ann_ret'], q5_stats['ann_ret']]
    monotonic = means[0] <= means[1] <= means[2]

    avg_q1_n = int(np.mean([m['q1_n'] for m in monthly_returns]))
    avg_q3_n = int(np.mean([m['q3_n'] for m in monthly_returns]))
    avg_q5_n = int(np.mean([m['q5_n'] for m in monthly_returns]))

    return {
        'q1': q1_stats, 'q3': q3_stats, 'q5': q5_stats, 'ls': ls_stats,
        'monotonic': monotonic,
        'avg_n': {'q1': avg_q1_n, 'q3': avg_q3_n, 'q5': avg_q5_n},
    }


# ============================================================
# Transaction cost analysis (with breakeven)
# ============================================================
def cost_analysis(period_predictions, period):
    cfg = FREQ_CONFIG[period]
    ann_factor = cfg["ann_factor"]
    sorted_keys = sorted(period_predictions.keys())

    all_rets = []
    for key in sorted_keys:
        entries = period_predictions[key]
        preds = np.array([e[1] for e in entries])
        rets_t1 = np.array([e[2].get(1, float('nan')) for e in entries])
        valid = ~np.isnan(rets_t1)
        if valid.sum() < cfg["min_stocks"]:
            continue
        preds = preds[valid]
        rets_t1 = rets_t1[valid]
        q80 = np.percentile(preds, 80)
        q20 = np.percentile(preds, 20)
        long_r = np.mean(rets_t1[preds >= q80]) if (preds >= q80).any() else 0.0
        short_r = np.mean(rets_t1[preds <= q20]) if (preds <= q20).any() else 0.0
        all_rets.append(float(long_r - short_r))

    if len(all_rets) < cfg["min_periods_bt"]:
        return []

    rets_arr = np.array(all_rets)

    results = []
    turnover = 0.5  # one-way turnover rate 50%
    for bps in COST_BPS:
        cost = turnover * 2 * bps / 10000  # round-trip
        net_rets = rets_arr - cost
        ann_ret = float(np.mean(net_rets) * ann_factor)
        ann_vol = float(np.std(rets_arr) * np.sqrt(ann_factor))
        net_sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
        results.append({'bps': bps, 'net_sharpe': net_sharpe, 'net_ann_ret': ann_ret})

    # Breakeven point search
    for bps in range(1, 500):
        cost = turnover * 2 * bps / 10000
        net_rets = rets_arr - cost
        ann_ret = float(np.mean(net_rets) * ann_factor)
        if ann_ret <= 0:
            results.append({'breakeven_bps': bps})
            break

    return results


# ============================================================
# Output display
# ============================================================
def print_period_results(period, ic_results, cv_results, bt, costs):
    cfg = FREQ_CONFIG[period]
    max_h = cfg["max_horizons"]
    label = cfg["label"]

    # IC decay
    print(f"\n{'─'*60}")
    print(f"  {label} T+1 ~ T+{max_h} Single-Period IC Decay")
    print(f"{'─'*60}")
    print(f"  {'Lag':>6} {'IC':>8} {'ICstd':>8} {'ICIR':>8} {'IC>0':>8} {'N':>6}")
    print(f"  {'─'*50}")
    for lag in range(1, max_h + 1):
        if ic_results[lag]:
            ics = [r['ic'] for r in ic_results[lag]]
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            ic_gt0 = np.mean([1 if x > 0 else 0 for x in ics])
            print(f"  T+{lag}  {mean_ic:+.4f}  {std_ic:.4f}  {icir:+.2f}  {ic_gt0:.1%}  {len(ics):>4}")

    # 5-Fold CV
    if cv_results:
        print(f"\n{'─'*60}")
        print(f"  5-Fold CV")
        print(f"{'─'*60}")
        cv_ics = [f['ic'] for f in cv_results if not np.isnan(f['ic'])]
        for f in cv_results:
            print(f"  Fold {f['fold']}  {f['period']:>20}  IC={f['ic']:+.4f}  ICIR={f['icir']:+.2f}  n={f['n_periods']}")
        if cv_ics:
            print(f"  Mean IC={np.mean(cv_ics):+.4f}")

    # Backtest
    if bt:
        print(f"\n{'─'*60}")
        print(f"  Long-Only Backtest (5-group, equal-weight)")
        print(f"{'─'*60}")
        print(f"  {'Group':>6} {'AnnRet':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'#Stocks':>8}")
        print(f"  {'─'*60}")
        for g, s in [('Q1', bt['q1']), ('Q3', bt['q3']), ('Q5', bt['q5']), ('LS', bt['ls'])]:
            print(f"  {g:>6} {s['ann_ret']:>+7.1%} {s['sharpe']:>8.2f} {s['max_dd']:>+7.1%} {s['win_rate']:>7.1%} {bt['avg_n'].get(g.lower(), 0):>8}")
        print(f"  Monotonicity: {'PASS' if bt['monotonic'] else 'FAIL'}")

    # Costs
    if costs:
        print(f"\n{'─'*60}")
        print(f"  Transaction Cost Sensitivity")
        print(f"{'─'*60}")
        for c in costs:
            if 'bps' in c:
                print(f"  {c['bps']}bp → Net Sharpe {c['net_sharpe']:+.2f} (Net AnnRet {c['net_ann_ret']:+.1%})")
            if 'breakeven_bps' in c:
                print(f"  Breakeven: {c['breakeven_bps']}bp")


def print_summary_table(all_results):
    """Three-frequency comparison summary table"""
    print(f"\n{'='*90}")
    print(f"  Three-Frequency Comparison Summary")
    print(f"{'='*90}")
    print(f"  {'Freq':>6} {'Stocks':>6} {'Samples':>8} {'Periods':>6} {'IC':>8} {'ICIR':>8} {'Q5 Sharpe':>10} {'LS Sharpe':>10}")
    print(f"  {'─'*80}")
    for period in ["monthly", "weekly", "daily"]:
        if period not in all_results:
            continue
        res = all_results[period]
        cfg = FREQ_CONFIG[period]
        t1 = res['ic_decay'].get('T+1', {})
        summary = res.get('_summary', {})
        print(f"  {cfg['label']:>6} {summary.get('stocks',0):>6} {summary.get('samples',0):>8} "
              f"{summary.get('periods',0):>6} {t1.get('ic',0):>+.4f} {t1.get('icir',0):>+.2f} "
              f"{summary.get('q5_sharpe',0):>10.2f} {summary.get('ls_sharpe',0):>10.2f}")
    print(f"{'='*90}")


# ============================================================
# Single frequency run
# ============================================================
def run_period(period):
    cfg = FREQ_CONFIG[period]
    label = cfg["label"]
    max_h = cfg["max_horizons"]

    print(f"\n{'='*70}")
    print(f"  {label} ({period}) -- v32 Full-Fix Evaluation")
    print(f"{'='*70}")

    print(f"\n[1/5] Loading data...")
    stock_data, industry_map = load_data(period)

    print(f"\n[2/5] Building cross-sections...")
    cross_sections = build_cross_sections(stock_data, period)
    cross_sections = neutralize_industry(cross_sections, industry_map)

    print(f"\n[3/5] Rolling training + cross-sectional IC...")
    ic_results, period_predictions = rolling_evaluation(cross_sections, period)

    print(f"\n[4/5] 5-Fold CV...")
    cv_results = time_series_cv(cross_sections, period)

    print(f"\n[5/5] Backtest + cost analysis...")
    bt = long_only_backtest(period_predictions, period)
    costs = cost_analysis(period_predictions, period)

    print_period_results(period, ic_results, cv_results, bt, costs)

    # Summary data
    t1_ics = [r['ic'] for r in ic_results[1]] if ic_results[1] else []
    total_samples = sum(
        len([x for x in section if 1 in x[2]])
        for section in cross_sections.values()
    )

    summary = {
        'stocks': len(stock_data),
        'samples': total_samples,
        'periods': len(cross_sections),
        'q5_sharpe': bt['q5']['sharpe'] if bt else 0,
        'ls_sharpe': bt['ls']['sharpe'] if bt else 0,
    }

    # Save
    output = {
        'period': period, 'label': label,
        'ic_decay': {f"T+{lag}": {
            'ic': float(np.mean([r['ic'] for r in ic_results[lag]])) if ic_results[lag] else 0.0,
            'ic_std': float(np.std([r['ic'] for r in ic_results[lag]])) if ic_results[lag] else 0.0,
            'icir': float(np.mean([r['ic'] for r in ic_results[lag]]) / max(np.std([r['ic'] for r in ic_results[lag]]), 1e-8)) if ic_results[lag] else 0.0,
            'ic_gt0': float(np.mean([1 if r['ic'] > 0 else 0 for r in ic_results[lag]])) if ic_results[lag] else 0.0,
            'n': len(ic_results[lag])}
            for lag in range(1, max_h + 1)},
        'cv': [{'fold': f['fold'], 'ic': f['ic'], 'icir': f['icir'], 'n': f['n_periods']} for f in cv_results],
        'backtest': {g: {'ann_ret': bt[g]['ann_ret'], 'sharpe': bt[g]['sharpe'], 'max_dd': bt[g]['max_dd']}
                      for g in ['q1', 'q3', 'q5', 'ls']} if bt else {},
        'monotonic': bt['monotonic'] if bt else None,
        'cost_analysis': costs,
        '_summary': summary,
    }
    out_path = os.path.join(OUTPUT_DIR, f"v32_{period}_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")

    return output


def main():
    periods = [p for p in sys.argv[1:] if p in FREQ_CONFIG] if len(sys.argv) > 1 else ["monthly", "weekly", "daily"]
    print(f"v32 Three-Frequency Full Evaluation | periods: {periods}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}
    for period in periods:
        try:
            all_results[period] = run_period(period)
        except Exception as e:
            print(f"\n  {period} failed: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) >= 2:
        print_summary_table(all_results)

    print(f"\nAll done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
