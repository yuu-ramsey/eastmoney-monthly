"""
Phase 5: FFT reconstruction — 30d→10d (keep only amplitude spectrum)
对比: 61维 baseline vs 41维 (FFT振幅10维)
Output: T+1~T+6 IC decay + 特征重要性
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb, xgboost as xgb, pywt

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'diagnosis'
OUT.mkdir(parents=True, exist_ok=True)
N_FFT = 10
DATE_FMT = '%Y-%m-%d %H:%M:%S'

# 41维特征名 (FFT 30d → 10d amplitude only)
FEATURE_NAMES_41 = [
    'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m',        # G1: 0-3
    'ma5_dev', 'ma20_dev', 'ma60_dev',                # G2: 4-6
    'dif', 'dea', 'macd_hist',                         # G3: 7-9
    'rsi14', 'bb_pos', 'vol_6m', 'atr14', 'amplitude', # G4: 10-14
    'above_ma20', 'above_ma60',                        # G5: 15-16
] + [f'fft_amp_{i+1}' for i in range(10)] + [         # FFT重构: 17-26 (10d)
    'vol_ratio', 'turnover', 'turnover_dev',            # G7: 27-32
    'vol_ma3_ratio', 'log_volume', 'log_turnover',
    'body_pct', 'price_pos', 'ma_spread',               # G7: 33-40
    'vol_12m', 'ma5_ma20_ratio', 'above_ma5',
    'up_streak', 'dn_streak',
]
assert len(FEATURE_NAMES_41) == 41, f"Expected 41, got {len(FEATURE_NAMES_41)}"


def ts():
    return time.strftime(DATE_FMT)


def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates == m], true[dates == m])[0]
           for m in np.unique(dates) if (dates == m).sum() >= 20]
    ics = np.array(ics)
    return np.mean(ics), np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0, ics


def cs_hit(pred, true, dates):
    hits = [np.mean((pred[dates == m] * true[dates == m]) > 0)
            for m in np.unique(dates) if (dates == m).sum() >= 20]
    return np.mean(hits) if hits else 0


def fft_amplitudes(p):
    """Returns前N_FFT个频峰的振幅(10维), 不做频率/相位"""
    x = np.arange(len(p))
    t = np.polyfit(x, p, 1)
    d = p - np.polyval(t, x)
    fp = np.fft.rfft(d)
    a = np.abs(fp)
    if len(a) <= 1:
        return np.zeros(N_FFT, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1][:N_FFT] + 1
    amps = [a[i] for i in pk]
    while len(amps) < N_FFT:
        amps.append(0)
    return np.array(amps[:N_FFT], dtype=np.float32)


def fft_full(p):
    """原始30维FFT (freq+amp+phase ×10), 与phase4_full.py一致"""
    x = np.arange(len(p))
    t = np.polyfit(x, p, 1)
    d = p - np.polyval(t, x)
    fp = np.fft.rfft(d)
    a = np.abs(fp)
    fq = np.fft.rfftfreq(len(d))
    if len(a) <= 1:
        return np.zeros(N_FFT * 3, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1][:N_FFT] + 1
    fs = []
    for i in pk:
        if i < len(fq):
            fs.extend([fq[i], a[i], np.angle(fp[i])])
    while len(fs) < N_FFT * 3:
        fs.extend([0, 0, 0])
    return np.array(fs[:N_FFT * 3], dtype=np.float32)


def wd(s):
    c = pywt.wavedec(s, 'db4', level=2)
    sigma = np.median(np.abs(c[-1])) / 0.6745
    th = sigma * np.sqrt(2 * np.log(len(s)))
    cd = [c[0]] + [pywt.threshold(cf, th, mode='soft') for cf in c[1:]]
    return pywt.waverec(cd, 'db4')[:len(s)]


def cross_sectional_neutralize(features, dates, neutralizer, ntype='categorical'):
    neutralized = features.copy()
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 50: continue
        if ntype == 'categorical':
            groups = neutralizer[mask]
            for g in np.unique(groups):
                gm = mask & (neutralizer == g)
                if gm.sum() >= 3: neutralized[gm] -= features[gm].mean(axis=0)
    return neutralized


def build_features():
    print(f"[{ts()}] 加载数据...", flush=True)
    conn = sqlite3.connect(str(DB))
    codes = [r[0] for r in conn.execute(
        'SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
    ind_map = {r[0]: r[1] for r in conn.execute(
        'SELECT stock_code, industry_code FROM stock_industry_mapping')}
    codes_with_ind = [c for c in codes if c in ind_map]
    params = ','.join('?' * len(codes_with_ind))
    df = pd.read_sql_query(
        f"SELECT code,date,open,high,low,close,volume,turnover_rate "
        f"FROM monthly_klines WHERE code IN ({params}) "
        f"AND date>='2005-01' ORDER BY code,date",
        conn, params=codes_with_ind)
    conn.close()
    codes_used = sorted(codes_with_ind)
    print(f"[{ts()}] {len(codes_used)} stocks, {len(df)} rows", flush=True)

    df['month'] = df['date'].str[:7]
    print(f"[{ts()}] 构建特征...", flush=True); t0 = time.time()
    # 61维 + 41维 同时构建
    flat61_list, flat41_list, y_list, dates_list, inds_list = [], [], [], [], []
    # T+1~T+6 forward returns
    fwd_dict = {h: [] for h in range(1, 7)}

    for code in codes_used:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n = len(c); cc = wd(c); industry = ind_map.get(code, 'unknown')
        ma5 = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        e12 = pd.Series(c).ewm(span=12).mean().values
        e26 = pd.Series(c).ewm(span=26).mean().values
        dif = e12 - e26; dea = pd.Series(dif).ewm(span=9).mean().values
        macd_hist = (dif - dea) * 2
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1 / 14).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1 / 14).mean().values
        rsi14 = np.nan_to_num(100 - 100 / (1 + avg_gain / np.maximum(avg_loss, 1e-8)), 50)
        bb_std = pd.Series(c).rolling(20).std().values
        bb_pos = np.nan_to_num((c - (ma20 - 2 * bb_std)) / np.maximum(4 * bb_std, 0.01), 0.5)
        trange = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
        atr14 = pd.Series(trange).rolling(14).mean().values
        vol_ma3 = pd.Series(v).rolling(3).mean().values
        vol_ma12 = pd.Series(v).rolling(12).mean().values
        p5h = pd.Series(c).rolling(60).max().values
        p5l = pd.Series(c).rolling(60).min().values
        body_pct = np.abs(c - o) / np.maximum(h - l, 0.01)
        up_streak = np.zeros(n); dn_streak = np.zeros(n)
        for i in range(1, n):
            up_streak[i] = up_streak[i - 1] + 1 if c[i] > c[i - 1] else 0
            dn_streak[i] = dn_streak[i - 1] + 1 if c[i] < c[i - 1] else 0

        for i in range(60, n - 6):
            if c[i] <= 0.01: continue
            if i + 6 >= n: continue

            # 共通部分 (非FFT)
            common = []
            common.extend([(c[i] - c[i - j]) / max(abs(c[i - j]), 0.01) if i >= j else 0 for j in [1, 3, 6, 12]])
            for ma in [ma5, ma20, ma60]:
                common.append((c[i] - ma[i]) / max(abs(c[i]), 0.01) if not np.isnan(ma[i]) else 0)
            common.extend([dif[i] if not np.isnan(dif[i]) else 0,
                           dea[i] if not np.isnan(dea[i]) else 0,
                           macd_hist[i] if not np.isnan(macd_hist[i]) else 0])
            common.append(rsi14[i] if not np.isnan(rsi14[i]) else 50)
            common.append(bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5)
            common.append(np.std(np.diff(c[max(0, i - 6):i + 1]) /
                                 np.maximum(np.abs(c[max(0, i - 5):i + 1]), 0.01)) if i >= 6 else 0)
            common.append(atr14[i] / max(abs(c[i]), 0.01) if not np.isnan(atr14[i]) else 0)
            common.append((h[i] - l[i]) / max(abs(c[i]), 0.01))
            common.append(1.0 if c[i] > ma20[i] else 0.0)
            common.append(1.0 if c[i] > ma60[i] else 0.0)

            # G7 量价
            g7 = [v[i] / max(vol_ma12[i], 1) - 1 if i >= 12 and vol_ma12[i] > 0 else 0,
                  tr[i] if not np.isnan(tr[i]) else 0,
                  tr[i] / max(np.mean(tr[max(0, i - 12):i + 1]), 0.001) - 1 if i >= 12 and not np.isnan(tr[i]) else 0,
                  (vol_ma3[i] / max(vol_ma12[i], 1) - 1) if i >= 12 and vol_ma12[i] > 0 else 0,
                  np.log1p(max(v[i], 1)),
                  np.log1p(max(tr[i] * 100, 1)) if not np.isnan(tr[i]) else 0,
                  body_pct[i] if not np.isnan(body_pct[i]) else 0,
                  (c[i] - p5l[i]) / max(p5h[i] - p5l[i], 0.01) if i >= 60 else 0.5,
                  (ma20[i] - ma60[i]) / max(abs(c[i]), 0.01) if i >= 60 else 0,
                  np.std(c[max(0, i - 12):i + 1]) / max(abs(c[i]), 0.01) if i >= 12 else 0,
                  ma5[i] / max(ma20[i], 0.01) - 1 if i >= 20 else 0,
                  1.0 if c[i] > ma5[i] else 0.0,
                  up_streak[i] / 12.0, dn_streak[i] / 12.0]

            # 61维版本: 30d FFT (freq+amp+phase)
            fft30 = fft_full(cc[i - 60 + 1:i + 1]).tolist()
            flat61_list.append(common + fft30 + g7)

            # 41维版本: 10d FFT (amplitude only)
            fft10 = fft_amplitudes(cc[i - 60 + 1:i + 1]).tolist()
            flat41_list.append(common + fft10 + g7)

            # T+3 label (训练用)
            fwd_ret_t3 = (c[i + 3] - c[i + 2]) / max(abs(c[i + 2]), 0.01)
            y_list.append(fwd_ret_t3)
            dates_list.append(g['month'].iloc[i])
            inds_list.append(industry)

            # T+1~T+6 forward returns (评估用)
            for lag in range(1, 7):
                if i + lag < n:
                    fr = (c[i + lag] - c[i + lag - 1]) / max(abs(c[i + lag - 1]), 0.01)
                    fwd_dict[lag].append(np.clip(fr, -2, 2))
                else:
                    fwd_dict[lag].append(np.nan)

    flat61 = np.array(flat61_list, dtype=np.float32)
    flat41 = np.array(flat41_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates_arr = np.array(dates_list)
    inds_arr = np.array(inds_list)

    # 清理
    valid = ~np.isnan(flat61).any(axis=1) & ~np.isnan(flat41).any(axis=1) & ~np.isnan(y) & (np.abs(y) <= 2)
    flat61 = flat61[valid]; flat41 = flat41[valid]
    y = y[valid]; dates_arr = dates_arr[valid]; inds_arr = inds_arr[valid]

    # Forward returns
    fwd_returns = {}
    for lag in range(1, 7):
        fwd_arr = np.array(fwd_dict[lag], dtype=np.float32)
        fwd_returns[lag] = fwd_arr[valid]

    print(f"[{ts()}] {len(y):,} 样本, 61d+41d ({time.time() - t0:.0f}s)", flush=True)
    return flat61, flat41, y, dates_arr, inds_arr, fwd_returns


def train_and_eval(X_train, y_train, X_test, y_test, dates_test, fwd_returns_test, label=""):
    sc = StandardScaler()
    Xt = sc.fit_transform(X_train); Xte = sc.transform(X_test)

    lgb_m = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
                              n_estimators=300, min_child_samples=20, subsample=0.8,
                              colsample_bytree=0.8, random_state=456, verbosity=-1, n_jobs=4)
    lgb_m.fit(Xt, y_train); p_lgb = lgb_m.predict(Xte)

    xgb_m = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.05,
                             n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                             random_state=456, verbosity=0, n_jobs=4)
    xgb_m.fit(Xt, y_train); p_xgb = xgb_m.predict(Xte)

    ridge_m = Ridge(alpha=1.0); ridge_m.fit(Xt, y_train); p_ridge = ridge_m.predict(Xte)

    ics_m = {}
    for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]:
        ic_tmp = np.mean([spearmanr(p[dates_test == m], y_test[dates_test == m])[0]
                          for m in np.unique(dates_test) if (dates_test == m).sum() >= 20])
        ics_m[n] = max(ic_tmp, 0)
    w_sum = sum(ics_m.values())
    if w_sum <= 0: ics_m, w_sum = {'LGB': 1.0, 'XGB': 1.0, 'Ridge': 1.0}, 3.0
    p_ens = sum(ics_m[n] * p for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]) / w_sum

    # IC decay T+1~T+6
    decay_rows = []
    for lag in range(1, 7):
        fwd = fwd_returns_test[lag]
        # 对齐预测 (pred是用T+3训练的, fwd是各期单月收益)
        ic_lag, icir_lag, ics_arr = cs_ic(p_ens, fwd, dates_test)
        hit_lag = cs_hit(p_ens, fwd, dates_test)
        decay_rows.append({
            'horizon': f'T+{lag}',
            'IC': round(ic_lag, 4),
            'IC_std': round(np.std(ics_arr), 4),
            'ICIR': round(icir_lag, 3),
            'Hit': round(hit_lag, 3),
            'IC_pos': round(np.mean(ics_arr > 0), 3),
        })

    ic_t3, icir_t3, _ = cs_ic(p_ens, y_test, dates_test)
    return {
        'label': label, 'IC': ic_t3, 'ICIR': icir_t3,
        'w_lgb': ics_m['LGB'], 'w_xgb': ics_m['XGB'], 'w_ridge': ics_m['Ridge'],
        'decay': decay_rows,
        'lgb': lgb_m, 'xgb': xgb_m, 'ridge': ridge_m,
    }


# ====== Main ======
if __name__ == '__main__':
    flat61, flat41, y, dates_arr, inds_arr, fwd_returns = build_features()

    print(f"[{ts()}] 行业中性化...", flush=True); t0 = time.time()
    flat61_ind = cross_sectional_neutralize(flat61.copy(), dates_arr, inds_arr, 'categorical')
    flat41_ind = cross_sectional_neutralize(flat41.copy(), dates_arr, inds_arr, 'categorical')
    print(f"[{ts()}] done ({time.time() - t0:.0f}s)", flush=True)

    tr_m = (dates_arr >= '2010-01') & (dates_arr <= '2014-12')
    te_m = (dates_arr >= '2015-01')

    X61_tr = flat61_ind[tr_m]; X61_te = flat61_ind[te_m]
    X41_tr = flat41_ind[tr_m]; X41_te = flat41_ind[te_m]
    y_tr = y[tr_m]; y_te = y[te_m]
    dates_te = dates_arr[te_m]
    # 对齐fwd_returns
    fwd_te = {lag: fwd_returns[lag][te_m] for lag in range(1, 7)}

    # ====== 训练 & 评估 ======
    print(f"\n{'=' * 70}")
    print("对比: 61维(FFT 30d) vs 41维(FFT 10d振幅谱)")
    print(f"{'=' * 70}")

    r61 = train_and_eval(X61_tr, y_tr, X61_te, y_te, dates_te, fwd_te, "61d_baseline")
    print(f"\n[61维] IC(T+3)={r61['IC']:+.4f}  ICIR={r61['ICIR']:+.3f}  "
          f"w=({r61['w_lgb']:.3f},{r61['w_xgb']:.3f},{r61['w_ridge']:.3f})")

    r41 = train_and_eval(X41_tr, y_tr, X41_te, y_te, dates_te, fwd_te, "41d_fft10")
    print(f"[41维] IC(T+3)={r41['IC']:+.4f}  ICIR={r41['ICIR']:+.3f}  "
          f"w=({r41['w_lgb']:.3f},{r41['w_xgb']:.3f},{r41['w_ridge']:.3f})")

    delta = r41['IC'] - r61['IC']
    print(f"ΔIC={delta:+.4f} ({delta/abs(r61['IC'])*100:+.1f}%)")

    # ====== IC Decay T+1~T+6 ======
    print(f"\n{'=' * 70}")
    print("IC Decay: T+1 ~ T+6 (单月收益口径, 模型用T+3训练)")
    print(f"{'=' * 70}")
    print(f"{'Horizon':<8s} {'61d_IC':>8s} {'61d_IR':>8s} {'61d_Hit':>8s} "
          f"{'41d_IC':>8s} {'41d_IR':>8s} {'41d_Hit':>8s} {'ΔIC':>8s}")
    print('-' * 70)
    for i in range(6):
        d61 = r61['decay'][i]; d41 = r41['decay'][i]
        dic = d41['IC'] - d61['IC']
        print(f"{d61['horizon']:<8s} {d61['IC']:+8.4f} {d61['ICIR']:+8.3f} {d61['Hit']:+8.3f} "
              f"{d41['IC']:+8.4f} {d41['ICIR']:+8.3f} {d41['Hit']:+8.3f} {dic:+8.4f}")

    # ====== Feature Importance (41维) ======
    print(f"\n{'=' * 70}")
    print("41维特征重要性 (按LGB gain排序)")
    print(f"{'=' * 70}")

    lgb_gain = r41['lgb'].feature_importances_
    xgb_gain_dict = r41['xgb'].get_booster().get_score(importance_type='gain')

    imp_rows = []
    for i in range(41):
        xgb_g = xgb_gain_dict.get(f'f{i}', 0)
        imp_rows.append({
            'idx': i, 'feature': FEATURE_NAMES_41[i],
            'lgb_gain': round(float(lgb_gain[i]), 6),
            'xgb_gain': round(float(xgb_g), 6),
        })
    imp_df = pd.DataFrame(imp_rows)
    imp_df = imp_df.sort_values('lgb_gain', ascending=False).reset_index(drop=True)

    # 分组汇总
    print(f"\n{'Group':<20s} {'LGB_gain':>10s} {'XGB_gain':>10s} {'N_feat':>8s}")
    print('-' * 50)
    groups_41 = {
        'G1_价格动量': range(0, 4),
        'G2_均线偏离': range(4, 7),
        'G3_MACD': range(7, 10),
        'G4_技术指标': range(10, 15),
        'G5_趋势二值': range(15, 17),
        'FFT_振幅10维': range(17, 27),
        'G7_量价K线': range(27, 41),
    }
    for gname, grange in groups_41.items():
        gdf = imp_df[imp_df['idx'].isin(grange)]
        print(f"{gname:<20s} {gdf['lgb_gain'].sum():>10.4f} {gdf['xgb_gain'].sum():>10.4f} {len(gdf):>8d}")

    # Top 20
    print(f"\nTop 20 (by LGB gain):")
    print(imp_df.head(20)[['idx', 'feature', 'lgb_gain', 'xgb_gain']].to_string(index=False))

    # 保存
    imp_df.to_csv(OUT / 'fft10_feature_importance.csv', index=False, encoding='utf-8-sig')

    decay_df = pd.DataFrame([{
        'horizon': d61['horizon'],
        'IC_61d': d61['IC'], 'ICIR_61d': d61['ICIR'], 'Hit_61d': d61['Hit'],
        'IC_41d': d41['IC'], 'ICIR_41d': d41['ICIR'], 'Hit_41d': d41['Hit'],
        'delta_IC': d41['IC'] - d61['IC'],
    } for d61, d41 in zip(r61['decay'], r41['decay'])])
    decay_df.to_csv(OUT / 'fft10_ic_decay.csv', index=False, encoding='utf-8-sig')

    summary = {
        'baseline_61d': {'IC': float(r61['IC']), 'ICIR': float(r61['ICIR'])},
        'fft10_41d': {'IC': float(r41['IC']), 'ICIR': float(r41['ICIR'])},
        'delta_IC': float(delta),
        'delta_IC_pct': float(delta / abs(r61['IC']) * 100),
        'fft10_group_lgb_gain': float(imp_df[imp_df['idx'].isin(range(17, 27))]['lgb_gain'].sum()),
        'fft30_group_lgb_gain_baseline': 7176.0,  # from step1.1
    }
    with open(OUT / 'fft10_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{ts()}] done. 结果: {OUT}")
