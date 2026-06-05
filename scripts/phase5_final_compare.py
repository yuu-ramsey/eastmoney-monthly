"""
Phase 5: Four-version final comparison
61d(baseline) / 52d(删G1+G5+G4部分) / 41d(FFT振幅10维) / 32d(合并精简)
统一5-Fold CV, IC + ICIR + IC>0
修复: IC>0 = 月份正IC占比 (非个股方向正确率)
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


def ts():
    return time.strftime(DATE_FMT)


def cs_ic_full(pred, true, dates):
    """Returns IC均值, ICIR, IC_std, IC>0(月份正IC占比), 月度ICs"""
    ics = []
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 20: continue
        ic = spearmanr(pred[mask], true[mask])[0]
        if not np.isnan(ic): ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0: return 0, 0, 0, 0, ics
    return (np.mean(ics), np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0,
            np.std(ics), np.mean(ics > 0), ics)


def fft_amplitudes(p):
    x = np.arange(len(p)); t = np.polyfit(x, p, 1); d = p - np.polyval(t, x)
    fp = np.fft.rfft(d); a = np.abs(fp)
    if len(a) <= 1: return np.zeros(N_FFT, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1][:N_FFT] + 1
    amps = [a[i] for i in pk]
    while len(amps) < N_FFT: amps.append(0)
    return np.array(amps[:N_FFT], dtype=np.float32)


def fft_full(p):
    x = np.arange(len(p)); t = np.polyfit(x, p, 1); d = p - np.polyval(t, x)
    fp = np.fft.rfft(d); a = np.abs(fp); fq = np.fft.rfftfreq(len(d))
    if len(a) <= 1: return np.zeros(N_FFT * 3, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1][:N_FFT] + 1; fs = []
    for i in pk:
        if i < len(fq): fs.extend([fq[i], a[i], np.angle(fp[i])])
    while len(fs) < N_FFT * 3: fs.extend([0, 0, 0])
    return np.array(fs[:N_FFT * 3], dtype=np.float32)


def wd(s):
    c = pywt.wavedec(s, 'db4', level=2)
    sigma = np.median(np.abs(c[-1])) / 0.6745; th = sigma * np.sqrt(2 * np.log(len(s)))
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


# 索引映射
# 61维 → 52维: 删 G1(0-3) + G4部分(10,11,14) + G5(15-16) = 9维
DROP_61 = {0, 1, 2, 3, 10, 11, 14, 15, 16}
KEEP_52 = [i for i in range(61) if i not in DROP_61]

# 41维 → 32维: 删同样的组 (G1:0-3, G4:10,11,14, G5:15-16)
# 41维中G7起始位置是27(非61维的47), 但G1/G4/G5位置不变
KEEP_32 = [i for i in range(41) if i not in DROP_61]

VERSION_NAMES = {
    '61d': '61维 baseline (FFT 30d)',
    '52d': '52维 精简 (删G1+G5+G4×3, FFT 30d)',
    '41d': '41维 FFT振幅10维',
    '32d': '32维 合并精简 (删G1+G5+G4×3, FFT振幅10维)',
}


def build_features():
    print(f"[{ts()}] Loaded数据...", flush=True)
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
    print(f"[{ts()}] 构建特征 (61d + 41d)...", flush=True); t0 = time.time()
    flat61_list, flat41_list, y_list, dates_list, inds_list = [], [], [], [], []

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
        e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
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
        vol_ma3 = pd.Series(v).rolling(3).mean().values; vol_ma12 = pd.Series(v).rolling(12).mean().values
        p5h = pd.Series(c).rolling(60).max().values; p5l = pd.Series(c).rolling(60).min().values
        body_pct = np.abs(c - o) / np.maximum(h - l, 0.01)
        up_streak = np.zeros(n); dn_streak = np.zeros(n)
        for i in range(1, n):
            up_streak[i] = up_streak[i - 1] + 1 if c[i] > c[i - 1] else 0
            dn_streak[i] = dn_streak[i - 1] + 1 if c[i] < c[i - 1] else 0

        for i in range(60, n - 6):
            if c[i] <= 0.01: continue
            if i + 3 >= n: continue
            fwd_ret = np.clip((c[i + 3] - c[i + 2]) / np.maximum(abs(c[i + 2]), 0.01), -2, 2)

            # 共通部分 (G1+G2+G3+G4+G5)
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

            flat61_list.append(common + fft_full(cc[i - 60 + 1:i + 1]).tolist() + g7)
            flat41_list.append(common + fft_amplitudes(cc[i - 60 + 1:i + 1]).tolist() + g7)
            y_list.append(fwd_ret)
            dates_list.append(g['month'].iloc[i])
            inds_list.append(industry)

    flat61 = np.array(flat61_list, dtype=np.float32)
    flat41 = np.array(flat41_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates_arr = np.array(dates_list)
    inds_arr = np.array(inds_list)

    v = ~np.isnan(flat61).any(axis=1) & ~np.isnan(flat41).any(axis=1) & ~np.isnan(y)
    flat61 = flat61[v]; flat41 = flat41[v]
    y = y[v]; dates_arr = dates_arr[v]; inds_arr = inds_arr[v]

    print(f"[{ts()}] {len(y):,} 样本, 61d={flat61.shape[1]}d 41d={flat41.shape[1]}d ({time.time() - t0:.0f}s)", flush=True)
    return flat61, flat41, y, dates_arr, inds_arr


def train_and_eval_cv(X_train, y_train, X_test, y_test, dates_test, label, n_folds=5):
    """训练 LGB+XGB+Ridge 集成, Returns全期 + 5-Fold CV 指标"""
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
        ic_tmp, _, _, _, _ = cs_ic_full(p, y_test, dates_test)
        ics_m[n] = max(ic_tmp, 0)
    w_sum = sum(ics_m.values())
    if w_sum <= 0: ics_m, w_sum = {'LGB': 1.0, 'XGB': 1.0, 'Ridge': 1.0}, 3.0
    p_ens = sum(ics_m[n] * p for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]) / w_sum

    # 全期
    ic_full, icir_full, ic_std, ic_pos, _ = cs_ic_full(p_ens, y_test, dates_test)

    # 5-Fold CV (时间序)
    all_months = np.unique(dates_test)
    fold_size = len(all_months) // n_folds
    cv_ics = []

    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(all_months)
        fold_months = all_months[start_idx:end_idx]
        fold_mask = np.isin(dates_test, fold_months)
        if fold_mask.sum() < 100: continue

        ic_f, _, _, _, _ = cs_ic_full(p_ens[fold_mask], y_test[fold_mask], dates_test[fold_mask])
        cv_ics.append({'fold': fold + 1, 'start': fold_months[0], 'end': fold_months[-1],
                        'IC': round(float(ic_f), 4), 'n_samples': int(fold_mask.sum())})

    cv_mean_ic = np.mean([c['IC'] for c in cv_ics]) if cv_ics else 0
    cv_ic_range = (min(c['IC'] for c in cv_ics), max(c['IC'] for c in cv_ics)) if cv_ics else (0, 0)
    cv_all_pos = all(c['IC'] > 0 for c in cv_ics) if cv_ics else False

    return {
        'label': label,
        'IC': float(ic_full), 'ICIR': float(icir_full), 'IC_std': float(ic_std),
        'IC>0': float(ic_pos),
        'w_lgb': float(ics_m['LGB']), 'w_xgb': float(ics_m['XGB']), 'w_ridge': float(ics_m['Ridge']),
        'cv_mean_IC': float(cv_mean_ic), 'cv_IC_range': cv_ic_range,
        'cv_all_pos': cv_all_pos, 'cv_details': cv_ics,
        'n_features': X_train.shape[1],
    }


# ====== Main ======
if __name__ == '__main__':
    flat61, flat41, y, dates_arr, inds_arr = build_features()

    print(f"[{ts()}] 行业中性化...", flush=True); t0 = time.time()
    flat61_ind = cross_sectional_neutralize(flat61.copy(), dates_arr, inds_arr, 'categorical')
    flat41_ind = cross_sectional_neutralize(flat41.copy(), dates_arr, inds_arr, 'categorical')
    print(f"[{ts()}] done ({time.time() - t0:.0f}s)", flush=True)

    tr_m = (dates_arr >= '2010-01') & (dates_arr <= '2014-12')
    te_m = (dates_arr >= '2015-01')

    # 4个版本的特征矩阵
    versions = {
        '61d': (flat61_ind[tr_m], flat61_ind[te_m], 61),
        '52d': (flat61_ind[tr_m][:, KEEP_52], flat61_ind[te_m][:, KEEP_52], 52),
        '41d': (flat41_ind[tr_m], flat41_ind[te_m], 41),
        '32d': (flat41_ind[tr_m][:, KEEP_32], flat41_ind[te_m][:, KEEP_32], 32),
    }

    y_tr = y[tr_m]; y_te = y[te_m]; dates_te = dates_arr[te_m]

    results = {}
    print(f"\n{'=' * 75}")
    print("四版本对比 (LGB+XGB+Ridge Ensemble, 行业中性化, T+3单月收益)")
    print(f"Train: 2010-2014, Test: 2015-01~2025-11, {len(np.unique(dates_te))} 测试月")
    print(f"{'=' * 75}")

    for vname, (X_tr, X_te, ndim) in versions.items():
        print(f"\n--- {VERSION_NAMES[vname]} ({ndim}d) ---", flush=True)
        r = train_and_eval_cv(X_tr, y_tr, X_te, y_te, dates_te, vname)
        results[vname] = r
        print(f"  IC={r['IC']:+.4f}  ICIR={r['ICIR']:+.3f}  IC_std={r['IC_std']:.4f}  "
              f"IC>0={r['IC>0']:.1%}  w=({r['w_lgb']:.3f},{r['w_xgb']:.3f},{r['w_ridge']:.3f})")
        print(f"  CV: mean_IC={r['cv_mean_IC']:+.4f}  range=[{r['cv_IC_range'][0]:+.4f}, {r['cv_IC_range'][1]:+.4f}]  "
              f"all_pos={r['cv_all_pos']}")

    # ====== 汇总表 ======
    print(f"\n{'=' * 100}")
    print("四方对比汇总")
    print(f"{'=' * 100}")
    print(f"{'Version':<8s} {'Dim':>4s} {'IC':>8s} {'ICIR':>8s} {'IC_std':>8s} {'IC>0':>8s} "
          f"{'CV_mean':>8s} {'CV_min':>8s} {'CV_max':>8s} {'CV_all+':>8s}")
    print('-' * 100)

    baseline_ic = results['61d']['IC']
    for vname in ['61d', '52d', '41d', '32d']:
        r = results[vname]
        cv_min = r['cv_IC_range'][0]; cv_max = r['cv_IC_range'][1]
        print(f"{vname:<8s} {r['n_features']:>4d} {r['IC']:+8.4f} {r['ICIR']:+8.3f} "
              f"{r['IC_std']:>8.4f} {r['IC>0']:>7.1%} "
              f"{r['cv_mean_IC']:+8.4f} {cv_min:+8.4f} {cv_max:+8.4f} "
              f"{str(r['cv_all_pos']):>8s}")

    # Δ vs baseline
    print(f"\n{'Version':<8s} {'ΔIC':>8s} {'ΔIC%':>8s} {'ΔICIR':>8s} {'ΔIC>0':>8s} {'ΔCV_mean':>8s}")
    print('-' * 50)
    for vname in ['52d', '41d', '32d']:
        r = results[vname]
        dic = r['IC'] - baseline_ic
        dic_pct = dic / abs(baseline_ic) * 100
        dir_ = r['ICIR'] - results['61d']['ICIR']
        dp = r['IC>0'] - results['61d']['IC>0']
        dcv = r['cv_mean_IC'] - results['61d']['cv_mean_IC']
        print(f"{vname:<8s} {dic:+8.4f} {dic_pct:+7.1f}% {dir_:+8.3f} {dp:+7.1%} {dcv:+8.4f}")

    # 按Fold展开
    print(f"\n{'=' * 80}")
    print("5-Fold CV 展开")
    print(f"{'=' * 80}")
    for i in range(5):
        fold_info = results['61d']['cv_details'][i]
        print(f"\nFold {fold_info['fold']} ({fold_info['start']}~{fold_info['end']}):")
        for vname in ['61d', '52d', '41d', '32d']:
            cv_d = results[vname]['cv_details'][i]
            print(f"  {vname}: IC={cv_d['IC']:+.4f}")

    # Save
    summary = {}
    for vname in ['61d', '52d', '41d', '32d']:
        r = results[vname]
        summary[vname] = {
            'desc': VERSION_NAMES[vname],
            'n_features': r['n_features'],
            'IC': r['IC'], 'ICIR': r['ICIR'], 'IC_std': r['IC_std'], 'IC>0': r['IC>0'],
            'weights': {'LGB': r['w_lgb'], 'XGB': r['w_xgb'], 'Ridge': r['w_ridge']},
            'cv_mean_IC': r['cv_mean_IC'], 'cv_IC_range': list(r['cv_IC_range']),
            'cv_all_pos': r['cv_all_pos'], 'cv_details': r['cv_details'],
        }

    summary_df = pd.DataFrame([{
        'version': vname,
        'n_features': summary[vname]['n_features'],
        'IC': summary[vname]['IC'],
        'ICIR': summary[vname]['ICIR'],
        'IC_std': summary[vname]['IC_std'],
        'IC>0': summary[vname]['IC>0'],
        'cv_mean_IC': summary[vname]['cv_mean_IC'],
        'cv_IC_min': summary[vname]['cv_IC_range'][0],
        'cv_IC_max': summary[vname]['cv_IC_range'][1],
        'cv_all_pos': summary[vname]['cv_all_pos'],
    } for vname in ['61d', '52d', '41d', '32d']])
    summary_df.to_csv(OUT / 'final_4way_compare.csv', index=False, encoding='utf-8-sig')

    with open(OUT / 'final_4way_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{ts()}] 四方对比done. 结果: {OUT}")
