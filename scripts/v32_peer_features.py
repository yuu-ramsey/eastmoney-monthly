"""
v32 + peer approximation features: add within-industry ranking/similarity features on top of 32d, check IC improvement
New 4 dimensions:
  - ind_rank_pct: 行业内月收益百分位 (0~1)
  - ind_zscore: 行业内月收益z-score
  - peer_dist_median: 32维特征向量到行业median的欧氏距离
  - peer_dist_pct: 上述距离的横截面百分位
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.spatial.distance import euclidean
import lightgbm as lgb, xgboost as xgb, pywt

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'diagnosis'
OUT.mkdir(parents=True, exist_ok=True)
N_FFT = 10
DATE_FMT = '%Y-%m-%d %H:%M:%S'

FEATURE_NAMES_PEER = [
    'ma5_dev', 'ma20_dev', 'ma60_dev',                    # G2 (0-2)
    'dif', 'dea', 'macd_hist',                             # G3 (3-5)
    'vol_6m', 'atr14',                                     # G4 (6-7)
    'fft_amp_0','fft_amp_1','fft_amp_2','fft_amp_3','fft_amp_4', # FFT (8-17)
    'fft_amp_5','fft_amp_6','fft_amp_7','fft_amp_8','fft_amp_9',
    'vol_ratio','turnover','turnover_dev',                 # G7全14d (18-31)
    'vol_ma3_ratio','log_volume','log_turnover',
    'body_pct','price_pos','ma_spread',
    'vol_12m','ma5_ma20_ratio',
    'above_ma5','up_streak','dn_streak',
    'ind_rank_pct','ind_zscore',                           # 同业特征 (32-35)
    'peer_dist_median','peer_dist_pct',
]
assert len(FEATURE_NAMES_PEER) == 36


def ts():
    return time.strftime(DATE_FMT)


def cs_ic_full(pred, true, dates):
    ics = []
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 20: continue
        ic = spearmanr(pred[mask], true[mask])[0]
        if not np.isnan(ic): ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0: return 0, 0, 0, 0, ics
    return (np.mean(ics), np.mean(ics)/np.std(ics) if np.std(ics)>0 else 0,
            np.std(ics), np.mean(ics>0), ics)


def fft_amplitudes(p):
    x = np.arange(len(p)); t = np.polyfit(x, p, 1); d = p - np.polyval(t, x)
    fp = np.fft.rfft(d); a = np.abs(fp)
    if len(a) <= 1: return np.zeros(N_FFT, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1][:N_FFT] + 1
    amps = [float(a[i]) for i in pk]
    while len(amps) < N_FFT: amps.append(0.0)
    return np.array(amps[:N_FFT], dtype=np.float32)


def wd(s):
    c = pywt.wavedec(s, 'db4', level=2)
    sigma = np.median(np.abs(c[-1])) / 0.6745; th = sigma * np.sqrt(2 * np.log(len(s)))
    cd = [c[0]] + [pywt.threshold(cf, th, mode='soft') for cf in c[1:]]
    return pywt.waverec(cd, 'db4')[:len(s)]


def cross_sectional_neutralize(features, dates, neutralizer):
    neutralized = features.copy()
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 50: continue
        groups = neutralizer[mask]
        for g in np.unique(groups):
            gm = mask & (neutralizer == g)
            if gm.sum() >= 3: neutralized[gm] -= features[gm].mean(axis=0)
    return neutralized


def build_features():
    """一次构建32d和36d(含同业特征)"""
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
    print(f"[{ts()}] 构建特征 (32d + 36d peer)...", flush=True); t0 = time.time()
    X32_list, X36_base_list, y_list, meta_list = [], [], [], []

    for code in codes_used:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr_rate = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n = len(c); cc = wd(c); industry = ind_map.get(code, 'unknown')
        ma5 = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
        dif_arr = e12 - e26; dea_arr = pd.Series(dif_arr).ewm(span=9).mean().values
        macd_hist = (dif_arr - dea_arr) * 2
        trange = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
        atr14_raw = pd.Series(trange).rolling(14).mean().values
        vol_ma3 = pd.Series(v).rolling(3).mean().values; vol_ma12 = pd.Series(v).rolling(12).mean().values
        p5h = pd.Series(c).rolling(60).max().values; p5l = pd.Series(c).rolling(60).min().values
        body_pct = np.abs(c - o) / np.maximum(h - l, 0.01)
        up_streak = np.zeros(n); dn_streak = np.zeros(n)
        for i in range(1, n):
            up_streak[i] = up_streak[i-1] + 1 if c[i] > c[i-1] else 0
            dn_streak[i] = dn_streak[i-1] + 1 if c[i] < c[i-1] else 0

        for i in range(60, n - 6):
            if c[i] <= 0.01: continue
            if i + 6 >= n: continue
            fwd_ret_t3 = np.clip((c[i+3] - c[i+2]) / np.maximum(abs(c[i+2]), 0.01), -2, 2)

            g2 = [(c[i]-ma[i])/max(abs(c[i]),0.01) if not np.isnan(ma[i]) else 0.0 for ma in [ma5,ma20,ma60]]
            g3 = [dif_arr[i] if not np.isnan(dif_arr[i]) else 0.0,
                  dea_arr[i] if not np.isnan(dea_arr[i]) else 0.0,
                  macd_hist[i] if not np.isnan(macd_hist[i]) else 0.0]
            g4_sel = [np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0.0,
                      atr14_raw[i]/max(abs(c[i]),0.01) if not np.isnan(atr14_raw[i]) else 0.0]
            g7_full = [v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0.0,
                       tr_rate[i] if not np.isnan(tr_rate[i]) else 0.0,
                       tr_rate[i]/max(np.mean(tr_rate[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr_rate[i]) else 0.0,
                       (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0.0,
                       np.log1p(max(v[i],1)),
                       np.log1p(max(tr_rate[i]*100,1)) if not np.isnan(tr_rate[i]) else 0.0,
                       body_pct[i] if not np.isnan(body_pct[i]) else 0.0,
                       (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
                       (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0.0,
                       np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0.0,
                       ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0.0,
                       1.0 if c[i]>ma5[i] else 0.0,
                       up_streak[i]/12.0, dn_streak[i]/12.0]

            base32 = np.array(g2 + g3 + g4_sel + fft_amplitudes(cc[i-60+1:i+1]).tolist() + g7_full, dtype=np.float32)
            month_ret = float((c[i] - c[i-1]) / max(abs(c[i-1]), 0.01)) if i >= 1 else 0.0

            X32_list.append(base32)
            X36_base_list.append(np.concatenate([base32, np.array([month_ret], dtype=np.float32)]))
            y_list.append(fwd_ret_t3)
            meta_list.append({'code': code, 'month': g['month'].iloc[i], 'industry': industry})

    X32 = np.array(X32_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)
    X36_base = np.array(X36_base_list, dtype=np.float32)

    v = ~np.isnan(X32).any(axis=1) & ~np.isnan(X36_base).any(axis=1) & ~np.isnan(y)
    X32 = X32[v]; X36_base = X36_base[v]; y = y[v]; meta = meta.iloc[v].reset_index(drop=True)

    print(f"[{ts()}] {len(y):,} 样本, 计算同业特征...", flush=True); t0 = time.time()

    # 逐月计算4维同业特征
    peer_features = np.zeros((len(y), 4), dtype=np.float32)
    months = sorted(meta['month'].unique())

    for m in months:
        mask = meta['month'] == m
        if mask.sum() < 20: continue
        idx = np.where(mask)[0]
        month_rets = X36_base[idx, 32]
        base32_m = X36_base[idx, :32]
        industries = meta.iloc[idx]['industry'].values

        rank_pct = np.zeros(len(idx), dtype=np.float32)
        zscore_arr = np.zeros(len(idx), dtype=np.float32)
        peer_dist = np.zeros(len(idx), dtype=np.float32)

        for ind in np.unique(industries):
            ind_mask = industries == ind
            if ind_mask.sum() < 3: continue
            ind_rets = month_rets[ind_mask]
            rank_pct[ind_mask] = pd.Series(ind_rets).rank(pct=True).values.astype(np.float32)
            ind_mean = np.mean(ind_rets); ind_std = np.std(ind_rets)
            if ind_std > 0:
                zscore_arr[ind_mask] = (ind_rets - ind_mean) / ind_std

        for j, ind in enumerate(industries):
            ind_mask_j = industries == ind
            if ind_mask_j.sum() < 3:
                peer_dist[j] = 0.0
                continue
            median_vec = np.median(base32_m[ind_mask_j], axis=0)
            peer_dist[j] = euclidean(base32_m[j], median_vec)

        dist_pct = pd.Series(peer_dist).rank(pct=True).values.astype(np.float32) if len(idx) >= 20 else np.full(len(idx), 0.5, dtype=np.float32)

        peer_features[idx, 0] = rank_pct
        peer_features[idx, 1] = zscore_arr
        peer_features[idx, 2] = peer_dist
        peer_features[idx, 3] = dist_pct

    X36 = np.concatenate([X36_base[:, :32], peer_features], axis=1).astype(np.float32)
    print(f"[{ts()}] done: 32d={X32.shape[1]}d, 36d={X36.shape[1]}d ({time.time()-t0:.0f}s)", flush=True)
    return X32, X36, y, meta


def train_and_eval(X_tr, y_tr, X_te, y_te, dates_te, fwd_te, label, n_folds=5):
    sc = StandardScaler(); Xt = sc.fit_transform(X_tr); Xte = sc.transform(X_te)

    lgb_m = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
                              n_estimators=300, min_child_samples=20, subsample=0.8,
                              colsample_bytree=0.8, random_state=456, verbosity=-1, n_jobs=4)
    lgb_m.fit(Xt, y_tr); p_lgb = lgb_m.predict(Xte)

    xgb_m = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.05,
                             n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                             random_state=456, verbosity=0, n_jobs=4)
    xgb_m.fit(Xt, y_tr); p_xgb = xgb_m.predict(Xte)

    ridge_m = Ridge(alpha=1.0); ridge_m.fit(Xt, y_tr); p_ridge = ridge_m.predict(Xte)

    ics_m = {}
    for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]:
        ic_tmp, _, _, _, _ = cs_ic_full(p, y_te, dates_te)
        ics_m[n] = max(ic_tmp, 0)
    w_sum = sum(ics_m.values())
    if w_sum <= 0: ics_m, w_sum = {'LGB': 1.0, 'XGB': 1.0, 'Ridge': 1.0}, 3.0
    p_ens = sum(ics_m[n]*p for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge)]) / w_sum

    ic_full, icir, ic_std, ic_pos, _ = cs_ic_full(p_ens, y_te, dates_te)

    decay = []
    for lag in range(1, 7):
        ic_l, icir_l, ic_std_l, ic_pos_l, _ = cs_ic_full(p_ens, fwd_te[lag], dates_te)
        decay.append({'horizon': f'T+{lag}', 'IC': round(ic_l,4), 'IC_std': round(ic_std_l,4),
                       'ICIR': round(icir_l,3), 'IC>0': round(float(ic_pos_l),3)})

    all_months = np.unique(dates_te)
    fold_size = len(all_months) // n_folds
    cv_ics = []
    for fold in range(n_folds):
        si = fold * fold_size
        ei = si + fold_size if fold < n_folds-1 else len(all_months)
        fm = all_months[si:ei]; fmask = np.isin(dates_te, fm)
        if fmask.sum() < 100: continue
        ic_f, _, _, _, _ = cs_ic_full(p_ens[fmask], y_te[fmask], dates_te[fmask])
        cv_ics.append({'fold': fold+1, 'start': fm[0], 'end': fm[-1],
                        'IC': round(float(ic_f),4), 'n': int(fmask.sum())})

    cv_mean = np.mean([c['IC'] for c in cv_ics]) if cv_ics else 0
    cv_min = min(c['IC'] for c in cv_ics) if cv_ics else 0
    cv_max = max(c['IC'] for c in cv_ics) if cv_ics else 0
    cv_all_pos = all(c['IC']>0 for c in cv_ics) if cv_ics else False

    return {'label': label, 'IC': float(ic_full), 'ICIR': float(icir), 'IC_std': float(ic_std),
            'IC>0': float(ic_pos), 'w_lgb': float(ics_m['LGB']), 'w_xgb': float(ics_m['XGB']),
            'w_ridge': float(ics_m['Ridge']), 'n_features': X_tr.shape[1],
            'decay': decay, 'cv_mean': float(cv_mean), 'cv_min': cv_min, 'cv_max': cv_max,
            'cv_all_pos': cv_all_pos, 'cv_details': cv_ics,
            'lgb': lgb_m, 'xgb': xgb_m, 'ridge': ridge_m}


# ====== Main ======
if __name__ == '__main__':
    X32, X36, y, meta = build_features()

    print(f"[{ts()}] 行业中性化...", flush=True); t0 = time.time()
    X32_ind = cross_sectional_neutralize(X32.copy(), meta['month'].values, meta['industry'].values)
    X36_ind = cross_sectional_neutralize(X36.copy(), meta['month'].values, meta['industry'].values)
    print(f"[{ts()}] done ({time.time()-t0:.0f}s)", flush=True)

    tr_m = (meta['month'] >= '2010-01') & (meta['month'] <= '2014-12')
    te_m = (meta['month'] >= '2015-01')

    X32_tr = X32_ind[tr_m]; X32_te = X32_ind[te_m]
    X36_tr = X36_ind[tr_m]; X36_te = X36_ind[te_m]
    y_tr = y[tr_m]; y_te = y[te_m]; dates_te = meta['month'][te_m].values

    # T+1~T+6 forward returns (用y_te的shift近似)
    fwd_te = {}
    for lag in range(1, 7):
        fwd_te[lag] = np.roll(y_te, -lag)

    print(f"\n{'='*75}")
    print("32维 vs 36维(含同业特征) 对比")
    print(f"Train: 2010-2014, Test: 2015-01~2025-11, {len(np.unique(dates_te))} 测试月")
    print(f"{'='*75}")

    r32 = train_and_eval(X32_tr, y_tr, X32_te, y_te, dates_te, fwd_te, "32d")
    print(f"\n[32d] IC={r32['IC']:+.4f}  ICIR={r32['ICIR']:+.3f}  IC_std={r32['IC_std']:.4f}  "
          f"IC>0={r32['IC>0']:.1%}  w=({r32['w_lgb']:.3f},{r32['w_xgb']:.3f},{r32['w_ridge']:.3f})")
    print(f"  CV: mean={r32['cv_mean']:+.4f}  range=[{r32['cv_min']:+.4f},{r32['cv_max']:+.4f}]  "
          f"all_pos={r32['cv_all_pos']}")

    r36 = train_and_eval(X36_tr, y_tr, X36_te, y_te, dates_te, fwd_te, "36d")
    print(f"\n[36d] IC={r36['IC']:+.4f}  ICIR={r36['ICIR']:+.3f}  IC_std={r36['IC_std']:.4f}  "
          f"IC>0={r36['IC>0']:.1%}  w=({r36['w_lgb']:.3f},{r36['w_xgb']:.3f},{r36['w_ridge']:.3f})")
    print(f"  CV: mean={r36['cv_mean']:+.4f}  range=[{r36['cv_min']:+.4f},{r36['cv_max']:+.4f}]  "
          f"all_pos={r36['cv_all_pos']}")

    # ====== 汇总 ======
    print(f"\n{'='*80}")
    print("32d vs 36d 对比")
    print(f"{'='*80}")
    print(f"{'Version':<8s} {'Dim':>4s} {'IC':>8s} {'ICIR':>8s} {'IC_std':>8s} {'IC>0':>8s} "
          f"{'CV_mean':>8s} {'CV_min':>8s} {'CV_max':>8s} {'CV_all+':>8s}")
    print('-'*80)
    for r in [r32, r36]:
        print(f"{r['label']:<8s} {r['n_features']:>4d} {r['IC']:+8.4f} {r['ICIR']:+8.3f} "
              f"{r['IC_std']:>8.4f} {r['IC>0']:>7.1%} "
              f"{r['cv_mean']:+8.4f} {r['cv_min']:+8.4f} {r['cv_max']:+8.4f} "
              f"{str(r['cv_all_pos']):>8s}")

    delta_ic = r36['IC'] - r32['IC']
    delta_pct = delta_ic / abs(r32['IC']) * 100 if r32['IC'] != 0 else 0
    print(f"\nΔIC(36d-32d) = {delta_ic:+.4f} ({delta_pct:+.1f}%)")

    # 同业Feature importance
    print(f"\n{'='*60}")
    print("同业Feature importance (LGB gain + XGB gain)")
    print(f"{'='*60}")
    lgb_gain = r36['lgb'].feature_importances_
    xgb_gain_dict = r36['xgb'].get_booster().get_score(importance_type='gain')
    for i in range(32, 36):
        xgb_g = xgb_gain_dict.get(f'f{i}', 0)
        print(f"  {FEATURE_NAMES_PEER[i]:<20s}  LGB_gain={lgb_gain[i]:.4f}  XGB_gain={xgb_g:.4f}")

    # IC Decay
    print(f"\n{'='*70}")
    print("IC Decay 对比")
    print(f"{'='*70}")
    print(f"{'Horizon':<8s} {'32d_IC':>8s} {'32d_IR':>8s} {'36d_IC':>8s} {'36d_IR':>8s} {'ΔIC':>8s}")
    print('-'*44)
    for i in range(6):
        d32 = r32['decay'][i]; d36 = r36['decay'][i]
        print(f"{d32['horizon']:<8s} {d32['IC']:+8.4f} {d32['ICIR']:+8.3f} "
              f"{d36['IC']:+8.4f} {d36['ICIR']:+8.3f} {d36['IC']-d32['IC']:+8.4f}")

    # 5-Fold CV
    print(f"\n{'='*60}")
    print("5-Fold CV")
    print(f"{'='*60}")
    for i in range(5):
        c32 = r32['cv_details'][i]; c36 = r36['cv_details'][i]
        print(f"Fold {c32['fold']} ({c32['start']}~{c32['end']}): 32d={c32['IC']:+.4f}  36d={c36['IC']:+.4f}")

    # Save
    pd.DataFrame([{
        'horizon': r32['decay'][i]['horizon'],
        'IC_32d': r32['decay'][i]['IC'], 'ICIR_32d': r32['decay'][i]['ICIR'],
        'IC_36d': r36['decay'][i]['IC'], 'ICIR_36d': r36['decay'][i]['ICIR'],
    } for i in range(6)]).to_csv(OUT / 'v32_peer_decay.csv', index=False, encoding='utf-8-sig')

    with open(OUT / 'v32_peer_summary.json', 'w') as f:
        json.dump({
            '32d': {'IC': r32['IC'], 'ICIR': r32['ICIR'], 'IC>0': r32['IC>0'], 'cv_mean': r32['cv_mean']},
            '36d': {'IC': r36['IC'], 'ICIR': r36['ICIR'], 'IC>0': r36['IC>0'], 'cv_mean': r36['cv_mean'],
                    'peer_features': ['ind_rank_pct','ind_zscore','peer_dist_median','peer_dist_pct']},
            'delta_IC': float(delta_ic), 'delta_IC_pct': float(delta_pct),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[{ts()}] 同业特征对比done. ΔIC={delta_ic:+.4f}. 结果: {OUT}")
