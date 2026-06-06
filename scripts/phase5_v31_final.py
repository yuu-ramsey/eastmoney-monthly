
"""
Phase 5: 31d final vs 61d baseline
FFT: Peak1(3d)+Peak2(3d)+Peaks3-10 stats(4d) = 10d
Drop: G1(4d)+G5(2d)+G4x3(rsi14/bb_pos/amplitude)+G7x1(above_ma5) = 10d
Keep: G2(3d)+G3(3d)+G4 pruned(2d)+G7 slim(13d) = 21d + FFT(10d) = 31d
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

FEATURE_NAMES_31 = [
    'ma5_dev', 'ma20_dev', 'ma60_dev',                       # G2 MA deviation (0-2)
    'dif', 'dea', 'macd_hist',                                # G3 MACD (3-5)
    'vol_6m', 'atr14',                                        # G4 pruned (6-7)
    'fft_p1_freq', 'fft_p1_amp', 'fft_p1_phase',             # FFT Peak1 (8-10)
    'fft_p2_freq', 'fft_p2_amp', 'fft_p2_phase',             # FFT Peak2 (11-13)
    'fft_amp_mean', 'fft_amp_std',                            # FFT Peaks3-10 stats (14-17)
    'fft_freq_median', 'fft_freq_range',
    'vol_ratio', 'turnover', 'turnover_dev',                  # G7 slim (18-30)
    'vol_ma3_ratio', 'log_volume', 'log_turnover',
    'body_pct', 'price_pos', 'ma_spread',
    'vol_12m', 'ma5_ma20_ratio',
    'up_streak', 'dn_streak',
]
assert len(FEATURE_NAMES_31) == 31, f"Expected 31, got {len(FEATURE_NAMES_31)}"


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


def fft_v31(p):
    """10d FFT: Peak1(freq+amp+phase) + Peak2(freq+amp+phase) + Peaks3-10 stats(amp_mean+amp_std+freq_median+freq_range)"""
    x = np.arange(len(p)); t = np.polyfit(x, p, 1); d = p - np.polyval(t, x)
    fp = np.fft.rfft(d); a = np.abs(fp); fq = np.fft.rfftfreq(len(d))
    if len(a) <= 1: return np.zeros(10, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1] + 1
    peaks = []
    for i in pk:
        if i < len(fq): peaks.append((float(fq[i]), float(a[i]), float(np.angle(fp[i]))))
        if len(peaks) >= 10: break
    while len(peaks) < 10: peaks.append((0.0, 0.0, 0.0))

    p1 = peaks[0]; p2 = peaks[1]
    p3_10 = peaks[2:10]
    f3_10 = [p[0] for p in p3_10]; a3_10 = [p[1] for p in p3_10]

    amp_mean = float(np.mean(a3_10)) if a3_10 else 0.0
    amp_std = float(np.std(a3_10)) if len(a3_10) > 1 else 0.0
    freq_median = float(np.median(f3_10)) if f3_10 else 0.0
    freq_range = float(max(f3_10) - min(f3_10)) if len(f3_10) > 1 else 0.0

    return np.array([p1[0], p1[1], p1[2], p2[0], p2[1], p2[2],
                     amp_mean, amp_std, freq_median, freq_range], dtype=np.float32)


def fft_full_30(p):
    """Original 30d FFT"""
    x = np.arange(len(p)); t = np.polyfit(x, p, 1); d = p - np.polyval(t, x)
    fp = np.fft.rfft(d); a = np.abs(fp); fq = np.fft.rfftfreq(len(d))
    if len(a) <= 1: return np.zeros(30, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1][:N_FFT] + 1; fs = []
    for i in pk:
        if i < len(fq): fs.extend([fq[i], a[i], np.angle(fp[i])])
    while len(fs) < 30: fs.extend([0, 0, 0])
    return np.array(fs[:30], dtype=np.float32)


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


def build_features():
    print(f"[{ts()}] Loading data...", flush=True)
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
    print(f"[{ts()}] Building features (61d + 31d)...", flush=True); t0 = time.time()
    flat61_list, flat31_list, y_list, dates_list, inds_list = [], [], [], [], []
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
        e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
        dif = e12 - e26; dea = pd.Series(dif).ewm(span=9).mean().values
        macd_hist = (dif - dea) * 2
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta > 0, delta, 0); loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
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
            up_streak[i] = up_streak[i-1] + 1 if c[i] > c[i-1] else 0
            dn_streak[i] = dn_streak[i-1] + 1 if c[i] < c[i-1] else 0

        for i in range(60, n - 6):
            if c[i] <= 0.01: continue
            if i + 6 >= n: continue
            fwd_ret_t3 = np.clip((c[i+3] - c[i+2]) / np.maximum(abs(c[i+2]), 0.01), -2, 2)

            # ====== 61d full version ======
            g1 = [(c[i]-c[i-j])/max(abs(c[i-j]),0.01) if i>=j else 0 for j in [1,3,6,12]]
            g2 = [(c[i]-ma[i])/max(abs(c[i]),0.01) if not np.isnan(ma[i]) else 0 for ma in [ma5,ma20,ma60]]
            g3 = [dif[i] if not np.isnan(dif[i]) else 0,
                  dea[i] if not np.isnan(dea[i]) else 0,
                  macd_hist[i] if not np.isnan(macd_hist[i]) else 0]
            g4 = [rsi14[i] if not np.isnan(rsi14[i]) else 50,
                  bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5,
                  np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0,
                  atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0,
                  (h[i]-l[i])/max(abs(c[i]),0.01)]
            g5 = [1.0 if c[i]>ma20[i] else 0.0, 1.0 if c[i]>ma60[i] else 0.0]

            # G7 full 14d
            g7_full = [v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0,
                       tr[i] if not np.isnan(tr[i]) else 0,
                       tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
                       (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0,
                       np.log1p(max(v[i],1)),
                       np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0,
                       body_pct[i] if not np.isnan(body_pct[i]) else 0,
                       (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
                       (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0,
                       np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0,
                       ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0,
                       1.0 if c[i]>ma5[i] else 0.0,
                       up_streak[i]/12.0, dn_streak[i]/12.0]

            flat61_list.append(g1 + g2 + g3 + g4 + g5 + fft_full_30(cc[i-60+1:i+1]).tolist() + g7_full)

            # ====== 31d pruned version ======
            # Drop G1(4d), G5(2d), G4 rsi14/bb_pos/amplitude(3d), G7 above_ma5(1d)
            # Keep: G2(3d) + G3(3d) + G4 pruned(vol_6m+atr14=2d) + FFTv31(10d) + G7 slim(13d)
            g4_sel = [np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0,
                      atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0]
            g7_slim = [v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0,
                       tr[i] if not np.isnan(tr[i]) else 0,
                       tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
                       (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0,
                       np.log1p(max(v[i],1)),
                       np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0,
                       body_pct[i] if not np.isnan(body_pct[i]) else 0,
                       (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
                       (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0,
                       np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0,
                       ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0,
                       up_streak[i]/12.0, dn_streak[i]/12.0]  # drop above_ma5

            flat31_list.append(g2 + g3 + g4_sel + fft_v31(cc[i-60+1:i+1]).tolist() + g7_slim)

            y_list.append(fwd_ret_t3)
            dates_list.append(g['month'].iloc[i])
            inds_list.append(industry)

            # T+1~T+6 forward returns
            for lag in range(1, 7):
                if i+lag < n:
                    fr = np.clip((c[i+lag]-c[i+lag-1])/np.maximum(abs(c[i+lag-1]),0.01), -2, 2)
                    fwd_dict[lag].append(fr)
                else:
                    fwd_dict[lag].append(np.nan)

    flat61 = np.array(flat61_list, dtype=np.float32)
    flat31 = np.array(flat31_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates_arr = np.array(dates_list)
    inds_arr = np.array(inds_list)

    v = ~np.isnan(flat61).any(axis=1) & ~np.isnan(flat31).any(axis=1) & ~np.isnan(y)
    flat61 = flat61[v]; flat31 = flat31[v]
    y = y[v]; dates_arr = dates_arr[v]; inds_arr = inds_arr[v]

    fwd_returns = {}
    for lag in range(1, 7):
        fwd_arr = np.array(fwd_dict[lag], dtype=np.float32)
        fwd_returns[lag] = fwd_arr[v]

    print(f"[{ts()}] {len(y):,} samples, 61d={flat61.shape[1]}d 31d={flat31.shape[1]}d ({time.time()-t0:.0f}s)", flush=True)
    return flat61, flat31, y, dates_arr, inds_arr, fwd_returns


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

    # Full period IC
    ic_full, icir, ic_std, ic_pos, _ = cs_ic_full(p_ens, y_te, dates_te)

    # IC Decay T+1~T+6
    decay = []
    for lag in range(1, 7):
        ic_l, icir_l, ic_std_l, ic_pos_l, _ = cs_ic_full(p_ens, fwd_te[lag], dates_te)
        decay.append({'horizon': f'T+{lag}', 'IC': round(ic_l,4), 'IC_std': round(ic_std_l,4),
                       'ICIR': round(icir_l,3), 'IC>0': round(float(ic_pos_l),3)})

    # 5-Fold CV
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
    flat61, flat31, y, dates_arr, inds_arr, fwd_returns = build_features()

    print(f"[{ts()}] Industry neutralizing...", flush=True); t0 = time.time()
    flat61_ind = cross_sectional_neutralize(flat61.copy(), dates_arr, inds_arr, 'categorical')
    flat31_ind = cross_sectional_neutralize(flat31.copy(), dates_arr, inds_arr, 'categorical')
    print(f"[{ts()}] done ({time.time()-t0:.0f}s)", flush=True)

    tr_m = (dates_arr>='2010-01') & (dates_arr<='2014-12')
    te_m = (dates_arr>='2015-01')
    X61_tr=flat61_ind[tr_m]; X61_te=flat61_ind[te_m]
    X31_tr=flat31_ind[tr_m]; X31_te=flat31_ind[te_m]
    y_tr=y[tr_m]; y_te=y[te_m]; dates_te=dates_arr[te_m]
    fwd_te={lag: fwd_returns[lag][te_m] for lag in range(1,7)}

    # ====== Training ======
    print(f"\n{'='*75}")
    print("31d Final vs 61d Baseline")
    print(f"Train: 2010-2014, Test: 2015-01~2025-11, {len(np.unique(dates_te))} test months")
    print(f"{'='*75}")

    r61 = train_and_eval(X61_tr, y_tr, X61_te, y_te, dates_te, fwd_te, "61d")
    print(f"\n[61d] IC={r61['IC']:+.4f}  ICIR={r61['ICIR']:+.3f}  IC_std={r61['IC_std']:.4f}  "
          f"IC>0={r61['IC>0']:.1%}  w=({r61['w_lgb']:.3f},{r61['w_xgb']:.3f},{r61['w_ridge']:.3f})")
    print(f"  CV: mean={r61['cv_mean']:+.4f}  range=[{r61['cv_min']:+.4f},{r61['cv_max']:+.4f}]  "
          f"all_pos={r61['cv_all_pos']}")

    r31 = train_and_eval(X31_tr, y_tr, X31_te, y_te, dates_te, fwd_te, "31d")
    print(f"\n[31d] IC={r31['IC']:+.4f}  ICIR={r31['ICIR']:+.3f}  IC_std={r31['IC_std']:.4f}  "
          f"IC>0={r31['IC>0']:.1%}  w=({r31['w_lgb']:.3f},{r31['w_xgb']:.3f},{r31['w_ridge']:.3f})")
    print(f"  CV: mean={r31['cv_mean']:+.4f}  range=[{r31['cv_min']:+.4f},{r31['cv_max']:+.4f}]  "
          f"all_pos={r31['cv_all_pos']}")

    # ====== IC Decay ======
    print(f"\n{'='*75}")
    print("IC Decay: T+1 ~ T+6 (single-month basis)")
    print(f"{'='*75}")
    print(f"{'Horizon':<8s} {'61d_IC':>8s} {'61d_IR':>8s} {'61d_IC>0':>10s} "
          f"{'31d_IC':>8s} {'31d_IR':>8s} {'31d_IC>0':>10s} {'delta_IC':>8s}")
    print('-'*75)
    for i in range(6):
        d61=r61['decay'][i]; d31=r31['decay'][i]; dic=d31['IC']-d61['IC']
        print(f"{d61['horizon']:<8s} {d61['IC']:+8.4f} {d61['ICIR']:+8.3f} {d61['IC>0']:>9.1%} "
              f"{d31['IC']:+8.4f} {d31['ICIR']:+8.3f} {d31['IC>0']:>9.1%} {dic:+8.4f}")

    # ====== 5-Fold CV Breakdown ======
    print(f"\n{'='*75}")
    print("5-Fold CV Breakdown")
    print(f"{'='*75}")
    for i in range(5):
        c61 = r61['cv_details'][i]; c31 = r31['cv_details'][i]
        print(f"  Fold {c61['fold']} ({c61['start']}~{c61['end']}): "
              f"61d={c61['IC']:+.4f}  31d={c31['IC']:+.4f}")

    # ====== Feature Importance (31d) ======
    print(f"\n{'='*75}")
    print("31d Feature Importance (LGB gain + XGB gain)")
    print(f"{'='*75}")
    lgb_gain = r31['lgb'].feature_importances_
    xgb_gain_dict = r31['xgb'].get_booster().get_score(importance_type='gain')

    imp_rows = []
    for i in range(31):
        xgb_g = xgb_gain_dict.get(f'f{i}', 0)
        imp_rows.append({'idx': i, 'feature': FEATURE_NAMES_31[i],
                         'lgb_gain': round(float(lgb_gain[i]), 4),
                         'xgb_gain': round(float(xgb_g), 4)})
    imp_df = pd.DataFrame(imp_rows).sort_values('lgb_gain', ascending=False)

    # Group summary
    groups_31 = {
        'G2_MA_deviation': range(0, 3), 'G3_MACD': range(3, 6),
        'G4_pruned': range(6, 8),
        'FFT_Peak1': range(8, 11), 'FFT_Peak2': range(11, 14),
        'FFT_P3-10_stats': range(14, 18), 'G7_slim': range(18, 31),
    }
    print(f"\n{'Group':<18s} {'LGB_gain':>10s} {'XGB_gain':>10s} {'N_feat':>6s}")
    print('-'*48)
    for gname, grange in groups_31.items():
        gdf = imp_df[imp_df['idx'].isin(grange)]
        print(f"{gname:<18s} {gdf['lgb_gain'].sum():>10.4f} {gdf['xgb_gain'].sum():>10.4f} {len(gdf):>6d}")

    print(f"\nTop 15 (LGB gain):")
    print(imp_df.head(15)[['feature', 'lgb_gain', 'xgb_gain']].to_string(index=False))

    # ====== Save ======
    delta_ic = r31['IC'] - r61['IC']
    delta_ic_pct = delta_ic / abs(r61['IC']) * 100

    # Decay
    decay_df = pd.DataFrame([{
        'horizon': d61['horizon'],
        'IC_61d': d61['IC'], 'ICIR_61d': d61['ICIR'], 'IC>0_61d': d61['IC>0'],
        'IC_31d': d31['IC'], 'ICIR_31d': d31['ICIR'], 'IC>0_31d': d31['IC>0'],
        'delta_IC': d31['IC']-d61['IC'],
    } for d61, d31 in zip(r61['decay'], r31['decay'])])
    decay_df.to_csv(OUT / 'v31_ic_decay.csv', index=False, encoding='utf-8-sig')

    # Feature importance
    imp_df.to_csv(OUT / 'v31_feature_importance.csv', index=False, encoding='utf-8-sig')

    # Summary
    summary = {
        'baseline_61d': {'IC': r61['IC'], 'ICIR': r61['ICIR'], 'IC_std': r61['IC_std'],
                         'IC>0': r61['IC>0'], 'cv_mean': r61['cv_mean'],
                         'cv_range': [r61['cv_min'], r61['cv_max']]},
        'v31': {'IC': r31['IC'], 'ICIR': r31['ICIR'], 'IC_std': r31['IC_std'],
                'IC>0': r31['IC>0'], 'cv_mean': r31['cv_mean'],
                'cv_range': [r31['cv_min'], r31['cv_max']]},
        'delta_IC': float(delta_ic), 'delta_IC_pct': float(delta_ic_pct),
        'n_features_removed': 30,
        'feature_list_31d': FEATURE_NAMES_31,
        'cv_details_31d': r31['cv_details'],
    }
    with open(OUT / 'v31_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{ts()}] 31d comparison done. delta_IC={delta_ic:+.4f} ({delta_ic_pct:+.1f}%). Results: {OUT}")
