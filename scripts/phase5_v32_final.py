"""
Phase 5: 32d final vs 31d vs 61d baseline three-way comparison
32维 = FFT振幅10d + G7全14d(含above_ma5) + 均线偏离3d + MACD3d + G4精选2d(ATR+vol_6m)
31维 = FFTv31(10d freq+amp+phase hybrid) + G7精简13d(删above_ma5) + 同上
统一: 5-Fold CV + T+1~T+6衰减 + IC_IR + 行业中性化
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

# ====== 特征名定义 ======
FEATURE_NAMES_32 = [
    # G2 均线偏离 (0-2)
    'ma5_dev', 'ma20_dev', 'ma60_dev',
    # G3 MACD (3-5)
    'dif', 'dea', 'macd_hist',
    # G4精选 (6-7)
    'vol_6m', 'atr14',
    # FFT振幅10维 (8-17)
    'fft_amp_0', 'fft_amp_1', 'fft_amp_2', 'fft_amp_3', 'fft_amp_4',
    'fft_amp_5', 'fft_amp_6', 'fft_amp_7', 'fft_amp_8', 'fft_amp_9',
    # G7全14维 (18-31)
    'vol_ratio', 'turnover', 'turnover_dev',
    'vol_ma3_ratio', 'log_volume', 'log_turnover',
    'body_pct', 'price_pos', 'ma_spread',
    'vol_12m', 'ma5_ma20_ratio',
    'above_ma5', 'up_streak', 'dn_streak',
]
assert len(FEATURE_NAMES_32) == 32, f"Expected 32, got {len(FEATURE_NAMES_32)}"

FEATURE_NAMES_31 = [
    'ma5_dev', 'ma20_dev', 'ma60_dev',                       # G2 均线偏离 (0-2)
    'dif', 'dea', 'macd_hist',                                # G3 MACD (3-5)
    'vol_6m', 'atr14',                                        # G4精选 (6-7)
    'fft_p1_freq', 'fft_p1_amp', 'fft_p1_phase',             # FFT Peak1 (8-10)
    'fft_p2_freq', 'fft_p2_amp', 'fft_p2_phase',             # FFT Peak2 (11-13)
    'fft_amp_mean', 'fft_amp_std',                            # FFT P3-10统计 (14-17)
    'fft_freq_median', 'fft_freq_range',
    'vol_ratio', 'turnover', 'turnover_dev',                  # G7精简13d (18-30)
    'vol_ma3_ratio', 'log_volume', 'log_turnover',
    'body_pct', 'price_pos', 'ma_spread',
    'vol_12m', 'ma5_ma20_ratio',
    'up_streak', 'dn_streak',
]
assert len(FEATURE_NAMES_31) == 31


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


# ====== FFT 函数 ======

def fft_amplitudes(p):
    """10维简单振幅谱"""
    x = np.arange(len(p)); t = np.polyfit(x, p, 1); d = p - np.polyval(t, x)
    fp = np.fft.rfft(d); a = np.abs(fp)
    if len(a) <= 1: return np.zeros(N_FFT, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1][:N_FFT] + 1
    amps = [float(a[i]) for i in pk]
    while len(amps) < N_FFT: amps.append(0.0)
    return np.array(amps[:N_FFT], dtype=np.float32)


def fft_v31(p):
    """31维版FFT: Peak1(freq+amp+phase) + Peak2(freq+amp+phase) + P3-10统计"""
    x = np.arange(len(p)); t = np.polyfit(x, p, 1); d = p - np.polyval(t, x)
    fp = np.fft.rfft(d); a = np.abs(fp); fq = np.fft.rfftfreq(len(d))
    if len(a) <= 1: return np.zeros(10, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1] + 1
    peaks = []
    for i in pk:
        if i < len(fq): peaks.append((float(fq[i]), float(a[i]), float(np.angle(fp[i]))))
        if len(peaks) >= 10: break
    while len(peaks) < 10: peaks.append((0.0, 0.0, 0.0))
    p1 = peaks[0]; p2 = peaks[1]; p3_10 = peaks[2:10]
    f3_10 = [p[0] for p in p3_10]; a3_10 = [p[1] for p in p3_10]
    amp_mean = float(np.mean(a3_10)) if a3_10 else 0.0
    amp_std = float(np.std(a3_10)) if len(a3_10) > 1 else 0.0
    freq_median = float(np.median(f3_10)) if f3_10 else 0.0
    freq_range = float(max(f3_10) - min(f3_10)) if len(f3_10) > 1 else 0.0
    return np.array([p1[0], p1[1], p1[2], p2[0], p2[1], p2[2],
                     amp_mean, amp_std, freq_median, freq_range], dtype=np.float32)


def fft_full_30(p):
    """原61维中的30维FFT: 10峰×(freq+amp+phase)"""
    x = np.arange(len(p)); t = np.polyfit(x, p, 1); d = p - np.polyval(t, x)
    fp = np.fft.rfft(d); a = np.abs(fp); fq = np.fft.rfftfreq(len(d))
    if len(a) <= 1: return np.zeros(30, dtype=np.float32)
    pk = np.argsort(a[1:])[::-1][:N_FFT] + 1; fs = []
    for i in pk:
        if i < len(fq): fs.extend([float(fq[i]), float(a[i]), float(np.angle(fp[i]))])
    while len(fs) < 30: fs.extend([0.0, 0.0, 0.0])
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
    """一次构建 61d / 31d / 32d 三套特征"""
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
    print(f"[{ts()}] 构建特征 (61d + 31d + 32d)...", flush=True); t0 = time.time()
    flat61_list, flat31_list, flat32_list, y_list, dates_list, inds_list = [], [], [], [], [], []
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
        dif_arr = e12 - e26; dea_arr = pd.Series(dif_arr).ewm(span=9).mean().values
        macd_hist = (dif_arr - dea_arr) * 2
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

            # ====== 共通特征计算 ======
            # G1 价格动量 (61d专用)
            g1 = [(c[i]-c[i-j])/max(abs(c[i-j]),0.01) if i>=j else 0.0 for j in [1,3,6,12]]
            # G2 均线偏离
            g2 = [(c[i]-ma[i])/max(abs(c[i]),0.01) if not np.isnan(ma[i]) else 0.0 for ma in [ma5,ma20,ma60]]
            # G3 MACD
            g3 = [dif_arr[i] if not np.isnan(dif_arr[i]) else 0.0,
                  dea_arr[i] if not np.isnan(dea_arr[i]) else 0.0,
                  macd_hist[i] if not np.isnan(macd_hist[i]) else 0.0]
            # G4 完整5维 (61d专用)
            g4_full = [rsi14[i] if not np.isnan(rsi14[i]) else 50.0,
                       bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5,
                       np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0.0,
                       atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0.0,
                       (h[i]-l[i])/max(abs(c[i]),0.01)]
            # G4 精选 (仅ATR+vol_6m, 31d/32d共用)
            g4_sel = [np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0.0,
                      atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0.0]
            # G5 趋势二值 (61d专用)
            g5 = [1.0 if c[i]>ma20[i] else 0.0, 1.0 if c[i]>ma60[i] else 0.0]

            # G7 完整14维 (61d+32d共用, 含 above_ma5)
            g7_full = [v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0.0,
                       tr[i] if not np.isnan(tr[i]) else 0.0,
                       tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0.0,
                       (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0.0,
                       np.log1p(max(v[i],1)),
                       np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0.0,
                       body_pct[i] if not np.isnan(body_pct[i]) else 0.0,
                       (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
                       (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0.0,
                       np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0.0,
                       ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0.0,
                       1.0 if c[i]>ma5[i] else 0.0,  # above_ma5
                       up_streak[i]/12.0, dn_streak[i]/12.0]

            # G7 精简13维 (31d专用, 删 above_ma5)
            g7_slim = [v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0.0,
                       tr[i] if not np.isnan(tr[i]) else 0.0,
                       tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0.0,
                       (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0.0,
                       np.log1p(max(v[i],1)),
                       np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0.0,
                       body_pct[i] if not np.isnan(body_pct[i]) else 0.0,
                       (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
                       (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0.0,
                       np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0.0,
                       ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0.0,
                       up_streak[i]/12.0, dn_streak[i]/12.0]

            window = cc[i-60+1:i+1]

            # 61维 = G1(4)+G2(3)+G3(3)+G4(5)+G5(2)+FFT30+G7全(14)
            flat61_list.append(g1 + g2 + g3 + g4_full + g5 + fft_full_30(window).tolist() + g7_full)

            # 31维 = G2(3)+G3(3)+G4精选(2)+FFTv31(10)+G7精简(13)
            flat31_list.append(g2 + g3 + g4_sel + fft_v31(window).tolist() + g7_slim)

            # 32维 = G2(3)+G3(3)+G4精选(2)+FFT振幅(10)+G7全(14)
            flat32_list.append(g2 + g3 + g4_sel + fft_amplitudes(window).tolist() + g7_full)

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
    flat32 = np.array(flat32_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates_arr = np.array(dates_list)
    inds_arr = np.array(inds_list)

    v = ~np.isnan(flat61).any(axis=1) & ~np.isnan(flat31).any(axis=1) & ~np.isnan(flat32).any(axis=1) & ~np.isnan(y)
    flat61 = flat61[v]; flat31 = flat31[v]; flat32 = flat32[v]
    y = y[v]; dates_arr = dates_arr[v]; inds_arr = inds_arr[v]

    fwd_returns = {}
    for lag in range(1, 7):
        fwd_arr = np.array(fwd_dict[lag], dtype=np.float32)
        fwd_returns[lag] = fwd_arr[v]

    print(f"[{ts()}] {len(y):,} 样本, 61d={flat61.shape[1]}d 31d={flat31.shape[1]}d 32d={flat32.shape[1]}d ({time.time()-t0:.0f}s)", flush=True)
    return flat61, flat31, flat32, y, dates_arr, inds_arr, fwd_returns


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

    # 全期 IC
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
    flat61, flat31, flat32, y, dates_arr, inds_arr, fwd_returns = build_features()

    print(f"[{ts()}] 行业中性化...", flush=True); t0 = time.time()
    flat61_ind = cross_sectional_neutralize(flat61.copy(), dates_arr, inds_arr, 'categorical')
    flat31_ind = cross_sectional_neutralize(flat31.copy(), dates_arr, inds_arr, 'categorical')
    flat32_ind = cross_sectional_neutralize(flat32.copy(), dates_arr, inds_arr, 'categorical')
    print(f"[{ts()}] done ({time.time()-t0:.0f}s)", flush=True)

    tr_m = (dates_arr>='2010-01') & (dates_arr<='2014-12')
    te_m = (dates_arr>='2015-01')
    X61_tr=flat61_ind[tr_m]; X61_te=flat61_ind[te_m]
    X31_tr=flat31_ind[tr_m]; X31_te=flat31_ind[te_m]
    X32_tr=flat32_ind[tr_m]; X32_te=flat32_ind[te_m]
    y_tr=y[tr_m]; y_te=y[te_m]; dates_te=dates_arr[te_m]
    fwd_te={lag: fwd_returns[lag][te_m] for lag in range(1,7)}

    # ====== 训练 ======
    print(f"\n{'='*75}")
    print("32维最终版 vs 31维 vs 61维baseline 三方对比")
    print(f"Train: 2010-2014, Test: 2015-01~2025-11, {len(np.unique(dates_te))} 测试月")
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

    r32 = train_and_eval(X32_tr, y_tr, X32_te, y_te, dates_te, fwd_te, "32d")
    print(f"\n[32d] IC={r32['IC']:+.4f}  ICIR={r32['ICIR']:+.3f}  IC_std={r32['IC_std']:.4f}  "
          f"IC>0={r32['IC>0']:.1%}  w=({r32['w_lgb']:.3f},{r32['w_xgb']:.3f},{r32['w_ridge']:.3f})")
    print(f"  CV: mean={r32['cv_mean']:+.4f}  range=[{r32['cv_min']:+.4f},{r32['cv_max']:+.4f}]  "
          f"all_pos={r32['cv_all_pos']}")

    # ====== 汇总表 ======
    print(f"\n{'='*100}")
    print("三方对比汇总")
    print(f"{'='*100}")
    print(f"{'Version':<8s} {'Dim':>4s} {'IC':>8s} {'ICIR':>8s} {'IC_std':>8s} {'IC>0':>8s} "
          f"{'CV_mean':>8s} {'CV_min':>8s} {'CV_max':>8s} {'CV_all+':>8s}")
    print('-'*100)
    for r in [r61, r31, r32]:
        print(f"{r['label']:<8s} {r['n_features']:>4d} {r['IC']:+8.4f} {r['ICIR']:+8.3f} "
              f"{r['IC_std']:>8.4f} {r['IC>0']:>7.1%} "
              f"{r['cv_mean']:+8.4f} {r['cv_min']:+8.4f} {r['cv_max']:+8.4f} "
              f"{str(r['cv_all_pos']):>8s}")

    # Δ vs 61d
    print(f"\n{'Version':<8s} {'ΔIC':>8s} {'ΔIC%':>8s} {'ΔICIR':>8s} {'ΔCV_mean':>8s}")
    print('-'*50)
    for r in [r31, r32]:
        dic = r['IC'] - r61['IC']
        dic_pct = dic / abs(r61['IC']) * 100 if r61['IC'] != 0 else 0
        dcv = r['cv_mean'] - r61['cv_mean']
        print(f"{r['label']:<8s} {dic:+8.4f} {dic_pct:+7.1f}% {r['ICIR']-r61['ICIR']:+8.3f} {dcv:+8.4f}")

    # ====== IC Decay T+1~T+6 ======
    print(f"\n{'='*100}")
    print("IC Decay: T+1 ~ T+6 (单月口径)")
    print(f"{'='*100}")
    hdr = f"{'Horizon':<8s}"
    for r in [r61, r31, r32]:
        hdr += f" {r['label']+'_IC':>8s} {r['label']+'_IR':>8s} {r['label']+'_IC>0':>10s}"
    print(hdr)
    print('-'*100)
    for i in range(6):
        line = f"{r61['decay'][i]['horizon']:<8s}"
        for r in [r61, r31, r32]:
            d = r['decay'][i]
            line += f" {d['IC']:+8.4f} {d['ICIR']:+8.3f} {d['IC>0']:>9.1%}"
        print(line)

    # ΔIC per horizon (vs 61d)
    print(f"\n{'Horizon':<8s} {'Δ31-61':>10s} {'Δ32-61':>10s} {'Δ32-31':>10s}")
    print('-'*42)
    for i in range(6):
        h = r61['decay'][i]['horizon']
        d31 = r31['decay'][i]['IC'] - r61['decay'][i]['IC']
        d32 = r32['decay'][i]['IC'] - r61['decay'][i]['IC']
        d32_31 = r32['decay'][i]['IC'] - r31['decay'][i]['IC']
        print(f"{h:<8s} {d31:+10.4f} {d32:+10.4f} {d32_31:+10.4f}")

    # ====== 5-Fold CV 展开 ======
    print(f"\n{'='*80}")
    print("5-Fold CV 展开")
    print(f"{'='*80}")
    for i in range(5):
        c61 = r61['cv_details'][i]; c31 = r31['cv_details'][i]; c32 = r32['cv_details'][i]
        print(f"Fold {c61['fold']} ({c61['start']}~{c61['end']}): "
              f"61d={c61['IC']:+.4f}  31d={c31['IC']:+.4f}  32d={c32['IC']:+.4f}")

    # ====== 32维Feature importance ======
    print(f"\n{'='*75}")
    print("32维Feature importance (LGB gain + XGB gain)")
    print(f"{'='*75}")
    lgb_gain = r32['lgb'].feature_importances_
    xgb_gain_dict = r32['xgb'].get_booster().get_score(importance_type='gain')

    imp_rows = []
    for i in range(32):
        xgb_g = xgb_gain_dict.get(f'f{i}', 0)
        imp_rows.append({'idx': i, 'feature': FEATURE_NAMES_32[i],
                         'lgb_gain': round(float(lgb_gain[i]), 4),
                         'xgb_gain': round(float(xgb_g), 4)})
    imp_df = pd.DataFrame(imp_rows).sort_values('lgb_gain', ascending=False)

    # 分组汇总
    groups_32 = {
        'G2_均线偏离': range(0, 3), 'G3_MACD': range(3, 6),
        'G4_精选': range(6, 8),
        'FFT_振幅10d': range(8, 18), 'G7_全14d': range(18, 32),
    }
    print(f"\n{'Group':<18s} {'LGB_gain':>10s} {'XGB_gain':>10s} {'N_feat':>6s}")
    print('-'*48)
    for gname, grange in groups_32.items():
        gdf = imp_df[imp_df['idx'].isin(grange)]
        print(f"{gname:<18s} {gdf['lgb_gain'].sum():>10.4f} {gdf['xgb_gain'].sum():>10.4f} {len(gdf):>6d}")

    print(f"\nTop 15 (LGB gain):")
    print(imp_df.head(15)[['feature', 'lgb_gain', 'xgb_gain']].to_string(index=False))

    # ====== 保存 ======
    # Decay
    decay_rows = []
    for i in range(6):
        decay_rows.append({
            'horizon': r61['decay'][i]['horizon'],
            'IC_61d': r61['decay'][i]['IC'], 'ICIR_61d': r61['decay'][i]['ICIR'], 'IC>0_61d': r61['decay'][i]['IC>0'],
            'IC_31d': r31['decay'][i]['IC'], 'ICIR_31d': r31['decay'][i]['ICIR'], 'IC>0_31d': r31['decay'][i]['IC>0'],
            'IC_32d': r32['decay'][i]['IC'], 'ICIR_32d': r32['decay'][i]['ICIR'], 'IC>0_32d': r32['decay'][i]['IC>0'],
        })
    pd.DataFrame(decay_rows).to_csv(OUT / 'v32_ic_decay.csv', index=False, encoding='utf-8-sig')

    # Feature importance
    imp_df.to_csv(OUT / 'v32_feature_importance.csv', index=False, encoding='utf-8-sig')

    # Summary
    summary = {}
    for r, names, fft_desc in [(r61, None, 'FFT30d(freq+amp+phase)'),
                                (r31, FEATURE_NAMES_31, 'FFTv31(Peak1+Peak2+P3-10统计)'),
                                (r32, FEATURE_NAMES_32, 'FFT振幅10d')]:
        summary[r['label']] = {
            'n_features': r['n_features'],
            'IC': r['IC'], 'ICIR': r['ICIR'], 'IC_std': r['IC_std'], 'IC>0': r['IC>0'],
            'weights': {'LGB': r['w_lgb'], 'XGB': r['w_xgb'], 'Ridge': r['w_ridge']},
            'cv_mean': r['cv_mean'], 'cv_range': [r['cv_min'], r['cv_max']],
            'cv_all_pos': r['cv_all_pos'], 'cv_details': r['cv_details'],
            'decay': r['decay'],
            'fft_desc': fft_desc,
        }
        if names is not None:
            summary[r['label']]['feature_names'] = names

    with open(OUT / 'v32_triple_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    delta_31 = r31['IC'] - r61['IC']
    delta_32 = r32['IC'] - r61['IC']
    delta_32_vs_31 = r32['IC'] - r31['IC']
    print(f"\n[{ts()}] 三方对比done.")
    print(f"  31d vs 61d: ΔIC={delta_31:+.4f} ({delta_31/abs(r61['IC'])*100:+.1f}%)")
    print(f"  32d vs 61d: ΔIC={delta_32:+.4f} ({delta_32/abs(r61['IC'])*100:+.1f}%)")
    print(f"  32d vs 31d: ΔIC={delta_32_vs_31:+.4f}")
    print(f"  结果: {OUT}")
