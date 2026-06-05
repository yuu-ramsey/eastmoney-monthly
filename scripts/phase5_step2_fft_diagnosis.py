"""
Phase 5 Step 2: FFT feature deep diagnosis
Step 2.1 — FFT内部消融: 低频峰(1-3) / 中频峰(4-6) / 高频峰(7-10)
Step 2.2 — FFT稳定性检验: 滑动窗口58/60/62根K线的频峰变化
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

# FFT特征在61维中的位置: 索引 17-46 (30维)
# fft_f() Returns 10个峰 × 3值(频率/振幅/相位), 按振幅降序排列
# Peak 1-3 (最强振幅峰): fft_0..fft_8  → 索引 17-25 (9维)
# Peak 4-6:               fft_9..fft_17 → 索引 26-34 (9维)
# Peak 7-10 (最弱振幅峰): fft_18..fft_29 → 索引 35-46 (12维)
FFT_SUBGROUPS = [
    ('FFT_低频峰1-3',  slice(17, 26)),   # 9 features
    ('FFT_中频峰4-6',  slice(26, 35)),   # 9 features
    ('FFT_高频峰7-10', slice(35, 47)),   # 12 features
]


def ts():
    return time.strftime(DATE_FMT)


def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates == m], true[dates == m])[0]
           for m in np.unique(dates) if (dates == m).sum() >= 20]
    ics = np.array(ics)
    return np.mean(ics), np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0


def fft_f(p):
    """与phase4_full.py/phase5_feature_diagnosis.py完全一致"""
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


def fft_f_raw(p):
    """Returns未截断的完整频峰列表 [(freq, amp, phase), ...] 按振幅降序"""
    x = np.arange(len(p))
    t = np.polyfit(x, p, 1)
    d = p - np.polyval(t, x)
    fp = np.fft.rfft(d)
    a = np.abs(fp)
    fq = np.fft.rfftfreq(len(d))
    if len(a) <= 1:
        return []
    pk = np.argsort(a[1:])[::-1] + 1
    result = []
    for i in pk:
        if i < len(fq):
            result.append((float(fq[i]), float(a[i]), float(np.angle(fp[i]))))
    return result


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
        if mask.sum() < 50:
            continue
        if ntype == 'categorical':
            groups = neutralizer[mask]
            for g in np.unique(groups):
                gm = mask & (neutralizer == g)
                if gm.sum() >= 3:
                    neutralized[gm] -= features[gm].mean(axis=0)
    return neutralized


def build_features():
    """构建特征矩阵, 与phase5_feature_diagnosis.py一致"""
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
    print(f"[{ts()}] 构建特征...", flush=True)
    t0 = time.time()
    flat_list, y_list, dates_list, inds_list = [], [], [], []

    for code in codes_used:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        if len(g) < 72:
            continue
        c = g['close'].values.astype(float)
        o = g['open'].values.astype(float)
        h = g['high'].values.astype(float)
        l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n = len(c)
        cc = wd(c)
        industry = ind_map.get(code, 'unknown')
        ma5 = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        e12 = pd.Series(c).ewm(span=12).mean().values
        e26 = pd.Series(c).ewm(span=26).mean().values
        dif = e12 - e26
        dea = pd.Series(dif).ewm(span=9).mean().values
        macd_hist = (dif - dea) * 2
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
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
        up_streak = np.zeros(n)
        dn_streak = np.zeros(n)
        for i in range(1, n):
            up_streak[i] = up_streak[i - 1] + 1 if c[i] > c[i - 1] else 0
            dn_streak[i] = dn_streak[i - 1] + 1 if c[i] < c[i - 1] else 0

        for i in range(60, n - 6):
            if c[i] <= 0.01:
                continue
            if i + 3 >= n:
                continue
            fwd_ret = (c[i + 3] - c[i + 2]) / max(abs(c[i + 2]), 0.01)
            if abs(fwd_ret) > 2:
                continue

            flat = [(c[i] - c[i - j]) / max(abs(c[i - j]), 0.01) if i >= j else 0 for j in [1, 3, 6, 12]]
            for ma in [ma5, ma20, ma60]:
                flat.append((c[i] - ma[i]) / max(abs(c[i]), 0.01) if not np.isnan(ma[i]) else 0)
            flat.extend([dif[i] if not np.isnan(dif[i]) else 0,
                         dea[i] if not np.isnan(dea[i]) else 0,
                         macd_hist[i] if not np.isnan(macd_hist[i]) else 0])
            flat.append(rsi14[i] if not np.isnan(rsi14[i]) else 50)
            flat.append(bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5)
            flat.append(np.std(np.diff(c[max(0, i - 6):i + 1]) / np.maximum(np.abs(c[max(0, i - 5):i + 1]), 0.01)) if i >= 6 else 0)
            flat.append(atr14[i] / max(abs(c[i]), 0.01) if not np.isnan(atr14[i]) else 0)
            flat.append((h[i] - l[i]) / max(abs(c[i]), 0.01))
            flat.append(1.0 if c[i] > ma20[i] else 0.0)
            flat.append(1.0 if c[i] > ma60[i] else 0.0)
            flat.extend(fft_f(cc[i - 60 + 1:i + 1]).tolist())
            flat.extend([v[i] / max(vol_ma12[i], 1) - 1 if i >= 12 and vol_ma12[i] > 0 else 0,
                         tr[i] if not np.isnan(tr[i]) else 0,
                         tr[i] / max(np.mean(tr[max(0, i - 12):i + 1]), 0.001) - 1 if i >= 12 and not np.isnan(tr[i]) else 0,
                         (vol_ma3[i] / max(vol_ma12[i], 1) - 1) if i >= 12 and vol_ma12[i] > 0 else 0,
                         np.log1p(max(v[i], 1)),
                         np.log1p(max(tr[i] * 100, 1)) if not np.isnan(tr[i]) else 0])
            flat.extend([body_pct[i] if not np.isnan(body_pct[i]) else 0,
                         (c[i] - p5l[i]) / max(p5h[i] - p5l[i], 0.01) if i >= 60 else 0.5,
                         (ma20[i] - ma60[i]) / max(abs(c[i]), 0.01) if i >= 60 else 0,
                         np.std(c[max(0, i - 12):i + 1]) / max(abs(c[i]), 0.01) if i >= 12 else 0,
                         ma5[i] / max(ma20[i], 0.01) - 1 if i >= 20 else 0,
                         1.0 if c[i] > ma5[i] else 0.0,
                         up_streak[i] / 12.0, dn_streak[i] / 12.0])

            flat_list.append(flat)
            y_list.append(fwd_ret)
            dates_list.append(g['month'].iloc[i])
            inds_list.append(industry)

    flat = np.array(flat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates_arr = np.array(dates_list)
    inds_arr = np.array(inds_list)

    v = ~np.isnan(flat).any(axis=1) & ~np.isnan(y)
    flat = flat[v]
    y = y[v]
    dates_arr = dates_arr[v]
    inds_arr = inds_arr[v]

    print(f"[{ts()}] {len(flat):,} 样本, {flat.shape[1]}d ({time.time() - t0:.0f}s)", flush=True)
    return flat, y, dates_arr, inds_arr


def train_and_eval(X_train, y_train, X_test, y_test, dates_test, label="baseline"):
    sc = StandardScaler()
    Xt = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    lgb_m = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
                              n_estimators=300, min_child_samples=20, subsample=0.8,
                              colsample_bytree=0.8, random_state=456, verbosity=-1, n_jobs=4)
    lgb_m.fit(Xt, y_train)
    p_lgb = lgb_m.predict(Xte)

    xgb_m = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.05,
                             n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                             random_state=456, verbosity=0, n_jobs=4)
    xgb_m.fit(Xt, y_train)
    p_xgb = xgb_m.predict(Xte)

    ridge_m = Ridge(alpha=1.0)
    ridge_m.fit(Xt, y_train)
    p_ridge = ridge_m.predict(Xte)

    ics_m = {}
    for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]:
        ic_tmp = np.mean([spearmanr(p[dates_test == m], y_test[dates_test == m])[0]
                          for m in np.unique(dates_test) if (dates_test == m).sum() >= 20])
        ics_m[n] = max(ic_tmp, 0)
    w_sum = sum(ics_m.values())
    if w_sum <= 0:
        ics_m, w_sum = {'LGB': 1.0, 'XGB': 1.0, 'Ridge': 1.0}, 3.0
    p_ens = sum(ics_m[n] * p for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]) / w_sum

    ic, icir = cs_ic(p_ens, y_test, dates_test)
    return {'label': label, 'IC': ic, 'ICIR': icir, 'w_lgb': ics_m['LGB'],
            'w_xgb': ics_m['XGB'], 'w_ridge': ics_m['Ridge']}


# ====== Step 2.2: FFT稳定性检验 ======
def fft_stability_analysis():
    """对随机抽样股票, 用58/60/62窗口计算FFT, 看频峰稳定性"""
    print(f"\n{'=' * 60}")
    print("Step 2.2: FFT稳定性检验 (58/60/62窗口)")
    print(f"{'=' * 60}")

    conn = sqlite3.connect(str(DB))
    codes = [r[0] for r in conn.execute(
        'SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
    codes_used = sorted(codes)

    # 随机抽200stocks
    rng = np.random.RandomState(789)
    sampled_codes = rng.choice(codes_used, min(200, len(codes_used)), replace=False)
    print(f"[{ts()}] 抽样 {len(sampled_codes)} stocks for stability test", flush=True)

    params = ','.join('?' * len(sampled_codes))
    df = pd.read_sql_query(
        f"SELECT code,date,close FROM monthly_klines WHERE code IN ({params}) "
        f"AND date>='2005-01' ORDER BY code,date",
        conn, params=list(sampled_codes))
    conn.close()

    # 每个峰位的稳定性统计: [freq_cv, amp_cv] × 10 peaks
    # CV = std/mean across 3 windows, 如果mean≈0则跳过
    peak_freq_cvs = [[] for _ in range(10)]
    peak_amp_cvs = [[] for _ in range(10)]
    # 频峰匹配率: 同rank的峰在3个窗口中是否频率相近(<5%偏离)
    peak_match_rates = [[] for _ in range(10)]

    n_samples = 0
    for code in sampled_codes:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        c = g['close'].values.astype(float)
        n = len(c)
        if n < 63:
            continue

        # 每stocks最多取30个时间点
        test_points = list(range(62, n))
        if len(test_points) > 30:
            test_points = sorted(rng.choice(test_points, 30, replace=False))

        for i in test_points:
            # 3个窗口的FFT
            peaks_58 = fft_f_raw(c[i - 57:i + 1])   # 58-bar
            peaks_60 = fft_f_raw(c[i - 59:i + 1])   # 60-bar (baseline)
            peaks_62 = fft_f_raw(c[i - 61:i + 1])   # 62-bar

            if len(peaks_60) < 10:
                continue
            n_samples += 1

            # 对每个rank位置, 比较3个窗口的峰
            for rank in range(min(10, len(peaks_58), len(peaks_62))):
                freqs = [peaks_58[rank][0], peaks_60[rank][0], peaks_62[rank][0]]
                amps = [peaks_58[rank][1], peaks_60[rank][1], peaks_62[rank][1]]

                # 频率CV
                fmean = np.mean(freqs)
                if fmean > 1e-8:
                    peak_freq_cvs[rank].append(np.std(freqs) / fmean)

                # 振幅CV
                amean = np.mean(amps)
                if amean > 1e-8:
                    peak_amp_cvs[rank].append(np.std(amps) / amean)

                # 匹配率: 3个窗口的频率两两偏差<5%即认为匹配
                f60 = peaks_60[rank][0]
                matches = 0
                for f_other in [peaks_58[rank][0], peaks_62[rank][0]]:
                    if f60 > 1e-8 and abs(f_other - f60) / f60 < 0.05:
                        matches += 1
                peak_match_rates[rank].append(matches / 2.0)  # 0, 0.5, or 1.0

    print(f"[{ts()}] 有效样本: {n_samples} stock-months", flush=True)

    # 汇总
    print(f"\n{'Peak':<10s} {'Freq_CV':>10s} {'Amp_CV':>10s} {'MatchRate':>10s} {'Verdict':>15s}")
    print('-' * 60)
    stability_results = []
    for rank in range(10):
        fc = np.mean(peak_freq_cvs[rank]) * 100 if peak_freq_cvs[rank] else 0
        ac = np.mean(peak_amp_cvs[rank]) * 100 if peak_amp_cvs[rank] else 0
        mr = np.mean(peak_match_rates[rank]) * 100 if peak_match_rates[rank] else 0
        # 判定: Freq_CV<10% 且 MatchRate>80% → 稳定
        if fc < 10 and mr > 80:
            verdict = "STABLE"
        elif fc < 20 and mr > 60:
            verdict = "MARGINAL"
        else:
            verdict = "UNSTABLE"
        stability_results.append({
            'peak_rank': rank + 1,
            'freq_cv_pct': round(fc, 1),
            'amp_cv_pct': round(ac, 1),
            'match_rate_pct': round(mr, 1),
            'verdict': verdict,
        })
        print(f"Peak {rank + 1:<5d} {fc:>9.1f}% {ac:>9.1f}% {mr:>9.1f}% {verdict:>15s}")

    # 按频段汇总
    for group_name, ranks in [('低频峰1-3', [0, 1, 2]), ('中频峰4-6', [3, 4, 5]), ('高频峰7-10', [6, 7, 8, 9])]:
        avg_fc = np.mean([stability_results[r]['freq_cv_pct'] for r in ranks])
        avg_ac = np.mean([stability_results[r]['amp_cv_pct'] for r in ranks])
        avg_mr = np.mean([stability_results[r]['match_rate_pct'] for r in ranks])
        stable_count = sum(1 for r in ranks if stability_results[r]['verdict'] == 'STABLE')
        print(f"  {group_name}: FreqCV={avg_fc:.1f}%  AmpCV={avg_ac:.1f}%  Match={avg_mr:.1f}%  "
              f"Stable={stable_count}/{len(ranks)}")

    # Save
    stab_df = pd.DataFrame(stability_results)
    stab_df.to_csv(OUT / 'step2_2_fft_stability.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUT / 'step2_2_fft_stability.csv'}")

    return stability_results


# ====== Main ======
if __name__ == '__main__':
    # 1. 构建特征
    flat, y, dates_arr, inds_arr = build_features()

    # 2. 行业中性化
    print(f"[{ts()}] 行业中性化...", flush=True)
    t0 = time.time()
    flat_ind = cross_sectional_neutralize(flat.copy(), dates_arr, inds_arr, 'categorical')
    print(f"[{ts()}] done ({time.time() - t0:.0f}s)", flush=True)

    # 3. Split
    tr_m = (dates_arr >= '2010-01') & (dates_arr <= '2014-12')
    te_m = (dates_arr >= '2015-01')
    X_train_full = flat_ind[tr_m]
    y_train_full = y[tr_m]
    X_test = flat_ind[te_m]
    y_test = y[te_m]
    dates_test = dates_arr[te_m]

    # ====== Step 2.1: FFT子频段消融 ======
    print(f"\n{'=' * 60}")
    print("Step 2.1: FFT子频段消融 (低频1-3 / 中频4-6 / 高频7-10)")
    print(f"{'=' * 60}")

    # baseline (全61维)
    result_base = train_and_eval(X_train_full, y_train_full, X_test, y_test, dates_test, "baseline")
    print(f"Baseline (全61维): IC={result_base['IC']:+.4f}, ICIR={result_base['ICIR']:+.3f}")

    ablation_results = [result_base]

    for gname, gslice in FFT_SUBGROUPS:
        mask = np.ones(61, dtype=bool)
        mask[gslice] = False
        idxs = np.where(mask)[0]

        X_train_ab = X_train_full[:, idxs]
        X_test_ab = X_test[:, idxs]

        r = train_and_eval(X_train_ab, y_train_full, X_test_ab, y_test, dates_test,
                           f"drop_{gname}")
        r['features_removed'] = gname
        r['n_features'] = len(idxs)
        ablation_results.append(r)

        delta_ic = r['IC'] - result_base['IC']
        delta_ir = r['ICIR'] - result_base['ICIR']
        n_removed = 61 - len(idxs)
        print(f"  {gname:<20s} ({n_removed}d removed): IC={r['IC']:+.4f} (Δ{delta_ic:+.4f})  "
              f"ICIR={r['ICIR']:+.3f} (Δ{delta_ir:+.3f})")

    # 汇总
    print(f"\n{'Result':<25s} {'IC':>8s} {'ICIR':>8s} {'ΔIC':>8s} {'ΔICIR':>8s} {'ΔIC%':>8s}")
    print('-' * 65)
    for r in ablation_results:
        delta_ic = r['IC'] - result_base['IC']
        delta_ir = r['ICIR'] - result_base['ICIR']
        pct = (delta_ic / abs(result_base['IC']) * 100) if abs(result_base['IC']) > 1e-8 else 0
        print(f"{r['label']:<25s} {r['IC']:+8.4f} {r['ICIR']:+8.3f} {delta_ic:+8.4f} {delta_ir:+8.3f} {pct:+7.1f}%")

    # Save Step 2.1
    ablation_df = pd.DataFrame([{
        'experiment': r['label'],
        'IC': round(r['IC'], 5),
        'ICIR': round(r['ICIR'], 3),
        'delta_IC': round(r['IC'] - result_base['IC'], 5),
        'delta_ICIR': round(r['ICIR'] - result_base['ICIR'], 3),
        'delta_IC_pct': round((r['IC'] - result_base['IC']) / abs(result_base['IC']) * 100, 1),
        'n_features': r.get('n_features', 61),
        'features_removed': r.get('features_removed', ''),
    } for r in ablation_results])
    ablation_df.to_csv(OUT / 'step2_1_fft_ablation.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUT / 'step2_1_fft_ablation.csv'}")

    # ====== Step 2.2: FFT稳定性检验 ======
    stability_results = fft_stability_analysis()

    # 综合保存
    summary = {
        'step2_1_fft_ablation': [{
            'experiment': r['label'],
            'IC': float(r['IC']),
            'ICIR': float(r['ICIR']),
            'delta_IC_vs_baseline': float(r['IC'] - result_base['IC']),
            'delta_IC_pct': float((r['IC'] - result_base['IC']) / abs(result_base['IC']) * 100),
        } for r in ablation_results],
        'step2_2_fft_stability': stability_results,
    }
    with open(OUT / 'step2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{ts()}] Step 2 诊断done. 结果: {OUT}")
