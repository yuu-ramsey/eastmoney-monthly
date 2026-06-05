"""
Phase 5: Pruned feature validation
砍掉: G5全部(2) + G1全部(4) + G4中rsi14/bb_pos/amplitude(3) = 9维
保留: atr14, vol_6m (来自G4)
对比 61维 vs 52维 的IC/ICIR
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

# 61维 → 52维: 砍掉9个特征
# G1(0-3): mom_1m/mom_3m/mom_6m/mom_12m — 全部砍
# G4(10-14): rsi14(10)/bb_pos(11)/amplitude(14) — 砍3个, 保留vol_6m(12)/atr14(13)
# G5(15-16): above_ma20/above_ma60 — 全部砍
DROP_INDICES = {0, 1, 2, 3, 10, 11, 14, 15, 16}
KEEP_INDICES = [i for i in range(61) if i not in DROP_INDICES]

KEPT_NAMES = [
    'ma5_dev', 'ma20_dev', 'ma60_dev',                # G2
    'dif', 'dea', 'macd_hist',                         # G3
    'vol_6m', 'atr14',                                 # G4 (精简)
] + [f'fft_{i}' for i in range(30)] + [                # G6
    'vol_ratio', 'turnover', 'turnover_dev',            # G7
    'vol_ma3_ratio', 'log_volume', 'log_turnover',
    'body_pct', 'price_pos', 'ma_spread',
    'vol_12m', 'ma5_ma20_ratio', 'above_ma5',
    'up_streak', 'dn_streak',
]
assert len(KEPT_NAMES) == 52, f"Expected 52, got {len(KEPT_NAMES)}"


def ts():
    return time.strftime(DATE_FMT)


def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates == m], true[dates == m])[0]
           for m in np.unique(dates) if (dates == m).sum() >= 20]
    ics = np.array(ics)
    return np.mean(ics), np.mean(ics) / np.std(ics) if np.std(ics) > 0 else 0


def fft_f(p):
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
    print(f"[{ts()}] 构建特征...", flush=True); t0 = time.time()
    flat_list, y_list, dates_list, inds_list = [], [], [], []

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
            if i + 3 >= n: continue
            fwd_ret = (c[i + 3] - c[i + 2]) / max(abs(c[i + 2]), 0.01)
            if abs(fwd_ret) > 2: continue

            # 构建全部61维特征
            flat_full = []
            flat_full.extend([(c[i] - c[i - j]) / max(abs(c[i - j]), 0.01) if i >= j else 0 for j in [1, 3, 6, 12]])  # G1:0-3
            for ma in [ma5, ma20, ma60]:
                flat_full.append((c[i] - ma[i]) / max(abs(c[i]), 0.01) if not np.isnan(ma[i]) else 0)  # G2:4-6
            flat_full.extend([dif[i] if not np.isnan(dif[i]) else 0,
                              dea[i] if not np.isnan(dea[i]) else 0,
                              macd_hist[i] if not np.isnan(macd_hist[i]) else 0])  # G3:7-9
            flat_full.append(rsi14[i] if not np.isnan(rsi14[i]) else 50)        # G4:10
            flat_full.append(bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5)     # G4:11
            flat_full.append(np.std(np.diff(c[max(0, i - 6):i + 1]) /
                                    np.maximum(np.abs(c[max(0, i - 5):i + 1]), 0.01)) if i >= 6 else 0)  # G4:12 vol_6m
            flat_full.append(atr14[i] / max(abs(c[i]), 0.01) if not np.isnan(atr14[i]) else 0)  # G4:13
            flat_full.append((h[i] - l[i]) / max(abs(c[i]), 0.01))             # G4:14 amplitude
            flat_full.append(1.0 if c[i] > ma20[i] else 0.0)                   # G5:15
            flat_full.append(1.0 if c[i] > ma60[i] else 0.0)                   # G5:16
            flat_full.extend(fft_f(cc[i - 60 + 1:i + 1]).tolist())             # G6:17-46
            flat_full.extend([v[i] / max(vol_ma12[i], 1) - 1 if i >= 12 and vol_ma12[i] > 0 else 0,
                              tr[i] if not np.isnan(tr[i]) else 0,
                              tr[i] / max(np.mean(tr[max(0, i - 12):i + 1]), 0.001) - 1 if i >= 12 and not np.isnan(tr[i]) else 0,
                              (vol_ma3[i] / max(vol_ma12[i], 1) - 1) if i >= 12 and vol_ma12[i] > 0 else 0,
                              np.log1p(max(v[i], 1)),
                              np.log1p(max(tr[i] * 100, 1)) if not np.isnan(tr[i]) else 0])  # G7:47-52
            flat_full.extend([body_pct[i] if not np.isnan(body_pct[i]) else 0,
                              (c[i] - p5l[i]) / max(p5h[i] - p5l[i], 0.01) if i >= 60 else 0.5,
                              (ma20[i] - ma60[i]) / max(abs(c[i]), 0.01) if i >= 60 else 0,
                              np.std(c[max(0, i - 12):i + 1]) / max(abs(c[i]), 0.01) if i >= 12 else 0,
                              ma5[i] / max(ma20[i], 0.01) - 1 if i >= 20 else 0,
                              1.0 if c[i] > ma5[i] else 0.0,
                              up_streak[i] / 12.0, dn_streak[i] / 12.0])  # G7:53-60

            flat_list.append(flat_full); y_list.append(fwd_ret)
            dates_list.append(g['month'].iloc[i]); inds_list.append(industry)

    flat = np.array(flat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates_arr = np.array(dates_list)
    inds_arr = np.array(inds_list)

    v = ~np.isnan(flat).any(axis=1) & ~np.isnan(y)
    flat = flat[v]; y = y[v]; dates_arr = dates_arr[v]; inds_arr = inds_arr[v]
    print(f"[{ts()}] {len(flat):,} 样本, {flat.shape[1]}d ({time.time() - t0:.0f}s)", flush=True)
    return flat, y, dates_arr, inds_arr


def train_and_eval(X_train, y_train, X_test, y_test, dates_test, label="baseline"):
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

    ic, icir = cs_ic(p_ens, y_test, dates_test)
    return {'label': label, 'IC': ic, 'ICIR': icir,
            'w_lgb': ics_m['LGB'], 'w_xgb': ics_m['XGB'], 'w_ridge': ics_m['Ridge']}


# ====== Main ======
if __name__ == '__main__':
    flat, y, dates_arr, inds_arr = build_features()

    print(f"[{ts()}] 行业中性化...", flush=True); t0 = time.time()
    flat_ind = cross_sectional_neutralize(flat.copy(), dates_arr, inds_arr, 'categorical')
    print(f"[{ts()}] done ({time.time() - t0:.0f}s)", flush=True)

    tr_m = (dates_arr >= '2010-01') & (dates_arr <= '2014-12')
    te_m = (dates_arr >= '2015-01')
    X_train_full = flat_ind[tr_m]; y_train_full = y[tr_m]
    X_test_full = flat_ind[te_m]; y_test = y[te_m]
    dates_test = dates_arr[te_m]

    # Baseline: 全61维
    print(f"\n{'=' * 60}")
    print("Comparison: 61维 vs 52维 (精简)")
    print(f"{'=' * 60}")

    r61 = train_and_eval(X_train_full, y_train_full, X_test_full, y_test, dates_test, "61d_baseline")
    print(f"[61维] IC={r61['IC']:+.4f}  ICIR={r61['ICIR']:+.3f}  "
          f"w=({r61['w_lgb']:.3f}, {r61['w_xgb']:.3f}, {r61['w_ridge']:.3f})")

    # 精简版: 52维
    X_train_52 = X_train_full[:, KEEP_INDICES]
    X_test_52 = X_test_full[:, KEEP_INDICES]

    r52 = train_and_eval(X_train_52, y_train_full, X_test_52, y_test, dates_test, "52d_pruned")
    print(f"[52维] IC={r52['IC']:+.4f}  ICIR={r52['ICIR']:+.3f}  "
          f"w=({r52['w_lgb']:.3f}, {r52['w_xgb']:.3f}, {r52['w_ridge']:.3f})")

    delta_ic = r52['IC'] - r61['IC']
    delta_ir = r52['ICIR'] - r61['ICIR']
    print(f"\nΔIC={delta_ic:+.4f} ({delta_ic / abs(r61['IC']) * 100:+.1f}%)  "
          f"ΔICIR={delta_ir:+.3f}")

    # 按fold验证
    print(f"\n{'=' * 60}")
    print("5-Fold CV 对比 (按时间切分)")
    print(f"{'=' * 60}")
    all_months = np.unique(dates_test)
    n_folds = 5
    fold_size = len(all_months) // n_folds

    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else len(all_months)
        fold_months = all_months[start_idx:end_idx]
        fold_mask = np.isin(dates_test, fold_months)

        if fold_mask.sum() < 100: continue

        ic61, _ = cs_ic(np.zeros(fold_mask.sum()), y_test[fold_mask], dates_test[fold_mask])
        # 用全量训练的模型预测fold内的样本
        sc61 = StandardScaler(); sc61.fit(X_train_full)
        X_fold_61 = sc61.transform(X_test_full[fold_mask])
        sc52 = StandardScaler(); sc52.fit(X_train_52)
        X_fold_52 = sc52.transform(X_test_52[fold_mask])

        # 重新在fold内算IC (用已训好的模型pred)
        # LGB
        lgb61 = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
                                  n_estimators=300, min_child_samples=20, subsample=0.8,
                                  colsample_bytree=0.8, random_state=456, verbosity=-1, n_jobs=4)
        lgb61.fit(sc61.fit_transform(X_train_full), y_train_full)
        p61_lgb = lgb61.predict(X_fold_61)

        lgb52 = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
                                  n_estimators=300, min_child_samples=20, subsample=0.8,
                                  colsample_bytree=0.8, random_state=456, verbosity=-1, n_jobs=4)
        lgb52.fit(sc52.fit_transform(X_train_52), y_train_full)
        p52_lgb = lgb52.predict(X_fold_52)

        ic61_fold, _ = cs_ic(p61_lgb, y_test[fold_mask], dates_test[fold_mask])
        ic52_fold, _ = cs_ic(p52_lgb, y_test[fold_mask], dates_test[fold_mask])

        fold_start = fold_months[0]; fold_end = fold_months[-1]
        delta_fold = ic52_fold - ic61_fold
        print(f"  Fold {fold + 1} ({fold_start}~{fold_end}): "
              f"61d IC={ic61_fold:+.4f}  52d IC={ic52_fold:+.4f}  Δ={delta_fold:+.4f}")

    # Save
    summary = {
        'baseline_61d': {'IC': float(r61['IC']), 'ICIR': float(r61['ICIR'])},
        'pruned_52d': {'IC': float(r52['IC']), 'ICIR': float(r52['ICIR'])},
        'delta_IC': float(delta_ic),
        'delta_IC_pct': float(delta_ic / abs(r61['IC']) * 100),
        'removed_features': ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m',
                             'rsi14', 'bb_pos', 'amplitude',
                             'above_ma20', 'above_ma60'],
        'kept_features': KEPT_NAMES,
        'n_kept': len(KEEP_INDICES),
    }
    with open(OUT / 'prune_validation.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{ts()}] 精简验证done. 结果: {OUT / 'prune_validation.json'}")
