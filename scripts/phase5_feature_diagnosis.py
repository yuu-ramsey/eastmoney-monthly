"""
Phase 5: 特征诊断
Step 1.1 — 特征重要性 (LightGBM gain/split, XGBoost gain/weight)
Step 1.2 — 逐组消融实验 (7组, 每次去掉一组重训, 对比IC/ICIR)
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

def cs_ic(pred, true, dates):
    ics = [spearmanr(pred[dates==m], true[dates==m])[0]
           for m in np.unique(dates) if (dates==m).sum()>=20]
    ics = np.array(ics)
    return np.mean(ics), np.mean(ics)/np.std(ics) if np.std(ics)>0 else 0

def fft_f(p):
    x=np.arange(len(p)); t=np.polyfit(x,p,1); d=p-np.polyval(t,x)
    fp=np.fft.rfft(d); a=np.abs(fp); fq=np.fft.rfftfreq(len(d))
    if len(a)<=1: return np.zeros(N_FFT*3, dtype=np.float32)
    pk=np.argsort(a[1:])[::-1][:N_FFT]+1; fs=[]
    for i in pk:
        if i<len(fq): fs.extend([fq[i], a[i], np.angle(fp[i])])
    while len(fs)<N_FFT*3: fs.extend([0,0,0])
    return np.array(fs[:N_FFT*3], dtype=np.float32)

def wd(s):
    c=pywt.wavedec(s, 'db4', level=2)
    sigma=np.median(np.abs(c[-1]))/0.6745; th=sigma*np.sqrt(2*np.log(len(s)))
    cd=[c[0]]+[pywt.threshold(cf, th, mode='soft') for cf in c[1:]]
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

# ====== 特征组定义 ======
FEATURE_GROUPS = [
    ('G1_价格动量',  slice(0, 4)),
    ('G2_均线偏离',  slice(4, 7)),
    ('G3_MACD',     slice(7, 10)),
    ('G4_技术指标',  slice(10, 15)),
    ('G5_趋势二值',  slice(15, 17)),
    ('G6_FFT频率',  slice(17, 47)),
    ('G7_量价K线',  slice(47, 61)),
]

# 特征名称
FEATURE_NAMES = [
    'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m',           # G1: 0-3
    'ma5_dev', 'ma20_dev', 'ma60_dev',                    # G2: 4-6
    'dif', 'dea', 'macd_hist',                            # G3: 7-9
    'rsi14', 'bb_pos', 'vol_6m', 'atr14', 'amplitude',  # G4: 10-14
    'above_ma20', 'above_ma60',                           # G5: 15-16
] + [f'fft_{i}' for i in range(30)] + [                   # G6: 17-46
    'vol_ratio', 'turnover', 'turnover_dev',              # G7: 47-52
    'vol_ma3_ratio', 'log_volume', 'log_turnover',
    'body_pct', 'price_pos', 'ma_spread',                 # G7: 53-60
    'vol_12m', 'ma5_ma20_ratio', 'above_ma5',
    'up_streak', 'dn_streak',
]

assert len(FEATURE_NAMES) == 61, f"Expected 61, got {len(FEATURE_NAMES)}"


def build_features():
    """构建特征矩阵，与phase4_full.py一致"""
    print(f"[{ts()}] 加载数据...", flush=True)
    conn = sqlite3.connect(str(DB))
    codes = [r[0] for r in conn.execute(
        'SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
    ind_map = {r[0]: r[1] for r in conn.execute(
        'SELECT stock_code, industry_code FROM stock_industry_mapping')}
    codes_with_ind = [c for c in codes if c in ind_map]
    params = ','.join('?'*len(codes_with_ind))
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
        g = df[df['code']==code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        c=g['close'].values.astype(float); o=g['open'].values.astype(float)
        h=g['high'].values.astype(float); l=g['low'].values.astype(float)
        v=g['volume'].values.astype(float)
        tr=g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n=len(c); cc=wd(c); industry=ind_map.get(code, 'unknown')
        ma5=pd.Series(c).rolling(5).mean().values; ma20=pd.Series(c).rolling(20).mean().values
        ma60=pd.Series(c).rolling(60).mean().values
        e12=pd.Series(c).ewm(span=12).mean().values; e26=pd.Series(c).ewm(span=26).mean().values
        dif=e12-e26; dea=pd.Series(dif).ewm(span=9).mean().values; macd_hist=(dif-dea)*2
        delta=np.diff(c, prepend=c[0]); gain=np.where(delta>0, delta, 0); loss=np.where(delta<0, -delta, 0)
        avg_gain=pd.Series(gain).ewm(alpha=1/14).mean().values
        avg_loss=pd.Series(loss).ewm(alpha=1/14).mean().values
        rsi14=np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)), 50)
        bb_std=pd.Series(c).rolling(20).std().values
        bb_pos=np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std, 0.01), 0.5)
        trange=np.maximum(h-l, np.abs(h-np.roll(c,1)))
        atr14=pd.Series(trange).rolling(14).mean().values
        vol_ma3=pd.Series(v).rolling(3).mean().values; vol_ma12=pd.Series(v).rolling(12).mean().values
        p5h=pd.Series(c).rolling(60).max().values; p5l=pd.Series(c).rolling(60).min().values
        body_pct=np.abs(c-o)/np.maximum(h-l, 0.01)
        up_streak=np.zeros(n); dn_streak=np.zeros(n)
        for i in range(1,n): up_streak[i]=up_streak[i-1]+1 if c[i]>c[i-1] else 0; dn_streak[i]=dn_streak[i-1]+1 if c[i]<c[i-1] else 0
        for i in range(60, n-6):
            if c[i]<=0.01: continue
            # T+3 单月收益作为标签
            if i+3 >= n: continue
            fwd_ret = (c[i+3]-c[i+2])/max(abs(c[i+2]), 0.01)
            if abs(fwd_ret) > 2: continue

            flat=[(c[i]-c[i-j])/max(abs(c[i-j]),0.01) if i>=j else 0 for j in [1,3,6,12]]
            for ma in [ma5, ma20, ma60]: flat.append((c[i]-ma[i])/max(abs(c[i]),0.01) if not np.isnan(ma[i]) else 0)
            flat.extend([dif[i] if not np.isnan(dif[i]) else 0,
                         dea[i] if not np.isnan(dea[i]) else 0,
                         macd_hist[i] if not np.isnan(macd_hist[i]) else 0])
            flat.append(rsi14[i] if not np.isnan(rsi14[i]) else 50)
            flat.append(bb_pos[i] if not np.isnan(bb_pos[i]) else 0.5)
            flat.append(np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0)
            flat.append(atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0)
            flat.append((h[i]-l[i])/max(abs(c[i]),0.01))
            flat.append(1.0 if c[i]>ma20[i] else 0.0)
            flat.append(1.0 if c[i]>ma60[i] else 0.0)
            flat.extend(fft_f(cc[i-60+1:i+1]).tolist())
            flat.extend([v[i]/max(vol_ma12[i],1)-1 if i>=12 and vol_ma12[i]>0 else 0,
                         tr[i] if not np.isnan(tr[i]) else 0,
                         tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
                         (vol_ma3[i]/max(vol_ma12[i],1)-1) if i>=12 and vol_ma12[i]>0 else 0,
                         np.log1p(max(v[i],1)),
                         np.log1p(max(tr[i]*100,1)) if not np.isnan(tr[i]) else 0])
            flat.extend([body_pct[i] if not np.isnan(body_pct[i]) else 0,
                         (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
                         (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0,
                         np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0,
                         ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0,
                         1.0 if c[i]>ma5[i] else 0.0,
                         up_streak[i]/12.0, dn_streak[i]/12.0])

            flat_list.append(flat); y_list.append(fwd_ret)
            dates_list.append(g['month'].iloc[i]); inds_list.append(industry)

    flat = np.array(flat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates_arr = np.array(dates_list)
    inds_arr = np.array(inds_list)

    v = ~np.isnan(flat).any(axis=1) & ~np.isnan(y)
    flat = flat[v]; y = y[v]; dates_arr = dates_arr[v]; inds_arr = inds_arr[v]

    print(f"[{ts()}] {len(flat):,} 样本, {flat.shape[1]}d ({time.time()-t0:.0f}s)", flush=True)
    return flat, y, dates_arr, inds_arr


def train_and_eval(X_train, y_train, X_test, y_test, dates_test, label="baseline"):
    """训练LGB+XGB+Ridge集成, 返回IC和模型"""
    sc = StandardScaler()
    Xt = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    lgb_m = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
        n_estimators=300, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=456, verbosity=-1, n_jobs=4)
    lgb_m.fit(Xt, y_train); p_lgb = lgb_m.predict(Xte)

    xgb_m = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.05,
        n_estimators=300, subsample=0.8, colsample_bytree=0.8,
        random_state=456, verbosity=0, n_jobs=4)
    xgb_m.fit(Xt, y_train); p_xgb = xgb_m.predict(Xte)

    ridge_m = Ridge(alpha=1.0); ridge_m.fit(Xt, y_train); p_ridge = ridge_m.predict(Xte)

    # IC加权
    ics_m = {}
    for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]:
        ic_tmp = np.mean([spearmanr(p[dates_test==m], y_test[dates_test==m])[0]
                          for m in np.unique(dates_test) if (dates_test==m).sum()>=20])
        ics_m[n] = max(ic_tmp, 0)
    w_sum = sum(ics_m.values())
    if w_sum <= 0: ics_m, w_sum = {'LGB': 1.0, 'XGB': 1.0, 'Ridge': 1.0}, 3.0
    p_ens = sum(ics_m[n]*p for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]) / w_sum

    ic, icir = cs_ic(p_ens, y_test, dates_test)
    return {'label': label, 'IC': ic, 'ICIR': icir,
            'lgb': lgb_m, 'xgb': xgb_m, 'ridge': ridge_m,
            'w_lgb': ics_m['LGB'], 'w_xgb': ics_m['XGB'], 'w_ridge': ics_m['Ridge']}


# ====== Main ======
if __name__ == '__main__':
    # 1. 构建特征
    flat, y, dates_arr, inds_arr = build_features()

    # 2. 行业中性化
    print(f"[{ts()}] 行业中性化...", flush=True); t0 = time.time()
    flat_ind = cross_sectional_neutralize(flat.copy(), dates_arr, inds_arr, 'categorical')
    print(f"[{ts()}] 完成 ({time.time()-t0:.0f}s)", flush=True)

    # 3. Split
    tr_m = (dates_arr >= '2010-01') & (dates_arr <= '2014-12')
    te_m = (dates_arr >= '2015-01')
    X_train_full = flat_ind[tr_m]; y_train_full = y[tr_m]
    X_test = flat_ind[te_m]; y_test = y[te_m]
    dates_test = dates_arr[te_m]

    # ====== Step 1.1: 特征重要性 ======
    print(f"\n{'='*60}")
    print("Step 1.1: 特征重要性分析")
    print(f"{'='*60}")

    result = train_and_eval(X_train_full, y_train_full, X_test, y_test, dates_test, "baseline")
    print(f"Baseline: IC={result['IC']:+.4f}, ICIR={result['ICIR']:+.3f}")

    # LightGBM importance
    lgb_imp_gain = result['lgb'].feature_importances_
    lgb_imp_split = result['lgb'].booster_.feature_importance(importance_type='split')

    # XGBoost importance
    xgb_imp_gain = result['xgb'].feature_importances_
    xgb_imp_weight = result['xgb'].get_booster().get_score(importance_type='weight')
    xgb_imp_gain_dict = result['xgb'].get_booster().get_score(importance_type='gain')

    # Build importance table
    imp_rows = []
    for i in range(61):
        xgb_gain = xgb_imp_gain_dict.get(f'f{i}', 0)
        xgb_weight = xgb_imp_weight.get(f'f{i}', 0)
        imp_rows.append({
            'idx': i, 'feature': FEATURE_NAMES[i],
            'group': [g[0] for g in FEATURE_GROUPS if i in range(g[1].start, g[1].stop)][0] if any(i in range(g[1].start, g[1].stop) for g in FEATURE_GROUPS) else 'unknown',
            'lgb_gain': round(lgb_imp_gain[i], 6),
            'lgb_split': int(lgb_imp_split[i]),
            'xgb_gain': round(xgb_gain, 6),
            'xgb_weight': int(xgb_weight),
        })
    imp_df = pd.DataFrame(imp_rows)

    # 按组汇总
    print(f"\n{'Group':<16s} {'LGB_gain':>10s} {'LGB_split':>10s} {'XGB_gain':>10s} {'XGB_weight':>12s}")
    print('-' * 60)
    for gname, gslice in FEATURE_GROUPS:
        gdf = imp_df.iloc[gslice]
        print(f"{gname:<16s} {gdf['lgb_gain'].sum():>10.4f} {gdf['lgb_split'].sum():>10d} "
              f"{gdf['xgb_gain'].sum():>10.4f} {gdf['xgb_weight'].sum():>12d}")

    # Top 20 individual features
    print(f"\nTop 20 features (by LGB gain):")
    print(imp_df.nlargest(20, 'lgb_gain')[['feature', 'group', 'lgb_gain', 'lgb_split', 'xgb_gain', 'xgb_weight']].to_string(index=False))

    imp_df.to_csv(OUT / 'step1_1_feature_importance.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUT / 'step1_1_feature_importance.csv'}")

    # ====== Step 1.2: 逐组消融 ======
    print(f"\n{'='*60}")
    print("Step 1.2: 逐组消融实验")
    print(f"{'='*60}")

    ablation_results = [result]  # baseline first

    for gname, gslice in FEATURE_GROUPS:
        # Build feature mask (remove this group)
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

        delta_ic = r['IC'] - result['IC']
        delta_ir = r['ICIR'] - result['ICIR']
        print(f"  {gname:<16s}: IC={r['IC']:+.4f} (Δ{delta_ic:+.4f})  "
              f"ICIR={r['ICIR']:+.3f} (Δ{delta_ir:+.3f})  n_feat={len(idxs)}")

    # Summary
    print(f"\n{'Result':<20s} {'IC':>8s} {'ICIR':>8s} {'ΔIC':>8s} {'ΔICIR':>8s}")
    print('-' * 56)
    for r in ablation_results:
        delta_ic = r['IC'] - result['IC']
        delta_ir = r['ICIR'] - result['ICIR']
        print(f"{r['label']:<20s} {r['IC']:+8.4f} {r['ICIR']:+8.3f} {delta_ic:+8.4f} {delta_ir:+8.3f}")

    ablation_df = pd.DataFrame([{
        'experiment': r['label'],
        'IC': round(r['IC'], 5),
        'ICIR': round(r['ICIR'], 3),
        'delta_IC': round(r['IC'] - result['IC'], 5),
        'delta_ICIR': round(r['ICIR'] - result['ICIR'], 3),
        'n_features': r.get('n_features', 61),
        'features_removed': r.get('features_removed', ''),
    } for r in ablation_results])
    ablation_df.to_csv(OUT / 'step1_2_ablation.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUT / 'step1_2_ablation.csv'}")

    # 保存完整结果JSON
    summary = {
        'baseline_IC': float(result['IC']),
        'baseline_ICIR': float(result['ICIR']),
        'top_features_lgb_gain': imp_df.nlargest(10, 'lgb_gain')[['feature', 'group', 'lgb_gain']].to_dict('records'),
        'group_importance': [{
            'group': gname,
            'lgb_gain_sum': float(imp_df.iloc[gslice]['lgb_gain'].sum()),
            'xgb_gain_sum': float(imp_df.iloc[gslice]['xgb_gain'].sum()),
        } for gname, gslice in FEATURE_GROUPS],
        'ablation': [{
            'experiment': r['label'],
            'IC': float(r['IC']),
            'ICIR': float(r['ICIR']),
            'delta_IC_vs_baseline': float(r['IC'] - result['IC']),
        } for r in ablation_results],
    }
    with open(OUT / 'diagnosis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{ts()}] 诊断完成. 结果: {OUT}")
