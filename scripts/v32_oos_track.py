"""
v32 OOS (Out-of-Sample) tracking: record monthly predictions+realized returns, rolling IC calculation
Usage:
  每月初:  python scripts/v32_oos_track.py predict   # 生成当月预测
  下月初:  python scripts/v32_oos_track.py realize   # 填写上月实现收益
  随时查:  python scripts/v32_oos_track.py report    # 滚动IC报告

数据文件: .eastmoney-ai/oos/v32_oos_tracker.json
records[]: {predict_month, generated_at, predictions{code:score},
            realized{T+1:{code:ret}, T+2:...}, realized_at}
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json, sys
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb, xgboost as xgb, pywt

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OOS_DIR = PROJECT / '.eastmoney-ai' / 'oos'
OOS_DIR.mkdir(parents=True, exist_ok=True)
OOS_FILE = OOS_DIR / 'v32_oos_tracker.json'
N_FFT = 10
DATE_FMT = '%Y-%m-%d %H:%M:%S'


def ts():
    return time.strftime(DATE_FMT)


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


def build_latest_features():
    """构建latest一个月的特征(仅当月截面)"""
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
    df['month'] = df['date'].str[:7]
    latest_month = df['month'].max()
    print(f"[{ts()}] latest月线: {latest_month}")
    X_list, meta_list = [], []
    for code in sorted(codes_with_ind):
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        i = len(g) - 1  # 最后一行
        if i < 60: continue
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr_rate = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n = len(c); cc = wd(c)
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
        up_s = 0; dn_s = 0
        for j in range(max(1, i-12), i+1):
            up_s = up_s + 1 if c[j] > c[j-1] else 0
            dn_s = dn_s + 1 if c[j] < c[j-1] else 0
        if c[i] <= 0.01: continue
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
                   up_s/12.0, dn_s/12.0]
        features = g2 + g3 + g4_sel + fft_amplitudes(cc[i-60+1:i+1]).tolist() + g7_full
        X_list.append(features)
        meta_list.append({'code': code, 'month': g['month'].iloc[i]})
    return np.array(X_list, dtype=np.float32), pd.DataFrame(meta_list), latest_month


def retrain_model():
    """用History data重训LGB+XGB+Ridge"""
    print(f"[{ts()}] 重训模型...", flush=True)
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
    df['month'] = df['date'].str[:7]
    X_list, y_list = [], []
    for code in sorted(codes_with_ind):
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr_rate = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n = len(c); cc = wd(c)
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
        up_arr = np.zeros(n); dn_arr = np.zeros(n)
        for qi in range(1, n):
            up_arr[qi] = up_arr[qi-1] + 1 if c[qi] > c[qi-1] else 0
            dn_arr[qi] = dn_arr[qi-1] + 1 if c[qi] < c[qi-1] else 0
        for i in range(60, n - 6):
            if c[i] <= 0.01: continue
            if i + 6 >= n: continue
            fwd = np.clip((c[i+3] - c[i+2]) / np.maximum(abs(c[i+2]), 0.01), -2, 2)
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
                       np.log1p(max(v[i],1)), np.log1p(max(tr_rate[i]*100,1)) if not np.isnan(tr_rate[i]) else 0.0,
                       body_pct[i] if not np.isnan(body_pct[i]) else 0.0,
                       (c[i]-p5l[i])/max(p5h[i]-p5l[i],0.01) if i>=60 else 0.5,
                       (ma20[i]-ma60[i])/max(abs(c[i]),0.01) if i>=60 else 0.0,
                       np.std(c[max(0,i-12):i+1])/max(abs(c[i]),0.01) if i>=12 else 0.0,
                       ma5[i]/max(ma20[i],0.01)-1 if i>=20 else 0.0,
                       1.0 if c[i]>ma5[i] else 0.0,
                       up_arr[i]/12.0, dn_arr[i]/12.0]
            X_list.append(g2 + g3 + g4_sel + fft_amplitudes(cc[i-60+1:i+1]).tolist() + g7_full)
            y_list.append(fwd)
    X = np.array(X_list, dtype=np.float32); y = np.array(y_list, dtype=np.float32)
    v = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[v]; y = y[v]
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    lgb_m = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
                              n_estimators=300, min_child_samples=20, subsample=0.8,
                              colsample_bytree=0.8, random_state=456, verbosity=-1, n_jobs=4)
    lgb_m.fit(Xs, y)
    xgb_m = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.05,
                             n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                             random_state=456, verbosity=0, n_jobs=4)
    xgb_m.fit(Xs, y)
    ridge_m = Ridge(alpha=1.0); ridge_m.fit(Xs, y)
    print(f"[{ts()}] 训练done ({len(y):,} 样本)", flush=True)
    return sc, lgb_m, xgb_m, ridge_m


def cmd_predict():
    """生成当月预测并追加到tracker"""
    print(f"[{ts()}] ===== OOS Predict =====")
    X_latest, meta, latest_month = build_latest_features()
    sc, lgb_m, xgb_m, ridge_m = retrain_model()
    Xs = sc.transform(X_latest)
    p_lgb = lgb_m.predict(Xs); p_xgb = xgb_m.predict(Xs); p_ridge = ridge_m.predict(Xs)
    p_ens = (p_lgb + p_xgb + p_ridge) / 3.0
    predictions = {}
    for i, row in meta.iterrows():
        predictions[row['code']] = round(float(p_ens[i]), 6)
    if OOS_FILE.exists():
        with open(OOS_FILE) as f: tracker = json.load(f)
    else:
        tracker = {'version': 'v32', 'start_month': latest_month, 'records': []}
    for rec in tracker['records']:
        if rec['predict_month'] == latest_month:
            print(f"  覆盖: {latest_month} 已存在")
            tracker['records'].remove(rec); break
    tracker['records'].append({
        'predict_month': latest_month, 'generated_at': ts(), 'predictions': predictions,
    })
    with open(OOS_FILE, 'w') as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)
    top10 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  当月: {latest_month}  |  {len(predictions)} 只股票")
    print(f"  Top 10: {top10}")
    print(f"  保存: {OOS_FILE}")


def cmd_realize():
    """填写上月预测的实现收益"""
    print(f"[{ts()}] ===== OOS Realize =====")
    if not OOS_FILE.exists():
        print("  ERROR: tracker 不存在"); return
    with open(OOS_FILE) as f: tracker = json.load(f)
    target = None
    for rec in reversed(tracker['records']):
        if 'realized' not in rec:
            target = rec; break
    if target is None:
        print("  所有记录已有实现收益"); return
    predict_month = target['predict_month']
    print(f"  填写 {predict_month} 实现收益...")
    conn = sqlite3.connect(str(DB))
    codes_str = ','.join(f"'{c}'" for c in target['predictions'].keys())
    df = pd.read_sql_query(
        f"SELECT code, date, close FROM monthly_klines "
        f"WHERE code IN ({codes_str}) AND date >= '{predict_month}' "
        f"ORDER BY code, date", conn)
    conn.close()
    realized = {f'T+{lag}': {} for lag in range(1, 7)}
    for code in target['predictions']:
        g = df[df['code'] == code].sort_values('date')
        if len(g) < 7: continue
        dates_m = g['date'].str[:7].values
        matches = np.where(dates_m == predict_month)[0]
        if len(matches) == 0: continue
        base_idx = matches[0]
        base_c = g.iloc[base_idx]['close']
        if base_c <= 0: continue
        for lag_idx, lag in enumerate(range(1, 7)):
            if base_idx + lag_idx + 1 < len(g):
                next_c = g.iloc[base_idx + lag_idx + 1]['close']
                if next_c > 0:
                    realized[f'T+{lag}'][code] = round(float((next_c - base_c) / abs(base_c)), 6)
    target['realized'] = realized; target['realized_at'] = ts()
    with open(OOS_FILE, 'w') as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)
    for lag in [1, 2, 3]:
        print(f"  T+{lag}: {len(realized[f'T+{lag}'])} 只")


def cmd_report():
    """滚动IC报告"""
    print(f"[{ts()}] ===== OOS 滚动IC =====")
    if not OOS_FILE.exists():
        print("  tracker 为空"); return
    with open(OOS_FILE) as f: tracker = json.load(f)
    complete = [r for r in tracker['records'] if 'realized' in r]
    print(f"  总: {len(tracker['records'])} 月, 已实现: {len(complete)}")
    if not complete:
        print("  暂无实现收益"); return
    print(f"\n{'Month':<10s} {'T+1_IC':>8s} {'T+1_N':>8s} {'T+3_IC':>8s} {'T+6_IC':>8s}")
    print('-'*46)
    all_t1, all_t3 = [], []
    for rec in complete:
        month = rec['predict_month']; preds = rec['predictions']; rea = rec['realized']
        vals = {}
        for lag in [1, 3, 6]:
            key = f'T+{lag}'
            common = list(set(preds.keys()) & set(rea.get(key, {}).keys()))
            if len(common) >= 20:
                p = [preds[c] for c in common]; r = [rea[key][c] for c in common]
                ic = spearmanr(p, r)[0]
                vals[lag] = {'IC': ic, 'n': len(common)}
            else:
                vals[lag] = {'IC': np.nan, 'n': 0}
        t1_ic = vals[1]['IC']; t3_ic = vals[3]['IC']; t6_ic = vals[6]['IC']
        if not np.isnan(t1_ic): all_t1.append(t1_ic)
        if not np.isnan(t3_ic): all_t3.append(t3_ic)
        print(f"{month:<10s} {t1_ic:+8.4f} {vals[1]['n']:>8d} {t3_ic:+8.4f} {t6_ic:+8.4f}")
    print(f"\n汇总:")
    for label, ics in [('T+1', all_t1), ('T+3', all_t3)]:
        if ics:
            a = np.array(ics)
            print(f"  {label}: mean={a.mean():+.4f}  ICIR={a.mean()/a.std():+.3f}  "
                  f"IC>0={np.mean(a>0):.1%}  n={len(a)}")
    print(f"  文件: {OOS_FILE}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    {'predict': cmd_predict, 'realize': cmd_realize, 'report': cmd_report}[cmd]()
