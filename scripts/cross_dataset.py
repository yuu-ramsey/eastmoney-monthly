# cross_dataset.py - multi-dataset multi-model IC comparison
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

OUT = Path('.eastmoney-ai/benchmark')
OUT.mkdir(parents=True, exist_ok=True)

# ---- helpers ----
def make_features(c, o, h, l, v):
    n = len(c)
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
    avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100 - 100/(1 + avg_gain/np.maximum(avg_loss, 1e-8)), 50)
    bb_std = pd.Series(c).rolling(20).std().values
    bb_pos = np.nan_to_num((c - (ma20 - 2*bb_std)) / np.maximum(4*bb_std, 0.01), 0.5)
    trange = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
    atr14 = pd.Series(trange).rolling(14).mean().values
    rows = []
    for i in range(60, n - 6):
        if c[i] <= 0.01:
            continue
        fwd = (c[i+3] - c[i]) / c[i]
        if abs(fwd) > 2:
            continue
        rows.append({
            'fwd_ret': fwd,
            'r1': (c[i]-c[i-1])/max(abs(c[i-1]), 0.01) if i >= 1 else 0,
            'r3': (c[i]-c[i-3])/max(abs(c[i-3]), 0.01) if i >= 3 else 0,
            'r6': (c[i]-c[i-6])/max(abs(c[i-6]), 0.01) if i >= 6 else 0,
            'r12': (c[i]-c[i-12])/max(abs(c[i-12]), 0.01) if i >= 12 else 0,
            'ma5d': (c[i]-ma5[i])/max(abs(c[i]), 0.01),
            'ma20d': (c[i]-ma20[i])/max(abs(c[i]), 0.01),
            'ma60d': (c[i]-ma60[i])/max(abs(c[i]), 0.01),
            'macd_dif': dif[i], 'macd_dea': dea[i], 'macd_hist': macd_hist[i],
            'rsi14': rsi14[i], 'bb_pos': bb_pos[i],
            'vol_6': np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]), 0.01)) if i >= 6 else 0,
            'atr_ratio': atr14[i]/max(abs(c[i]), 0.01) if not np.isnan(atr14[i]) else 0,
            'hilo': (h[i]-l[i])/max(abs(c[i]), 0.01),
            'above_ma20': 1.0 if c[i] > ma20[i] else 0.0,
            'above_ma60': 1.0 if c[i] > ma60[i] else 0.0,
        })
    return rows

# ---- datasets ----
def load_ashare():
    db = '.eastmoney-ai/db/klines-v2.sqlite'
    conn = sqlite3.connect(db)
    codes = [r[0] for r in conn.execute(
        'SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84'
    ).fetchall()]
    params = ','.join('?' * len(codes))
    df = pd.read_sql_query(
        f"SELECT code,date,open,high,low,close,volume,turnover_rate "
        f"FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' "
        f"ORDER BY code,date", conn, params=codes
    )
    conn.close()
    rows = []
    for code in codes:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        if len(g) < 72:
            continue
        c = g['close'].values.astype(float)
        o = g['open'].values.astype(float)
        h = g['high'].values.astype(float)
        l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        feat_rows = make_features(c, o, h, l, v)
        for idx, r in enumerate(feat_rows):
            r['date'] = g['date'].iloc[60 + idx]
        rows.extend(feat_rows)
    data = pd.DataFrame(rows).dropna()
    feat_cols = [c for c in data.columns if c not in ['date', 'fwd_ret']]
    train = data[(data['date'] >= '2015-01') & (data['date'] <= '2021-12')]
    test = data[data['date'] >= '2024-01']
    return train, test, feat_cols, f'A-share ({len(codes)} stocks)'

def load_spy():
    f = OUT / 'spy_monthly.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=['Date'])
    df.columns = [c.lower() for c in df.columns]
    c = df['close'].values.astype(float)
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    v = df['volume'].values.astype(float)
    rows = make_features(c, o, h, l, v)
    data = pd.DataFrame(rows).dropna()
    feat_cols = [c for c in data.columns if c not in ['fwd_ret']]
    split = int(len(data) * 0.7)
    return data.iloc[:split], data.iloc[split:], feat_cols, f'SPY ({len(data)} rows)'

def load_ff3():
    f = OUT / 'ff3_factors.csv'
    if not f.exists():
        return None
    text = f.read_text().strip().split('\n')
    data_start = 0
    for i, line in enumerate(text):
        if 'Mkt-RF' in line and 'SMB' in line:
            data_start = i + 1
            break
    rows = []
    for i in range(data_start, len(text)):
        parts = text[i].strip().split(',')
        if len(parts) < 4:
            continue
        try:
            rows.append({
                'mkt_rf': float(parts[1]) / 100,
                'smb': float(parts[2]) / 100,
                'hml': float(parts[3]) / 100,
                'rf': float(parts[4]) / 100 if len(parts) > 4 else 0,
            })
        except ValueError:
            continue
    if len(rows) < 36:
        return None
    df = pd.DataFrame(rows)
    for lag in [1, 3, 6, 12]:
        df[f'mkt_lag{lag}'] = df['mkt_rf'].shift(lag)
        df[f'smb_lag{lag}'] = df['smb'].shift(lag)
        df[f'hml_lag{lag}'] = df['hml'].shift(lag)
    df['mkt_vol12'] = df['mkt_rf'].rolling(12).std()
    df['smb_vol12'] = df['smb'].rolling(12).std()
    df['fwd_ret'] = df['mkt_rf'].shift(-3)
    df = df.dropna()
    feat_cols = [c for c in df.columns if c not in ['fwd_ret']]
    split = int(len(df) * 0.7)
    return df.iloc[:split], df.iloc[split:], feat_cols, f'FF3 ({len(df)} months)'

# ---- main ----
def main():
    print('=' * 60)
    print('Multi-Dataset Multi-Model IC Comparison')
    print('=' * 60)

    datasets = {}
    for name, loader in [('A-share', load_ashare), ('SPY', load_spy), ('FF3', load_ff3)]:
        print(f'\nLoading {name}...')
        try:
            result = loader()
            if result is None:
                print(f'  Data not available')
                continue
            train, test, feats, label = result
            print(f'  {label}: train={len(train):,} test={len(test):,} features={len(feats)}')
            datasets[name] = (train, test, feats)
        except Exception as e:
            print(f'  Failed: {e}')

    all_results = {}
    for ds_name, (train, test, feats) in datasets.items():
        print(f'\n{"-"*60}')
        print(f'Dataset: {ds_name}')
        X_tr = train[feats].values.astype(float)
        y_tr = train['fwd_ret'].values.astype(float)
        X_te = test[feats].values.astype(float)
        y_te = test['fwd_ret'].values.astype(float)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        results = {}
        for name, model_cls, use_scaled in [
            ('Ridge      ', Ridge(alpha=1.0, random_state=456), True),
            ('Linear     ', LinearRegression(), True),
            ('LightGBM   ', lgb.LGBMRegressor(objective='regression', num_leaves=63,
                learning_rate=0.03, n_estimators=200, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8, random_state=456,
                verbosity=-1, n_jobs=4), False),
            ('MLP(64,32) ', MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                alpha=0.001, batch_size=64, max_iter=300, early_stopping=True,
                random_state=456), True),
            ('RandomForest', RandomForestRegressor(n_estimators=200, max_depth=8,
                min_samples_leaf=10, random_state=456, n_jobs=4), False),
        ]:
            t0 = time.time()
            try:
                Xt = X_tr_s if use_scaled else X_tr
                Xe = X_te_s if use_scaled else X_te
                model_cls.fit(Xt, y_tr)
                pred = model_cls.predict(Xe)
                if ds_name == 'A-share' and 'date' in test.columns:
                    ics = []
                    for m in test['date'].unique():
                        mask = test['date'] == m
                        if mask.sum() >= 20:
                            ics.append(spearmanr(
                                pred[mask], test.loc[mask, 'fwd_ret'].values
                            )[0])
                else:
                    ic = spearmanr(pred, y_te)[0]
                    ics = [ic] if not np.isnan(ic) else []
                avg_ic = np.mean(ics) if ics else np.nan
                icir = avg_ic / np.std(ics) if ics and np.std(ics) > 0 else np.nan
                hit = np.mean((pred > 0) == (y_te > 0))
                results[name.strip()] = {
                    'IC': round(float(avg_ic), 4),
                    'ICIR': round(float(icir), 4),
                    'hit': round(float(hit), 3),
                    'time': round(time.time() - t0, 1),
                }
                print(f'  {name} IC={avg_ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f}')
            except Exception as e:
                print(f'  {name} ERROR: {e}')
                results[name.strip()] = {'error': str(e)}
        all_results[ds_name] = results

    # Summary
    print(f'\n{"="*60}')
    print('IC Matrix')
    sep = '-' * 60
    header = f'{"Model":<14s}'
    for ds in datasets:
        header += f' {ds:>14s}'
    print(header)
    print(sep)
    for mn in ['Ridge', 'Linear', 'LightGBM', 'MLP(64,32)', 'RandomForest']:
        row = f'{mn:<14s}'
        for ds in datasets:
            ic = all_results[ds].get(mn, {}).get('IC', np.nan)
            row += f' {ic:+10.4f}  ' if not np.isnan(ic) else '       N/A    '
        print(row)

    with open(OUT / 'cross_dataset_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved: {OUT / "cross_dataset_results.json"}')

if __name__ == '__main__':
    main()
