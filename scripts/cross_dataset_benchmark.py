"""Multi-dataset × multi-model comparison (no Qlib needed, pure sklearn+LightGBM)
数据集: A股月线(2247只) + SPY月线(Yahoo) + Fama-French 3/5因子(Ken French)
模型: Ridge / Linear / LightGBM / MLP / RandomForest
指标: IC / ICIR / Hit Rate
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json, zipfile, io
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

OUT = Path('.eastmoney-ai/benchmark')
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. A股月线
# ═══════════════════════════════════════════════════════════════════════════

def load_ashare():
    db = '.eastmoney-ai/db/klines-v2.sqlite'
    conn = sqlite3.connect(db)
    codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
    ind_map = {r[0]:r[1] for r in conn.execute('SELECT stock_code, industry_code FROM stock_industry_mapping')}
    params = ','.join('?'*len(codes))
    df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date", conn, params=codes)
    conn.close()
    rows = []
    for code in codes:
        g = df[df['code']==code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        dates = g['date'].tolist(); industry = ind_map.get(code, 'unknown')
        ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values; ma60 = pd.Series(c).rolling(60).mean().values
        e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
        dif = e12-e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif-dea)*2
        delta = np.diff(c, prepend=c[0]); gain = np.where(delta>0,delta,0); loss = np.where(delta<0,-delta,0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values; avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
        rsi14 = np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)), 50)
        bb_std = pd.Series(c).rolling(20).std().values; bb_pos = np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01), 0.5)
        trange = np.maximum(h-l, np.abs(h-np.roll(c,1))); atr14 = pd.Series(trange).rolling(14).mean().values
        for i in range(60, len(g)-6):
            if c[i] <= 0.01: continue
            fwd = (c[i+3]-c[i])/c[i]
            if abs(fwd) > 2: continue
            rows.append({'date':dates[i],'fwd_ret':fwd,
                'r1':(c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0,
                'r3':(c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0,
                'r6':(c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0,
                'r12':(c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0,
                'ma5d':(c[i]-ma5[i])/max(abs(c[i]),0.01), 'ma20d':(c[i]-ma20[i])/max(abs(c[i]),0.01),
                'ma60d':(c[i]-ma60[i])/max(abs(c[i]),0.01), 'macd_dif':dif[i],'macd_dea':dea[i],'macd_hist':macd_hist[i],
                'rsi14':rsi14[i],'bb_pos':bb_pos[i],
                'vol_6':np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0,
                'atr_ratio':atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0,
                'hilo':(h[i]-l[i])/max(abs(c[i]),0.01),
                'above_ma20':1.0 if c[i]>ma20[i] else 0.0, 'above_ma60':1.0 if c[i]>ma60[i] else 0.0,
            })
    data = pd.DataFrame(rows).dropna()
    feat_cols = [c for c in data.columns if c not in ['date','fwd_ret']]
    train = data[(data['date']>='2015-01')&(data['date']<='2021-12')]
    test = data[data['date']>='2024-01']
    return train, test, feat_cols, f'A股月线({len(codes)}只)'

# ═══════════════════════════════════════════════════════════════════════════
# 2. SPY 月线
# ═══════════════════════════════════════════════════════════════════════════

def load_spy():
    df = pd.read_csv(OUT/'spy_monthly.csv', parse_dates=['Date'])
    df.columns = [c.lower() for c in df.columns]
    c = df['close'].values.astype(float); o = df['open'].values.astype(float)
    h = df['high'].values.astype(float); l = df['low'].values.astype(float)
    v = df['volume'].values.astype(float); n = len(c)
    ma5 = pd.Series(c).rolling(5).mean().values; ma20 = pd.Series(c).rolling(20).mean().values; ma60 = pd.Series(c).rolling(60).mean().values
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    dif = e12-e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif-dea)*2
    delta = np.diff(c, prepend=c[0]); gain = np.where(delta>0,delta,0); loss = np.where(delta<0,-delta,0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values; avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
    rsi14 = np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)), 50)
    bb_std = pd.Series(c).rolling(20).std().values; bb_pos = np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01), 0.5)
    trange = np.maximum(h-l, np.abs(h-np.roll(c,1))); atr14 = pd.Series(trange).rolling(14).mean().values
    rows = []
    for i in range(60, n-6):
        if c[i] <= 0.01: continue
        fwd = (c[i+3]-c[i])/c[i]
        if abs(fwd) > 2: continue
        rows.append({'date':str(df['date'].iloc[i])[:10],'fwd_ret':fwd,
            'r1':(c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0,
            'r3':(c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0,
            'r6':(c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0,
            'r12':(c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0,
            'ma5d':(c[i]-ma5[i])/max(abs(c[i]),0.01), 'ma20d':(c[i]-ma20[i])/max(abs(c[i]),0.01),
            'ma60d':(c[i]-ma60[i])/max(abs(c[i]),0.01), 'macd_dif':dif[i],'macd_dea':dea[i],'macd_hist':macd_hist[i],
            'rsi14':rsi14[i],'bb_pos':bb_pos[i],
            'vol_6':np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0,
            'atr_ratio':atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0,
            'hilo':(h[i]-l[i])/max(abs(c[i]),0.01),
            'above_ma20':1.0 if c[i]>ma20[i] else 0.0, 'above_ma60':1.0 if c[i]>ma60[i] else 0.0,
        })
    data = pd.DataFrame(rows).dropna()
    feat_cols = [c for c in data.columns if c not in ['date','fwd_ret']]
    split = int(len(data)*0.7)
    return data.iloc[:split], data.iloc[split:], feat_cols, f'SPY月线({len(data)}条)'

# ═══════════════════════════════════════════════════════════════════════════
# 3. Fama-French 因子预测
# ═══════════════════════════════════════════════════════════════════════════

def load_ff_factors():
    """用 FF3 因子历史值预测未来市场超额收益"""
    f = OUT/'ff3_factors.csv'
    if not f.exists(): return None
    lines = f.read_text().strip().split('\n')
    # 找到数据起始行
    data_start = 0
    header_idx = 0
    for i, line in enumerate(lines):
        if 'Mkt-RF' in line and 'SMB' in line:
            header_idx = i
            data_start = i+1
            break
    header = lines[header_idx].strip().split(',')
    # 找到年/月列
    rows = []
    for i in range(data_start, len(lines)):
        parts = lines[i].strip().split(',')
        if len(parts) < 4: continue
        try:
            mkt_rf = float(parts[1])/100
            smb = float(parts[2])/100
            hml = float(parts[3])/100
            rf = float(parts[4])/100 if len(parts)>4 else 0
            rows.append({'mkt_rf':mkt_rf, 'smb':smb, 'hml':hml, 'rf':rf})
        except: continue
    if len(rows) < 36: return None
    # 构建滚动特征（过去12个月的因子值+波动率）
    df = pd.DataFrame(rows)
    for lag in [1,3,6,12]:
        df[f'mkt_lag{lag}'] = df['mkt_rf'].shift(lag)
        df[f'smb_lag{lag}'] = df['smb'].shift(lag)
        df[f'hml_lag{lag}'] = df['hml'].shift(lag)
    df['mkt_vol12'] = df['mkt_rf'].rolling(12).std()
    df['smb_vol12'] = df['smb'].rolling(12).std()
    # 前向收益
    df['fwd_ret'] = df['mkt_rf'].shift(-3)  # 预测3个月后
    df['date'] = range(len(df))
    df = df.dropna()
    feat_cols = [c for c in df.columns if c not in ['date','fwd_ret']]
    split = int(len(df)*0.7)
    return df.iloc[:split], df.iloc[split:], feat_cols, f'Fama-French3因子({len(df)}月)'

# ═══════════════════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════════════════

def eval_model(train, test, feat_cols, ds_name):
    X_tr = train[feat_cols].values.astype(float); y_tr = train['fwd_ret'].values.astype(float)
    X_te = test[feat_cols].values.astype(float); y_te = test['fwd_ret'].values.astype(float)
    sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    results = {}
    for name, model_cls, use_scaled in [
        ('Ridge', Ridge(alpha=1.0,random_state=456), True),
        ('Linear', LinearRegression(), True),
        ('LightGBM', lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,n_estimators=200,min_child_samples=10,subsample=0.8,colsample_bytree=0.8,random_state=456,verbosity=-1,n_jobs=4), False),
        ('MLP(64,32)', MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,batch_size=64,max_iter=300,early_stopping=True,random_state=456), True),
        ('RandomForest', RandomForestRegressor(n_estimators=200,max_depth=8,min_samples_leaf=10,random_state=456,n_jobs=4), False),
    ]:
        t0 = time.time()
        try:
            Xt = X_tr_s if use_scaled else X_tr
            Xe = X_te_s if use_scaled else X_te
            model_cls.fit(Xt, y_tr)
            pred = model_cls.predict(Xe)
            # 截面IC（仅多资产）
            if ds_name.startswith('A股'):
                ics = [spearmanr(pred[test['date']==m], test.loc[test['date']==m,'fwd_ret'].values)[0] for m in test['date'].unique() if (test['date']==m).sum()>=20]
            else:
                ic = spearmanr(pred, y_te)[0]
                ics = [ic] if not np.isnan(ic) else []
            avg_ic = np.mean(ics) if ics else np.nan
            hit = np.mean((pred>0) == (y_te>0))
            results[name] = {'IC':round(float(avg_ic),4), 'ICIR':round(float(avg_ic/np.std(ics)) if ics and np.std(ics)>0 else np.nan,4), 'hit_rate':round(float(hit),3), 'time_s':round(time.time()-t0,1), 'n_months':len(ics)}
        except Exception as e:
            results[name] = {'error':str(e)}
    return results

# ═══════════════════════════════════════════════════════════════════════════
# Main flow
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print('='*70)
    print('多数据集 × 多模型 IC 对比')
    print('='*70)

    datasets = {}
    for loader, key in [(load_ashare,'A股月线'), (load_spy,'SPY月线'), (load_ff_factors,'FF3因子')]:
        try:
            print(f'\n加载 {key}...')
            result = loader()
            if result is None:
                print(f'  数据不可用，跳过')
                continue
            train, test, feats, label = result
            print(f'  {label} | 训练{len(train):,} 测试{len(test):,} 特征{len(feats)}')
            datasets[key] = (train, test, feats, label)
        except Exception as e:
            print(f'  failed: {e}')

    all_results = {}
    for ds_key, (train, test, feats, ds_label) in datasets.items():
        print('\n' + '─'*70)
        print(f'数据集: {ds_label}')
        r = eval_model(train, test, feats, ds_key)
        all_results[ds_key] = r
        for name, m in r.items():
            if 'error' in m: print(f'  {name:<15s} ERROR: {m[\"error\"]}')
            else: print(f'  {name:<15s} IC={m[\"IC\"]:+.4f} ICIR={m[\"ICIR\"]:+.4f} hit={m[\"hit_rate\"]:.3f} ({m[\"time_s\"]}s)')

    # 汇总
    print(f'\n{\"=\"*70}')
    print('IC 矩阵')
    model_names = ['Ridge','Linear','LightGBM','MLP(64,32)','RandomForest']
    header = f'{\"模型\":<15s}'
    for ds in datasets: header += f' {ds:<18s}'
    print(header)
    print('-'*len(header))
    for mn in model_names:
        row = f'{mn:<15s}'
        for ds in datasets:
            ic = all_results[ds].get(mn,{}).get('IC',np.nan)
            row += f' {ic:+.4f}  ' if not np.isnan(ic) else ' N/A   '
        print(row)

    with open(OUT/'cross_dataset_results.json','w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n保存到 {OUT/\"cross_dataset_results.json\"}')

if __name__ == '__main__':
    main()
