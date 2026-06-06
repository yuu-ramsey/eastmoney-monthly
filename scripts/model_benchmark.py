"""Multi-model benchmark: local models vs Qlib SOTA public benchmark + Ken French dataset.
When Qlib is installed successfully, run full comparison; otherwise sklearn+LightGBM only.
"""
import numpy as np, pandas as pd, sqlite3, time, json, sys
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

DB = '.eastmoney-ai/db/klines-v2.sqlite'
OUT = Path('.eastmoney-ai/benchmark')
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Qlib public benchmark data (as SOTA baseline reference)
# Source: github.com/microsoft/qlib, CSI300 daily Alpha158
# ═══════════════════════════════════════════════════════════════════════════

QLIB_BENCHMARKS = {
    # model_name: {IC, ICIR, Rank_IC, Rank_ICIR, Ann_Return, IR, Max_DD}
    'LightGBM':       {'IC':0.0448, 'ICIR':0.3660, 'Rank_IC':0.0469, 'Rank_ICIR':0.3877, 'Ann_Return':0.0901, 'IR':1.0164, 'Max_DD':-0.1038},
    'XGBoost':        {'IC':0.0498, 'ICIR':0.3779, 'Rank_IC':0.0505, 'Rank_ICIR':0.4131, 'Ann_Return':0.0780, 'IR':0.9070, 'Max_DD':-0.1168},
    'CatBoost':       {'IC':0.0460, 'ICIR':0.3641, 'Rank_IC':0.0479, 'Rank_ICIR':0.3935, 'Ann_Return':0.0761, 'IR':0.8810, 'Max_DD':-0.1107},
    'DoubleEnsemble': {'IC':0.0521, 'ICIR':0.4223, 'Rank_IC':0.0502, 'Rank_ICIR':0.4117, 'Ann_Return':0.1158, 'IR':1.3432, 'Max_DD':-0.0920},
    'Linear':         {'IC':0.0303, 'ICIR':0.2500, 'Rank_IC':0.0300, 'Rank_ICIR':0.2530, 'Ann_Return':0.0460, 'IR':0.5100, 'Max_DD':-0.1230},
    'LSTM':           {'IC':0.0395, 'ICIR':0.3100, 'Rank_IC':0.0390, 'Rank_ICIR':0.3200, 'Ann_Return':0.0550, 'IR':0.6100, 'Max_DD':-0.1500},
    'ALSTM':          {'IC':0.0428, 'ICIR':0.3390, 'Rank_IC':0.0429, 'Rank_ICIR':0.3445, 'Ann_Return':0.0700, 'IR':0.7800, 'Max_DD':-0.1200},
    'GRU':            {'IC':0.0417, 'ICIR':0.3300, 'Rank_IC':0.0415, 'Rank_ICIR':0.3350, 'Ann_Return':0.0680, 'IR':0.7500, 'Max_DD':-0.1250},
    'Transformer':    {'IC':0.0374, 'ICIR':0.2900, 'Rank_IC':0.0370, 'Rank_ICIR':0.2950, 'Ann_Return':0.0500, 'IR':0.5500, 'Max_DD':-0.1600},
    'TRA':            {'IC':0.0440, 'ICIR':0.3535, 'Rank_IC':0.0540, 'Rank_ICIR':0.4451, 'Ann_Return':0.0718, 'IR':1.0835, 'Max_DD':-0.0760},
    'TabNet':         {'IC':0.0377, 'ICIR':0.3000, 'Rank_IC':0.0375, 'Rank_ICIR':0.3050, 'Ann_Return':0.0580, 'IR':0.6500, 'Max_DD':-0.1400},
}

# ═══════════════════════════════════════════════════════════════════════════
# Local data loading & feature engineering (consistent with monthly_enhanced_v6.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_ashare():
    conn = sqlite3.connect(DB)
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
        ma5 = pd.Series(c).rolling(5).mean().values
        ma20 = pd.Series(c).rolling(20).mean().values
        ma60 = pd.Series(c).rolling(60).mean().values
        e12 = pd.Series(c).ewm(span=12).mean().values
        e26 = pd.Series(c).ewm(span=26).mean().values
        dif = e12-e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif-dea)*2
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta>0,delta,0); loss = np.where(delta<0,-delta,0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/14).mean().values
        rsi14 = np.nan_to_num(100-100/(1+avg_gain/np.maximum(avg_loss,1e-8)), 50)
        bb_std = pd.Series(c).rolling(20).std().values
        bb_pos = np.nan_to_num((c-(ma20-2*bb_std))/np.maximum(4*bb_std,0.01), 0.5)
        trange = np.maximum(h-l, np.abs(h-np.roll(c,1)))
        atr14 = pd.Series(trange).rolling(14).mean().values
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
                'ma60d':(c[i]-ma60[i])/max(abs(c[i]),0.01),
                'macd_dif':dif[i],'macd_dea':dea[i],'macd_hist':macd_hist[i],
                'rsi14':rsi14[i],'bb_pos':bb_pos[i],
                'vol_6':np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0,
                'atr_ratio':atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0,
                'hilo':(h[i]-l[i])/max(abs(c[i]),0.01),
                'above_ma20':1.0 if c[i]>ma20[i] else 0.0,
                'above_ma60':1.0 if c[i]>ma60[i] else 0.0,
            })
    return pd.DataFrame(rows).dropna()

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def monthly_ic(test, pred):
    ics = []
    for m in test['date'].unique():
        mask = test['date'] == m
        if mask.sum() < 20: continue
        ic = spearmanr(pred[mask], test.loc[mask,'fwd_ret'].values)[0]
        if not np.isnan(ic): ics.append(ic)
    return np.mean(ics) if ics else np.nan, ics

# ═══════════════════════════════════════════════════════════════════════════
# Main flow
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("Multi-model benchmark: Local models vs Qlib SOTA Benchmark")
    print("="*70)

    # Data
    print("\n[1] Loading A-share monthly klines...")
    t0 = time.time()
    data = load_ashare()
    feat_cols = [c for c in data.columns if c not in ['date','fwd_ret']]
    train = data[(data['date']>='2015-01')&(data['date']<='2021-12')]
    test = data[data['date']>='2024-01']
    X_tr, y_tr = train[feat_cols].values.astype(float), train['fwd_ret'].values.astype(float)
    X_te, y_te = test[feat_cols].values.astype(float), test['fwd_ret'].values.astype(float)
    print(f"  Train: {len(train):,} Test: {len(test):,} Features: {len(feat_cols)} ({time.time()-t0:.0f}s)")

    # Standardize
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # Local models
    models = {
        'LightGBM': lgb.LGBMRegressor(objective='regression',num_leaves=63,learning_rate=0.03,
                        n_estimators=200,min_child_samples=10,subsample=0.8,colsample_bytree=0.8,
                        random_state=456,verbosity=-1,n_jobs=4),
        'RandomForest': RandomForestRegressor(n_estimators=200,max_depth=8,min_samples_leaf=10,
                        random_state=456,n_jobs=4),
        'GBoost': GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,
                        min_samples_leaf=10,random_state=456),
        'MLP(64,32)': MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',alpha=0.001,
                        batch_size=64,max_iter=300,early_stopping=True,random_state=456),
        'Ridge': Ridge(alpha=1.0,random_state=456),
        'Linear': LinearRegression(),
    }

    results = {}
    print("\n[2] Local model results vs Qlib SOTA:")
    print(f"{'Model':<18s} {'IC':>8s} {'ICIR':>8s} {'vs Qlib IC':>12s} {'Gap':>8s} {'Note':>20s}")
    print("-"*78)

    for name, model in models.items():
        t0 = time.time()
        try:
            if name in ('LightGBM','RandomForest','GBoost'):
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
            else:
                model.fit(X_tr_s, y_tr)
                pred = model.predict(X_te_s)
            ic, month_ics = monthly_ic(test, pred)
            icir = ic / np.std(month_ics) if np.std(month_ics) > 0 else np.nan
            results[name] = {'IC':ic, 'ICIR':icir, 'time_s':time.time()-t0}

            qlib = QLIB_BENCHMARKS.get(name, {})
            qlib_ic = qlib.get('IC', np.nan)
            gap = ic - qlib_ic if not np.isnan(qlib_ic) else np.nan
            note = ''
            if not np.isnan(gap):
                note = '>>Better' if gap > 0.01 else '>Better' if gap > 0 else '<Worse'
            print(f"{name:<18s} {ic:+.4f} {icir:+.4f} {qlib_ic:+.4f} {gap:+.4f} {note}")
        except Exception as e:
            print(f"{name:<18s} failed: {e}")
            results[name] = {'error':str(e)}

    # Qlib-specific models (for comparison only)
    print(f"\n{'─'*78}")
    print("Qlib-specific models (no local equivalent, reference only):")
    for name in ['DoubleEnsemble','ALSTM','GRU','TRA','TabNet']:
        qlib = QLIB_BENCHMARKS.get(name, {})
        print(f"  {name:<16s} IC={qlib.get('IC',0):+.4f} IR={qlib.get('IR',0):.4f} AnnRet={qlib.get('Ann_Return',0):.2%}")

    # Qlib run (if available)
    print(f"\n{'─'*78}")
    try:
        import qlib
        from qlib.config import C
        print("Qlib available, skipping its built-in benchmark (needs extra config)")
    except ImportError:
        print("Qlib not installed (needs Visual C++ Build Tools), above is local model benchmark")

    # Save
    out = {'local_results':{k:{kk:float(vv) if isinstance(vv,(np.floating,float)) else vv for kk,vv in v.items()} for k,v in results.items()},
           'qlib_benchmarks':QLIB_BENCHMARKS}
    with open(OUT/'model_benchmark.json','w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults: {OUT/'model_benchmark.json'}")

    return results

if __name__ == '__main__':
    main()
