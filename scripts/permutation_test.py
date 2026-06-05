"""Permutation test: verify statistical significance of monthly LightGBM IC=0.063.
方法：截面打乱 fwd_ret（破坏 X→y 关系）+ 价格序列打乱（破坏时序结构）
"""
import numpy as np, pandas as pd, sqlite3, time, json
from pathlib import Path
from scipy.stats import spearmanr
import lightgbm as lgb

DB = '.eastmoney-ai/db/klines-v2.sqlite'
N_PERM = 500  # 截面打乱次数
N_PRICE_PERM = 50  # 价格打乱次数（较慢）

# ── 特征工程（与 monthly_enhanced_v6.py 完全一致）─────────────────────
def build_features(df, codes, ind_map):
    rows = []
    for code in codes:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
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
        dif = e12 - e26; dea = pd.Series(dif).ewm(span=9).mean().values; macd_hist = (dif - dea) * 2
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
            fwd_ret = (c[i+3]-c[i])/c[i]
            if abs(fwd_ret) > 2: continue
            rows.append({'code':code,'date':dates[i],'fwd_ret':fwd_ret,
                'r1':(c[i]-c[i-1])/max(abs(c[i-1]),0.01) if i>=1 else 0,
                'r3':(c[i]-c[i-3])/max(abs(c[i-3]),0.01) if i>=3 else 0,
                'r6':(c[i]-c[i-6])/max(abs(c[i-6]),0.01) if i>=6 else 0,
                'r12':(c[i]-c[i-12])/max(abs(c[i-12]),0.01) if i>=12 else 0,
                'ma5d':(c[i]-ma5[i])/max(abs(c[i]),0.01),
                'ma20d':(c[i]-ma20[i])/max(abs(c[i]),0.01),
                'ma60d':(c[i]-ma60[i])/max(abs(c[i]),0.01),
                'macd_dif':dif[i],'macd_dea':dea[i],'macd_hist':macd_hist[i],
                'rsi14':rsi14[i],'bb_pos':bb_pos[i],
                'vol_6':np.std(np.diff(c[max(0,i-6):i+1])/np.maximum(np.abs(c[max(0,i-5):i+1]),0.01)) if i>=6 else 0,
                'atr_ratio':atr14[i]/max(abs(c[i]),0.01) if not np.isnan(atr14[i]) else 0,
                'vol_chg':np.mean(v[max(0,i-3):i+1])/max(np.mean(v[max(0,i-12):i+1]),1)-1 if i>=12 else 0,
                'to':tr[i] if not np.isnan(tr[i]) else 0,
                'to_chg':tr[i]/max(np.mean(tr[max(0,i-12):i+1]),0.001)-1 if i>=12 and not np.isnan(tr[i]) else 0,
                'hilo':(h[i]-l[i])/max(abs(c[i]),0.01),
                'body':abs(c[i]-o[i])/max(abs(c[i]),0.01),
                'above_ma20':1.0 if c[i]>ma20[i] else 0.0,
                'above_ma60':1.0 if c[i]>ma60[i] else 0.0,
                'sector_id':hash(industry)%31/31.0})
    return pd.DataFrame(rows).dropna()

# ── LightGBM 训练 + 月截面 IC ──────────────────────────────────────────
def train_eval(train, test, feat_cols, seed=456):
    model = lgb.LGBMRegressor(
        objective='regression', metric='l1', num_leaves=63, learning_rate=0.03,
        n_estimators=200, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, random_state=seed, verbosity=-1, n_jobs=4)
    model.fit(train[feat_cols].values, train['fwd_ret'].values)
    pred = model.predict(test[feat_cols].values)
    ics = []
    for m in test['date'].unique():
        mask = test['date'] == m
        if mask.sum() < 20: continue
        ics.append(spearmanr(pred[mask], test.loc[mask,'fwd_ret'].values)[0])
    return np.mean(ics) if ics else np.nan

# ── Main flow ─────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("月线 LightGBM 排列检验")
    print(f"截面打乱: {N_PERM}次 | 价格打乱: {N_PRICE_PERM}次")
    print("="*60)

    # 加载
    t0 = time.time()
    conn = sqlite3.connect(DB)
    codes = [r[0] for r in conn.execute('SELECT code FROM monthly_klines GROUP BY code HAVING COUNT(*)>=84').fetchall()]
    ind_map = {r[0]:r[1] for r in conn.execute('SELECT stock_code, industry_code FROM stock_industry_mapping')}
    params = ','.join('?'*len(codes))
    df = pd.read_sql_query(f"SELECT code,date,open,high,low,close,volume,turnover_rate FROM monthly_klines WHERE code IN ({params}) AND date>='2010-01' ORDER BY code,date", conn, params=codes)
    conn.close()
    print(f"[1/4] 数据: {len(codes)}只股票, {len(df):,}行 ({time.time()-t0:.0f}s)")

    # 真实 IC
    print("[2/4] 真实 IC 基线...")
    data = build_features(df, codes, ind_map)
    feat_cols = [c for c in data.columns if c not in ['code','date','fwd_ret']]
    train = data[(data['date']>='2015-01')&(data['date']<='2021-12')]
    test = data[data['date']>='2024-01']
    real_ic = train_eval(train, test, feat_cols)
    print(f"  真实 IC = {real_ic:+.4f} | 训练: {len(train):,} 测试: {len(test):,} 特征: {len(feat_cols)}")

    # 截面打乱排列
    print(f"\n[3/4] 截面打乱排列 ({N_PERM}次)...")
    perm_ics, better = [], 0
    t0 = time.time()
    for n in range(N_PERM):
        test_p = test.copy()
        for m in test_p['date'].unique():
            mask = test_p['date'] == m
            test_p.loc[mask,'fwd_ret'] = test_p.loc[mask,'fwd_ret'].sample(frac=1,random_state=None).values
        ic = train_eval(train, test_p, feat_cols, seed=n)
        perm_ics.append(ic)
        if ic >= real_ic: better += 1
        if (n+1)%100==0:
            e = time.time()-t0
            print(f"  [{n+1}/{N_PERM}] IC={ic:+.4f} better={better}/{n+1} ({e:.0f}s ETA{(e/(n+1)*(N_PERM-n-1)):.0f}s)")
    p_cross = better/N_PERM

    # 价格打乱排列
    print(f"\n[4/4] 价格打乱排列 ({N_PRICE_PERM}次)...")
    price_ics, price_better = [], 0
    t0 = time.time()
    for n in range(N_PRICE_PERM):
        df_p = df.copy()
        for code in codes:
            m = df_p['code']==code
            if m.sum()<72: continue
            g = df_p.loc[m].sort_values('date')
            p = g['close'].values.astype(float)
            rets = np.diff(np.log(np.maximum(p,0.01)))
            np.random.shuffle(rets)
            new_p = [p[0]]
            for r in rets: new_p.append(new_p[-1]*np.exp(r))
            scale = np.array(new_p)/np.maximum(p,0.01)
            for col in ['open','high','low','close']:
                df_p.loc[m,col] = g[col].values*scale
        dp = build_features(df_p, codes, ind_map)
        tp = dp[(dp['date']>='2015-01')&(dp['date']<='2021-12')]
        tep = dp[dp['date']>='2024-01']
        ic = train_eval(tp, tep, feat_cols, seed=n)
        price_ics.append(ic)
        if ic >= real_ic: price_better += 1
        e = time.time()-t0
        print(f"  [{n+1}/{N_PRICE_PERM}] IC={ic:+.4f} better={price_better}/{n+1} ({e:.0f}s)")
    p_price = price_better/N_PRICE_PERM

    # 结果
    print("\n"+"="*60)
    print("结果")
    print(f"  真实 IC:                 {real_ic:+.4f}")
    print(f"  截面打乱 P-value:        {p_cross:.4f} {'**显著' if p_cross<0.05 else '不显著'}")
    print(f"  价格打乱 P-value:        {p_price:.4f} {'**显著' if p_price<0.05 else '不显著'}")
    print(f"  截面打乱 IC 95%分位:     {np.percentile(perm_ics,95):+.4f}")
    print(f"  截面打乱 IC 99%分位:     {np.percentile(perm_ics,99):+.4f}")
    print(f"  截面打乱 IC mean±std:   {np.mean(perm_ics):+.4f}±{np.std(perm_ics):.4f}")

    out = {k:v for k,v in locals().items() if isinstance(v,(int,float,str,bool))}
    out['perm_ics'] = [float(x) for x in perm_ics]
    out['price_perm_ics'] = [float(x) for x in price_ics]
    Path('.eastmoney-ai/eval').mkdir(parents=True, exist_ok=True)
    with open('.eastmoney-ai/eval/permutation_test.json','w') as f:
        json.dump(out, f, indent=2)
    print(f"\n保存到 .eastmoney-ai/eval/permutation_test.json")

if __name__ == '__main__':
    main()
