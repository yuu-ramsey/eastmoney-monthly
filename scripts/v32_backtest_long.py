"""
v32 pure long backtest: 32d factor Q5 long portfolio + price limit + liquidity filter + monthly rebalance
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
OUT = PROJECT / '.eastmoney-ai' / 'backtest'
OUT.mkdir(parents=True, exist_ok=True)
N_FFT = 10
DATE_FMT = '%Y-%m-%d %H:%M:%S'

# Filter Parameters
MIN_DAILY_AMOUNT = 5_000_000   # 日均成交额 > 500万
LIMIT_UP_DOWN = 0.10           # 涨跌停 ±10%
TOP_QUINTILE = 0.20            # Q5 = top 20%
SLIPPAGE = 0.003               # 滑点
COMMISSION = 0.00025            # 手续费 (万2.5)


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


def cross_sectional_neutralize(features, dates, neutralizer):
    neutralized = features.copy()
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() < 50: continue
        groups = neutralizer[mask]
        for g in np.unique(groups):
            gm = mask & (neutralizer == g)
            if gm.sum() >= 3: neutralized[gm] -= features[gm].mean(axis=0)
    return neutralized


def load_daily_filters():
    """Loaded每日涨跌停和成交额数据, 按月聚合为筛选标记"""
    print(f"[{ts()}] Loaded日线数据用于筛选...", flush=True); t0 = time.time()
    conn = sqlite3.connect(str(DB))
    df = pd.read_sql_query("""
        SELECT code, date, amount, change_percent
        FROM daily_klines
        WHERE date >= '2014-01' AND amount > 0
        ORDER BY code, date
    """, conn)
    conn.close()

    df['month'] = df['date'].str[:7]

    # 按月+股票聚合
    monthly_flags = df.groupby(['code', 'month']).agg(
        avg_amount=('amount', 'mean'),
        hit_limit_up=('change_percent', lambda x: (x >= 9.9).any()),
        hit_limit_down=('change_percent', lambda x: (x <= -9.9).any()),
        n_days=('date', 'count'),
    ).reset_index()

    monthly_flags['liquid'] = monthly_flags['avg_amount'] >= MIN_DAILY_AMOUNT
    monthly_flags['not_limit'] = ~(monthly_flags['hit_limit_up'] | monthly_flags['hit_limit_down'])
    monthly_flags['pass_filter'] = monthly_flags['liquid'] & monthly_flags['not_limit']

    print(f"[{ts()}] 日线筛选就绪: {len(monthly_flags):,} 条月记录 ({time.time()-t0:.0f}s)", flush=True)
    return monthly_flags


def build_32d_features():
    """构建32维特征, Returns特征矩阵 + 元数据"""
    print(f"[{ts()}] Loaded月线数据...", flush=True)
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
    print(f"[{ts()}] 构建32维特征...", flush=True); t0 = time.time()
    X_list, y_list, meta_list = [], [], []

    for code in codes_used:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        if len(g) < 72: continue
        c = g['close'].values.astype(float); o = g['open'].values.astype(float)
        h = g['high'].values.astype(float); l = g['low'].values.astype(float)
        v = g['volume'].values.astype(float)
        tr_rate = g['turnover_rate'].values.astype(float) if 'turnover_rate' in g.columns else np.zeros(len(g))
        n = len(c); cc = wd(c); industry = ind_map.get(code, 'unknown')
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
        up_streak = np.zeros(n); dn_streak = np.zeros(n)
        for i in range(1, n):
            up_streak[i] = up_streak[i-1] + 1 if c[i] > c[i-1] else 0
            dn_streak[i] = dn_streak[i-1] + 1 if c[i] < c[i-1] else 0

        for i in range(60, n - 6):
            if c[i] <= 0.01: continue
            if i + 6 >= n: continue
            fwd_ret_t3 = np.clip((c[i+3] - c[i+2]) / np.maximum(abs(c[i+2]), 0.01), -2, 2)

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
                       up_streak[i]/12.0, dn_streak[i]/12.0]

            features = g2 + g3 + g4_sel + fft_amplitudes(cc[i-60+1:i+1]).tolist() + g7_full
            X_list.append(features)
            y_list.append(fwd_ret_t3)
            meta_list.append({'code': code, 'month': g['month'].iloc[i], 'industry': industry,
                              'close': float(c[i])})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)

    v = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[v]; y = y[v]; meta = meta.iloc[v].reset_index(drop=True)

    print(f"[{ts()}] {len(y):,} 样本, {X.shape[1]}d ({time.time()-t0:.0f}s)", flush=True)
    return X, y, meta


# ====== Main ======
if __name__ == '__main__':
    # ---- 1. 构建特征 ----
    X, y, meta = build_32d_features()

    # ---- 2. 行业中性化 ----
    print(f"[{ts()}] 行业中性化...", flush=True); t0 = time.time()
    X_ind = cross_sectional_neutralize(X.copy(), meta['month'].values, meta['industry'].values)
    print(f"[{ts()}] done ({time.time()-t0:.0f}s)", flush=True)

    # ---- 3. Train/Test split ----
    tr_m = (meta['month'] >= '2010-01') & (meta['month'] <= '2014-12')
    te_m = (meta['month'] >= '2015-01')

    X_tr = X_ind[tr_m]; X_te = X_ind[te_m]
    y_tr = y[tr_m]; y_te = y[te_m]
    meta_te = meta[te_m].reset_index(drop=True)

    # ---- 4. 训练集成模型 ----
    print(f"\n{'='*60}")
    print("训练 LGB+XGB+Ridge Ensemble (32维)")
    print(f"Train: {X_tr.shape[0]:,} 样本, Test: {X_te.shape[0]:,} 样本")
    print(f"{'='*60}")

    sc = StandardScaler()
    Xt = sc.fit_transform(X_tr); Xte = sc.transform(X_te)

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
        ic_tmp, _, _, _, _ = cs_ic_full(p, y_te, meta_te['month'].values)
        ics_m[n] = max(ic_tmp, 0)
    w_sum = sum(ics_m.values())
    if w_sum <= 0: ics_m, w_sum = {'LGB': 1.0, 'XGB': 1.0, 'Ridge': 1.0}, 3.0
    p_ens = sum(ics_m[n]*p for n,p in [('LGB',p_lgb),('XGB',p_xgb),('Ridge',p_ridge)]) / w_sum

    ic_ens, icir_ens, _, ic_pos, _ = cs_ic_full(p_ens, y_te, meta_te['month'].values)
    print(f"\nEnsemble: IC={ic_ens:+.4f}  ICIR={icir_ens:+.3f}  IC>0={ic_pos:.1%}")
    print(f"集成权重: LGB={ics_m['LGB']:.3f}  XGB={ics_m['XGB']:.3f}  Ridge={ics_m['Ridge']:.3f}")

    # ---- 5. Loaded日线筛选数据 ----
    daily_flags = load_daily_filters()

    # ---- 6. 构建预测表 ----
    meta_te['pred'] = p_ens
    pred_table = meta_te[['code', 'month', 'close', 'pred']].copy()

    # ---- 7. 逐月回测 ----
    print(f"\n{'='*60}")
    print(f"Q5 多头回测 (top {int(TOP_QUINTILE*100)}%, 等权, 月度调仓)")
    print(f"筛选: 日均成交额>{MIN_DAILY_AMOUNT/1e6:.0f}万, 无涨跌停")
    print(f"成本: 滑点{SLIPPAGE:.1%}, 手续费{COMMISSION:.4%}")
    print(f"{'='*60}")

    test_months = sorted(meta_te['month'].unique())
    portfolio_returns = []
    equity_curve = [1.0]
    holdings_log = []

    for t, cur_month in enumerate(test_months[:-1]):
        next_month = test_months[t + 1]

        # 当月预测
        cur_preds = pred_table[pred_table['month'] == cur_month].copy()
        if len(cur_preds) < 50: continue

        # 合并筛选标记
        cur_flags = daily_flags[daily_flags['month'] == cur_month][['code', 'pass_filter', 'avg_amount']]
        cur_preds = cur_preds.merge(cur_flags, on='code', how='left')
        cur_preds['pass_filter'] = cur_preds['pass_filter'].fillna(False)

        # 筛选: 流动性 + 无涨跌停
        filtered = cur_preds[cur_preds['pass_filter']].copy()
        if len(filtered) < 20: continue

        # Q5: top 20%
        top_n = max(int(len(filtered) * TOP_QUINTILE), 5)
        q5 = filtered.nlargest(top_n, 'pred')

        # 下月实际收益
        next_preds = pred_table[pred_table['month'] == next_month][['code', 'close']]
        q5 = q5.merge(next_preds, on='code', how='inner', suffixes=('_cur', '_next'))
        if len(q5) < 5: continue

        q5['ret'] = (q5['close_next'] - q5['close_cur']) / q5['close_cur']

        # 等权收益 (扣成本)
        gross_ret = q5['ret'].mean()
        if t > 0 and prev_holdings:
            cur_codes = set(q5['code'].values)
            turnover = len(cur_codes - prev_holdings) / len(prev_holdings) if prev_holdings else 1.0
            net_ret = gross_ret - turnover * (SLIPPAGE + COMMISSION * 2)
        else:
            net_ret = gross_ret - (SLIPPAGE + COMMISSION * 2)

        portfolio_returns.append({
            'month': cur_month,
            'next_month': next_month,
            'n_stocks': len(q5),
            'gross_ret': float(gross_ret),
            'net_ret': float(net_ret),
        })
        equity_curve.append(equity_curve[-1] * (1 + net_ret))
        prev_holdings = set(q5['code'].values)

        if (t + 1) % 24 == 0:
            print(f"  [{cur_month}] n={len(q5):>3d}  net={net_ret:+.3%}  equity={equity_curve[-1]:.3f}")

    # ---- 8. 绩效统计 ----
    if not portfolio_returns:
        print("ERROR: 无有效回测月份")
        exit(1)

    rets = pd.Series([r['net_ret'] for r in portfolio_returns])
    eq = pd.Series(equity_curve)

    n_months = len(rets)
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum_ret = eq.iloc[-1] - 1.0
    cummax = eq.expanding().max()
    dd = (eq - cummax) / cummax
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0
    win_rate = (rets > 0).mean()
    avg_n = np.mean([r['n_stocks'] for r in portfolio_returns])

    print(f"\n{'='*60}")
    print("回测绩效 (纯多头 Q5)")
    print(f"{'='*60}")
    print(f"  回测期: {portfolio_returns[0]['month']} ~ {portfolio_returns[-1]['next_month']}")
    print(f"  总月份: {n_months}")
    print(f"  accumulated净收益: {cum_ret:+.2%}")
    print(f"  年化收益:   {ann_ret:+.2%}")
    print(f"  年化波动:   {ann_vol:.2%}")
    print(f"  Sharpe:     {sharpe:.3f}")
    print(f"  最大回撤:   {max_dd:.2%}")
    print(f"  Calmar:     {calmar:.3f}")
    print(f"  月胜率:     {win_rate:.1%}")
    print(f"  平均持仓:   {avg_n:.0f} 只")

    # ---- 9. 逐年统计 ----
    print(f"\n{'='*60}")
    print("逐年统计")
    print(f"{'='*60}")
    print(f"{'Year':<6s} {'Months':>6s} {'Ann_Ret':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Win':>8s}")
    print('-'*56)
    yearly = {}
    for r in portfolio_returns:
        y = r['month'][:4]
        if y not in yearly: yearly[y] = []
        yearly[y].append(r['net_ret'])

    for year in sorted(yearly):
        yr = pd.Series(yearly[year])
        ann_r = yr.mean() * 12
        ann_v = yr.std() * np.sqrt(12)
        sr = ann_r / ann_v if ann_v > 0 else 0
        eq_y = (1 + yr).cumprod()
        dd_y = (eq_y / eq_y.expanding().max() - 1).min()
        wr = (yr > 0).mean()
        print(f"{year:<6s} {len(yr):>6d} {ann_r:>+7.1%} {ann_v:>7.1%} {sr:>+7.2f} {dd_y:>7.1%} {wr:>7.1%}")

    # ---- 10. 保存 ----
    pd.DataFrame(portfolio_returns).to_csv(OUT / 'v32_backtest_monthly.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'equity': equity_curve}).to_csv(OUT / 'v32_backtest_equity.csv', index=False, encoding='utf-8-sig')

    summary = {
        'version': 'v32', 'factor_dim': 32,
        'strategy': 'Q5 long-only, equal-weight, monthly rebalance',
        'filters': {'min_daily_amount': MIN_DAILY_AMOUNT, 'no_limit_up_down': True},
        'costs': {'slippage': SLIPPAGE, 'commission': COMMISSION},
        'period': f"{portfolio_returns[0]['month']} ~ {portfolio_returns[-1]['next_month']}",
        'n_months': n_months, 'cum_return': float(cum_ret),
        'ann_return': float(ann_ret), 'ann_vol': float(ann_vol),
        'sharpe': float(sharpe), 'max_drawdown': float(max_dd),
        'calmar': float(calmar), 'win_rate': float(win_rate),
        'avg_positions': float(avg_n),
        'ensemble_weights': {'LGB': float(ics_m['LGB']), 'XGB': float(ics_m['XGB']), 'Ridge': float(ics_m['Ridge'])},
    }
    with open(OUT / 'v32_backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[{ts()}] 回测done. 结果: {OUT}")
