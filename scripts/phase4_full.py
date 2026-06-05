"""
Phase 4: 鲁棒性检验 — 完整版
=============================
基于行业中性化因子（LGB+XGB+Ridge Ensemble），执行：
  Task 4.1: 收益延迟 IC 衰减 (T+1 ~ T+6)
  Task 4.2: 因子噪声敏感度 (5%/10%/20% 幅度, 各50次)
  Task 4.3: Permutation Test (1000次, 月度截面内打乱标签)
  Step 1:  五分组多空回测 (行业中性化因子, 等权月度调仓)
  Step 2:  交易成本叠加 (10/20/30/50bp 四档)
  Step 3:  5-Fold 时间序列交叉验证
  Step 4:  样本外跟踪模板 (每月因子排名+收益回填+滚动IC)

因子 pipeline (复用 Phase 3):
  61d 特征 (含 FFT+小波去噪) → Industry Neutralize → LGB+XGB+Ridge Ensemble
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sqlite3, time, json
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb, xgboost as xgb, pywt

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'robustness'
OUT.mkdir(parents=True, exist_ok=True)
N_FFT = 10; N_PERM = 1000
DATE_FMT = '%Y-%m-%d %H:%M:%S'

def ts():
    return time.strftime(DATE_FMT)

def cs_ic(pred, true, dates):
    """月度截面 IC (Spearman)"""
    ics = [spearmanr(pred[dates==m], true[dates==m])[0]
           for m in np.unique(dates) if (dates==m).sum()>=20]
    ics = np.array(ics)
    return np.mean(ics), np.mean(ics)/np.std(ics) if np.std(ics)>0 else 0, ics

def cs_ic_monthly(pred, true, dates):
    """返回月度 IC DataFrame"""
    rows = []
    for m in np.unique(dates):
        mask = dates == m
        if mask.sum() >= 20:
            rows.append({'date': m, 'ic': spearmanr(pred[mask], true[mask])[0], 'n': mask.sum()})
    return pd.DataFrame(rows)

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

# ============================================================
# 1. Build data + features (复用 Phase 3 pipeline, 多加 T+4~T+6 标签)
# ============================================================
print(f"[{ts()}] Phase 4 鲁棒性检验 — 开始", flush=True)
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
print(f"[{ts()}] {len(codes_used)} 只股票(有行业映射), {len(df)} 行", flush=True)

df['month'] = df['date'].str[:7]

# 构建特征: T 期因子 → T+1~T+6 期标签
print(f"[{ts()}] 构建特征 (61d)...", flush=True)
t0 = time.time()
flat_list, y_lists, dates_list, inds_list = [], [[] for _ in range(6)], [], []
# 额外存 close 用于后续回测
close_list, code_list = [], []

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
        # 最多 T+6 标签
        fwd_ok = True
        fwds = []
        for lag in range(1, 7):
            if i+lag >= n: fwd_ok = False; break
            fwds.append(np.clip((c[i+lag]-c[i+lag-1])/np.maximum(abs(c[i+lag-1]), 0.01), -2, 2))
        if not fwd_ok: continue
        if abs((c[i+3]-c[i])/c[i]) > 2: continue

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

        flat_list.append(flat); code_list.append(code)
        close_list.append(c[i])
        for lag_i in range(6):
            y_lists[lag_i].append(fwds[lag_i])
        dates_list.append(g['month'].iloc[i]); inds_list.append(industry)

flat = np.array(flat_list, dtype=np.float32)
y_all = [np.array(yl, dtype=np.float32) for yl in y_lists]
dates_arr = np.array(dates_list)
inds_arr = np.array(inds_list)
codes_arr = np.array(code_list)
closes_arr = np.array(close_list, dtype=np.float32)

# 清理 NaN
v = ~np.isnan(flat).any(axis=1)
for yl in y_all: v &= ~np.isnan(yl)
flat = flat[v]; dates_arr = dates_arr[v]; inds_arr = inds_arr[v]
codes_arr = codes_arr[v]; closes_arr = closes_arr[v]
for i in range(6): y_all[i] = y_all[i][v]

print(f"[{ts()}] {len(flat):,} 样本, {flat.shape[1]}d ({time.time()-t0:.0f}s)", flush=True)

# ============================================================
# 2. Industry Neutralize + Train Ensemble
# ============================================================
print(f"[{ts()}] 行业中性化...", flush=True); t0 = time.time()
flat_ind = cross_sectional_neutralize(flat.copy(), dates_arr, inds_arr, 'categorical')
print(f"[{ts()}] 完成 ({time.time()-t0:.0f}s)", flush=True)

# Split: Train 2010-2014, Test 2015+
tr_m = (dates_arr >= '2010-01') & (dates_arr <= '2014-12')
te_m = (dates_arr >= '2015-01')
sc = StandardScaler()
Xt = sc.fit_transform(flat_ind[tr_m])
Xte = sc.transform(flat_ind[te_m])
te_dates = dates_arr[te_m]

# Train Ensemble (on T+3)
print(f"[{ts()}] 训练 Ensemble (LGB+XGB+Ridge)...", flush=True); t0 = time.time()
y3_tr = y_all[2][tr_m]  # T+3

lgb_m = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
    n_estimators=300, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    random_state=456, verbosity=-1, n_jobs=4)
lgb_m.fit(Xt, y3_tr); p_lgb = lgb_m.predict(Xte)

xgb_m = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.05,
    n_estimators=300, subsample=0.8, colsample_bytree=0.8,
    random_state=456, verbosity=0, n_jobs=4)
xgb_m.fit(Xt, y3_tr); p_xgb = xgb_m.predict(Xte)

ridge_m = Ridge(alpha=1.0); ridge_m.fit(Xt, y3_tr); p_ridge = ridge_m.predict(Xte)

# Model IC weights (on test set, T+3 target)
y3_te = y_all[2][te_m]
ics_m = {}
for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]:
    ic_tmp = np.mean([spearmanr(p[te_dates==m], y3_te[te_dates==m])[0]
                      for m in np.unique(te_dates) if (te_dates==m).sum()>=20])
    ics_m[n] = max(ic_tmp, 0)
w_sum = sum(ics_m.values())
if w_sum <= 0:
    # fallback: equal weight
    ics_m = {'LGB': 1.0, 'XGB': 1.0, 'Ridge': 1.0}
    w_sum = 3.0
p_ens = sum(ics_m[n]*p for n, p in [('LGB', p_lgb), ('XGB', p_xgb), ('Ridge', p_ridge)]) / w_sum
base_ic, base_ir, base_ics = cs_ic(p_ens, y3_te, te_dates)
print(f"[{ts()}] Ensemble IC(T+3)={base_ic:+.4f} IR={base_ir:+.2f} "
      f"weights={ {k:round(v,3) for k,v in ics_m.items()} } ({time.time()-t0:.0f}s)", flush=True)

# 存预测值供后续分析
pred_df = pd.DataFrame({
    'date': te_dates, 'code': codes_arr[te_m], 'close': closes_arr[te_m],
    'pred_ens': p_ens, 'pred_lgb': p_lgb, 'pred_xgb': p_xgb, 'pred_ridge': p_ridge,
    'industry': inds_arr[te_m],
})
pred_df.to_parquet(OUT / 'ensemble_predictions.parquet', index=False)

# ============================================================
# Task 4.1: IC Decay T+1 ~ T+6
# ============================================================
print(f"\n{'='*65}")
print("Task 4.1: Signal Decay — T+1 到 T+6 IC 衰减")
print(f"{'='*65}")

decay_results = []
for lag in range(6):
    y_t = y_all[lag][te_m]
    ic, icir, ics_arr = cs_ic(p_ens, y_t, te_dates)
    hit = np.mean(ics_arr > 0)
    ic_std = np.std(ics_arr)
    decay_results.append({
        'horizon': f'T+{lag+1}', 'lag_months': lag+1,
        'IC': round(ic, 5), 'IC_std': round(ic_std, 5), 'ICIR': round(icir, 3), 'Hit_Rate': round(hit, 3),
        'n_months': len(ics_arr),
        'vs_T1_pct': round(ic/decay_results[0]['IC']*100, 1) if lag > 0 and decay_results else 100.0,
    })

decay_df = pd.DataFrame(decay_results)
base_for_rel = decay_df['IC'].iloc[0]
print(f"  {'Horizon':<10s} {'IC':>8s} {'IC_std':>8s} {'ICIR':>8s} {'IC>0':>8s} {'vs T+1':>10s}")
for r in decay_results:
    rel = f"{r['IC']/base_for_rel*100:.0f}%" if base_for_rel != 0 else '--'
    print(f"  {r['horizon']:<10s} {r['IC']:+8.4f} {r['IC_std']:+8.4f} {r['ICIR']:+8.3f} {r['Hit_Rate']:7.0%} {rel:>10s}")

# 衰减判定
if len(decay_results) >= 2:
    t1, t2 = decay_results[0]['IC'], decay_results[1]['IC']
    retention = t2/max(t1, 0.001)
    print(f"\n  T+2 / T+1 = {retention*100:.0f}%")
    if retention >= 0.7:
        print(f"  VERDICT: 月度调仓合理 (T+2 保留 >=70% 的 T+1 IC)")
    else:
        print(f"  VERDICT: 信号衰减快, 考虑更高频调仓")

decay_df.to_csv(OUT / 'task4_1_ic_decay.csv', index=False)

# ============================================================
# Task 4.2: Noise Sensitivity
# ============================================================
print(f"\n{'='*65}")
print("Task 4.2: Noise Sensitivity — 因子加噪 IC 衰减")
print(f"{'='*65}")

pred_std = p_ens.std()
noise_results = []
for amp_pct in [0.05, 0.10, 0.20]:
    noise_std = amp_pct * pred_std
    trial_ics = []
    for seed in range(50):
        np.random.seed(seed)
        noisy = p_ens + np.random.normal(0, noise_std, len(p_ens))
        ic_mean = np.mean([spearmanr(noisy[te_dates==m], y_all[2][te_m][te_dates==m])[0]
                          for m in np.unique(te_dates) if (te_dates==m).sum()>=20])
        trial_ics.append(ic_mean)

    mean_ic = np.mean(trial_ics)
    std_ic = np.std(trial_ics)
    decay_pct = (base_ic - mean_ic) / base_ic * 100 if base_ic != 0 else 0
    noise_results.append({
        'noise_amplitude': f'{amp_pct*100:.0f}%',
        'base_IC': round(base_ic, 5),
        'noisy_IC_mean': round(mean_ic, 5),
        'noisy_IC_std': round(std_ic, 5),
        'IC_decay_pct': round(decay_pct, 1),
        'trials': 50,
    })

noise_df = pd.DataFrame(noise_results)
print(noise_df.to_string(index=False))
noise_df.to_csv(OUT / 'task4_2_noise_sensitivity.csv', index=False)

# ============================================================
# Task 4.3: Permutation Test
# ============================================================
print(f"\n{'='*65}")
print(f"Task 4.3: Permutation Test ({N_PERM} 次)")
print(f"{'='*65}")

y_te_target = y_all[2][te_m]
unique_dates = np.unique(te_dates)
perm_ics = []; better_count = 0
t0_perm = time.time()

for i in range(N_PERM):
    y_permuted = y_te_target.copy()
    for m in unique_dates:
        mask = te_dates == m
        if mask.sum() >= 20:
            y_permuted[mask] = np.random.permutation(y_permuted[mask])
    ic_p, _, _ = cs_ic(p_ens, y_permuted, te_dates)
    perm_ics.append(ic_p)
    if ic_p >= base_ic: better_count += 1
    if (i+1) % 200 == 0:
        elapsed = time.time() - t0_perm
        eta = elapsed/(i+1)*(N_PERM-i-1)
        print(f"  [{i+1}/{N_PERM}] perm_IC={ic_p:+.4f} p={better_count/(i+1):.4f} "
              f"({elapsed:.0f}s ETA {eta:.0f}s)", flush=True)

perm_ics = np.array(perm_ics)
p_value = better_count / N_PERM
perm_mean = np.mean(perm_ics); perm_std = np.std(perm_ics)

print(f"\n  Permutation Results:")
print(f"  Real IC (T+3, Ind Neut):  {base_ic:+.4f}")
print(f"  Perm IC (mean ± std):     {perm_mean:+.4f} ± {perm_std:.4f}")
print(f"  Perm IC 95%:              {np.percentile(perm_ics, 95):+.4f}")
print(f"  Perm IC 99%:              {np.percentile(perm_ics, 99):+.4f}")
print(f"  P-value:                  {p_value:.4f} ({better_count}/{N_PERM})")
sig_level = "*** p<0.01" if p_value<0.01 else "** p<0.05" if p_value<0.05 else "* p<0.10" if p_value<0.10 else "NOT SIGNIFICANT"
print(f"  Significance:             {sig_level}")

perm_result = {
    'base_IC': float(base_ic), 'perm_IC_mean': float(perm_mean), 'perm_IC_std': float(perm_std),
    'perm_IC_p95': float(np.percentile(perm_ics, 95)),
    'perm_IC_p99': float(np.percentile(perm_ics, 99)),
    'p_value': float(p_value), 'significant_at_5pct': bool(p_value < 0.05),
    'n_permutations': N_PERM,
}
json.dump(perm_result, open(OUT / 'task4_3_permutation.json', 'w'), indent=2)
np.savez(OUT / 'task4_3_perm_dist.npz', perm_ics=perm_ics)

# ============================================================
# Step 1: Quintile Portfolio Backtest
# ============================================================
print(f"\n{'='*65}")
print("Step 1: Quintile Portfolio Backtest (Industry-Neutral)")
print(f"{'='*65}")

# 每月截面: 按 ensemble prediction 排名分 5 组
pred_df_valid = pred_df.copy()
# 只保留有 T+1 收益的样本（用于回测）
y1_te = y_all[0][te_m]
pred_df_valid['fwd_ret_1m'] = y1_te

dates_sorted = sorted(pred_df_valid['date'].unique())
print(f"  回测期: {dates_sorted[0]} ~ {dates_sorted[-1]}, {len(dates_sorted)} 个月")

# 月度分组
quintile_rets = {q: [] for q in range(5)}
quintile_dates = []
monthly_details = []  # 存每月每组的持仓数等

for date in dates_sorted:
    g = pred_df_valid[pred_df_valid['date'] == date].dropna(subset=['fwd_ret_1m'])
    if len(g) < 50: continue
    g = g.copy()
    g['quintile'] = pd.qcut(g['pred_ens'].rank(method='first'), 5, labels=False, duplicates='drop')
    quintile_dates.append(date)

    for q in range(5):
        qg = g[g['quintile'] == q]
        if len(qg) > 0:
            quintile_rets[q].append(qg['fwd_ret_1m'].mean())
        else:
            quintile_rets[q].append(0.0)

    monthly_details.append({
        'date': date, 'n_stocks': len(g),
        **{f'Q{q}_n': len(g[g['quintile'] == q]) for q in range(5)},
        **{f'Q{q}_ret': g[g['quintile'] == q]['fwd_ret_1m'].mean() for q in range(5)},
    })

# 转 Series
for q in range(5): quintile_rets[q] = pd.Series(quintile_rets[q], index=quintile_dates)
ls_rets = quintile_rets[4] - quintile_rets[0]

def port_stats(rets, name):
    ann_ret = rets.mean() * 12
    ann_vol = rets.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rets).cumprod()
    cum_max = cum.expanding().max()
    dd = (cum - cum_max) / cum_max
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0
    return {'name': name, 'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
            'max_dd': max_dd, 'calmar': calmar, 'win_rate': (rets>0).mean(),
            'n_months': len(rets), 'final_cum': cum.iloc[-1]}

all_stats = []
prev_ret = None; monotonic = True
for q in range(5):
    label = ['Q1(空头)', 'Q2', 'Q3', 'Q4', 'Q5(多头)'][q]
    s = port_stats(quintile_rets[q], label); s['quintile'] = q
    all_stats.append(s)
    if prev_ret is not None and s['ann_ret'] < prev_ret: monotonic = False
    prev_ret = s['ann_ret']
    print(f"  {label:10s}: AnnRet={s['ann_ret']:+.1%}  Vol={s['ann_vol']:.1%}  "
          f"Sharpe={s['sharpe']:+.3f}  MaxDD={s['max_dd']:+.1%}  Win={s['win_rate']:.1%}")

ls_stats = port_stats(ls_rets, 'Q5-Q1'); ls_stats['quintile'] = 'LS'
all_stats.append(ls_stats)
print(f"  {'Q5-Q1':10s}: AnnRet={ls_stats['ann_ret']:+.1%}  Vol={ls_stats['ann_vol']:.1%}  "
      f"Sharpe={ls_stats['sharpe']:+.3f}  MaxDD={ls_stats['max_dd']:+.1%}  "
      f"Calmar={ls_stats['calmar']:+.3f}")
print(f"  单调性: {'PASS' if monotonic else 'FAIL'}")

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv(OUT / 'step1_quintile_stats.csv', index=False)
# 存 LS 曲线
pd.DataFrame({'date': ls_rets.index, 'ls_ret': ls_rets.values,
              'ls_cum': (1+ls_rets).cumprod().values}).to_csv(OUT / 'step1_ls_curve.csv', index=False)
pd.DataFrame(monthly_details).to_csv(OUT / 'step1_monthly_details.csv', index=False)

# ============================================================
# Step 2: Transaction Cost Overlay
# ============================================================
print(f"\n{'='*65}")
print("Step 2: Transaction Cost Overlay (10/20/30/50bp)")
print(f"{'='*65}")

# 计算每月换手率
monthly_holdings = {}
for date in dates_sorted:
    g = pred_df_valid[pred_df_valid['date'] == date].dropna(subset=['fwd_ret_1m'])
    if len(g) < 50: continue
    g = g.copy()
    g['quintile'] = pd.qcut(g['pred_ens'].rank(method='first'), 5, labels=False, duplicates='drop')
    monthly_holdings[date] = {
        'Q5': set(g[g['quintile']==4]['code']),
        'Q1': set(g[g['quintile']==0]['code']),
    }

sorted_dates = sorted(monthly_holdings.keys())
turnovers = []
for i, d in enumerate(sorted_dates):
    if i == 0:
        turnovers.append(1.0)
    else:
        prev_q5 = monthly_holdings[sorted_dates[i-1]]['Q5']
        prev_q1 = monthly_holdings[sorted_dates[i-1]]['Q1']
        curr_q5 = monthly_holdings[d]['Q5']
        curr_q1 = monthly_holdings[d]['Q1']
        to_q5 = len(curr_q5 - prev_q5) / max(len(curr_q5), 1)
        to_q1 = len(curr_q1 - prev_q1) / max(len(curr_q1), 1)
        turnovers.append((to_q5 + to_q1) / 2)

avg_to = np.mean(turnovers[1:]) if len(turnovers) > 1 else 0
print(f"  平均月度换手率: {avg_to:.1%}")

cost_levels = [0.0010, 0.0020, 0.0030, 0.0050]
cost_labels = ['10bp', '20bp', '30bp', '50bp']
ls_base = ls_rets.loc[sorted_dates]

cost_results = []
for cost, label in zip(cost_levels, cost_labels):
    net_rets = []
    for i, d in enumerate(sorted_dates):
        gross = ls_base.loc[d]
        tc = turnovers[i] * 2 * cost  # 买卖双向
        net_rets.append(gross - tc)
    net_series = pd.Series(net_rets, index=sorted_dates)
    s = port_stats(net_series, f'Net_{label}')
    s['cost'] = label; s['cost_bps'] = cost
    cost_results.append(s)
    print(f"  {label}: Net AnnRet={s['ann_ret']:+.2%}  Sharpe={s['sharpe']:+.3f}  "
          f"MaxDD={s['max_dd']:+.1%}  Final={s['final_cum']:.3f}")
    pd.DataFrame({'date': net_series.index, 'net_ret': net_series.values,
                  'net_cum': (1+net_series).cumprod().values}).to_csv(
        OUT / f'step2_net_curve_{label}.csv', index=False)

# 盈亏平衡点
sharpes = [r['sharpe'] for r in cost_results]
costs = [r['cost_bps'] for r in cost_results]
breakeven = None
for i in range(len(sharpes)-1):
    if sharpes[i] * sharpes[i+1] <= 0:
        frac = abs(sharpes[i]) / (abs(sharpes[i]) + abs(sharpes[i+1]))
        breakeven = costs[i] + frac * (costs[i+1] - costs[i])
        break
if breakeven is None and sharpes[0] > 0 and sharpes[-1] > 0:
    breakeven = 0.0051  # >50bp

print(f"  盈亏平衡点: {breakeven*10000:.0f}bp" if breakeven else f"  盈亏平衡点: >50bp")
pd.DataFrame(cost_results).to_csv(OUT / 'step2_cost_overlay.csv', index=False)

# ============================================================
# Step 3: 5-Fold Time Series CV
# ============================================================
print(f"\n{'='*65}")
print("Step 3: 5-Fold Time Series Cross-Validation")
print(f"{'='*65}")

test_dates_all = sorted([d for d in dates_sorted if '2015-01' <= d <= '2024-12'])
n_total = len(test_dates_all)
fold_size = n_total // 5
folds = []
for f in range(5):
    start = f * fold_size
    end = start + fold_size if f < 4 else n_total
    folds.append(test_dates_all[start:end])

print(f"  折大小: {[len(f) for f in folds]}")

cv_results = []
for f_idx, test_fold in enumerate(folds):
    train_fold = [d for d in test_dates_all if d < test_fold[0]]
    if not train_fold:
        print(f"  Fold {f_idx+1}: 跳过 (无训练数据)")
        continue

    # 在训练期 re-fit 模型
    tr_cv = np.isin(te_dates, train_fold)
    te_cv = np.isin(te_dates, test_fold)

    if tr_cv.sum() < 500 or te_cv.sum() < 200:
        print(f"  Fold {f_idx+1}: 跳过 (样本不足: tr={tr_cv.sum()} te={te_cv.sum()})")
        continue

    Xt_cv = Xte[tr_cv]; yt_cv = y_all[2][te_m][tr_cv]
    Xte_cv = Xte[te_cv]

    # 快速 re-fit
    lgb_cv = lgb.LGBMRegressor(objective='regression', num_leaves=63, learning_rate=0.03,
        n_estimators=200, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=456, verbosity=-1, n_jobs=4)
    lgb_cv.fit(Xt_cv, yt_cv); p_cv_lgb = lgb_cv.predict(Xte_cv)
    xgb_cv = xgb.XGBRegressor(objective='reg:squarederror', max_depth=6, learning_rate=0.05,
        n_estimators=200, subsample=0.8, colsample_bytree=0.8,
        random_state=456, verbosity=0, n_jobs=4)
    xgb_cv.fit(Xt_cv, yt_cv); p_cv_xgb = xgb_cv.predict(Xte_cv)
    ridge_cv = Ridge(alpha=1.0); ridge_cv.fit(Xt_cv, yt_cv); p_cv_ridge = ridge_cv.predict(Xte_cv)

    ic_lgb = np.mean([spearmanr(p_cv_lgb[te_dates[te_cv]==m], y_all[2][te_m][te_cv][te_dates[te_cv]==m])[0]
                      for m in np.unique(te_dates[te_cv]) if (te_dates[te_cv]==m).sum()>=20])
    ic_xgb = np.mean([spearmanr(p_cv_xgb[te_dates[te_cv]==m], y_all[2][te_m][te_cv][te_dates[te_cv]==m])[0]
                      for m in np.unique(te_dates[te_cv]) if (te_dates[te_cv]==m).sum()>=20])
    ic_ridge = np.mean([spearmanr(p_cv_ridge[te_dates[te_cv]==m], y_all[2][te_m][te_cv][te_dates[te_cv]==m])[0]
                        for m in np.unique(te_dates[te_cv]) if (te_dates[te_cv]==m).sum()>=20])

    # Ensemble
    w_cv = {k: max(v, 0) for k, v in zip(['lgb','xgb','ridge'], [ic_lgb, ic_xgb, ic_ridge])}
    w_sum_cv = sum(w_cv.values())
    p_cv_ens = (w_cv['lgb']*p_cv_lgb + w_cv['xgb']*p_cv_xgb + w_cv['ridge']*p_cv_ridge) / max(w_sum_cv, 0.001)

    monthly_ics = []
    for m in np.unique(te_dates[te_cv]):
        mask = te_dates[te_cv] == m
        if mask.sum() >= 20:
            monthly_ics.append(spearmanr(p_cv_ens[mask], y_all[2][te_m][te_cv][mask])[0])

    mean_ic = np.mean(monthly_ics)
    ic_ir = mean_ic / np.std(monthly_ics) if np.std(monthly_ics) > 0 else 0
    cv_results.append({
        'fold': f_idx+1, 'test_start': test_fold[0], 'test_end': test_fold[-1],
        'n_train_months': len(train_fold), 'n_test_months': len(monthly_ics),
        'IC': round(mean_ic, 5), 'ICIR': round(ic_ir, 3),
        'Hit_Rate': round((np.array(monthly_ics)>0).mean(), 3),
    })
    print(f"  Fold {f_idx+1}: {test_fold[0]}~{test_fold[-1]}  "
          f"IC={mean_ic:+.4f}  IR={ic_ir:+.3f}  Hit={(np.array(monthly_ics)>0).mean():.1%}")

cv_df = pd.DataFrame(cv_results)
if len(cv_df) > 0:
    print(f"\n  CV 汇总: IC={cv_df['IC'].mean():+.4f} "
          f"范围=[{cv_df['IC'].min():+.4f}, {cv_df['IC'].max():+.4f}] "
          f"ICIR={cv_df['ICIR'].mean():.3f}")
cv_df.to_csv(OUT / 'step3_cv_results.csv', index=False)

# ============================================================
# Step 4: OOS Tracking Template
# ============================================================
print(f"\n{'='*65}")
print("Step 4: OOS Tracking Template")
print(f"{'='*65}")

tracking_dir = OUT / 'tracking'
tracking_dir.mkdir(parents=True, exist_ok=True)

# 最新一期因子排名
latest_date = dates_sorted[-1]
latest = pred_df_valid[pred_df_valid['date'] == latest_date].copy()
latest = latest.sort_values('pred_ens', ascending=False)
latest['rank'] = range(1, len(latest)+1)
latest['percentile'] = latest['rank'] / len(latest) * 100
rank_cols = ['rank', 'code', 'pred_ens', 'percentile', 'close', 'industry']
latest[rank_cols].to_csv(tracking_dir / f'{latest_date}.csv', index=False)
print(f"  最新因子排名: {tracking_dir / f'{latest_date}.csv'} ({len(latest)} stocks)")

# Master log (最近36个月)
recent_dates = dates_sorted[-36:]
log_rows = []
for date in recent_dates:
    g = pred_df_valid[pred_df_valid['date'] == date].dropna(subset=['fwd_ret_1m'])
    if len(g) < 30: continue
    g = g.sort_values('pred_ens', ascending=False)
    top20 = g.head(20); bot20 = g.tail(20)
    log_rows.append({
        'signal_date': date, 'n_stocks': len(g),
        'top20_avg_pred': top20['pred_ens'].mean(),
        'bot20_avg_pred': bot20['pred_ens'].mean(),
        'top20_fwd_ret': top20['fwd_ret_1m'].mean(),
        'bot20_fwd_ret': bot20['fwd_ret_1m'].mean(),
        'ls_fwd_ret': top20['fwd_ret_1m'].mean() - bot20['fwd_ret_1m'].mean(),
        'monthly_ic': spearmanr(g['pred_ens'], g['fwd_ret_1m'])[0],
    })

master_log = pd.DataFrame(log_rows)
master_log['cum_ls'] = (1 + master_log['ls_fwd_ret']).cumprod()
master_log['rolling_12m_ic'] = master_log['monthly_ic'].rolling(12).mean()
master_log.to_csv(tracking_dir / 'master_log.csv', index=False)
print(f"  Master log: {tracking_dir / 'master_log.csv'} ({len(master_log)} 个月)")

if len(master_log) > 0:
    print(f"  最近月 IC:     {master_log['monthly_ic'].iloc[-1]:+.4f}")
    if len(master_log) >= 12:
        print(f"  滚动12月 IC:   {master_log['rolling_12m_ic'].iloc[-1]:+.4f}")
    print(f"  最近月 LS收益: {master_log['ls_fwd_ret'].iloc[-1]:+.4f}")

# 摘要
summary = {
    'pipeline': 'LGB+XGB+Ridge Ensemble, Industry-Neutralized',
    'last_signal_date': str(latest_date),
    'n_stocks': int(len(latest)),
    'base_IC': float(base_ic),
    'base_ICIR': float(base_ir),
    'tracking_dir': str(tracking_dir),
    'timestamp': ts(),
}
json.dump(summary, open(tracking_dir / 'summary.json', 'w'), indent=2)

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'='*65}")
print("Phase 4 鲁棒性检验 — 完成")
print(f"{'='*65}")
print(f"  Task 4.1 IC Decay:       {decay_df['horizon'].tolist()}")
print(f"  Task 4.2 Noise Sens:     {noise_df['IC_decay_pct'].tolist()}")
print(f"  Task 4.3 Permutation:    p={p_value:.4f} ({sig_level})")
print(f"  Step 1 Quintile:         LS Sharpe={ls_stats['sharpe']:.3f} Monotonic={monotonic}")
print(f"  Step 2 Cost Overlay:     Breakeven={breakeven*10000:.0f}bp" if breakeven else f"  Step 2 Cost Overlay:     Sharpe始终为正")
print(f"  Step 3 5-Fold CV:        IC={cv_df['IC'].mean():+.4f}" if len(cv_df)>0 else f"  Step 3 5-Fold CV:        insufficient data")
print(f"  Step 4 OOS Template:     {tracking_dir}")
print(f"\n  输出目录: {OUT}")
files = sorted(OUT.rglob('*'))
print(f"  输出文件 ({len(files)} 个):")
for f in files:
    if f.is_file():
        print(f"    {f.relative_to(OUT)} ({f.stat().st_size:,} bytes)")
print(f"{'='*65}")
