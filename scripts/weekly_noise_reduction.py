"""Weekly noise diagnosis + denoising plan: systematically solve high noise + complex model overfitting

Diagnostic dimensions:
  1. Target noise: 13-week return variance decomposition (market beta vs stock alpha)
  2. Feature noise: autocorrelation decay of each feature (signal persistence)
  3. Overfitting diagnosis: model capacity vs effective sample size
  4. Cross-sectional normalization: whether signal strengthens after removing market volatility

Denoising approaches:
  A. Cross-sectional rank normalization (convert features/targets to rank percentiles per period)
  B. Target smoothing (discounted multi-horizon return vs single-point return)
  C. Model capacity reduction (small model + strong regularization)
  D. Sample denoising (volatility-weighted training loss)
"""
import numpy as np, pandas as pd, sqlite3, warnings
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_score, recall_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

# ======== 0. Load data (reuse weekly_daily_bridge feature build logic) ========
print("0/5 Loading & building features...")

conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
w_df = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume FROM weekly_klines
    WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01-01'
    ORDER BY code, date
""", conn, params=stocks)
conn.close()
w_df['date'] = w_df['date'].astype(str)

# Daily aggregation (v6: time-safe, with in_sample flag)
sig_path = OUT / 'daily_signals_v6.parquet'
if not sig_path.exists():
    sig_path = OUT / 'daily_signals.parquet'  # fallback to old
    print("  WARN: v6 signals not found, using old daily_signals.parquet")
daily_sig = pd.read_parquet(sig_path)
daily_sig['date_dt'] = pd.to_datetime(daily_sig['date'])
daily_sig['week_id'] = daily_sig['date_dt'].dt.isocalendar().year.astype(str) + '-W' + \
    daily_sig['date_dt'].dt.isocalendar().week.astype(str).str.zfill(2)
# When aggregating, exclude in_sample flag (only aggregate out-of-sample signals)
has_in_sample = 'in_sample' in daily_sig.columns
daily_agg = daily_sig.groupby(['code', 'week_id']).agg(
    ds_mean=('score', 'mean'), ds_std=('score', 'std'),
    ds_last=('score', 'last'), ds_pos=('score', lambda x: (x > 0).mean()),
    ds_is_clean=('in_sample', lambda x: not any(x)) if has_in_sample else ('score', lambda x: True),
).reset_index()
if not has_in_sample:
    daily_agg['ds_is_clean'] = False  # old signals: assume contaminated
clean_pct = daily_agg['ds_is_clean'].mean() * 100 if 'ds_is_clean' in daily_agg.columns else 0
print(f"  Clean daily signals (v6, OOS): {clean_pct:.0f}% of weeks")

w_df['date_dt'] = pd.to_datetime(w_df['date'])
w_df['week_id'] = w_df['date_dt'].dt.isocalendar().year.astype(str) + '-W' + \
    w_df['date_dt'].dt.isocalendar().week.astype(str).str.zfill(2)
w_df = w_df.merge(daily_agg, on=['code', 'week_id'], how='left')

def compute_features(g):
    n = len(g)
    closes = g['close'].values.astype(float)
    highs = g['high'].values.astype(float)
    lows = g['low'].values.astype(float)
    vols = g['volume'].values.astype(float)
    rows = []

    for i in range(52, n - 13):
        if closes[i] <= 0.01: continue

        # Weekly momentum
        ret_1w = (closes[i] - closes[i-1]) / max(closes[i-1], 0.01) if i >= 1 else 0
        ret_4w = (closes[i] - closes[i-4]) / max(closes[i-4], 0.01) if i >= 4 else 0
        ret_13w = (closes[i] - closes[i-13]) / max(closes[i-13], 0.01) if i >= 13 else 0
        ret_26w = (closes[i] - closes[i-26]) / max(closes[i-26], 0.01) if i >= 26 else 0
        ret_52w = (closes[i] - closes[i-52]) / max(closes[i-52], 0.01) if i >= 52 else 0

        # 52-week position
        h52 = np.max(highs[max(0,i-52):i+1]); l52 = np.min(lows[max(0,i-52):i+1])
        pos_52w = (closes[i] - l52) / max(h52 - l52, 0.01)
        dist_52h = (h52 - closes[i]) / max(h52, 0.01)

        # Volatility
        rets_13 = np.diff(closes[max(0,i-13):i+1]) / np.maximum(closes[max(0,i-13):i], 0.01)
        vol_13w = np.std(rets_13) if len(rets_13) >= 3 else 0
        rets_52 = np.diff(closes[max(0,i-52):i+1]) / np.maximum(closes[max(0,i-52):i], 0.01)
        vol_52w = np.std(rets_52) if len(rets_52) >= 10 else 0.001
        vol_ratio = vol_13w / max(vol_52w, 0.001)

        # Volume ratio
        vol_4w_avg = np.mean(vols[max(0,i-4):i+1])
        vol_ratio_q = vols[i] / max(vol_4w_avg, 1)

        # MA position
        ma20 = np.mean(closes[max(0,i-20):i+1])
        ma60 = np.mean(closes[max(0,i-60):i+1]) if i >= 60 else ma20
        ma_pos_20 = (closes[i] - ma20) / max(closes[i], 0.01)
        ma_pos_60 = (closes[i] - ma60) / max(closes[i], 0.01)

        # MA alignment
        if i >= 60:
            ma5 = np.mean(closes[max(0,i-5):i+1]); ma10 = np.mean(closes[max(0,i-10):i+1])
            ma_align = 1.0 if ma5 > ma10 > ma20 > ma60 else (-1.0 if ma5 < ma10 < ma20 < ma60 else 0.0)
        else:
            ma_align = 0.0

        # MACD
        close_series = pd.Series(closes[max(0,i-52):i+1])
        e12 = close_series.ewm(span=12).mean().iloc[-1]
        e26 = close_series.ewm(span=26).mean().iloc[-1]
        macd_line = e12 - e26

        # Weekly range
        week_range = (highs[i] - lows[i]) / max(closes[i], 0.01)

        # Daily aggregates
        ds_mean = g['ds_mean'].iloc[i]; ds_std = g['ds_std'].iloc[i]
        ds_last = g['ds_last'].iloc[i]; ds_pos = g['ds_pos'].iloc[i]
        ds_mean = 0.0 if pd.isna(ds_mean) else ds_mean
        ds_std = 0.0 if pd.isna(ds_std) else ds_std
        ds_last = 0.0 if pd.isna(ds_last) else ds_last
        ds_pos = 0.5 if pd.isna(ds_pos) else ds_pos

        # Target: raw 13-week return + discounted multi-horizon return (gamma=0.9, 3 months=13 weeks)
        fwd_raw = (closes[i+13] - closes[i]) / max(closes[i], 0.01)
        fwd_raw = np.clip(fwd_raw, -2, 2)

        # Discounted multi-horizon target (13w + gamma*26w + gamma^2*39w + gamma^3*52w)
        fwd_disc = fwd_raw
        for k, horizon in enumerate([26, 39, 52], 1):
            if i + horizon < n:
                r = (closes[i+horizon] - closes[i]) / max(closes[i], 0.01)
                fwd_disc += (0.9 ** k) * np.clip(r, -2, 2)
        fwd_disc = np.clip(fwd_disc, -3, 3)

        rows.append({
            'code': g['code'].iloc[0], 'date': g['date'].iloc[i],
            'ret_1w': ret_1w, 'ret_4w': ret_4w, 'ret_13w': ret_13w,
            'ret_26w': ret_26w, 'ret_52w': ret_52w,
            'pos_52w': pos_52w, 'dist_52h': dist_52h,
            'vol_13w': vol_13w, 'vol_ratio': vol_ratio, 'vol_ratio_q': vol_ratio_q,
            'ma_pos_20': ma_pos_20, 'ma_pos_60': ma_pos_60,
            'ma_align': ma_align, 'macd_line': macd_line, 'week_range': week_range,
            'ds_mean': ds_mean, 'ds_std': ds_std, 'ds_last': ds_last, 'ds_pos': ds_pos,
            'fwd_raw': fwd_raw, 'fwd_disc': fwd_disc,
        })
    return rows

all_rows = []
for code in stocks:
    g = w_df[w_df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 66: continue
    all_rows.extend(compute_features(g))

df = pd.DataFrame(all_rows)
FEATURES = ['ret_1w', 'ret_4w', 'ret_13w', 'ret_26w', 'ret_52w',
            'pos_52w', 'dist_52h', 'vol_13w', 'vol_ratio', 'vol_ratio_q',
            'ma_pos_20', 'ma_pos_60', 'ma_align', 'macd_line', 'week_range',
            'ds_mean', 'ds_std', 'ds_last', 'ds_pos']
# Baseline features without daily signal (safe feature set)
SAFE_FEATURES = [f for f in FEATURES if not f.startswith('ds_')]

for col in FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)
# Zero out contaminated daily signals (keep only out-of-sample)
if 'ds_is_clean' in df.columns:
    for col in ['ds_mean', 'ds_std', 'ds_last', 'ds_pos']:
        if col in df.columns:
            df.loc[~df['ds_is_clean'], col] = 0.0
    print(f"  Zeroed ds_* for {((~df['ds_is_clean']).sum())} contaminated weeks")

# Strict NaN removal (target cols + feature cols)
df = df.dropna(subset=['fwd_raw', 'fwd_disc'])
for col in FEATURES:
    df = df[np.isfinite(df[col])]

print(f"  Rows: {len(df)}, Stocks: {df.code.nunique()}, Dates: {df.date.min()}~{df.date.max()}")
print(f"  fwd_raw NaN: {df['fwd_raw'].isna().sum()}, Inf: {(~np.isfinite(df['fwd_raw'])).sum()}")
print(f"  fwd_disc NaN: {df['fwd_disc'].isna().sum()}, Inf: {(~np.isfinite(df['fwd_disc'])).sum()}")

# ======== 1. Noise diagnosis ========
print("\n" + "="*70)
print("1. Noise Diagnosis")
print("="*70)

# 1a. Target noise: market beta vs stock alpha
print("\n1a. Target noise decomposition (cross-sectional variance of 13-week forward returns)")
dates_sorted = sorted(df['date'].unique())
total_var, mkt_var, idio_var = [], [], []
for dt in dates_sorted[::4]:  # sample every 4 weeks
    te = df[df['date'] == dt]
    if len(te) < 50: continue
    fwd = te['fwd_raw'].values
    fwd = fwd[np.isfinite(fwd)]
    if len(fwd) < 20: continue
    mkt_ret = np.mean(fwd)
    total_var.append(np.var(fwd))
    mkt_var.append(mkt_ret ** 2)
    idio_var.append(np.var(fwd - mkt_ret))

if total_var:
    print(f"  Total variance:      mean={np.mean(total_var):.4f}")
    print(f"  Market beta variance:    mean={np.mean(mkt_var):.4f} ({100*np.mean(mkt_var)/np.mean(total_var):.1f}%)")
    print(f"  Stock alpha variance:    mean={np.mean(idio_var):.4f} ({100*np.mean(idio_var)/np.mean(total_var):.1f}%)")
else:
    print("  Variance decomposition: no valid data")
print(f"  -> High stock-alpha variance share = large cross-sectional predictability space; high market-beta share = cross-sectional signal submerged by market")

# 1b. Feature autocorrelation (signal persistence)
print("\n1b. Feature autocorrelation decay (signal persistence diagnosis)")
# Take a representative stock, check feature autocorrelation
for code in ['000001', '600519']:
    g = df[df['code'] == code].sort_values('date')
    if len(g) < 100: continue
    print(f"\n  {code}:")
    for feat in ['ret_13w', 'ma_pos_20', 'macd_line', 'pos_52w', 'vol_ratio']:
        vals = g[feat].values[-100:]
        vals = vals - np.mean(vals)
        ac1 = np.corrcoef(vals[:-1], vals[1:])[0,1] if len(vals) > 2 else 0
        ac4 = np.corrcoef(vals[:-4], vals[4:])[0,1] if len(vals) > 5 else 0
        ac13 = np.corrcoef(vals[:-13], vals[13:])[0,1] if len(vals) > 14 else 0
        print(f"    {feat:15s}: AC(1)={ac1:+.3f}  AC(4)={ac4:+.3f}  AC(13)={ac13:+.3f}")

# 1c. Effective sample size estimation
print("\n1c. Effective sample size vs model complexity")
n_stocks = df.code.nunique()
n_weeks_total = len(df)
n_independent = n_stocks * (len(dates_sorted) / 13)  # every 13 weeks ~ independent sample
print(f"  Total samples:     {n_weeks_total:,}")
print(f"  Est. independent:  {n_independent:.0f} (one independent observation per 13 weeks)")
print(f"  Stocks per period: {n_stocks}")
print(f"  -> LSTM-7 params: ~2M, effective samples: ~{n_independent:.0f} -- severe over-parameterization")

# ======== 2. Denoising A: cross-sectional rank normalization ========
print("\n" + "="*70)
print("2. Denoising A: Cross-Sectional Rank Normalization")
print("="*70)

def cross_sectional_rank(df_in, feats, target_col='fwd_raw'):
    """Convert features and target to [0,1] rank percentiles per period"""
    df_out = df_in.copy()
    for col in feats + [target_col]:
        df_out[col + '_rank'] = np.nan
    for dt, grp in df_out.groupby('date'):
        if len(grp) < 30: continue
        for col in feats + [target_col]:
            ranks = grp[col].rank(pct=True)
            df_out.loc[grp.index, col + '_rank'] = ranks
    for col in feats + [target_col]:
        df_out[col + '_rank'] = df_out[col + '_rank'].fillna(0.5)
    return df_out

df_rank = cross_sectional_rank(df, FEATURES, 'fwd_raw')
RANK_FEATS = [f + '_rank' for f in FEATURES]

# ======== 3. Denoising B: discounted multi-horizon target ========
print("\n3. Denoising B: Discounted Multi-Horizon Target")
print(f"  fwd_raw:  mean={df['fwd_raw'].mean():.4f}, std={df['fwd_raw'].std():.4f}")
print(f"  fwd_disc: mean={df['fwd_disc'].mean():.4f}, std={df['fwd_disc'].std():.4f}")
# Correlation
valid = df['fwd_raw'].notna() & df['fwd_disc'].notna()
r = pearsonr(df.loc[valid, 'fwd_raw'], df.loc[valid, 'fwd_disc'])[0]
print(f"  Pearson r(raw, disc) = {r:.4f}")

# ======== 4. Comprehensive Walk-Forward Comparison ========
print("\n" + "="*70)
print("4. Walk-Forward: Denoising Approach Comparison")
print("="*70)

# Approach matrix
configs = {
    'raw_target+raw_feat':   ('fwd_raw', FEATURES, df),
    'raw_target+rank_feat':  ('fwd_raw', RANK_FEATS, df_rank),
    'disc_target+raw_feat':  ('fwd_disc', FEATURES, df),
    'disc_target+rank_feat': ('fwd_disc', RANK_FEATS, df_rank),
}

# Ridge alpha sweep small -> large
ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

# LGB config sweep small -> large
lgb_configs = {
    'tiny':  dict(n_estimators=30, max_depth=2, learning_rate=0.02, min_child_samples=50),
    'small': dict(n_estimators=50, max_depth=3, learning_rate=0.03, min_child_samples=30),
    'medium':dict(n_estimators=80, max_depth=4, learning_rate=0.03, min_child_samples=20),
    'large': dict(n_estimators=150, max_depth=6, learning_rate=0.05, min_child_samples=10),
}

months = sorted(df['date'].unique())

# Test representative configs only
test_configs = [
    # (target_key, feat_list, df_source, model_type, model_params, label)
    ('fwd_raw', SAFE_FEATURES, df, 'ridge', {'alpha': 1.0}, 'B0: raw+safe Ridge (no ds)'),
    ('fwd_raw', FEATURES, df, 'ridge', {'alpha': 1.0}, 'B1: raw+raw Ridge(a=1)'),
    ('fwd_raw', FEATURES, df, 'ridge', {'alpha': 10.0}, 'B2: raw+raw Ridge(a=10)'),
    ('fwd_raw', RANK_FEATS, df_rank, 'ridge', {'alpha': 1.0}, 'A1: raw+rank Ridge'),
    ('fwd_disc', SAFE_FEATURES, df, 'ridge', {'alpha': 1.0}, 'B4: disc+safe Ridge (no ds)'),
    ('fwd_disc', FEATURES, df, 'ridge', {'alpha': 1.0}, 'B3: disc+raw Ridge'),
    ('fwd_disc', RANK_FEATS, df_rank, 'ridge', {'alpha': 1.0}, 'A2: disc+rank Ridge'),
    ('fwd_raw', SAFE_FEATURES, df, 'lgb', lgb_configs['small'], 'C0: raw+safe LGB-small'),
    ('fwd_raw', FEATURES, df, 'lgb', lgb_configs['small'], 'C1: raw+raw LGB-small'),
    ('fwd_raw', FEATURES, df, 'lgb', lgb_configs['tiny'], 'C2: raw+raw LGB-tiny'),
    ('fwd_disc', SAFE_FEATURES, df, 'lgb', lgb_configs['small'], 'C5: disc+safe LGB-small'),
    ('fwd_disc', FEATURES, df, 'lgb', lgb_configs['small'], 'C4: disc+raw LGB-small'),
]

all_results = {}
f1_results = {}  # label -> {f1:[], prec:[], rec:[], ic:[], n:[]}
for target_key, feats, df_src, model, params, label in test_configs:
    results = []
    fold_metrics = {'f1': [], 'precision': [], 'recall': [], 'ic': []}
    for i, month in enumerate(months):
        if month < '2018-01-01': continue
        tr = df_src[df_src['date'] < month]
        te = df_src[df_src['date'] == month]
        if len(tr) < 1000 or len(te) < 10: continue

        X_tr = tr[feats].values.astype(np.float64)
        X_te = te[feats].values.astype(np.float64)
        y_tr = tr[target_key].values.astype(np.float64)
        y_te = te[target_key].values.astype(np.float64)

        # NaN/inf safety
        if np.any(~np.isfinite(X_tr)) or np.any(~np.isfinite(X_te)): continue
        if np.any(~np.isfinite(y_tr)) or np.any(~np.isfinite(y_te)): continue

        try:
            if model == 'ridge':
                m = Ridge(**params).fit(X_tr, y_tr)
                p = m.predict(X_te)
            else:
                m = lgb.LGBMRegressor(**params, subsample=0.7, colsample_bytree=0.7,
                                       reg_alpha=0.5, reg_lambda=0.5,
                                       random_state=42, verbosity=-1).fit(X_tr, y_tr)
                p = m.predict(X_te)
            if len(p) > 10:
                ic = spearmanr(p, y_te)[0]
                if not np.isnan(ic): results.append(ic)
                # F1: predicted direction vs actual direction
                pred_dir = (p > 0).astype(int)
                true_dir = (y_te > 0).astype(int)
                if len(np.unique(pred_dir)) > 1 and len(np.unique(true_dir)) > 1:
                    fold_metrics['f1'].append(f1_score(true_dir, pred_dir, zero_division=0))
                    fold_metrics['precision'].append(precision_score(true_dir, pred_dir, zero_division=0))
                    fold_metrics['recall'].append(recall_score(true_dir, pred_dir, zero_division=0))
                    fold_metrics['ic'].append(ic)
        except Exception:
            pass

    all_results[label] = results
    f1_results[label] = fold_metrics
    if results:
        print(f"  {label:35s} IC={np.mean(results):.4f} +- {np.std(results):.4f} n={len(results)}")

# ======== Baselines (no training needed) ========
print(f"\n  --- Baselines (no training) ---")

for bl_name, bl_predictor in [
    ('BASE: Always-Up',        lambda te: np.ones(len(te))),
    ('BASE: Naive Mom(4w)',    lambda te: te['ret_4w'].values),
    ('BASE: Naive Mom(13w)',   lambda te: te['ret_13w'].values),
]:
    bl_ic, bl_f1 = [], {'f1': [], 'precision': [], 'recall': [], 'ic': []}
    for month in months:
        if month < '2018-01-01': continue
        te = df[df['date'] == month]
        if len(te) < 10: continue
        y_te = te['fwd_disc'].values.astype(np.float64)
        if np.any(~np.isfinite(y_te)): continue
        p = bl_predictor(te).astype(np.float64)
        if len(p) > 10:
            ic = spearmanr(p, y_te)[0]
            if not np.isnan(ic): bl_ic.append(ic)
            pred_dir = (p > 0).astype(int)
            true_dir = (y_te > 0).astype(int)
            if len(np.unique(pred_dir)) > 1 and len(np.unique(true_dir)) > 1:
                bl_f1['f1'].append(f1_score(true_dir, pred_dir, zero_division=0))
                bl_f1['precision'].append(precision_score(true_dir, pred_dir, zero_division=0))
                bl_f1['recall'].append(recall_score(true_dir, pred_dir, zero_division=0))
                bl_f1['ic'].append(ic)
    all_results[bl_name] = bl_ic
    f1_results[bl_name] = bl_f1
    if bl_ic:
        print(f"  {bl_name:35s} IC={np.mean(bl_ic):.4f} +- {np.std(bl_ic):.4f} n={len(bl_ic)}")

# ======== 5. Final report ========
print(f"\n{'='*70}")
print("FINAL: Denoising Approach Effectiveness Ranking")
print(f"{'='*70}")
print(f"{'Rank':<6} {'Method':<35} {'IC Mean':>10} {'IC Std':>10} {'N':>6} {'Win Rate':>10}")
print(f"{'-'*6} {'-'*35} {'-'*10} {'-'*10} {'-'*6} {'-'*10}")

ranked = sorted(all_results.items(), key=lambda x: np.mean(x[1]) if x[1] else -999, reverse=True)
for rank, (label, vals) in enumerate(ranked, 1):
    if not vals: continue
    m = np.mean(vals); s = np.std(vals)
    wr = np.mean([1 if v > 0 else 0 for v in vals])
    print(f"  {rank:<6} {label:<35} {m:10.4f} {s:10.4f} {len(vals):6} {wr:10.2%}")

# Compare to baseline
print(f"\n  Baseline LSTM-7:     IC=0.0074")
print(f"  Baseline LightGBM:   IC=0.0447")
if ranked:
    best_name, best_vals = ranked[0]
    print(f"  Best (denoised):    IC={np.mean(best_vals):.4f} ({best_name})")
    gain = np.mean(best_vals) / 0.0074
    print(f"  vs LSTM-7 baseline: {gain:.0f}x")

# ======== F1 Score summary ========
print(f"\n{'='*70}")
print("F1 Score: Directional Prediction Accuracy (predicted up/down vs actual up/down)")
print(f"{'='*70}")
print(f"{'Rank':<6} {'Method':<35} {'F1':>8} {'Precision':>10} {'Recall':>8} {'IC':>8} {'N Folds':>8}")
print(f"{'-'*6} {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

f1_ranked = sorted(f1_results.items(), key=lambda x: np.mean(x[1]['f1']) if x[1]['f1'] else 0, reverse=True)
for rank, (label, m) in enumerate(f1_ranked, 1):
    if not m['f1']: continue
    f1_m = np.mean(m['f1'])
    p_m = np.mean(m['precision'])
    r_m = np.mean(m['recall'])
    ic_m = np.mean(m['ic'])
    print(f"  {rank:<6} {label:<35} {f1_m:8.4f} {p_m:10.4f} {r_m:8.4f} {ic_m:8.4f} {len(m['f1']):8}")

print(f"\n{'='*70}")
