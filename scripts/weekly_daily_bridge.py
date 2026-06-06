"""Weekly IC improvement plan: daily signal aggregation bridge + enriched features + multi-model comparison
Core hypothesis: daily LSTM (IC=0.141) already has predictive power; aggregating to weekly level can transfer signal

Approach:
  A. Direct aggregation: daily score -> weekly agg -> evaluate IC (zero cost)
  B. Aggregate signal + weekly tech features -> Ridge/LightGBM -> walk-forward IC
  C. Classification target: 5-class classification vs regression comparison
"""
import numpy as np, pandas as pd, sqlite3, warnings
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import lightgbm as lgb

warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'

# ======== 1. Load data ========
print("1/4 Loading data...")

# Daily signals
daily_sig = pd.read_parquet(OUT / 'daily_signals.parquet')
daily_sig['date'] = daily_sig['date'].astype(str)
print(f"  Daily signals: {len(daily_sig)} rows, {daily_sig.code.nunique()} stocks")

# Weekly klines
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute(
    "SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
w_df = pd.read_sql_query(f"""
    SELECT code, date, open, high, low, close, volume
    FROM weekly_klines
    WHERE code IN ({','.join('?'*len(stocks))})
    AND date >= '2010-01-01'
    ORDER BY code, date
""", conn, params=stocks)
conn.close()
w_df['date'] = w_df['date'].astype(str)
print(f"  Weekly klines: {len(w_df)} rows, {w_df.code.nunique()} stocks")

# ======== 2. Aggregate daily signals to weekly ========
print("\n2/4 Aggregating daily signals to weekly level...")

# For each weekly bar, find corresponding daily signals (weekly date is the last trading day of that week)
daily_sig['date_dt'] = pd.to_datetime(daily_sig['date'])
w_df['date_dt'] = pd.to_datetime(w_df['date'])

# More robust approach: for each week, aggregate all daily signals within that week
# First build week_id: YYYY-WW
daily_sig['week_id'] = daily_sig['date_dt'].dt.isocalendar().year.astype(str) + '-W' + \
    daily_sig['date_dt'].dt.isocalendar().week.astype(str).str.zfill(2)
w_df['week_id'] = w_df['date_dt'].dt.isocalendar().year.astype(str) + '-W' + \
    w_df['date_dt'].dt.isocalendar().week.astype(str).str.zfill(2)

# Aggregate daily signals for each (code, week_id)
daily_agg = daily_sig.groupby(['code', 'week_id']).agg(
    ds_mean=('score', 'mean'),
    ds_std=('score', 'std'),
    ds_last=('score', 'last'),
    ds_max=('score', 'max'),
    ds_min=('score', 'min'),
    ds_pos_ratio=('score', lambda x: (x > 0).mean()),
    ds_days=('score', 'count'),
).reset_index()

# Compute intra-week trend: difference between last 3 days avg and first 3 days avg
def week_trend(group):
    if len(group) < 5:
        return 0.0
    first_half = group['score'].iloc[:len(group)//2].mean()
    second_half = group['score'].iloc[len(group)//2:].mean()
    return second_half - first_half

daily_trend = daily_sig.groupby(['code', 'week_id']).apply(week_trend).reset_index(name='ds_trend')

daily_agg = daily_agg.merge(daily_trend, on=['code', 'week_id'], how='left')
daily_agg['ds_trend'] = daily_agg['ds_trend'].fillna(0)

print(f"  Weekly aggregates: {len(daily_agg)} rows")
print(f"  Coverage: {daily_agg.ds_days.mean():.1f} days/week avg")

# Merge daily aggregates to weekly klines
w_df = w_df.merge(daily_agg, on=['code', 'week_id'], how='left')

# ======== 3. Build weekly features + targets ========
print("\n3/4 Building weekly features & targets...")

def compute_weekly_features(g):
    """Build weekly feature matrix for a single stock"""
    n = len(g)
    closes = g['close'].values.astype(float)
    highs = g['high'].values.astype(float)
    lows = g['low'].values.astype(float)
    volumes = g['volume'].values.astype(float)

    rows = []
    for i in range(52, n - 13):  # need 52-week lookback + 13-week forward
        if closes[i] <= 0.01:
            continue

        # -- Daily aggregate signals (already merged) --
        ds_mean = g['ds_mean'].iloc[i]
        ds_std = g['ds_std'].iloc[i]
        ds_last = g['ds_last'].iloc[i]
        ds_trend = g['ds_trend'].iloc[i]
        ds_pos = g['ds_pos_ratio'].iloc[i]

        # Validity flag: whether this week has daily signals
        has_daily = 1.0 if not pd.isna(ds_mean) else 0.0
        ds_mean = 0.0 if pd.isna(ds_mean) else ds_mean
        ds_std = 0.0 if pd.isna(ds_std) else ds_std
        ds_last = 0.0 if pd.isna(ds_last) else ds_last
        ds_trend = 0.0 if pd.isna(ds_trend) else ds_trend
        ds_pos = 0.5 if pd.isna(ds_pos) else ds_pos

        # -- Weekly momentum --
        ret_1w = (closes[i] - closes[i-1]) / max(closes[i-1], 0.01) if i >= 1 else 0
        ret_4w = (closes[i] - closes[i-4]) / max(closes[i-4], 0.01) if i >= 4 else 0
        ret_13w = (closes[i] - closes[i-13]) / max(closes[i-13], 0.01) if i >= 13 else 0
        ret_26w = (closes[i] - closes[i-26]) / max(closes[i-26], 0.01) if i >= 26 else 0
        ret_52w = (closes[i] - closes[i-52]) / max(closes[i-52], 0.01) if i >= 52 else 0

        # -- 52-week position --
        h52 = np.max(highs[max(0,i-52):i+1])
        l52 = np.min(lows[max(0,i-52):i+1])
        pos_52w = (closes[i] - l52) / max(h52 - l52, 0.01)
        dist_52h = (h52 - closes[i]) / max(h52, 0.01)

        # -- Volatility features --
        rets_13w = np.diff(closes[max(0,i-13):i+1]) / np.maximum(closes[max(0,i-13):i], 0.01)
        vol_13w = np.std(rets_13w) if len(rets_13w) >= 3 else 0
        rets_52w = np.diff(closes[max(0,i-52):i+1]) / np.maximum(closes[max(0,i-52):i], 0.01)
        vol_52w = np.std(rets_52w) if len(rets_52w) >= 10 else 0
        vol_ratio = vol_13w / max(vol_52w, 0.001)  # recent volatility vs long-term

        # -- Volume features --
        vol_4w_avg = np.mean(volumes[max(0,i-4):i+1])
        vol_ratio_q = volumes[i] / max(vol_4w_avg, 1)

        # -- Weekly technical indicators --
        # MA position
        ma20 = np.mean(closes[max(0,i-20):i+1])
        ma60 = np.mean(closes[max(0,i-60):i+1]) if i >= 60 else ma20
        ma_pos_20 = (closes[i] - ma20) / max(closes[i], 0.01)
        ma_pos_60 = (closes[i] - ma60) / max(closes[i], 0.01)

        # MA alignment
        if i >= 60:
            ma5 = np.mean(closes[max(0,i-5):i+1])
            ma10 = np.mean(closes[max(0,i-10):i+1])
            ma_align = 1.0 if ma5 > ma10 > ma20 > ma60 else (-1.0 if ma5 < ma10 < ma20 < ma60 else 0.0)
        else:
            ma_align = 0.0

        # Weekly MACD
        close_series = pd.Series(closes[max(0,i-52):i+1])
        e12 = close_series.ewm(span=12).mean().iloc[-1]
        e26 = close_series.ewm(span=26).mean().iloc[-1]
        macd_line = e12 - e26

        # -- Weekly range --
        week_range = (highs[i] - lows[i]) / max(closes[i], 0.01)

        # -- Target: 13-week forward return --
        fwd_ret = (closes[i+13] - closes[i]) / max(closes[i], 0.01)
        fwd_ret = np.clip(fwd_ret, -2, 2)

        # Classification labels
        if fwd_ret >= 0.15:
            label = 2  # strong_bull
        elif fwd_ret >= 0.05:
            label = 1  # bull
        elif fwd_ret > -0.05:
            label = 0  # neutral
        elif fwd_ret > -0.15:
            label = -1  # bear
        else:
            label = -2  # strong_bear

        rows.append({
            'code': g['code'].iloc[0],
            'date': g['date'].iloc[i],
            # Daily aggregate signals (6-dim)
            'ds_mean': ds_mean, 'ds_std': ds_std, 'ds_last': ds_last,
            'ds_trend': ds_trend, 'ds_pos': ds_pos, 'has_daily': has_daily,
            # Weekly momentum (5-dim)
            'ret_1w': ret_1w, 'ret_4w': ret_4w, 'ret_13w': ret_13w, 'ret_26w': ret_26w, 'ret_52w': ret_52w,
            # Position (2-dim)
            'pos_52w': pos_52w, 'dist_52h': dist_52h,
            # Volatility (2-dim)
            'vol_13w': vol_13w, 'vol_ratio': vol_ratio,
            # Volume (1-dim)
            'vol_ratio_q': vol_ratio_q,
            # Technical indicators (4-dim)
            'ma_pos_20': ma_pos_20, 'ma_pos_60': ma_pos_60, 'ma_align': ma_align,
            'macd_line': macd_line,
            # Weekly range (1-dim)
            'week_range': week_range,
            # Target
            'fwd_ret': fwd_ret, 'label': label,
        })

    return rows

all_rows = []
stock_count = 0
for code in stocks:
    g = w_df[w_df['code'] == code].sort_values('date').reset_index(drop=True)
    if len(g) < 66:  # need 52 lookback + 13 forward
        continue
    stock_count += 1
    rows = compute_weekly_features(g)
    all_rows.extend(rows)

# Feature set definitions (must define before NaN filling)
DAILY_FEATS = ['ds_mean', 'ds_std', 'ds_last', 'ds_trend', 'ds_pos', 'has_daily']
MOM_FEATS = ['ret_1w', 'ret_4w', 'ret_13w', 'ret_26w', 'ret_52w']
POS_FEATS = ['pos_52w', 'dist_52h']
VOL_FEATS = ['vol_13w', 'vol_ratio']
TECH_FEATS = ['ma_pos_20', 'ma_pos_60', 'ma_align', 'macd_line', 'week_range', 'vol_ratio_q']

ALL_FEATS = DAILY_FEATS + MOM_FEATS + POS_FEATS + VOL_FEATS + TECH_FEATS
BASELINE_FEATS = MOM_FEATS + POS_FEATS + TECH_FEATS  # 15-dim (no daily signal)
ENRICHED_FEATS = DAILY_FEATS + BASELINE_FEATS  # 21-dim (with daily signal)

df = pd.DataFrame(all_rows).dropna(subset=['fwd_ret'])
# Fill NaN features (some weekly bars may be missing daily signals)
for col in ALL_FEATS:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)
print(f"  Feature rows: {len(df)}, stocks: {df.code.nunique()}")
print(f"  NaN check: {df[ENRICHED_FEATS].isna().sum().sum()}")
print(f"  Label dist: {df.label.value_counts().sort_index().to_dict()}")
print(f"  Target stats: mean={df.fwd_ret.mean():.4f}, std={df.fwd_ret.std():.4f}")

# ======== 4. Walk-Forward Evaluation ========
print("\n4/4 Walk-Forward Evaluation")
print("="*80)

months = sorted(df['date'].unique())
results = {
    'baseline_ridge': [],   # Ridge + baseline features
    'enriched_ridge': [],   # Ridge + with daily aggregation
    'enriched_lgb': [],     # LightGBM + with daily aggregation
    'daily_agg_only': [],   # Pure daily aggregate signal (zero cost)
}

for i, month in enumerate(months):
    if month < '2018-01-01':
        continue

    tr = df[df['date'] < month]
    te = df[df['date'] == month]
    if len(tr) < 1000 or len(te) < 10:
        continue

    y_tr = tr['fwd_ret'].values.astype(np.float64)
    y_te = te['fwd_ret'].values.astype(np.float64)

    # A. Pure daily aggregation: ds_mean directly as weekly signal
    ds_pred = te['ds_mean'].fillna(0).values.astype(np.float64)
    if len(ds_pred) > 10:
        ic = spearmanr(ds_pred, y_te)[0]
        if not np.isnan(ic):
            results['daily_agg_only'].append(ic)

    # B. Ridge + baseline features
    Xb_tr = tr[BASELINE_FEATS].values.astype(np.float64)
    Xb_te = te[BASELINE_FEATS].values.astype(np.float64)
    ridge_b = Ridge(alpha=1.0).fit(Xb_tr, y_tr)
    p_b = ridge_b.predict(Xb_te)
    if len(p_b) > 10:
        ic = spearmanr(p_b, y_te)[0]
        if not np.isnan(ic):
            results['baseline_ridge'].append(ic)

    # C. Ridge + with daily aggregation
    Xe_tr = tr[ENRICHED_FEATS].values.astype(np.float64)
    Xe_te = te[ENRICHED_FEATS].values.astype(np.float64)
    ridge_e = Ridge(alpha=1.0).fit(Xe_tr, y_tr)
    p_e = ridge_e.predict(Xe_te)
    if len(p_e) > 10:
        ic = spearmanr(p_e, y_te)[0]
        if not np.isnan(ic):
            results['enriched_ridge'].append(ic)

    # D. LightGBM + with daily aggregation
    try:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=80, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=0.5,
            min_child_samples=30, random_state=42, verbosity=-1,
        ).fit(Xe_tr, y_tr)
        p_lgb = lgb_model.predict(Xe_te)
        if len(p_lgb) > 10:
            ic = spearmanr(p_lgb, y_te)[0]
            if not np.isnan(ic):
                results['enriched_lgb'].append(ic)
    except Exception:
        pass

    if i % 24 == 0:
        def avg_last(r, n=6):
            vals = results[r]
            return np.mean(vals[-n:]) if len(vals) >= n else (np.mean(vals) if vals else 0)
        print(f"  {month}: agg={avg_last('daily_agg_only'):.4f} "
              f"baseR={avg_last('baseline_ridge'):.4f} "
              f"richR={avg_last('enriched_ridge'):.4f} "
              f"richL={avg_last('enriched_lgb'):.4f}")

# ======== Final report ========
print(f"\n{'='*80}")
print("FINAL: Weekly IC Improvement Report")
print(f"{'='*80}")
print(f"{'Method':<25} {'IC Mean':>10} {'IC Std':>10} {'N':>8} {'Win Rate':>10}")
print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

for name, vals in results.items():
    if vals:
        m = np.mean(vals)
        s = np.std(vals)
        wr = np.mean([1 if v > 0 else 0 for v in vals])
        print(f"  {name:<25} {m:10.4f} {s:10.4f} {len(vals):8} {wr:10.2%}")

# Compare to baseline (LSTM-7 IC=0.007)
print(f"\n  Baseline: existing LSTM-7 weekly IC = 0.0074")
best_name = max(results.items(), key=lambda x: np.mean(x[1]) if x[1] else -999)[0]
best_ic = np.mean(results[best_name]) if results[best_name] else 0
print(f"  Best: {best_name} IC = {best_ic:.4f} (improvement {best_ic/0.0074:.0f}x)")

# ======== Feature importance ========
if results['enriched_lgb']:
    print(f"\n{'='*60}")
    print("LightGBM Feature Importance (full dataset training)")
    X_all = df[ENRICHED_FEATS].values.astype(np.float64)
    y_all = df['fwd_ret'].values.astype(np.float64)
    final_lgb = lgb.LGBMRegressor(
        n_estimators=80, max_depth=4, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=0.5,
        min_child_samples=30, random_state=42, verbosity=-1,
    ).fit(X_all, y_all)

    importances = final_lgb.feature_importances_
    idx = np.argsort(importances)[::-1]
    print(f"{'Feature':<20} {'Importance':>12}")
    print(f"{'-'*20} {'-'*12}")
    for i in idx:
        print(f"  {ENRICHED_FEATS[i]:<20} {importances[i]:12.4f}")

print(f"\n{'='*80}")
print("Done.")
