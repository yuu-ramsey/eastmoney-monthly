"""Step 1 核实: walk-forward split / skipped stocks / y pre-clip distribution"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent.parent.parent / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
LOOKBACK = 60

conn = sqlite3.connect(str(DB_PATH))

# All HS300 stocks
stocks_all = pd.read_sql_query(
    "SELECT DISTINCT stock_code FROM stock_industry_mapping", conn
)['stock_code'].tolist()
print(f"HS300 stocks (mapped): {len(stocks_all)}")

# Get klines
df = pd.read_sql_query("""
    SELECT m.code, m.date, m.close FROM monthly_klines m
    WHERE m.code IN (SELECT DISTINCT stock_code FROM stock_industry_mapping)
    ORDER BY m.code, m.date
""", conn)
conn.close()

# ============================================================
# Problem 1: Walk-forward date ranges
# ============================================================
print("\n" + "=" * 60)
print("问题 1: Walk-forward split 日期验证")
print("=" * 60)

# Simulate the same sequence generation logic
all_seq_dates = []  # (code, cutoffDate, y3_date, y6_date)

for code, group in df.groupby('code'):
    group = group.sort_values('date').reset_index(drop=True)
    closes = group['close'].values.astype(np.float64)
    dates = group['date'].tolist()
    n = len(group)

    if n < LOOKBACK + 12:
        continue

    for i in range(LOOKBACK - 1, n - 12):
        if closes[i] <= 0.01:
            continue
        y3_date = dates[i + 3] if i + 3 < n else 'N/A'
        y6_date = dates[i + 6] if i + 6 < n else 'N/A'
        all_seq_dates.append({
            'code': code,
            'cutoff': dates[i],
            'y3_date': y3_date,
            'y6_date': y6_date,
        })

seqs = pd.DataFrame(all_seq_dates)
print(f"Total sequences: {len(seqs)}")

# Split (same logic as dataset.py)
n = len(seqs)
n_train = int(n * 0.55)
n_val = int(n * 0.70)
n_val = min(max(n_val, n_train + 100), n - 100)

train_seqs = seqs.iloc[:n_train]
val_seqs = seqs.iloc[n_train:n_val]
test_seqs = seqs.iloc[n_val:]

for name, s in [('Train', train_seqs), ('Val', val_seqs), ('Test', test_seqs)]:
    print(f"\n{name}: {len(s)} sequences")
    print(f"  cutoff range: {s['cutoff'].min()} ~ {s['cutoff'].max()}")
    print(f"  unique stocks: {s['code'].nunique()}")

# Check overlap
train_dates = set(train_seqs['cutoff'].unique())
val_dates = set(val_seqs['cutoff'].unique())
test_dates = set(test_seqs['cutoff'].unique())

train_val_overlap = train_dates & val_dates
val_test_overlap = val_dates & test_dates
train_test_overlap = train_dates & test_dates

print(f"\n日期重叠检查:")
print(f"  Train ∩ Val: {len(train_val_overlap)} dates")
if train_val_overlap:
    print(f"    {sorted(train_val_overlap)[:5]}...")
print(f"  Val ∩ Test: {len(val_test_overlap)} dates")
if val_test_overlap:
    print(f"    {sorted(val_test_overlap)[:5]}...")
print(f"  Train ∩ Test: {len(train_test_overlap)} dates")
if train_test_overlap:
    print(f"    {sorted(train_test_overlap)[:5]}...")

# Check expected rules
train_max = train_seqs['cutoff'].max()
val_min = val_seqs['cutoff'].min()
val_max = val_seqs['cutoff'].max()
test_min = test_seqs['cutoff'].min()

overlap = (train_max >= val_min) or (val_max >= test_min) or (train_max >= test_min)
print(f"\n训练期约束: Train max cutoff={train_max}")
print(f"验证期约束: Val cutoff=[{val_min}, {val_max}]")
print(f"测试期约束: Test min cutoff={test_min}")

if train_max >= '2022-01':
    print("⚠ 数据穿越风险: Train 包含 2022+ 数据!")
else:
    print("✓ Train cutoff ≤ 2021-12")

# ============================================================
# Problem 2: skipped stocks
# ============================================================
print("\n" + "=" * 60)
print("问题 2: 42 skipped stocks")
print("=" * 60)

skipped = []
for code in stocks_all:
    group = df[df['code'] == code].sort_values('date')
    if len(group) < LOOKBACK + 12:
        skipped.append({
            'code': code,
            'n_months': len(group),
            'date_range': f"{group['date'].iloc[0] if len(group) > 0 else 'N/A'} ~ {group['date'].iloc[-1] if len(group) > 0 else 'N/A'}",
            'reason': 'too_short' if len(group) > 0 else 'no_data',
        })

print(f"Skipped: {len(skipped)}")
if skipped:
    df_skipped = pd.DataFrame(skipped)
    print("\nBy reason:")
    print(df_skipped['reason'].value_counts())
    print("\nTop 10 by n_months:")
    top10 = df_skipped.sort_values('n_months', ascending=False).head(10)
    for _, r in top10.iterrows():
        print(f"  {r['code']}: {r['n_months']} months, {r['date_range']}")

    # Survivorship bias check
    print("\n上市时间分布 (skipped stocks):")
    if len(df_skipped) > 0 and 'date_range' in df_skipped.columns:
        start_years = df_skipped['date_range'].str.extract(r'^(\d{4})')[0]
        print(f"  Start years: {start_years.value_counts().sort_index().to_dict()}")

print(f"\n覆盖率: {len(stocks_all) - len(skipped)}/{len(stocks_all)} = {(len(stocks_all) - len(skipped)) / len(stocks_all) * 100:.1f}%")

# ============================================================
# Problem 3: y pre-clip distribution
# ============================================================
print("\n" + "=" * 60)
print("问题 3: y 标签 clip 前分布")
print("=" * 60)

y3_all = []
y6_all = []

for code, group in df.groupby('code'):
    group = group.sort_values('date').reset_index(drop=True)
    closes = group['close'].values.astype(np.float64)
    n = len(group)
    if n < LOOKBACK + 12:
        continue
    for i in range(LOOKBACK - 1, n - 12):
        if closes[i] <= 0.01:
            continue
        y3 = (closes[i + 3] - closes[i]) / closes[i] if i + 3 < n else np.nan
        y6 = (closes[i + 6] - closes[i]) / closes[i] if i + 6 < n else np.nan
        y3_all.append(y3)
        y6_all.append(y6)

y3_arr = np.array(y3_all)
y6_arr = np.array(y6_all)

# Filter NaN
y3_clean = y3_arr[~np.isnan(y3_arr)]
y6_clean = y6_arr[~np.isnan(y6_arr)]

for name, arr in [('y3 (3m fwd)', y3_clean), ('y6 (6m fwd)', y6_clean)]:
    pcts = [1, 5, 25, 50, 75, 95, 99]
    values = np.percentile(arr, pcts)
    print(f"\n{name}: n={len(arr)}")
    print(f"  mean={arr.mean():.4f} std={arr.std():.4f}")
    for p, v in zip(pcts, values):
        print(f"  p{p:2d}: {v:.4f} ({v*100:.1f}%)")

    # Clip analysis
    clip_neg = (arr < -2.0).sum()
    clip_pos = (arr > 2.0).sum()
    total_clip = clip_neg + clip_pos
    pct_clip = total_clip / len(arr) * 100
    print(f"  < -2.0 (clip): {clip_neg} ({clip_neg/len(arr)*100:.2f}%)")
    print(f"  > +2.0 (clip): {clip_pos} ({clip_pos/len(arr)*100:.2f}%)")
    print(f"  total clipped: {total_clip} ({pct_clip:.2f}%)")
    if pct_clip > 5:
        print(f"  ⚠ 超过 5% 阈值!")

    # Extreme samples (pre-clip)
    extremes = arr[np.abs(arr) > 5.0]
    if len(extremes) > 0:
        print(f"  |y| > 5.0 (extreme): {len(extremes)} ({len(extremes)/len(arr)*100:.2f}%)")

print("\nDone.")
