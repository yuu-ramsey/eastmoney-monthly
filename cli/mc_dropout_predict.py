"""MC Dropout batch inference — Loads daily model, N forward passes, outputs uncertainty quantification results
Usage:
  python cli/mc_dropout_predict.py --codes 000001,000002  # specific stocks
  python cli/mc_dropout_predict.py --limit 100             # first N stocks
  python cli/mc_dropout_predict.py --all                   # all stocks
  python cli/mc_dropout_predict.py --latest                # latest trading day only
Output: .eastmoney-ai/lstm/mc_dropout_signals.parquet
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, sqlite3, argparse, time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
MODEL_PATH = OUT / 'models_v2' / 'daily_lstm7.pt'
LOOKBACK = 252
MC_SAMPLES = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== Model Loading ========
sys_path = str(PROJECT / 'lib' / 'lstm')
import sys
sys.path.insert(0, sys_path)
from model_v2 import create_model

print(f"Device: {DEVICE}")
print(f"Loading model from {MODEL_PATH}...")
model = create_model('LSTM-2', 10).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# ======== Feature Engineering ========
def rolling_zscore_vec(s, w=60):
    return ((s - s.rolling(w, min_periods=w).mean()) / s.rolling(w, min_periods=w).std()).fillna(0).values

def build_features(df_group):
    """Build feature matrix from single stock daily data"""
    closes = df_group['close'].values.astype(float)
    opens = df_group['open'].values.astype(float)
    highs = df_group['high'].values.astype(float)
    lows = df_group['low'].values.astype(float)
    vols = df_group['volume'].values.astype(float)
    n = len(df_group)

    feats = np.zeros((n, 21), dtype=np.float32)
    s = pd.Series(closes)
    feats[:, 0] = rolling_zscore_vec(s)
    feats[:, 1] = rolling_zscore_vec(pd.Series(opens))
    feats[:, 2] = rolling_zscore_vec(pd.Series(highs))
    feats[:, 3] = rolling_zscore_vec(pd.Series(lows))
    feats[:, 4] = rolling_zscore_vec(pd.Series(vols))
    e12 = s.ewm(span=12).mean().values
    e26 = s.ewm(span=26).mean().values
    feats[:, 5] = np.nan_to_num(e12 - e26, 0)
    feats[:, 6] = np.nan_to_num(pd.Series(feats[:, 5]).ewm(span=9).mean().values, 0)
    feats[:, 7] = (feats[:, 5] - feats[:, 6]) * 2
    feats[:, 8] = np.nan_to_num(50, 50)
    feats[:, 18] = np.nan_to_num((closes - s.rolling(20).mean().values) / np.maximum(closes, 0.01), 0)
    feats[:, 19] = np.nan_to_num((closes - s.rolling(60).mean().values) / np.maximum(closes, 0.01), 0)
    feats[:, 20] = hash(df_group['code'].iloc[0]) % 31 / 31.0
    feats = np.nan_to_num(feats, 0.0)
    return feats[:, :10]  # only first 10 dimensions

# ======== MC Dropout Inference ========
def mc_predict_batch(model, X_batch, n_samples=MC_SAMPLES):
    """Batch MC Dropout: N forward passes, dropout kept active"""
    model.train()  # dropout active
    all_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = model(X_batch).cpu().numpy()  # (B, 2)
            all_samples.append(p)
    samples = np.stack(all_samples, axis=0)  # (N, B, 2)
    mean = samples.mean(axis=0)  # (B, 2)
    std = samples.std(axis=0)    # (B, 2)
    return mean, std

def confidence_penalty_vec(mean, std, strength=1.0):
    """Vectorized confidence penalty"""
    eps = 1e-8
    cv = std / (np.abs(mean) + eps)
    penalty = 1.0 / (1.0 + strength * cv)
    adjusted = mean * penalty
    confidence = np.clip(1.0 - cv, 0.0, 1.0)
    return adjusted, penalty, confidence, cv

def uncertainty_level_vec(cv):
    """Vectorized uncertainty grading"""
    result = np.full(cv.shape[0], 'medium', dtype=object)
    result[cv < 0.3] = 'low'
    result[cv >= 0.7] = 'high'
    return result

# ======== Main Flow ========
def main():
    parser = argparse.ArgumentParser(description='MC Dropout batch inference')
    parser.add_argument('--codes', type=str, default='', help='comma-separated stock codes')
    parser.add_argument('--limit', type=int, default=0, help='limit stock count')
    parser.add_argument('--all', action='store_true', help='all stocks')
    parser.add_argument('--latest', action='store_true', help='latest trading day only')
    parser.add_argument('--batch-size', type=int, default=256, help='GPU inference batch size')
    parser.add_argument('--n-samples', type=int, default=MC_SAMPLES, help='MC sample count')
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB))
    all_stocks = [r[0] for r in conn.execute(
        "SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]

    if args.codes:
        codes = [c.strip() for c in args.codes.split(',')]
    elif args.all:
        codes = all_stocks
    elif args.limit > 0:
        codes = all_stocks[:args.limit]
    else:
        codes = all_stocks[:50]  # default: first 50 stocks

    print(f"Target stocks: {len(codes)}")

    # Load daily kline data
    d_df = pd.read_sql_query(f"""
        SELECT code, date, open, high, low, close, volume
        FROM daily_klines
        WHERE code IN ({','.join('?' * len(codes))})
        AND date >= '2010-01-01'
        ORDER BY code, date
    """, conn, params=codes)
    conn.close()
    print(f"Daily rows: {len(d_df)}")

    all_rows = []
    stock_count = 0

    for code in codes:
        g = d_df[d_df['code'] == code].sort_values('date').reset_index(drop=True)
        if len(g) < LOOKBACK + 63:
            continue

        stock_count += 1
        n = len(g)
        dates = g['date'].tolist()

        # Build features
        feats = build_features(g)

        # Build sequence batch
        seqs = []
        seq_dates = []
        if args.latest:
            # Latest date only
            i = n - 1
            if i >= LOOKBACK - 1:
                seqs.append(feats[i - LOOKBACK + 1:i + 1])
                seq_dates.append(dates[i])
        else:
            for i in range(LOOKBACK - 1, n):
                seqs.append(feats[i - LOOKBACK + 1:i + 1])
                seq_dates.append(dates[i])

        if not seqs:
            continue

        # Batch GPU inference
        all_mean, all_std = [], []
        for j in range(0, len(seqs), args.batch_size):
            batch = torch.from_numpy(np.array(seqs[j:j + args.batch_size])).float().to(DEVICE)
            mean, std = mc_predict_batch(model, batch, args.n_samples)
            all_mean.append(mean)
            all_std.append(std)

        means = np.concatenate(all_mean, axis=0)  # (n_dates, 2)
        stds = np.concatenate(all_std, axis=0)

        # Confidence penalty
        y3_adj, y3_penalty, y3_conf, y3_cv = confidence_penalty_vec(means[:, 0], stds[:, 0])
        y6_adj, y6_penalty, y6_conf, y6_cv = confidence_penalty_vec(means[:, 1], stds[:, 1])
        # Only use y3 for overall_confidence; y6 is the dummy target during training (always 0), not used
        overall_conf = y3_conf
        avg_cv = y3_cv
        ulevel = uncertainty_level_vec(avg_cv)

        for i in range(len(seq_dates)):
            all_rows.append({
                'code': code,
                'date': seq_dates[i],
                'y3_mean': float(means[i, 0]),
                'y3_std': float(stds[i, 0]),
                'y3_adjusted': float(y3_adj[i]),
                'y3_confidence': float(y3_conf[i]),
                'y6_mean': float(means[i, 1]),
                'y6_std': float(stds[i, 1]),
                'y6_adjusted': float(y6_adj[i]),
                'y6_confidence': float(y6_conf[i]),
                'overall_confidence': float(overall_conf[i]),
                'uncertainty_level': str(ulevel[i]),
                'signal': float(y3_adj[i]),
                'signal_raw': float(means[i, 0]),
            })

        if stock_count % 20 == 0:
            print(f"  Progress: {stock_count}/{len(codes)} stocks, {len(all_rows)} signals")

    # Save
    df_out = pd.DataFrame(all_rows)
    out_path = OUT / 'mc_dropout_signals.parquet'
    df_out.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df_out)} signals to {out_path}")

    # Summary statistics
    print(f"\n=== Uncertainty Distribution ===")
    if len(df_out) > 0:
        for level in ['low', 'medium', 'high']:
            count = (df_out['uncertainty_level'] == level).sum()
            print(f"  {level}: {count} ({100*count/len(df_out):.1f}%)")
        print(f"  mean confidence: {df_out['overall_confidence'].mean():.3f}")
        print(f"  mean y3_std: {df_out['y3_std'].mean():.5f}")
        print(f"  mean signal: {df_out['signal'].mean():.4f}")
        print(f"  signal std: {df_out['signal'].std():.4f}")

if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time() - t0:.0f}s")
