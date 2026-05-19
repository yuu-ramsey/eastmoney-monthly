"""Phase 19 v2: 5 signals + market timing + IC-IR weighting + 4 config comparison"""
import sqlite3, torch, numpy as np, pandas as pd, sys, json
from pathlib import Path
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

PROJECT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT_DIR = PROJECT / '.eastmoney-ai' / 'backtest'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = PROJECT / '.eastmoney-ai' / 'lstm' / 'models' / 'lstm_best.pt'

# Add lstm path
sys.path.insert(0, str(PROJECT / 'lib' / 'lstm'))
from model import LSTMBaseline

SLIPPAGE = 0.003; COMMISSION = 0.00025; TOP_K = 20; LOOKBACK = 60

# ============================================================
# 1. Data Loading
# ============================================================
def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query("""
        SELECT code, date, open, high, low, close, volume
        FROM monthly_klines WHERE code IN (SELECT DISTINCT stock_code FROM stock_industry_mapping)
        AND date >= '2010-01' ORDER BY code, date
    """, conn)
    ind = pd.read_sql_query("SELECT stock_code, industry_code FROM stock_industry_mapping", conn)
    conn.close()
    return df, dict(zip(ind['stock_code'], ind['industry_code']))

# ============================================================
# 2. LSTM Signal (from checkpoint)
# ============================================================
class LSTMSignal:
    def __init__(self):
        self.model = LSTMBaseline(21, 64, 1, 0.2)
        self.model.load_state_dict(torch.load(str(MODEL_PATH), map_location='cpu'))
        self.model.eval()

    def predict(self, closes, opens, highs, lows, volumes, industry_id=0):
        """Input: raw arrays of length >= 60. Output: y3_pred normalized to [-1,1]"""
        if len(closes) < LOOKBACK:
            return 0.0
        n = len(closes)
        # Build 21 features (simplified: use z-score of raw values)
        feats = np.zeros((n, 21), dtype=np.float32)
        for i in range(LOOKBACK, n):
            w_close = closes[max(0,i-LOOKBACK):i]
            w_open = opens[max(0,i-LOOKBACK):i]
            w_high = highs[max(0,i-LOOKBACK):i]
            w_low = lows[max(0,i-LOOKBACK):i]
            w_vol = volumes[max(0,i-LOOKBACK):i]
            m, s = np.mean(w_close), np.std(w_close) + 1e-8
            feats[i, 0] = (closes[i] - m) / s
            m, s = np.mean(w_open), np.std(w_open) + 1e-8
            feats[i, 1] = (opens[i] - m) / s
            m, s = np.mean(w_high), np.std(w_high) + 1e-8
            feats[i, 2] = (highs[i] - m) / s
            m, s = np.mean(w_low), np.std(w_low) + 1e-8
            feats[i, 3] = (lows[i] - m) / s
            m, s = np.mean(w_vol), np.std(w_vol) + 1e-8
            feats[i, 4] = (volumes[i] - m) / s
            # Simple MACD
            if i >= 26:
                e12 = pd.Series(closes[:i+1]).ewm(span=12).mean().values[-1]
                e26 = pd.Series(closes[:i+1]).ewm(span=26).mean().values[-1]
                feats[i, 5] = e12 - e26
        feats = np.nan_to_num(feats, 0.0)

        # Sliding window predictions
        preds = []
        for i in range(LOOKBACK, n):
            seq = feats[i-LOOKBACK+1:i+1]
            if seq.shape[0] < LOOKBACK:
                continue
            with torch.no_grad():
                x = torch.from_numpy(seq).float().unsqueeze(0)
                p = self.model(x).numpy()[0]
                preds.append(p[0])  # y3
        if not preds:
            return 0.0
        # Normalize last prediction to [-1, 1]
        return np.clip(preds[-1] * 3, -1.0, 1.0)

# ============================================================
# 3. Signal Computation (per stock, all months)
# ============================================================
def compute_signals(df, stock_to_ind, lstm_signal, lstm_lookup=None):
    """Compute 4 signals + LSTM for each stock/month. Returns dict of DataFrames."""
    stocks = sorted(df['code'].unique())
    if lstm_lookup is None:
        import pandas as pd
        lstm_df = pd.read_parquet(str(PROJECT / '.eastmoney-ai' / 'lstm' / 'monthly_lstm_signals_v2.parquet'))
        lstm_lookup = {}
        for _, r in lstm_df.iterrows():
            lstm_lookup[(r['code'], r['month'])] = r['lstm_signal']

    sig_tech = {}
    sig_res = {}
    sig_sec = {}
    sig_lstm = {}
    market_timing = {}

    print(f"Computing signals for {len(stocks)} stocks...")
    processed = 0
    for code in stocks:
        g = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        closes = g['close'].values.astype(float)
        highs = g['high'].values.astype(float)
        lows = g['low'].values.astype(float)
        volumes = g['volume'].values.astype(float)
        dates = g['date'].tolist()
        n = len(g)
        if n < 60: continue

        industry = stock_to_ind.get(code, None)
        tech_vals, res_vals, sec_vals, lstm_vals = {}, {}, {}, {}

        for i in range(60, n):
            d = dates[i]
            w_close = closes[:i+1]

            # Tech
            ma20 = np.mean(w_close[-20:])
            ma60 = np.mean(w_close[-60:]) if i >= 60 else ma20
            e12 = pd.Series(w_close).ewm(span=12).mean().values[-1]
            e26 = pd.Series(w_close).ewm(span=26).mean().values[-1]
            dif = e12 - e26
            # RSI
            delta = np.diff(w_close, prepend=w_close[0])
            gain = np.where(delta>0, delta, 0)
            loss = np.where(delta<0, -delta, 0)
            ag = pd.Series(gain).ewm(alpha=1/14).mean().values[-1]
            al = pd.Series(loss).ewm(alpha=1/14).mean().values[-1]
            rsi = 100 - 100/(1 + ag/max(al, 1e-8)) if al > 0 else 50

            s = 0.0
            s += np.clip(dif*3, -0.3, 0.3)
            s += 0.2 if rsi < 30 else (-0.2 if rsi > 70 else 0)
            s += 0.15 if w_close[-1] > ma20 else -0.15
            s += 0.15 if w_close[-1] > ma60 else -0.15
            tech_vals[d] = np.clip(s, -1, 1)

            # Resonance reverse
            slope = np.polyfit(range(5), w_close[-5:], 1)[0]
            macd_pos = dif > 0
            s2 = 0.0
            if w_close[-1] > ma60 and slope > 0 and macd_pos:
                s2 = -0.5
            elif w_close[-1] < ma60 and slope < 0 and not macd_pos:
                s2 = 0.5
            res_vals[d] = s2

            # Sector alpha 24m reverse
            s3 = 0.0
            if i >= 24:
                sr = (w_close[-1] - w_close[-25]) / max(w_close[-25], 0.01)
                s3 = np.clip(-sr * 5, -0.5, 0.5)
            sec_vals[d] = s3

            # LSTM (from pre-computed monthly signals)
            s4 = lstm_lookup.get((code, d), 0.0)
            lstm_vals[d] = np.clip(s4, -1, 1)

        sig_tech[code] = pd.Series(tech_vals)
        sig_res[code] = pd.Series(res_vals)
        sig_sec[code] = pd.Series(sec_vals)
        sig_lstm[code] = pd.Series(lstm_vals)

        processed += 1
        if processed % 50 == 0:
            print(f"  {processed}/{len(stocks)}")

    # Market timing: equal-weight universe close vs MA60
    # Use average close from all stocks' raw data
    mkt_closes = {}
    for code in stocks:
        g = df[df['code'] == code].sort_values('date')
        for _, row in g.iterrows():
            d = row['date']; c = row['close']
            if pd.notna(c) and c > 0:
                mkt_closes[d] = mkt_closes.get(d, []) + [c]
    mkt_avg = pd.Series({d: np.mean(vals) for d, vals in mkt_closes.items()}).sort_index()
    mkt_ma60 = mkt_avg.rolling(60).mean()
    for d in mkt_avg.index:
        if pd.notna(mkt_ma60[d]) and pd.notna(mkt_avg[d]):
            market_timing[d] = 1.0 if mkt_avg[d] > mkt_ma60[d] else 0.3

    return (
        pd.DataFrame(sig_tech).sort_index(),
        pd.DataFrame(sig_res).sort_index(),
        pd.DataFrame(sig_sec).sort_index(),
        pd.DataFrame(sig_lstm).sort_index(),
        pd.Series(market_timing).sort_index(),
    )

# ============================================================
# 4. IC-IR Weighting
# ============================================================
def compute_ic_weights(signals_dict, returns, lookback=24):
    """Compute IC-IR weights for each signal using trailing 24-month IC."""
    # signals_dict: {name: DataFrame (months × stocks)}
    weights_history = {}
    common_months = sorted(set.intersection(*[set(s.index) for s in signals_dict.values()]))

    for t_idx, month in enumerate(common_months):
        if t_idx < lookback + 1:
            weights_history[month] = {k: 0.25 for k in signals_dict}
            continue

        # Compute IC for each signal over trailing window
        ics = {}
        for name, sig_df in signals_dict.items():
            window_months = common_months[t_idx-lookback:t_idx]
            sig_vals = []
            ret_vals = []
            for wm in window_months:
                if wm not in sig_df.index or wm not in returns.index:
                    continue
                s = sig_df.loc[wm].dropna()
                r = returns.loc[wm].dropna()
                common = s.index.intersection(r.index)
                if len(common) < 20: continue
                ic, _ = spearmanr(s[common].values, r[common].values)
                if not np.isnan(ic):
                    sig_vals.extend(s[common].values)
                    ret_vals.extend(r[common].values)
            if len(sig_vals) > 20:
                ic, _ = spearmanr(sig_vals, ret_vals)
                ics[name] = abs(ic) if not np.isnan(ic) else 0.0
            else:
                ics[name] = 0.0

        total_ic = sum(ics.values()) + 1e-8
        weights_history[month] = {k: v / total_ic for k, v in ics.items()}

    return weights_history

# ============================================================
# 5. Backtest
# ============================================================
def run_backtest(prices, signal_df, market_timing=None, name=''):
    """signal_df: aggregated signal matrix (months × stocks)"""
    common = sorted(set(prices.index) & set(signal_df.index))
    print(f"  Dates: prices[{prices.index[0]}~{prices.index[-1]}] sig[{signal_df.index[0]}~{signal_df.index[-1]}] common={len(common)}")
    if len(common) < 2: return None, None, None
    rets, eq, turns = [], [1.0], []
    for i in range(len(common)-1):
        d, nd = common[i], common[i+1]
        sig = signal_df.loc[d].dropna()
        if len(sig) < TOP_K: continue
        top = sig.nlargest(TOP_K)
        w = pd.Series(1.0/TOP_K, index=top.index)

        # Market timing
        mt = 1.0
        if market_timing is not None and d in market_timing.index:
            mt = market_timing[d]
        w *= mt  # Scale by position size

        # Returns
        port_ret = 0.0
        for c in w.index:
            p0 = prices.at[d, c] if c in prices.columns else np.nan
            p1 = prices.at[nd, c] if c in prices.columns else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                port_ret += w[c] * (p1 - p0) / p0

        # Turnover cost
        if i > 0:
            prev = signal_df.loc[common[max(0,i-1)]].dropna()
            if len(prev) >= TOP_K:
                to = len(set(top.index) - set(prev.nlargest(TOP_K).index)) / TOP_K
                port_ret -= to * (SLIPPAGE + COMMISSION * 2) * mt
                turns.append(to)

        rets.append(port_ret)
        eq.append(eq[-1] * (1 + port_ret))

    rets = pd.Series(rets, index=common[1:])
    eq = pd.Series(eq, index=common[:len(eq)])
    ann_r = rets.mean()*12; ann_v = rets.std()*np.sqrt(12)
    sr = ann_r/ann_v if ann_v>0 else 0
    cm = eq.expanding().max(); dd = (eq-cm)/cm; mdd = dd.min()
    cal = ann_r/abs(mdd) if abs(mdd)>0 else 0
    return rets, eq, {'sharpe':sr, 'max_dd':mdd, 'calmar':cal, 'ann_ret':ann_r, 'ann_vol':ann_v, 'win':(rets>0).mean()}

# ============================================================
# 6. Main
# ============================================================
def main():
    print("Loading...")
    df, stock_to_ind = load_data()
    prices = df.pivot(index='date', columns='code', values='close').sort_index().ffill()
    returns_df = prices.pct_change()
    print(f"Prices: {prices.shape}, Returns: {returns_df.shape}")


    print("Computing signals...")
    s_tech, s_res, s_sec, s_lstm, mkt_timing = compute_signals(df, stock_to_ind, None)
    print(f"Signals: tech={s_tech.shape}, res={s_res.shape}, sec={s_sec.shape}, lstm={s_lstm.shape}, mt={len(mkt_timing)}")

    signals_dict = {'tech': s_tech, 'res': s_res, 'sec': s_sec, 'lstm': s_lstm}
    sig_names = list(signals_dict.keys())

    # IC-IR weights (walk-forward)
    print("Computing IC-IR weights...")
    ic_weights = compute_ic_weights(signals_dict, returns_df)

    # Aggregate: equal-weight and IC-IR
    print("Aggregating...")
    sig_ew = (sum(s.astype(float) for s in signals_dict.values()) / len(signals_dict)).astype(float)
    sig_icir = pd.DataFrame(0.0, index=sig_ew.index, columns=sig_ew.columns).astype(float)
    for month in sig_ew.index:
        if month in ic_weights:
            w = ic_weights[month]
            total = sum(w.values())
            if total > 0:
                row = pd.Series(0.0, index=sig_ew.columns)
                for name in sig_names:
                    if month in signals_dict[name].index:
                        row = row.add(signals_dict[name].loc[month].fillna(0) * w[name] / total, fill_value=0)
                sig_icir.loc[month] = row

    # Run 4 configs
    configs = [
        ('v1 EW', sig_ew, None),
        ('v2 EW+Timing', sig_ew, mkt_timing),
        ('v3 ICIR', sig_icir, None),
        ('v4 ICIR+Timing', sig_icir, mkt_timing),
    ]

    results = {}
    for label, sig, mt in configs:
        print(f"\n{'='*60}")
        print(f"IN-SAMPLE: {label} (2015-2023)")
        print(f"{'='*60}")
        r_is, eq_is, m_is = run_backtest(prices, sig.loc['2015-01':'2023-12'], mt, label)

        print(f"\nLIVE TEST: {label} (2024-2025)")
        r_lt, eq_lt, m_lt = run_backtest(prices, sig.loc['2024-01':'2025-12'], mt, label)

        results[label] = {'is': m_is or {'sharpe':0,'max_dd':0,'ann_ret':0},
                          'lt': m_lt or {'sharpe':0,'max_dd':0,'ann_ret':0}}
        eq_is.to_csv(OUT_DIR / f'equity_{label}_is.csv')
        eq_lt.to_csv(OUT_DIR / f'equity_{label}_lt.csv')

    # Benchmark
    bench = returns_df.mean(axis=1).dropna()
    bench_is = bench.loc['2015-01':'2023-12']
    bench_sr = bench_is.mean()*12 / (bench_is.std()*np.sqrt(12)) if bench_is.std()>0 else 0
    bench_ar = bench_is.mean()*12

    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'IS Sharpe':>10} {'IS MaxDD':>10} {'IS AnnRet':>10} {'LT Sharpe':>10} {'LT MaxDD':>10} {'LT AnnRet':>10} {'Alpha':>8}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for label, sig, mt in configs:
        m = results[label]
        alpha = m['is']['ann_ret'] - bench_ar
        print(f"{label:<20} {m['is']['sharpe']:10.3f} {m['is']['max_dd']:9.1%} {m['is']['ann_ret']:9.1%} "
              f"{m['lt']['sharpe']:10.3f} {m['lt']['max_dd']:9.1%} {m['lt']['ann_ret']:9.1%} {alpha:7.1%}")
    print(f"{'HS300 EW':<20} {bench_sr:10.3f} {'—':>10} {bench_ar:9.1%}")

    # Kill switch
    best_lt_sr = max(results.values(), key=lambda m: m['lt']['sharpe'])
    lsr = best_lt_sr['lt']['sharpe']
    best_label = [l for l, _ in configs if results[l] == best_lt_sr][0]
    print(f"\nBest Live Sharpe: {lsr:.3f} ({best_label})")
    if lsr > 0.9: print("KILL SWITCH: PASS → Phase 20")
    elif lsr > 0.7: print("KILL SWITCH: MARGINAL → pause discussion")
    else: print("KILL SWITCH: FAIL → pivot AI assistant positioning")

if __name__ == '__main__':
    main()
