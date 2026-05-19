"""Final Sprint: HMM Regime Switching — 4-state detection + strategy switching.
Project-level kill switch: 7-window stress test."""
import numpy as np, pandas as pd, sqlite3, torch, torch.nn as nn
from pathlib import Path
from hmmlearn import hmm

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
REGIME_DIR = PROJECT / '.eastmoney-ai' / 'regime'
REGIME_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = 20

print("Loading data...")
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
prices = pd.read_sql_query(f"SELECT code, date, close FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2005-01'", conn, params=stocks)
conn.close()
price_matrix = prices.pivot(index='date', columns='code', values='close').sort_index().ffill()
returns_matrix = price_matrix.pct_change()
hs300_rets = returns_matrix.mean(axis=1).dropna()  # equal-weight proxy

lstm = pd.read_parquet(OUT / 'monthly_lstm_signals_v2.parquet')
signal_df = pd.pivot_table(lstm, index='month', columns='code', values='lstm_signal', aggfunc='mean').ffill().fillna(0)

# ============================================================
# Task 1: HMM Regime Detection (walk-forward)
# ============================================================
print("\nTask 1: HMM Regime Detection...")

def compute_hmm_features(rets_series):
    """Compute regime features: trailing 12m return, vol, skew, sharpe"""
    rets = rets_series.values
    n = len(rets)
    feats = np.zeros((n, 4))
    for i in range(12, n):
        w = rets[i-11:i+1]
        feats[i, 0] = np.mean(w) * 12       # ann return
        feats[i, 1] = np.std(w) * np.sqrt(12)  # ann vol
        feats[i, 2] = pd.Series(w).skew() if len(w) > 2 else 0
        feats[i, 3] = feats[i, 0] / max(feats[i, 1], 0.01)  # Sharpe
    return feats

features = compute_hmm_features(hs300_rets)
dates = hs300_rets.index.tolist()

# Walk-forward HMM: retrain every 12 months
regime_history = {}
for i in range(60, len(features)):
    train_data = features[max(0,i-60):i]  # trailing 5 years
    train_data = train_data[~np.isnan(train_data).any(axis=1)]
    if len(train_data) < 24: continue

    try:
        model = hmm.GaussianHMM(n_components=4, covariance_type='diag', n_iter=100, random_state=42)
        model.fit(train_data)
        # Predict current state
        state = model.predict(features[i:i+1])[0]
        # Map state to regime based on state's mean return
        state_means = model.means_[:, 0]  # mean return for each state
        # Sort states by return: panic < bear < sideways < bull
        sorted_states = np.argsort(state_means)
        regime_map = {sorted_states[0]: 'panic', sorted_states[1]: 'bear',
                      sorted_states[2]: 'sideways', sorted_states[3]: 'bull'}
        regime_history[dates[i]] = regime_map.get(state, 'sideways')
    except:
        # Fallback: simple rule-based
        ann_ret = features[i, 0]; ann_vol = features[i, 1]; sharpe = features[i, 3]
        if sharpe > 1.0: regime_history[dates[i]] = 'bull'
        elif sharpe > 0.0: regime_history[dates[i]] = 'sideways'
        elif sharpe > -1.0: regime_history[dates[i]] = 'bear'
        else: regime_history[dates[i]] = 'panic'

regime_df = pd.DataFrame([{'date': k, 'regime': v} for k, v in regime_history.items()])
regime_df.to_parquet(REGIME_DIR / 'regime_history.parquet')
print(f"Regime history: {len(regime_df)} months")
for y in ['2008', '2015', '2018', '2019', '2020', '2022', '2024']:
    year_regimes = regime_df[regime_df['date'].str.startswith(y)]['regime'].value_counts().to_dict()
    print(f"  {y}: {year_regimes}")

# ============================================================
# Task 2: Regime-Switching Strategy
# ============================================================
print("\nTask 2: Regime-Switching Strategy...")

# Simplified RL Policy (reuse from B2)
class SimpleRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(TOP_K*5+5, 64), nn.ReLU(), nn.Linear(64, TOP_K))
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def train_simple_rl(returns, signals, train_months):
    policy = SimpleRL()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for i, month in enumerate(train_months[:-1]):
        if i < 12: continue
        sig = signals.loc[month].dropna()
        if len(sig) < TOP_K: continue
        top = sig.nlargest(TOP_K)
        codes = top.index.tolist()
        feats = []
        for j, c in enumerate(codes):
            r1m = returns.at[month, c] if c in returns.columns and month in returns.index else 0
            r3m = returns.loc[train_months[max(0,i-3)]:month, c].mean() if c in returns.columns else 0
            vol = returns.loc[train_months[max(0,i-6)]:month, c].std() if c in returns.columns else 0.1
            feats.extend([sig[c], r1m or 0, r3m or 0, vol or 0.1, j/TOP_K])
        mkt_r = returns.loc[month].mean() if month in returns.index else 0
        state = torch.tensor(feats + [mkt_r, 0, 0.1, 0, 1], dtype=torch.float32).unsqueeze(0)
        w = policy(state).squeeze(0)
        next_m = train_months[i+1]
        port_ret = 0.0
        for j, c in enumerate(codes):
            if c in returns.columns and next_m in returns.index:
                r = returns.at[next_m, c]
                if pd.notna(r): port_ret += w[j].item() * r
        loss = -port_ret * torch.log(w.sum() + 1e-8)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    return policy

def compute_strategy_return(month, next_month, signals, returns, regime, policy_rl, prev_weights=None):
    """Returns: strategy_return, weights_used, turnover"""
    sig = signals.loc[month].dropna()
    if len(sig) < TOP_K: return 0.0, None, 0.0

    top = sig.nlargest(TOP_K)
    codes = top.index.tolist()

    if regime == 'bull':
        # 100% HS300 ETF → equal-weight all stocks
        w = pd.Series(1.0/len(codes), index=codes)
    elif regime == 'bear':
        # 30% RL + 70% cash
        feats = [];
        for j, c in enumerate(codes):
            feats.extend([sig[c], 0.0, 0.0, 0.1, j/TOP_K])
        state = torch.tensor(feats + [0,0,0.1,0,1], dtype=torch.float32).unsqueeze(0)
        rl_w = policy_rl(state).squeeze(0).detach().numpy()
        w = pd.Series(rl_w * 0.3, index=codes)
    elif regime == 'sideways':
        # 80% EW pool
        w = pd.Series(0.8/TOP_K, index=codes)
    else:  # panic → 100% cash
        return 0.0, None, 0.0

    # Position smoothing: cap change from previous month
    if prev_weights is not None:
        for c in w.index:
            if c in prev_weights.index:
                w[c] = np.clip(w[c], prev_weights.get(c, 0) - 0.3, prev_weights.get(c, 0) + 0.3)

    # Compute return
    port_ret = 0.0
    for c in w.index:
        if c in returns.columns and next_month in returns.index:
            ret = returns.at[next_month, c]
            if pd.notna(ret): port_ret += w[c] * ret

    # Transaction costs
    cost = 0.004  # 0.4% slippage
    if prev_weights is not None:
        turnover = sum(abs(w.get(c, 0) - prev_weights.get(c, 0)) for c in set(w.index) | set(prev_weights.index))
        port_ret -= turnover * cost

    return port_ret, w, 0.0

# ============================================================
# Task 3: 7-Window Stress Test
# ============================================================
print("\nTask 3: 7-Window Stress Test...")

windows = [
    ('2018', '2014-01', '2017-12', '2018-01', '2018-12', '熊市'),
    ('2019', '2015-01', '2018-12', '2019-01', '2019-12', '牛市'),
    ('2020', '2016-01', '2019-12', '2020-01', '2020-12', '牛市'),
    ('2021', '2017-01', '2020-12', '2021-01', '2021-12', '结构性牛'),
    ('2022', '2018-01', '2021-12', '2022-01', '2022-12', '熊市'),
    ('2023', '2019-01', '2022-12', '2023-01', '2023-12', '震荡熊'),
    ('2024-25', '2020-01', '2023-12', '2024-01', '2025-12', '牛市'),
]

all_months = sorted(set(price_matrix.index) & set(signal_df.index))

def compute_metrics(rets):
    pr = np.array(rets)
    if len(pr) < 3: return 0, 0, 0
    sr = pr.mean()*12/(pr.std()*np.sqrt(12)) if pr.std()>0 else 0
    eq = np.cumprod(1+pr)
    mdd = float((eq/np.maximum.accumulate(eq)-1).min())
    return sr, mdd, pr.mean()*12

print(f"\n{'='*90}")
print(f"{'Window':<10} {'Market':<10} {'Regime SR':>9} {'EW SR':>8} {'HS300 SR':>9} {'Δ Reg-HS':>9} {'Δ Reg-EW':>9}")
print(f"{'-'*10} {'-'*10} {'-'*9} {'-'*8} {'-'*9} {'-'*9} {'-'*9}")

all_results = []
for name, tr_start, tr_end, te_start, te_end, market in windows:
    train_months = [m for m in all_months if tr_start <= m <= tr_end]
    test_months = [m for m in all_months if te_start <= m <= te_end]

    if len(train_months) < 24 or len(test_months) < 6: continue

    # Train RL policy for this window
    policy_rl = train_simple_rl(returns_matrix, signal_df, train_months)

    # Run regime-switching strategy
    regime_rets, ew_rets = [], []
    prev_w = None
    for i, month in enumerate(test_months[:-1]):
        next_month = test_months[i+1]
        regime = regime_history.get(month, 'sideways')

        r_ret, w, _ = compute_strategy_return(month, next_month, signal_df, returns_matrix, regime, policy_rl, prev_w)
        regime_rets.append(r_ret)
        prev_w = w

        # EW baseline
        sig = signal_df.loc[month].dropna()
        if len(sig) >= TOP_K:
            top = sig.nlargest(TOP_K)
            ew_r = 0.0
            for c in top.index:
                if c in returns_matrix.columns and next_month in returns_matrix.index:
                    ret = returns_matrix.at[next_month, c]
                    if pd.notna(ret): ew_r += ret / TOP_K
            ew_rets.append(ew_r)

    # HS300 index for the same period
    hs300_win = np.array([hs300_rets.at[m] for m in test_months[1:] if m in hs300_rets.index][:len(regime_rets)])
    regime_rets = regime_rets[:len(hs300_win)]

    sr_r, dd_r, ret_r = compute_metrics(regime_rets)
    sr_e, _, _ = compute_metrics(ew_rets[:len(hs300_win)])
    sr_h, _, _ = compute_metrics(hs300_win)

    d_reg_hs = sr_r - sr_h
    d_reg_ew = sr_r - sr_e
    all_results.append({'window': name, 'market': market, 'regime_sr': sr_r, 'ew_sr': sr_e,
                        'hs300_sr': sr_h, 'delta': d_reg_hs, 'delta_ew': d_reg_ew})

    print(f"{name:<10} {market:<10} {sr_r:9.3f} {sr_e:8.3f} {sr_h:9.3f} {d_reg_hs:+9.3f} {d_reg_ew:+9.3f}")

# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{'='*80}")
print("PROJECT FINAL VERDICT")
print(f"{'='*80}")

n_better = sum(1 for r in all_results if r['delta'] > 0)
n_total = len(all_results)
avg_delta = np.mean([r['delta'] for r in all_results])
big_bear = sum(1 for r in all_results if '熊' in r['market'] and r['delta'] > 0.2)
bull_ok = sum(1 for r in all_results if '牛' in r['market'] and r['delta'] > -0.5)

print(f"  Regime > HS300 in {n_better}/{n_total} windows")
print(f"  Average Δ: {avg_delta:+.3f}")
print(f"  Bear windows with Δ>0.20: {big_bear}")
print(f"  Bull windows with Δ>-0.50: {bull_ok}")

if n_better >= 5 and avg_delta > 0.2 and bull_ok >= 2:
    print(f"\n  VERDICT A: TRUE ALPHA → Phase 20")
elif n_better >= 3 and avg_delta > 0.0:
    print(f"\n  VERDICT B: REGIME DETECTION VALID but strategy needs tuning")
else:
    print(f"\n  VERDICT C: COMPLETE FAILURE → Phase 23 pivot")

pd.DataFrame(all_results).to_csv(REGIME_DIR / 'final_verdict.csv', index=False)
print(f"\nSaved final_verdict.csv")
