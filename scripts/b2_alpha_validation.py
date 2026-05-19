"""B2 Alpha Validation: Multi-period robustness + IR + 2018 bear stress test.
Determine: real alpha or bull market beta."""
import numpy as np, pandas as pd, sqlite3, torch, torch.nn as nn
from pathlib import Path
from collections import deque

PROJECT = Path(__file__).parent.parent
DB = PROJECT / '.eastmoney-ai' / 'db' / 'klines-v2.sqlite'
OUT = PROJECT / '.eastmoney-ai' / 'lstm'
TOP_K = 20

print("Loading data...")
conn = sqlite3.connect(str(DB))
stocks = [r[0] for r in conn.execute("SELECT DISTINCT stock_code FROM stock_industry_mapping").fetchall()]
prices = pd.read_sql_query(f"SELECT code, date, close FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2010-01'", conn, params=stocks)
conn.close()
price_matrix = prices.pivot(index='date', columns='code', values='close').sort_index().ffill()
returns_matrix = price_matrix.pct_change()

lstm = pd.read_parquet(OUT / 'monthly_lstm_signals_v2.parquet')
signal_df = pd.pivot_table(lstm, index='month', columns='code', values='lstm_signal', aggfunc='mean').ffill().fillna(0)

# HS300 equal-weight index proxy (from available stocks)
def get_hs300_index_returns():
    idx_rets = returns_matrix.mean(axis=1)
    return idx_rets.dropna()

hs300_rets = get_hs300_index_returns()

# Simple RL Policy
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(TOP_K*5 + 5, 128), nn.ReLU(), nn.Linear(128, TOP_K))
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

def train_rl_policy(returns, signals, train_months):
    """Train RL policy on given training months"""
    policy = PolicyNet()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    rewards_history = []

    for i, month in enumerate(train_months[:-1]):
        if i < 12: continue
        sig = signals.loc[month].dropna()
        if len(sig) < TOP_K: continue

        top = sig.nlargest(TOP_K)
        codes = top.index.tolist()
        stock_feats = []
        for j, c in enumerate(codes):
            ret_1m = returns.at[month, c] if c in returns.columns and month in returns.index else 0
            ret_3m = returns.loc[train_months[max(0,i-3)]:month, c].mean() if c in returns.columns else 0
            vol = returns.loc[train_months[max(0,i-6)]:month, c].std() if c in returns.columns else 0.1
            stock_feats.extend([sig[c], ret_1m or 0, ret_3m or 0, vol or 0.1, j/TOP_K])

        mkt_ret = returns.loc[month].mean() if month in returns.index else 0
        mkt_feats = [mkt_ret, 0.0, 0.1, 0.0, 1.0]
        state = torch.tensor(stock_feats + mkt_feats, dtype=torch.float32).unsqueeze(0)

        weights = policy(state).squeeze(0)

        next_month = train_months[i+1]
        port_ret = 0.0
        for j, c in enumerate(codes):
            if c in returns.columns and next_month in returns.index:
                ret = returns.at[next_month, c]
                if pd.notna(ret): port_ret += weights[j].item() * ret

        rewards_history.append(port_ret)

        if len(rewards_history) >= 6 and i % 6 == 0:
            baseline = np.mean(rewards_history[-12:]) if len(rewards_history) >= 12 else 0
            loss = 0.0
            for r in rewards_history[-6:]:
                loss += -(r - baseline) * torch.log(weights.sum() + 1e-8)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    return policy

def evaluate_policy(policy, signals, returns, test_months, use_rl=True):
    """Evaluate policy on test months, return monthly returns"""
    port_rets = []
    for i, month in enumerate(test_months[:-1]):
        sig = signals.loc[month].dropna()
        if len(sig) < TOP_K: continue
        top = sig.nlargest(TOP_K)
        codes = top.index.tolist()

        if use_rl:
            stock_feats = []
            for j, c in enumerate(codes):
                stock_feats.extend([sig[c], 0.0, 0.0, 0.1, j/TOP_K])
            state = torch.tensor(stock_feats + [0,0,0.1,0,1], dtype=torch.float32).unsqueeze(0)
            weights = policy(state).squeeze(0).detach().numpy()
            w = pd.Series(weights, index=codes)
        else:
            w = pd.Series(1.0/TOP_K, index=codes)

        next_month = test_months[i+1]
        port_ret = 0.0
        for c in w.index:
            if c in returns.columns and next_month in returns.index:
                ret = returns.at[next_month, c]
                if pd.notna(ret): port_ret += w[c] * ret
        port_rets.append(port_ret)

    return np.array(port_rets)

# Define OOS test windows
windows = [
    ('2019', '2015-01', '2018-12', '2019-01', '2019-12', '牛市反弹'),
    ('2020', '2016-01', '2019-12', '2020-01', '2020-12', '牛市'),
    ('2021', '2017-01', '2020-12', '2021-01', '2021-12', '结构性牛'),
    ('2022', '2018-01', '2021-12', '2022-01', '2022-12', '熊市'),
    ('2023', '2019-01', '2022-12', '2023-01', '2023-12', '震荡熊'),
    ('2024-25', '2020-01', '2023-12', '2024-01', '2025-12', '牛市'),
]

def compute_metrics(rets, hs300):
    sr = np.mean(rets)*12/(np.std(rets)*np.sqrt(12)) if np.std(rets)>0 else 0
    ann_r = np.mean(rets)*12
    eq = np.cumprod(1+rets)
    mdd = float((eq/np.maximum.accumulate(eq)-1).min())
    calmar = ann_r/abs(mdd) if abs(mdd)>0 else 0

    # IR vs HS300
    active = rets - hs300
    ir = np.mean(active)*12/(np.std(active)*np.sqrt(12)) if np.std(active)>0 else 0
    return sr, mdd, calmar, ann_r, ir

results = []
all_months = sorted(set(price_matrix.index) & set(signal_df.index))
all_months = [m for m in all_months if m >= '2015-01']

print(f"{'='*80}")
print(f"{'Window':<10} {'Market':<10} {'RL SR':>7} {'EW SR':>7} {'HS300':>7} {'RL-HS300':>9} {'IR':>7} {'RL DD':>7}")
print(f"{'-'*10} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*7}")

for name, train_start, train_end, test_start, test_end, market in windows:
    train_months = [m for m in all_months if train_start <= m <= train_end]
    test_months = [m for m in all_months if test_start <= m <= test_end]

    if len(train_months) < 24 or len(test_months) < 6: continue

    policy = train_rl_policy(returns_matrix, signal_df, train_months)

    rl_rets = evaluate_policy(policy, signal_df, returns_matrix, test_months, use_rl=True)
    ew_rets = evaluate_policy(policy, signal_df, returns_matrix, test_months, use_rl=False)

    # HS300 index returns for same period (simple numpy alignment)
    hs300_vals = []
    for m in test_months[1:]:
        if m in hs300_rets.index:
            hs300_vals.append(hs300_rets.at[m])
    hs300_aligned = np.array(hs300_vals[:len(rl_rets)])
    min_len = min(len(rl_rets), len(hs300_aligned))
    rl_rets = rl_rets[:min_len]
    hs300_aligned = hs300_aligned[:min_len]

    if len(rl_rets) < 3: continue
    sr_rl, dd_rl, cal_rl, ret_rl, ir_rl = compute_metrics(rl_rets, hs300_aligned)
    sr_ew, dd_ew, cal_ew, ret_ew, ir_ew = compute_metrics(ew_rets[:min_len], hs300_aligned)
    sr_hs, _, _, _, _ = compute_metrics(hs300_aligned, np.zeros_like(hs300_aligned))

    delta = sr_rl - sr_hs
    results.append({'window': name, 'market': market, 'rl_sr': sr_rl, 'ew_sr': sr_ew,
                    'hs300_sr': sr_hs, 'delta': delta, 'ir': ir_rl, 'rl_dd': dd_rl, 'rl_rets': rl_rets,
                    'hs300_rets': hs300_aligned})

    print(f"{name:<10} {market:<10} {sr_rl:7.3f} {sr_ew:7.3f} {sr_hs:7.3f} {delta:+9.3f} {ir_rl:7.3f} {dd_rl:6.1%}")

# ======== Bear market stress test: 2018 ========
print(f"\n{'='*80}")
print("BEAR MARKET STRESS TEST: 2018 (HS300 -25%)")
print(f"{'='*80}")

bear_train = [m for m in all_months if '2014-01' <= m <= '2017-12']
bear_test = [m for m in all_months if '2018-01' <= m <= '2018-12']
policy_2018 = train_rl_policy(returns_matrix, signal_df, bear_train)
rl_2018 = evaluate_policy(policy_2018, signal_df, returns_matrix, bear_test, use_rl=True)
ew_2018 = evaluate_policy(policy_2018, signal_df, returns_matrix, bear_test, use_rl=False)
hs300_2018 = np.array([hs300_rets.at[m] for m in bear_test[1:] if m in hs300_rets.index][:len(rl_2018)])

sr_rl18, dd_rl18, _, ret_rl18, ir_rl18 = compute_metrics(rl_2018, hs300_2018)
sr_ew18, dd_ew18, _, ret_ew18, _ = compute_metrics(ew_2018, hs300_2018)
sr_hs18, dd_hs18, _, ret_hs18, _ = compute_metrics(hs300_2018, np.zeros_like(hs300_2018))

print(f"  RL Policy:   SR={sr_rl18:.3f} DD={dd_rl18:.1%} Ret={ret_rl18:.1%} IR={ir_rl18:.3f}")
print(f"  EW Pool:     SR={sr_ew18:.3f} DD={dd_ew18:.1%} Ret={ret_ew18:.1%}")
print(f"  HS300 Index: SR={sr_hs18:.3f} DD={dd_hs18:.1%} Ret={ret_hs18:.1%}")
print(f"  RL-HS300 Δ:  {sr_rl18-sr_hs18:+.3f}")

# ======== Final verdict ========
print(f"\n{'='*80}")
print("VERDICT")
print(f"{'='*80}")

valid_results = [r for r in results if r is not None]
n_better = sum(1 for r in valid_results if r['rl_sr'] > r['ew_sr'])
n_total = len(valid_results)
avg_ir = np.mean([r['ir'] for r in valid_results])
has_bear_boost = any(r['delta'] > 0.10 and '熊' in r['market'] for r in valid_results)
bear_delta = sr_rl18 - sr_hs18

print(f"  RL > EW in {n_better}/{n_total} windows")
print(f"  Average IR: {avg_ir:.3f}")
print(f"  Bear market RL-HS300 Δ: {bear_delta:+.3f}")
print(f"  Has bear window with Δ>0.10: {has_bear_boost}")

if n_better >= n_total * 0.75 and avg_ir > 0.5 and (has_bear_boost or bear_delta > 0.05):
    print(f"\n  VERDICT A: TRUE ALPHA confirmed → Phase 20")
elif n_better < n_total * 0.5 and avg_ir < 0.3:
    print(f"\n  VERDICT B: PSEUDO ALPHA (bull beta) → Risk-First path")
else:
    print(f"\n  VERDICT C: REGIME-DEPENDENT → further analysis needed")

# Save results
pd.DataFrame([{k: v for k, v in r.items() if k not in ['rl_rets', 'hs300_rets']} for r in results]).to_csv(OUT / 'b2_alpha_validation.csv', index=False)
print(f"\nSaved b2_alpha_validation.csv")
