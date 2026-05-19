"""B2: RL Portfolio — Policy Gradient optimizer for Top-K stock selection.
State = signals + trailing metrics, Action = Top-20 weight adjustments.
Reward = risk-adjusted monthly return."""
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
prices = pd.read_sql_query(f"SELECT code, date, close FROM monthly_klines WHERE code IN ({','.join('?'*len(stocks))}) AND date >= '2014-12'", conn, params=stocks)
conn.close()
price_matrix = prices.pivot(index='date', columns='code', values='close').sort_index().ffill()
returns_matrix = price_matrix.pct_change()

lstm = pd.read_parquet(OUT / 'monthly_lstm_signals_v2.parquet')
signal_df = pd.pivot_table(lstm, index='month', columns='code', values='lstm_signal', aggfunc='mean').ffill().fillna(0)

# Policy Network: state → action (weight vector over stocks)
# State: for each stock: [lstm_signal, trailing_ret_1m, trailing_ret_3m, trailing_vol, is_top20]
# Simplified: aggregate state + policy over Top-20 stocks

class PolicyNetwork(nn.Module):
    def __init__(self, n_stocks, hidden=128):
        super().__init__()
        # Aggregate market state
        self.market_net = nn.Sequential(nn.Linear(5, 32), nn.ReLU())
        # Per-stock state: signal, ret1m, ret3m, vol, rank → combined
        self.stock_net = nn.Sequential(nn.Linear(TOP_K * 5, hidden), nn.ReLU(), nn.Linear(hidden, TOP_K))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, market_state, stock_features):
        # market_state: (5,) — mkt ret, mkt vol, dd, timing, bull_bear
        # stock_features: (TOP_K, 5) — signal, ret1m, ret3m, vol, rank
        mkt_emb = self.market_net(market_state)
        stock_flat = stock_features.flatten().unsqueeze(0)
        logits = self.stock_net(stock_flat)
        weights = self.softmax(logits)
        return weights.squeeze(0)

# Simplified RL: use REINFORCE with baseline
def train_rl():
    common_months = sorted(set(price_matrix.index) & set(signal_df.index))
    common_months = [m for m in common_months if '2015' <= m[:4] <= '2023']  # Train: 2015-2023

    policy = PolicyNetwork(TOP_K)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    episode_rewards = []
    all_actions = []
    all_rewards = []

    for i, month in enumerate(common_months[:-1]):
        if i < 12: continue  # warmup

        # Get signals for this month
        sig = signal_df.loc[month].dropna()
        if len(sig) < TOP_K: continue

        # Top-K stocks by signal
        top_stocks = sig.nlargest(TOP_K)
        stock_codes = top_stocks.index.tolist()

        # Build state features
        # Market state
        mkt_ret_1m = returns_matrix.loc[month].mean() if month in returns_matrix.index else 0
        mkt_ret_3m = returns_matrix.loc[common_months[max(0,i-3)]:month].mean().mean() if i >= 3 else 0
        mkt_vol = returns_matrix.loc[common_months[max(0,i-6)]:month].std().mean() if i >= 6 else 0.1
        mkt_dd = 0.0  # simplified
        mkt_timing = 1.0 if mkt_ret_3m > 0 else 0.5

        market_state = torch.tensor([mkt_ret_1m, mkt_ret_3m, mkt_vol, mkt_dd, mkt_timing], dtype=torch.float32)

        # Stock features per top-K stock
        stock_feats = torch.zeros(TOP_K, 5)
        for j, code in enumerate(stock_codes):
            signal_val = sig[code]
            ret_1m = returns_matrix.at[month, code] if code in returns_matrix.columns and month in returns_matrix.index else 0
            ret_3m = returns_matrix.loc[common_months[max(0,i-3)]:month, code].mean() if code in returns_matrix.columns else 0
            vol = returns_matrix.loc[common_months[max(0,i-6)]:month, code].std() if code in returns_matrix.columns else 0.1
            stock_feats[j] = torch.tensor([signal_val, ret_1m or 0, ret_3m or 0, vol or 0.1, j/TOP_K])

        # Get policy weights
        weights = policy(market_state, stock_feats)

        # Execute: compute reward as risk-adjusted return
        next_month = common_months[i+1]
        monthly_ret = 0.0
        for j, code in enumerate(stock_codes):
            if code in returns_matrix.columns and next_month in returns_matrix.index:
                ret = returns_matrix.at[next_month, code]
                if pd.notna(ret):
                    monthly_ret += weights[j].item() * ret

        # Reward = monthly return - 0.5 * volatility penalty
        reward = monthly_ret - 0.5 * abs(monthly_ret) * mkt_vol

        all_rewards.append(reward)
        all_actions.append((market_state, stock_feats, weights))

        # Train every 6 months
        if len(all_rewards) >= 6 and i % 6 == 0:
            # Compute baseline (moving average)
            baseline = np.mean(all_rewards[-12:]) if len(all_rewards) >= 12 else 0
            policy_loss = 0.0
            for (ms, sf, w), r in zip(all_actions[-6:], all_rewards[-6:]):
                advantage = r - baseline
                log_prob = torch.log(w + 1e-8).sum()
                policy_loss += -log_prob * advantage

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if i % 24 == 0:
                avg_r = np.mean(all_rewards[-12:]) if len(all_rewards) >= 12 else np.mean(all_rewards)
                print(f"  {month}: avg_reward={avg_r:.4f}, policy_loss={policy_loss.item():.4f}")

    return policy, all_rewards

print("Training RL policy...")
policy, train_rewards = train_rl()
print(f"Train complete. {len(train_rewards)} steps, avg reward={np.mean(train_rewards):.4f}")

# Evaluate: compare RL vs EW on OOS (2024-2025)
print("\nEvaluating RL vs EW on OOS (2024-2025)...")
common_months = sorted(set(price_matrix.index) & set(signal_df.index))
oos_months = [m for m in common_months if m >= '2024-01']

def backtest_policy(policy, use_rl=True):
    port_rets, eq, turnover = [], [1.0], []
    for i, month in enumerate(oos_months[:-1]):
        sig = signal_df.loc[month].dropna()
        if len(sig) < TOP_K: continue
        top_stocks = sig.nlargest(TOP_K)

        if use_rl:
            # Build state and get RL weights
            stock_feats = torch.zeros(TOP_K, 5)
            stock_codes = top_stocks.index.tolist()
            for j, code in enumerate(stock_codes):
                stock_feats[j] = torch.tensor([sig[code], 0.0, 0.0, 0.1, j/TOP_K])
            market_state = torch.zeros(5)
            weights = policy(market_state, stock_feats).detach().numpy()
            w = pd.Series(weights, index=top_stocks.index)
        else:
            w = pd.Series(1.0/TOP_K, index=top_stocks.index)

        next_month = oos_months[i+1]
        port_ret = 0.0
        for code in w.index:
            p0 = price_matrix.at[month, code] if code in price_matrix.columns else np.nan
            p1 = price_matrix.at[next_month, code] if code in price_matrix.columns else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                port_ret += w[code] * (p1-p0)/p0

        port_rets.append(port_ret)
        eq.append(eq[-1] * (1 + port_ret))

    pr = np.array(port_rets)
    sr = pr.mean()*12/(pr.std()*np.sqrt(12)) if pr.std()>0 else 0
    ann_r = pr.mean()*12
    mdd = float((np.array(eq) / np.maximum.accumulate(np.array(eq)) - 1).min())
    calmar = ann_r/abs(mdd) if abs(mdd)>0 else 0
    return sr, mdd, calmar, ann_r

sr_rl, dd_rl, cal_rl, ret_rl = backtest_policy(policy, use_rl=True)
sr_ew, dd_ew, cal_ew, ret_ew = backtest_policy(policy, use_rl=False)

print(f"\n{'='*60}")
print("B2 FINAL: RL Portfolio (OOS 2024-2025)")
print(f"{'='*60}")
print(f"  EW Baseline: SR={sr_ew:.3f} DD={dd_ew:.1%} Calmar={cal_ew:.3f} Ret={ret_ew:.1%}")
print(f"  RL Policy:   SR={sr_rl:.3f} DD={dd_rl:.1%} Calmar={cal_rl:.3f} Ret={ret_rl:.1%}")
delta = sr_rl - sr_ew
print(f"  Δ Sharpe: {delta:+.3f}")
if sr_rl > 1.0: print("KILL SWITCH: PASS > 1.0")
elif sr_rl > 0.8: print("KILL SWITCH: MARGINAL 0.8-1.0")
else: print("KILL SWITCH: FAIL < 0.8")
