"""Random portfolio control group validation
Based on Bilibili BV1ckrZBNEXg methodology:
- Generate N random equal-sized portfolios
- Compute random portfolio return distribution
- Check if strategy return exceeds random_mean + 3σ
- Output significance report

Usage:
  python cli/random_portfolio_test.py --dataset .eastmoney-ai/eval/dataset-v6.json
  python cli/random_portfolio_test.py --dataset ... --n-random 500 --threshold 3.0
  python cli/random_portfolio_test.py --results .eastmoney-ai/eval/results.json  # includes LLM prediction results
"""
import json, numpy as np, argparse, math
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent


def load_testpoints(dataset_path):
    """Load test points, returns [{stockCode, actualReturn, groundTruth, ...}]"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_points = data.get('testPoints', data if isinstance(data, list) else [])
    return test_points, data.get('stocks', [])


def load_results(results_path):
    """Load LLM evaluation results, returns [{stockCode, predictedSignal, ...}]"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('results', data if isinstance(data, list) else [])


def map_signal_to_bucket(signal):
    """Signal -> numeric bucket"""
    m = {'strong_bull': 2, 'bull': 1, 'neutral': 0, 'bear': -1, 'strong_bear': -2}
    return m.get(signal, 0)


def random_portfolio_test(returns, n_random=500, threshold=3.0, seed=42):
    """
    Core validation: strategy portfolio vs random portfolios

    Args:
        returns: np.array, actual returns of strategy-selected stocks
        n_random: number of random portfolios
        threshold: sigma multiple threshold (default 3.0)
        seed: random seed

    Returns:
        dict with: strategy_mean, random_mean, random_std, z_score,
                   significant, p_value_approx, n_stocks
    """
    rng = np.random.RandomState(seed)
    n_stocks = len(returns)
    strategy_mean = float(np.mean(returns))

    if n_stocks < 5:
        return {
            'strategy_mean': strategy_mean,
            'random_mean': None, 'random_std': None,
            'z_score': None, 'significant': False,
            'p_value_approx': None, 'n_stocks': n_stocks,
            'error': 'Insufficient sample size (< 5)',
        }

    # Generate N random portfolios (bootstrap sampling, size = n_stocks)
    random_means = np.zeros(n_random)
    for i in range(n_random):
        sample = rng.choice(returns, size=n_stocks, replace=True)
        random_means[i] = np.mean(sample)

    random_mean = float(np.mean(random_means))
    random_std = float(np.std(random_means))

    if random_std < 1e-10:
        return {
            'strategy_mean': strategy_mean,
            'random_mean': random_mean, 'random_std': random_std,
            'z_score': None, 'significant': False,
            'p_value_approx': None, 'n_stocks': n_stocks,
            'error': 'Random portfolio std is 0',
        }

    z_score = (strategy_mean - random_mean) / random_std
    significant = z_score > threshold
    # One-sided p-value approximation (normal distribution)
    p_approx = 1.0 - float(0.5 * (1.0 + math.erf(z_score / math.sqrt(2))))

    return {
        'strategy_mean': strategy_mean,
        'random_mean': random_mean,
        'random_std': random_std,
        'z_score': z_score,
        'significant': significant,
        'p_value_approx': p_approx,
        'n_stocks': n_stocks,
        'n_random': n_random,
        'threshold': threshold,
    }


def random_portfolio_test_external(returns, universe_returns, n_random=500, threshold=3.0, seed=42):
    """
    External universe version: randomly sample equal-sized portfolios from universe_returns
    (Stricter than bootstrap, since strategy stocks cannot repeat in random portfolios)

    Args:
        returns: actual returns of strategy-selected stocks
        universe_returns: actual returns of all universe stocks
    """
    rng = np.random.RandomState(seed)
    n_stocks = len(returns)
    strategy_mean = float(np.mean(returns))
    universe = np.array(universe_returns)

    if n_stocks < 5 or len(universe) < n_stocks:
        return {
            'strategy_mean': strategy_mean,
            'random_mean': None, 'random_std': None,
            'z_score': None, 'significant': False,
            'n_stocks': n_stocks,
            'error': 'Insufficient sample size',
        }

    random_means = np.zeros(n_random)
    for i in range(n_random):
        sample = rng.choice(universe, size=n_stocks, replace=False)
        random_means[i] = np.mean(sample)

    random_mean = float(np.mean(random_means))
    random_std = float(np.std(random_means))

    if random_std < 1e-10:
        return {'strategy_mean': strategy_mean, 'random_mean': random_mean,
                'random_std': random_std, 'z_score': None, 'significant': False,
                'n_stocks': n_stocks, 'error': 'Random portfolio std is 0'}

    z_score = (strategy_mean - random_mean) / random_std
    significant = z_score > threshold
    p_approx = 1.0 - float(0.5 * (1.0 + math.erf(z_score / math.sqrt(2))))

    return {
        'strategy_mean': strategy_mean,
        'random_mean': random_mean,
        'random_std': random_std,
        'z_score': z_score,
        'significant': significant,
        'p_value_approx': p_approx,
        'n_stocks': n_stocks,
        'n_universe': len(universe),
        'n_random': n_random,
        'threshold': threshold,
    }


def format_report(result, label='Strategy'):
    """Format validation report"""
    if result.get('error'):
        return f"[{label}] Validation failed: {result['error']}\n"

    sig = 'SIGNIFICANT' if result['significant'] else 'NOT SIGNIFICANT'
    lines = [
        f"\n{'='*60}",
        f"  Random Portfolio Control Group Validation — {label}",
        f"{'='*60}",
        f"  Sample size:        {result['n_stocks']} stocks",
        f"  Strategy mean ret:  {result['strategy_mean']:.2f}%",
        f"  Random mean ret:    {result['random_mean']:.2f}%",
        f"  Random std:         {result['random_std']:.2f}%",
        f"  Z-score:            {result['z_score']:.2f} (threshold {result['threshold']}σ)",
        f"  Approx p-value:     {result['p_value_approx']:.4f}",
        f"  Significance:       {sig}",
    ]
    if result.get('n_universe'):
        lines.append(f"  Universe stocks:    {result['n_universe']}")
    lines.append(f"  Random portfolios:  {result.get('n_random', 0)}")
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Random portfolio control group validation')
    parser.add_argument('--dataset', type=str, required=True, help='eval dataset JSON path')
    parser.add_argument('--results', type=str, default='', help='LLM evaluation results JSON (optional, includes predictedSignal)')
    parser.add_argument('--n-random', type=int, default=500, help='number of random portfolios')
    parser.add_argument('--threshold', type=float, default=3.0, help='sigma threshold')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_points, stocks = load_testpoints(args.dataset)
    print(f"Dataset: {len(test_points)} test points, {len(stocks)} stocks")

    # ---- Mode 1: Based on groundTruth signals (no LLM dependency) ----
    print("\n=== Mode 1: Based on groundTruth signals ===")
    # Bull group: groundTruth = strong_bull or bull
    bull_returns = [tp['actualReturn'] for tp in test_points
                    if tp.get('groundTruth') in ('strong_bull', 'bull')]
    bear_returns = [tp['actualReturn'] for tp in test_points
                    if tp.get('groundTruth') in ('strong_bear', 'bear')]
    all_returns = [tp['actualReturn'] for tp in test_points]

    if bull_returns:
        result = random_portfolio_test_external(
            np.array(bull_returns), all_returns,
            n_random=args.n_random, threshold=args.threshold, seed=args.seed)
        print(format_report(result, 'Bull Portfolio (groundTruth)'))

    if bear_returns:
        result = random_portfolio_test_external(
            np.array(bear_returns), all_returns,
            n_random=args.n_random, threshold=args.threshold, seed=args.seed)
        print(format_report(result, 'Bear Portfolio (groundTruth)'))

    # Long-short portfolio
    if bull_returns and bear_returns:
        long_short = np.mean(bull_returns) - np.mean(bear_returns)
        # Random long-short: randomly select two equal-sized groups
        rng = np.random.RandomState(args.seed)
        n_bull, n_bear = len(bull_returns), len(bear_returns)
        random_ls = np.zeros(args.n_random)
        for i in range(args.n_random):
            long_sample = rng.choice(all_returns, size=n_bull, replace=False)
            short_sample = rng.choice(all_returns, size=n_bear, replace=False)
            random_ls[i] = np.mean(long_sample) - np.mean(short_sample)
        ls_mean = float(np.mean(random_ls))
        ls_std = float(np.std(random_ls))
        ls_z = (long_short - ls_mean) / ls_std if ls_std > 1e-10 else None
        ls_sig = ls_z > args.threshold if ls_z is not None else False
        print(f"\n  Long-Short Portfolio Validation:")
        print(f"  LS return diff:  {long_short:.2f}%")
        print(f"  Random LS mean:  {ls_mean:.2f}%")
        print(f"  Random LS std:   {ls_std:.2f}%")
        print(f"  Z-score:         {ls_z:.2f}" if ls_z else "  Z-score: N/A")
        print(f"  Conclusion: {'SIGNIFICANT' if ls_sig else 'NOT SIGNIFICANT'}")

    # ---- Mode 2: Based on LLM predicted signals (if results provided) ----
    if args.results:
        results = load_results(args.results)
        if results:
            print(f"\n=== Mode 2: Based on LLM predicted signals ({len(results)} results) ===")
            predicted_bull = [r for r in results
                              if r.get('predictedSignal') in ('strong_bull', 'bull')]
            predicted_bear = [r for r in results
                              if r.get('predictedSignal') in ('strong_bear', 'bear')]

            # Need actualReturn: match from test_points
            tp_map = {tp['id']: tp.get('actualReturn', tp.get('alpha', 0))
                      for tp in test_points}

            if predicted_bull:
                bull_rets = [tp_map.get(r.get('testPointId', ''), 0) for r in predicted_bull]
                bull_rets = [r for r in bull_rets if r != 0]
                if bull_rets:
                    result = random_portfolio_test_external(
                        np.array(bull_rets), all_returns,
                        n_random=args.n_random, threshold=args.threshold, seed=args.seed)
                    print(format_report(result, 'LLM Bull Portfolio'))

            if predicted_bear:
                bear_rets = [tp_map.get(r.get('testPointId', ''), 0) for r in predicted_bear]
                bear_rets = [r for r in bear_rets if r != 0]
                if bear_rets:
                    result = random_portfolio_test_external(
                        np.array(bear_rets), all_returns,
                        n_random=args.n_random, threshold=args.threshold, seed=args.seed)
                    print(format_report(result, 'LLM Bear Portfolio'))

    # ---- Summary Report ----
    print(f"{'='*60}")
    print(f"  Params: {args.n_random} random portfolios, {args.threshold}σ threshold, seed={args.seed}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
