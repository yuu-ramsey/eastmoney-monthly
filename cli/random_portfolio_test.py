"""随机组合对照组验证
基于 B站 BV1ckrZBNEXg 方法论：
- 生成 N 组随机等规模组合
- 计算随机组合收益分布
- 检查策略收益是否超过 random_mean + 3σ
- 输出显著性报告

用法：
  python cli/random_portfolio_test.py --dataset .eastmoney-ai/eval/dataset-v6.json
  python cli/random_portfolio_test.py --dataset ... --n-random 500 --threshold 3.0
  python cli/random_portfolio_test.py --results .eastmoney-ai/eval/results.json  # 含 LLM 预测结果
"""
import json, numpy as np, argparse, math
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent


def load_testpoints(dataset_path):
    """加载测试点，返回 [{stockCode, actualReturn, groundTruth, ...}]"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_points = data.get('testPoints', data if isinstance(data, list) else [])
    return test_points, data.get('stocks', [])


def load_results(results_path):
    """加载 LLM 评估结果，返回 [{stockCode, predictedSignal, ...}]"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('results', data if isinstance(data, list) else [])


def map_signal_to_bucket(signal):
    """信号 → 数值桶"""
    m = {'strong_bull': 2, 'bull': 1, 'neutral': 0, 'bear': -1, 'strong_bear': -2}
    return m.get(signal, 0)


def random_portfolio_test(returns, n_random=500, threshold=3.0, seed=42):
    """
    核心验证：策略组合 vs 随机组合

    Args:
        returns: np.array, 策略选中股票的实际收益率
        n_random: 随机组合数量
        threshold: σ 倍数阈值（默认 3.0）
        seed: 随机种子

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
            'error': '样本量不足（< 5）',
        }

    # 生成 N 组随机组合（bootstrap 抽样，size = n_stocks）
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
            'error': '随机组合标准差为 0',
        }

    z_score = (strategy_mean - random_mean) / random_std
    significant = z_score > threshold
    # 单侧 p 值近似（正态分布）
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
    外部宇宙版：从 universe_returns 中随机抽取等规模组合
    （比 bootstrap 更严格，因为不允许策略股票在随机组合中重复出现）

    Args:
        returns: 策略选中股票的实际收益率
        universe_returns: 全部宇宙股票的实际收益率
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
            'error': '样本量不足',
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
                'n_stocks': n_stocks, 'error': '随机组合标准差为 0'}

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


def format_report(result, label='策略'):
    """格式化验证报告"""
    if result.get('error'):
        return f"[{label}] 验证失败: {result['error']}\n"

    sig = '✅ 显著' if result['significant'] else '❌ 不显著'
    lines = [
        f"\n{'='*60}",
        f"  随机组合对照组验证 — {label}",
        f"{'='*60}",
        f"  样本量:        {result['n_stocks']} 只",
        f"  策略均值收益:  {result['strategy_mean']:.2f}%",
        f"  随机均值收益:  {result['random_mean']:.2f}%",
        f"  随机标准差:    {result['random_std']:.2f}%",
        f"  Z-score:       {result['z_score']:.2f} (阈值 {result['threshold']}σ)",
        f"  近似 p 值:     {result['p_value_approx']:.4f}",
        f"  显著性结论:    {sig}",
    ]
    if result.get('n_universe'):
        lines.append(f"  宇宙股票数:    {result['n_universe']}")
    lines.append(f"  随机组合数:    {result.get('n_random', 0)}")
    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='随机组合对照组验证')
    parser.add_argument('--dataset', type=str, required=True, help='eval 数据集 JSON 路径')
    parser.add_argument('--results', type=str, default='', help='LLM 评估结果 JSON（可选，含 predictedSignal）')
    parser.add_argument('--n-random', type=int, default=500, help='随机组合数量')
    parser.add_argument('--threshold', type=float, default=3.0, help='σ 阈值')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    test_points, stocks = load_testpoints(args.dataset)
    print(f"数据集: {len(test_points)} 个测试点, {len(stocks)} 只股票")

    # ---- 模式 1：基于 groundTruth 信号（不依赖 LLM 预测） ----
    print("\n=== 模式 1：基于 groundTruth 信号 ===")
    # 看多组：groundTruth = strong_bull 或 bull
    bull_returns = [tp['actualReturn'] for tp in test_points
                    if tp.get('groundTruth') in ('strong_bull', 'bull')]
    bear_returns = [tp['actualReturn'] for tp in test_points
                    if tp.get('groundTruth') in ('strong_bear', 'bear')]
    all_returns = [tp['actualReturn'] for tp in test_points]

    if bull_returns:
        result = random_portfolio_test_external(
            np.array(bull_returns), all_returns,
            n_random=args.n_random, threshold=args.threshold, seed=args.seed)
        print(format_report(result, '看多组合 (groundTruth)'))

    if bear_returns:
        result = random_portfolio_test_external(
            np.array(bear_returns), all_returns,
            n_random=args.n_random, threshold=args.threshold, seed=args.seed)
        print(format_report(result, '看空组合 (groundTruth)'))

    # 多空组合
    if bull_returns and bear_returns:
        long_short = np.mean(bull_returns) - np.mean(bear_returns)
        # 随机多空：随机选两组等规模
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
        print(f"\n  多空组合验证:")
        print(f"  多空收益差:    {long_short:.2f}%")
        print(f"  随机多空均值:  {ls_mean:.2f}%")
        print(f"  随机多空 std:  {ls_std:.2f}%")
        print(f"  Z-score:       {ls_z:.2f}" if ls_z else "  Z-score: N/A")
        print(f"  结论: {'✅ 显著' if ls_sig else '❌ 不显著'}")

    # ---- 模式 2：基于 LLM 预测信号（如果有 results） ----
    if args.results:
        results = load_results(args.results)
        if results:
            print(f"\n=== 模式 2：基于 LLM 预测信号 ({len(results)} 结果) ===")
            predicted_bull = [r for r in results
                              if r.get('predictedSignal') in ('strong_bull', 'bull')]
            predicted_bear = [r for r in results
                              if r.get('predictedSignal') in ('strong_bear', 'bear')]

            # 需要 actualReturn：从 test_points 匹配
            tp_map = {tp['id']: tp.get('actualReturn', tp.get('alpha', 0))
                      for tp in test_points}

            if predicted_bull:
                bull_rets = [tp_map.get(r.get('testPointId', ''), 0) for r in predicted_bull]
                bull_rets = [r for r in bull_rets if r != 0]
                if bull_rets:
                    result = random_portfolio_test_external(
                        np.array(bull_rets), all_returns,
                        n_random=args.n_random, threshold=args.threshold, seed=args.seed)
                    print(format_report(result, 'LLM看多组合'))

            if predicted_bear:
                bear_rets = [tp_map.get(r.get('testPointId', ''), 0) for r in predicted_bear]
                bear_rets = [r for r in bear_rets if r != 0]
                if bear_rets:
                    result = random_portfolio_test_external(
                        np.array(bear_rets), all_returns,
                        n_random=args.n_random, threshold=args.threshold, seed=args.seed)
                    print(format_report(result, 'LLM看空组合'))

    # ---- 综合报告 ----
    print(f"{'='*60}")
    print(f"  参数: {args.n_random} 组随机组合, {args.threshold}σ 阈值, seed={args.seed}")
    print(f"  时间: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
