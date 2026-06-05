"""MC Dropout 不确定性量化 + 置信度惩罚
基于 Andrej Karpathy 推荐的 LSTM 股票预测方法：
- 推理时保留 Dropout，N 次前向传播收集预测分布
- 标准差量化不确定性
- 置信度惩罚: 不确定性越高，信号衰减越强
"""
import torch, numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT / '.eastmoney-ai' / 'lstm' / 'models_v2'


def mc_predict(model, X_tensor, n_samples=50):
    """MC Dropout 推理：N 次前向传播，dropout 保持开启

    Args:
        model: nn.Module（调用前需已加载权重）
        X_tensor: torch.Tensor, shape (1, seq_len, input_dim)
        n_samples: 前向传播次数

    Returns:
        mean: np.ndarray, shape (1, n_targets), 预测均值
        std:  np.ndarray, shape (1, n_targets), 预测标准差（不确定性）
        samples: np.ndarray, shape (n_samples, 1, n_targets), 全部样本
    """
    model.train()  # dropout 保持活跃
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(model(X_tensor).cpu().numpy())
    samples = np.stack(samples, axis=0)  # (n_samples, 1, n_targets)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return mean, std, samples


def confidence_penalty(mean, std, strength=1.0):
    """置信度惩罚：高不确定性 → 信号向 0 收缩

    公式：
        cv = std / (|mean| + ε)          — 变异系数
        penalty = 1 / (1 + strength × cv) — 惩罚因子 ∈ (0, 1]
        adjusted = mean × penalty         — 惩罚后信号
        confidence = clip(1 - cv, 0, 1)   — 置信度 ∈ [0, 1]

    Args:
        mean: np.ndarray, 预测均值
        std:  np.ndarray, 预测标准差
        strength: 惩罚强度，越大越保守（默认 1.0）

    Returns:
        dict: {adjusted, penalty, confidence, cv} — 每个都是 np.ndarray
    """
    eps = 1e-8
    cv = std / (np.abs(mean) + eps)
    penalty = 1.0 / (1.0 + strength * cv)
    adjusted = mean * penalty
    confidence = np.clip(1.0 - cv, 0.0, 1.0)
    return {
        'adjusted': adjusted,
        'penalty': penalty,
        'confidence': confidence,
        'cv': cv,
    }


def predict_signal_with_uncertainty(model, X_array, n_samples=50,
                                     penalty_strength=1.0, device='cuda'):
    """端到端：从 numpy 数组到带不确定性的信号

    Args:
        model: nn.Module（已加载权重）
        X_array: np.ndarray, shape (seq_len, input_dim) 单条序列
        n_samples: MC 采样次数
        penalty_strength: 置信度惩罚强度
        device: 'cuda' | 'cpu'

    Returns:
        dict with keys:
            y3_mean, y3_std, y3_adjusted, y3_confidence
            y6_mean, y6_std, y6_adjusted, y6_confidence
            overall_confidence — y3 和 y6 置信度的均值
            signal — 综合信号（y3 调整后），正=看多，负=看空
            uncertainty_level — 'low' | 'medium' | 'high'
            n_samples
    """
    X_tensor = torch.from_numpy(X_array.astype(np.float32)).unsqueeze(0).to(device)
    mean, std, samples = mc_predict(model, X_tensor, n_samples)

    # y3 / y6 的置信度惩罚
    y3 = confidence_penalty(mean[0, 0], std[0, 0], penalty_strength)
    y6 = confidence_penalty(mean[0, 1], std[0, 1], penalty_strength)

    overall_conf = float((y3['confidence'] + y6['confidence']) / 2.0)

    # 不确定性分级
    avg_cv = float((y3['cv'] + y6['cv']) / 2.0)
    if avg_cv < 0.3:
        uncertainty_level = 'low'
    elif avg_cv < 0.7:
        uncertainty_level = 'medium'
    else:
        uncertainty_level = 'high'

    return {
        'y3_mean': float(mean[0, 0]),
        'y3_std': float(std[0, 0]),
        'y3_adjusted': float(y3['adjusted']),
        'y3_confidence': float(y3['confidence']),
        'y6_mean': float(mean[0, 1]),
        'y6_std': float(std[0, 1]),
        'y6_adjusted': float(y6['adjusted']),
        'y6_confidence': float(y6['confidence']),
        'overall_confidence': overall_conf,
        'signal': float(y3['adjusted']),
        'signal_raw': float(mean[0, 0]),
        'uncertainty_level': uncertainty_level,
        'n_samples': n_samples,
    }


def format_uncertainty_for_prompt(result):
    """将 MC Dropout 结果格式化为可注入 prompt 的 dict"""
    if not result:
        return None

    ulevel = result.get('uncertainty_level', 'medium')
    level_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
    level_desc = {
        'low': '模型预测一致性强，信号可信度较高',
        'medium': '模型预测存在分歧，信号需结合技术面验证',
        'high': '模型预测分歧大，信号不可靠，以技术分析为主',
    }

    return {
        'lstm_signal': result['signal'],
        'lstm_signal_raw': result['signal_raw'],
        'y3_mean': result['y3_mean'],
        'y3_std': result['y3_std'],
        'y6_mean': result['y6_mean'],
        'y6_std': result['y6_std'],
        'overall_confidence': result['overall_confidence'],
        'uncertainty_level': result['uncertainty_level'],
        'uncertainty_emoji': level_emoji.get(ulevel, '🟡'),
        'uncertainty_desc': level_desc.get(ulevel, ''),
        'mc_samples': result['n_samples'],
    }
