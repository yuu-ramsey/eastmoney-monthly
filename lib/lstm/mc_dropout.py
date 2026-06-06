"""MC Dropout uncertainty quantification + confidence penalty
Based on Andrej Karpathy's recommended LSTM stock prediction method:
- Keep Dropout active during inference, N forward passes collect prediction distribution
- Standard deviation quantifies uncertainty
- Confidence penalty: higher uncertainty → stronger signal decay
"""
import torch, numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT / '.eastmoney-ai' / 'lstm' / 'models_v2'


def mc_predict(model, X_tensor, n_samples=50):
    """MC Dropout inference: N forward passes, dropout kept active

    Args:
        model: nn.Module (must have weights loaded before calling)
        X_tensor: torch.Tensor, shape (1, seq_len, input_dim)
        n_samples: number of forward passes

    Returns:
        mean: np.ndarray, shape (1, n_targets), prediction mean
        std:  np.ndarray, shape (1, n_targets), prediction standard deviation (uncertainty)
        samples: np.ndarray, shape (n_samples, 1, n_targets), all samples
    """
    model.train()  # dropout kept active
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(model(X_tensor).cpu().numpy())
    samples = np.stack(samples, axis=0)  # (n_samples, 1, n_targets)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return mean, std, samples


def confidence_penalty(mean, std, strength=1.0):
    """Confidence penalty: high uncertainty → signal shrinks toward 0

    Formula:
        cv = std / (|mean| + ε)          — coefficient of variation
        penalty = 1 / (1 + strength × cv) — penalty factor ∈ (0, 1]
        adjusted = mean × penalty         — penalized signal
        confidence = clip(1 - cv, 0, 1)   — confidence ∈ [0, 1]

    Args:
        mean: np.ndarray, prediction mean
        std:  np.ndarray, prediction standard deviation
        strength: penalty strength, higher = more conservative (default 1.0)

    Returns:
        dict: {adjusted, penalty, confidence, cv} — each is np.ndarray
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
    """End-to-end: from numpy array to signal with uncertainty

    Args:
        model: nn.Module (weights already loaded)
        X_array: np.ndarray, shape (seq_len, input_dim) single sequence
        n_samples: MC sampling count
        penalty_strength: confidence penalty strength
        device: 'cuda' | 'cpu'

    Returns:
        dict with keys:
            y3_mean, y3_std, y3_adjusted, y3_confidence
            y6_mean, y6_std, y6_adjusted, y6_confidence
            overall_confidence — mean of y3 and y6 confidence
            signal — composite signal (y3 adjusted), positive=bullish, negative=bearish
            uncertainty_level — 'low' | 'medium' | 'high'
            n_samples
    """
    X_tensor = torch.from_numpy(X_array.astype(np.float32)).unsqueeze(0).to(device)
    mean, std, samples = mc_predict(model, X_tensor, n_samples)

    # Confidence penalty for y3 / y6
    y3 = confidence_penalty(mean[0, 0], std[0, 0], penalty_strength)
    y6 = confidence_penalty(mean[0, 1], std[0, 1], penalty_strength)

    overall_conf = float((y3['confidence'] + y6['confidence']) / 2.0)

    # Uncertainty grading
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
    """Format MC Dropout result as a dict injectable into prompts"""
    if not result:
        return None

    ulevel = result.get('uncertainty_level', 'medium')
    level_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
    level_desc = {
        'low': 'Model predictions highly consistent, signal reliability high',
        'medium': 'Model predictions show divergence, signal needs technical verification',
        'high': 'Model predictions highly divergent, signal unreliable, rely on technical analysis',
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
