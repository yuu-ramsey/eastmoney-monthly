"""
Signal generator - aggregates multiple independent Kronos K-line predictions into trading signals

Input: list of N prediction DataFrames (independent samples, each sample_count=1)
Output: dict { direction, predicted_change_pct, predicted_high,
             predicted_low, confidence, volatility }

置信度：N 次采样中，方向与最终判定一致的比例
波动率：N 次采样预测涨跌幅的标准差

注：采样循环（orchestrator）不在此模块，由上层负责跑 N 次 predict(sample_count=1)。
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


import numpy as np
import pandas as pd


def generate_signal(
    predictions: list[pd.DataFrame],
    threshold: float = 2.0,
) -> dict:
    """
    将多次独立采样的预测结果聚合为单一交易信号。

    Args:
        predictions: N 个预测 DataFrame 列表，每个来自一次独立的 predict(sample_count=1) 调用。
                     每份 DataFrame 按时间升序，columns 含 open/high/low/close。
        threshold: 判定 flat 的阈值（%），|涨跌幅| < threshold → "flat"

    Returns:
        dict:
            - direction: "up" / "down" / "flat"
            - predicted_change_pct: 预测涨跌幅 %（N 次均值）
            - predicted_high: 预测期最高价（N 次均值）
            - predicted_low: 预测期最低价（N 次均值）
            - predicted_close: 预测期末收盘价（N 次均值）
            - confidence: 方向一致性 0-1（与最终方向相同的采样占比）
            - volatility: 预测涨跌幅标准差 %
            - sample_count: 采样次数
            - up_count / down_count / flat_count: 各方向采样计数
    """
    if not predictions:
        raise ValueError("predictions 列表不能为空")

    # 提取每轮采样的关键指标
    changes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []

    for pred_df in predictions:
        # 预测期内涨跌幅：首根预测 K 线开盘 → 末根预测 K 线收盘
        open_first = float(pred_df["open"].iloc[0])
        close_last = float(pred_df["close"].iloc[-1])
        change_pct = (close_last - open_first) / open_first * 100.0

        changes.append(change_pct)
        highs.append(float(pred_df["high"].max()))
        lows.append(float(pred_df["low"].min()))
        closes.append(close_last)

    changes_arr = np.array(changes)
    median_change = float(np.median(changes_arr))
    mean_change = float(np.mean(changes_arr))
    std_change = float(np.std(changes_arr, ddof=1)) if len(changes_arr) > 1 else 0.0

    # 方向判定用均值（更敏感），报告用中位数（抗 outlier）
    if abs(median_change) < threshold:
        direction = "flat"
    elif median_change > 0:
        direction = "up"
    else:
        direction = "down"

    # 置信度：与最终判定方向一致的采样占比
    if direction == "up":
        consistent_count = int((changes_arr > 0).sum())
    elif direction == "down":
        consistent_count = int((changes_arr < 0).sum())
    else:
        consistent_count = int((abs(changes_arr) < threshold).sum())
    confidence = consistent_count / len(changes_arr)

    # 方向计数（按 threshold 阈值统计每个采样的独立方向）
    up_count = int((changes_arr > threshold).sum())
    down_count = int((changes_arr < -threshold).sum())
    flat_count = len(changes_arr) - up_count - down_count

    return {
        "direction": direction,
        "predicted_change_pct": round(median_change, 2),
        "predicted_high": round(float(np.median(highs)), 2),
        "predicted_low": round(float(np.median(lows)), 2),
        "predicted_close": round(float(np.median(closes)), 2),
        "confidence": round(confidence, 4),
        "volatility": round(std_change, 4),
        "sample_count": len(predictions),
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": flat_count,
    }


# ============================================================
# 验证入口
# ============================================================

if __name__ == "__main__":
    # 伪造 20 次独立采样的预测数据来验证信号生成逻辑
    np.random.seed(42)
    n_samples = 20

    fake_predictions: list[pd.DataFrame] = []
    for i in range(n_samples):
        dates = pd.date_range(start="2026-06-01", periods=3, freq="MS")
        # 模拟偏上涨：首根开盘 ≈ 1400，末根收盘在 1400~1460 间波动
        open_first = 1400.0 + np.random.randn() * 5
        close_last = open_first + 18.0 + np.random.randn() * 15  # 偏正
        df = pd.DataFrame({
            "open":  [open_first, close_last - 5, close_last - 3],
            "high":  [open_first + 10, close_last + 5, close_last + 8],
            "low":   [open_first - 10, close_last - 15, close_last - 12],
            "close": [open_first + 3, close_last - 2, close_last],
            "volume": [5e7, 6e7, 5.5e7],
            "amount": [7e10, 8.5e10, 7.8e10],
        }, index=dates)
        fake_predictions.append(df)

    signal = generate_signal(fake_predictions)

    print("=" * 60)
    print("Kronos 信号生成器验证")
    print(f"采样次数: {signal['sample_count']}（模拟偏上涨场景）")
    print("=" * 60)
    for k, v in signal.items():
        print(f"  {k:24s}: {v}")

    # 验证一：计数完整性
    assert signal["up_count"] + signal["down_count"] + signal["flat_count"] == n_samples
    # 验证二：方向与计数一致
    if signal["direction"] == "up":
        assert signal["predicted_change_pct"] >= 2.0
        assert signal["up_count"] >= signal["down_count"]
    elif signal["direction"] == "down":
        assert signal["predicted_change_pct"] <= -2.0
        assert signal["down_count"] >= signal["up_count"]
    else:
        assert abs(signal["predicted_change_pct"]) < 2.0

    # 验证三：confidence 在合理范围
    assert 0.0 <= signal["confidence"] <= 1.0

    print("\n所有断言通过.")
