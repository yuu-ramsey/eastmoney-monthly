"""
Signal generator - aggregates multiple independent Kronos K-line predictions into trading signals

Input: list of N prediction DataFrames (independent samples, each sample_count=1)
Output: dict { direction, predicted_change_pct, predicted_high,
             predicted_low, confidence, volatility }

Confidence: proportion of N samples whose direction matches the final verdict
Volatility: standard deviation of predicted change percentages across N samples

Note: the sampling loop (orchestrator) is not in this module; the caller runs N predict(sample_count=1) calls.
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
    Aggregate multiple independent sample predictions into a single trading signal.

    Args:
        predictions: list of N prediction DataFrames, each from one independent predict(sample_count=1) call.
                     Each DataFrame sorted by time ascending, columns include open/high/low/close.
        threshold: threshold for "flat" classification (%), |change_pct| < threshold -> "flat"

    Returns:
        dict:
            - direction: "up" / "down" / "flat"
            - predicted_change_pct: predicted change % (mean of N samples)
            - predicted_high: highest price in prediction period (mean of N samples)
            - predicted_low: lowest price in prediction period (mean of N samples)
            - predicted_close: closing price at end of prediction period (mean of N samples)
            - confidence: directional consistency 0-1 (proportion of samples matching final direction)
            - volatility: standard deviation of predicted change percentages %
            - sample_count: number of sampling runs
            - up_count / down_count / flat_count: per-direction sample counts
    """
    if not predictions:
        raise ValueError("predictions list cannot be empty")

    # Extract key metrics from each sampling round
    changes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []

    for pred_df in predictions:
        # Change over prediction period: first predicted candle open -> last predicted candle close
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

    # Use median for direction decision (outlier-resistant); report also uses median
    if abs(median_change) < threshold:
        direction = "flat"
    elif median_change > 0:
        direction = "up"
    else:
        direction = "down"

    # Confidence: proportion of samples matching the final direction verdict
    if direction == "up":
        consistent_count = int((changes_arr > 0).sum())
    elif direction == "down":
        consistent_count = int((changes_arr < 0).sum())
    else:
        consistent_count = int((abs(changes_arr) < threshold).sum())
    confidence = consistent_count / len(changes_arr)

    # Direction counts (count each sample's independent direction by threshold)
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
# Verification entry point
# ============================================================

if __name__ == "__main__":
    # Generate 20 fake independent sample predictions to verify signal generation logic
    np.random.seed(42)
    n_samples = 20

    fake_predictions: list[pd.DataFrame] = []
    for i in range(n_samples):
        dates = pd.date_range(start="2026-06-01", periods=3, freq="MS")
        # Simulate bullish bias: first candle open ~1400, last candle close fluctuates in 1400~1460 range
        open_first = 1400.0 + np.random.randn() * 5
        close_last = open_first + 18.0 + np.random.randn() * 15  # upward bias
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
    print("Kronos Signal Generator Verification")
    print(f"Sample count: {signal['sample_count']} (simulated bullish scenario)")
    print("=" * 60)
    for k, v in signal.items():
        print(f"  {k:24s}: {v}")

    # Assertion 1: count completeness
    assert signal["up_count"] + signal["down_count"] + signal["flat_count"] == n_samples
    # Assertion 2: direction consistent with counts
    if signal["direction"] == "up":
        assert signal["predicted_change_pct"] >= 2.0
        assert signal["up_count"] >= signal["down_count"]
    elif signal["direction"] == "down":
        assert signal["predicted_change_pct"] <= -2.0
        assert signal["down_count"] >= signal["up_count"]
    else:
        assert abs(signal["predicted_change_pct"]) < 2.0

    # Assertion 3: confidence within reasonable range
    assert 0.0 <= signal["confidence"] <= 1.0

    print("\nAll assertions passed.")
