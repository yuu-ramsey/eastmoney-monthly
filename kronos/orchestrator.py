"""
Orchestrator - chains data adapter -> prediction -> signal generation into a complete analysis pipeline

Fully local offline: read SQLite -> model inference -> signal aggregation, no network IO.
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


from typing import Optional

import pandas as pd

from .data_adapter import load_monthly_klines
from .predictor import KronosPredictor
from .signal_generator import generate_signal


def run_analysis(
    code: str,
    predictor: KronosPredictor,
    x_timestamp: str,
    y_timestamp: str,
    pred_len: int = 3,
    context_len: int = 256,
    top_k: int = 30,
    top_p: float = 0.85,
    temperature: float = 0.6,
    n_samples: int = 30,
    db_path: Optional[str] = None,
) -> dict:
    """
    Complete analysis pipeline for a single stock.

    Args:
        code: stock code
        predictor: KronosPredictor instance with loaded model
        x_timestamp: historical data cutoff time
        y_timestamp: prediction start time
        pred_len: number of predicted K-lines
        context_len: number of historical K-lines to use (context window)
        n_samples: number of independent sampling runs

    Returns:
        signal dict + stock metadata
    """
    df = load_monthly_klines(code, db_path=db_path, min_records=context_len)

    predictions: list[pd.DataFrame] = []
    for _ in range(n_samples):
        pred_df = predictor.predict(
            df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            context_len=context_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            sample_count=1,
        )
        predictions.append(pred_df)

    signal = generate_signal(predictions)

    # Attach metadata
    signal["code"] = code
    signal["data_range"] = (
        f"{df.index.min().strftime('%Y-%m')} → "
        f"{df.index.max().strftime('%Y-%m')}"
    )
    signal["data_count"] = len(df)
    signal["pred_len"] = pred_len

    return signal
