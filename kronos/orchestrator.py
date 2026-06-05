"""
编排器 — 串联数据适配 → 预测 → 信号生成的完整分析流程

纯本地离线：读取 SQLite → 模型推理 → 信号聚合，不涉及任何网络 IO。
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
    单只股票完整分析流程。

    Args:
        code: 股票代码
        predictor: 已加载模型的 KronosPredictor 实例
        x_timestamp: 历史数据截止时间
        y_timestamp: 预测起始时间
        pred_len: 预测 K 线数量
        context_len: 使用的历史 K 线数量（上下文窗口）
        n_samples: 独立采样次数

    Returns:
        signal dict + 股票元信息
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

    # 附加元信息
    signal["code"] = code
    signal["data_range"] = (
        f"{df.index.min().strftime('%Y-%m')} → "
        f"{df.index.max().strftime('%Y-%m')}"
    )
    signal["data_count"] = len(df)
    signal["pred_len"] = pred_len

    return signal
