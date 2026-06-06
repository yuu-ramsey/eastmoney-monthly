"""
Kronos local offline prediction CLI

Usage:
    python cli_predict.py 600519                    # text output
    python cli_predict.py 600519 --json             # JSON output (for Node.js consumption)
    python cli_predict.py 600519 --n-samples 20     # custom sample count
    python cli_predict.py 600519 --pred-len 6       # predict 6 monthly candles

Prerequisite: run download_weights.py first to download pretrained weights to kronos/weights/
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Project root directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from kronos.data_adapter import DEFAULT_DB_PATH, load_monthly_klines
from kronos.orchestrator import run_analysis
from kronos.predictor import KronosPredictor
from kronos.tokenizer import KronosTokenizer
from kronos.transformer import Kronos

_KRONOS_DIR = Path(__file__).resolve().parent
_TOKENIZER_DIR = _KRONOS_DIR / "weights" / "tokenizer"
_MODEL_DIR = _KRONOS_DIR / "weights" / "model"


def load_predictor(device: str = "cuda") -> KronosPredictor:
    """Load tokenizer + model weights"""
    if not _TOKENIZER_DIR.exists():
        raise FileNotFoundError(
            f"Tokenizer weights not found: {_TOKENIZER_DIR}\n"
            f"Please run first: python -m kronos.download_weights"
        )
    if not _MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model weights not found: {_MODEL_DIR}\n"
            f"Please run first: python -m kronos.download_weights"
        )

    print(f"Loaded tokenizer: {_TOKENIZER_DIR}", file=sys.stderr)
    tokenizer = KronosTokenizer.from_pretrained(str(_TOKENIZER_DIR))
    print(f"Loaded model: {_MODEL_DIR}", file=sys.stderr)
    model = Kronos.from_pretrained(str(_MODEL_DIR))

    predictor = KronosPredictor(tokenizer, model, device=device)
    print(f"Device: {predictor.device}", file=sys.stderr)
    return predictor


def format_text(signal: dict) -> str:
    """signal dict -> human-readable text"""
    direction_label = {"up": "Bullish", "down": "Bearish", "flat": "Sideways"}

    lines = [
        f"code: {signal['code']}",
        f"Data: {signal['data_count']} records ({signal['data_range']})",
        f"Prediction: next {signal['pred_len']} months",
        "",
        f"Direction: {direction_label.get(signal['direction'], signal['direction'])}",
        f"Change: {signal['predicted_change_pct']:+.2f}%",
        f"High: {signal['predicted_high']:.2f}",
        f"Low: {signal['predicted_low']:.2f}",
        f"Confidence: {signal['confidence']:.2%}",
        f"Volatility: {signal['volatility']:.2f}%",
        f"Samples: {signal['sample_count']} "
        f"(up{signal['up_count']}/down{signal['down_count']}/flat{signal['flat_count']})",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Kronos local offline prediction")
    parser.add_argument("code", help="stock code, e.g. 600519")
    parser.add_argument("--json", action="store_true", help="JSON output (for Node.js parsing)")
    parser.add_argument("--n-samples", type=int, default=30, help="sample count (default: 30)")
    parser.add_argument("--pred-len", type=int, default=3, help="prediction months (default: 3)")
    parser.add_argument("--context-len", type=int, default=256, help="context window length (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.6, help="sampling temperature (default: 0.6)")
    parser.add_argument("--top-k", type=int, default=30, help="top-k filtering (default: 30)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--x-timestamp", default=None, help="historical cutoff time (default: auto-use latest month)")
    parser.add_argument("--y-timestamp", default=None, help="prediction start time (default: x_timestamp + 1 month)")
    parser.add_argument("--db", default=None, help="SQLite database path")
    args = parser.parse_args()

    # In JSON mode, suppress all non-JSON output to stdout
    _real_stdout = sys.stdout
    if args.json:
        sys.stdout = open(os.devnull, 'w')

    try:
        # Load model
        try:
            predictor = load_predictor(args.device)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

        # Automatically infer timestamp
        df_meta = load_monthly_klines(args.code, db_path=args.db, min_records=12)
        latest_date = df_meta.index.max()

        if args.x_timestamp is None:
            # Default: use the last completed month as x_ts.
            # If the current month is not yet finished (day < 28), the last monthly candle
            # is a partial month, so exclude it.
            now = datetime.now()
            latest_month = latest_date.month
            latest_year = latest_date.year
            if latest_year == now.year and latest_month == now.month and now.day < 28:
                if len(df_meta) >= 2:
                    x_ts = df_meta.index[-2].strftime("%Y-%m-%d")
                else:
                    x_ts = latest_date.strftime("%Y-%m-%d")
            else:
                x_ts = latest_date.strftime("%Y-%m-%d")
        else:
            x_ts = args.x_timestamp

        if args.y_timestamp is None:
            if latest_date.month == 12:
                y_ts = f"{latest_date.year + 1}-01-01"
            else:
                y_ts = f"{latest_date.year}-{latest_date.month + 1:02d}-01"
        else:
            y_ts = args.y_timestamp

        # Predict
        signal = run_analysis(
            code=args.code,
            predictor=predictor,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=args.pred_len,
            context_len=args.context_len,
            temperature=args.temperature,
            top_k=args.top_k,
            n_samples=args.n_samples,
            db_path=args.db,
        )
        signal["timestamp"] = datetime.now().isoformat()
    finally:
        if args.json:
            sys.stdout = _real_stdout

    if args.json:
        output = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in signal.items()
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(format_text(signal))


if __name__ == "__main__":
    main()
