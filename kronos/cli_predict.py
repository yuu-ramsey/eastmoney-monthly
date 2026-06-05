"""
Kronos local offline prediction CLI

Usage:
    python cli_predict.py 600519                    # text output
    python cli_predict.py 600519 --json             # JSON 输出（供 Node.js 消费）
    python cli_predict.py 600519 --n-samples 20     # 自定义采样次数
    python cli_predict.py 600519 --pred-len 6       # 预测 6 根月线

前置条件: 先运行 download_weights.py download预训练权重到 kronos/weights/
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

# 项目根目录
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
    """加载 tokenizer + model 权重"""
    if not _TOKENIZER_DIR.exists():
        raise FileNotFoundError(
            f"Tokenizer 权重未找到: {_TOKENIZER_DIR}\n"
            f"请先运行: python -m kronos.download_weights"
        )
    if not _MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model 权重未找到: {_MODEL_DIR}\n"
            f"请先运行: python -m kronos.download_weights"
        )

    print(f"加载 tokenizer: {_TOKENIZER_DIR}", file=sys.stderr)
    tokenizer = KronosTokenizer.from_pretrained(str(_TOKENIZER_DIR))
    print(f"加载 model: {_MODEL_DIR}", file=sys.stderr)
    model = Kronos.from_pretrained(str(_MODEL_DIR))

    predictor = KronosPredictor(tokenizer, model, device=device)
    print(f"设备: {predictor.device}", file=sys.stderr)
    return predictor


def format_text(signal: dict) -> str:
    """signal dict → 人类可读文本"""
    direction_label = {"up": "看多", "down": "看空", "flat": "震荡"}

    lines = [
        f"代码: {signal['code']}",
        f"数据: {signal['data_count']} 条 ({signal['data_range']})",
        f"预测: 未来 {signal['pred_len']} 个月",
        "",
        f"方向: {direction_label.get(signal['direction'], signal['direction'])}",
        f"涨跌幅: {signal['predicted_change_pct']:+.2f}%",
        f"最高价: {signal['predicted_high']:.2f}",
        f"最低价: {signal['predicted_low']:.2f}",
        f"置信度: {signal['confidence']:.2%}",
        f"波动率: {signal['volatility']:.2f}%",
        f"采样: {signal['sample_count']} 次 "
        f"(涨{signal['up_count']}/跌{signal['down_count']}/平{signal['flat_count']})",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Kronos 本地离线预测")
    parser.add_argument("code", help="股票代码，如 600519")
    parser.add_argument("--json", action="store_true", help="JSON 输出（供 Node.js 解析）")
    parser.add_argument("--n-samples", type=int, default=30, help="采样次数 (default: 30)")
    parser.add_argument("--pred-len", type=int, default=3, help="预测月数 (default: 3)")
    parser.add_argument("--context-len", type=int, default=256, help="上下文窗口长度 (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度 (default: 0.6)")
    parser.add_argument("--top-k", type=int, default=30, help="top-k 过滤 (default: 30)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--x-timestamp", default=None, help="historical截止时间 (default: 自动取最新月)")
    parser.add_argument("--y-timestamp", default=None, help="预测起始时间 (default: x_timestamp + 1月)")
    parser.add_argument("--db", default=None, help="SQLite 数据库路径")
    args = parser.parse_args()

    # JSON 模式下抑制所有非 JSON 输出到 stdout
    _real_stdout = sys.stdout
    if args.json:
        sys.stdout = open(os.devnull, 'w')

    try:
        # 加载模型
        try:
            predictor = load_predictor(args.device)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

        # 自动推断时间戳
        df_meta = load_monthly_klines(args.code, db_path=args.db, min_records=12)
        latest_date = df_meta.index.max()

        if args.x_timestamp is None:
            # 默认取最后完整月作为 x_ts。
            # 若当月未结束（<28 日），最后一条月线是 partial month，排除之。
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

        # 预测
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
