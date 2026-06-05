"""
Kronos weight download CLI — downloads pretrained models from HuggingFace Hub

Usage:
    python -m kronos.download_weights                        # 下载 tokenizer + model
    python -m kronos.download_weights --tokenizer-only       # 仅下载 tokenizer
    python -m kronos.download_weights --model-only           # 仅下载 model
    python -m kronos.download_weights --model-size large     # 下载 Kronos-large
    python -m kronos.download_weights --no-verify            # 跳过验证
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


import argparse
import sys
from pathlib import Path

import torch

# 添加父目录以支持独立运行
_KRONOS_DIR = Path(__file__).resolve().parent
if str(_KRONOS_DIR) not in sys.path:
    sys.path.insert(0, str(_KRONOS_DIR))

from kronos.tokenizer import KronosTokenizer
from kronos.transformer import Kronos

# HuggingFace 模型仓库
REPO_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
REPO_MODEL_BASE = "NeoQuasar/Kronos-base"
REPO_MODEL_LARGE = "NeoQuasar/Kronos-large"

# 本地权重保存路径
WEIGHTS_DIR = _KRONOS_DIR / "weights"
TOKENIZER_DIR = WEIGHTS_DIR / "tokenizer"
MODEL_DIR = WEIGHTS_DIR / "model"


def download_tokenizer() -> Path:
    """下载 KronosTokenizer 权重，返回保存路径"""
    print(f"[1/4] 从 {REPO_TOKENIZER} 加载 tokenizer 配置...")
    tokenizer = KronosTokenizer.from_pretrained(REPO_TOKENIZER)
    print(f"  配置: s1_bits={tokenizer.s1_bits}, s2_bits={tokenizer.s2_bits}, "
          f"d_model={tokenizer.d_model}, codebook_dim={tokenizer.codebook_dim}")

    print(f"[2/4] 保存到 {TOKENIZER_DIR}...")
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(TOKENIZER_DIR))
    print("  完成.")
    return TOKENIZER_DIR


def download_model(model_size: str = "base") -> Path:
    """下载 Kronos 预测模型权重，返回保存路径"""
    repo = REPO_MODEL_LARGE if model_size == "large" else REPO_MODEL_BASE
    print(f"[3/4] 从 {repo} 加载 model 配置...")
    model = Kronos.from_pretrained(repo)
    print(f"  配置: s1_bits={model.s1_bits}, s2_bits={model.s2_bits}, "
          f"d_model={model.d_model}, n_layers={model.n_layers}, n_heads={model.n_heads}")

    print(f"[4/4] 保存到 {MODEL_DIR}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    print("  完成.")
    return MODEL_DIR


def verify(tokenizer_path: Path, model_path: Path) -> bool:
    """
    验证权重加载：创建 tokenizer + model 实例并跑一次 dummy forward pass。

    Returns:
        True 表示验证通过，False 表示失败
    """
    print("\n========================================")
    print("验证权重加载...")
    print("========================================")

    try:
        print("  加载 tokenizer...")
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))

        print("  加载 model...")
        model = Kronos.from_pretrained(str(model_path))

        # 合成测试数据
        B, T = 2, 10
        x = torch.randn(B, T, 6)
        s1_ids = torch.randint(0, 2 ** tokenizer.s1_bits - 1, (B, T))
        s2_ids = torch.randint(0, 2 ** tokenizer.s2_bits - 1, (B, T))

        print("  Tokenizer encode...")
        with torch.no_grad():
            indices = tokenizer.encode(x)
        assert indices.shape == (B, T), f"encode shape 错误: {indices.shape}"

        print("  Tokenizer decode...")
        with torch.no_grad():
            recon = tokenizer.decode(indices)
        assert recon.shape == (B, T, 6), f"decode shape 错误: {recon.shape}"

        print("  Model forward...")
        with torch.no_grad():
            s1_logits, s2_logits = model(s1_ids, s2_ids)
        assert s1_logits.shape == (B, T, 2 ** model.s1_bits), \
            f"s1_logits shape 错误: {s1_logits.shape}"
        assert s2_logits.shape == (B, T, 2 ** model.s2_bits), \
            f"s2_logits shape 错误: {s2_logits.shape}"

        print("  验证通过!")
        return True

    except Exception as e:
        print(f"  验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="下载 Kronos 预训练权重 (HuggingFace)"
    )
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="仅下载 tokenizer 权重")
    parser.add_argument("--model-only", action="store_true",
                        help="仅下载 model 权重")
    parser.add_argument("--model-size", choices=["base", "large"],
                        default="base", help="model 规模 (default: base)")
    parser.add_argument("--no-verify", action="store_true",
                        help="下载后不跑验证")
    args = parser.parse_args()

    tokenizer_path: Path | None = None
    model_path: Path | None = None

    if not args.model_only:
        tokenizer_path = download_tokenizer()

    if not args.tokenizer_only:
        model_path = download_model(args.model_size)

    if not args.no_verify:
        tp = tokenizer_path or TOKENIZER_DIR
        mp = model_path or MODEL_DIR
        ok = verify(tp, mp)
        if not ok:
            sys.exit(1)

    print("\n全部完成.")
    if tokenizer_path:
        print(f"  Tokenizer: {tokenizer_path}")
    if model_path:
        print(f"  Model:     {model_path}")


if __name__ == "__main__":
    main()
