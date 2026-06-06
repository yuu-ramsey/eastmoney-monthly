"""
Kronos weight download CLI - downloads pretrained models from HuggingFace Hub

Usage:
    python -m kronos.download_weights                        # download tokenizer + model
    python -m kronos.download_weights --tokenizer-only       # download tokenizer only
    python -m kronos.download_weights --model-only           # download model only
    python -m kronos.download_weights --model-size large     # download Kronos-large
    python -m kronos.download_weights --no-verify            # skip verification
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to support standalone execution
_KRONOS_DIR = Path(__file__).resolve().parent
if str(_KRONOS_DIR) not in sys.path:
    sys.path.insert(0, str(_KRONOS_DIR))

from kronos.tokenizer import KronosTokenizer
from kronos.transformer import Kronos

# HuggingFace model repositories
REPO_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
REPO_MODEL_BASE = "NeoQuasar/Kronos-base"
REPO_MODEL_LARGE = "NeoQuasar/Kronos-large"

# Local weight save paths
WEIGHTS_DIR = _KRONOS_DIR / "weights"
TOKENIZER_DIR = WEIGHTS_DIR / "tokenizer"
MODEL_DIR = WEIGHTS_DIR / "model"


def download_tokenizer() -> Path:
    """Download KronosTokenizer weights, return save path"""
    print(f"[1/4] Loading tokenizer config from {REPO_TOKENIZER}...")
    tokenizer = KronosTokenizer.from_pretrained(REPO_TOKENIZER)
    print(f"  Config: s1_bits={tokenizer.s1_bits}, s2_bits={tokenizer.s2_bits}, "
          f"d_model={tokenizer.d_model}, codebook_dim={tokenizer.codebook_dim}")

    print(f"[2/4] Saving to {TOKENIZER_DIR}...")
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(TOKENIZER_DIR))
    print("  Done.")
    return TOKENIZER_DIR


def download_model(model_size: str = "base") -> Path:
    """Download Kronos prediction model weights, return save path"""
    repo = REPO_MODEL_LARGE if model_size == "large" else REPO_MODEL_BASE
    print(f"[3/4] Loading model config from {repo}...")
    model = Kronos.from_pretrained(repo)
    print(f"  Config: s1_bits={model.s1_bits}, s2_bits={model.s2_bits}, "
          f"d_model={model.d_model}, n_layers={model.n_layers}, n_heads={model.n_heads}")

    print(f"[4/4] Saving to {MODEL_DIR}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    print("  Done.")
    return MODEL_DIR


def verify(tokenizer_path: Path, model_path: Path) -> bool:
    """
    Verify weight loading: create tokenizer + model instances and run a dummy forward pass.

    Returns:
        True if verification passed, False if failed
    """
    print("\n========================================")
    print("Verifying weight loading...")
    print("========================================")

    try:
        print("  Loading tokenizer...")
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))

        print("  Loading model...")
        model = Kronos.from_pretrained(str(model_path))

        # Synthetic test data
        B, T = 2, 10
        x = torch.randn(B, T, 6)
        s1_ids = torch.randint(0, 2 ** tokenizer.s1_bits - 1, (B, T))
        s2_ids = torch.randint(0, 2 ** tokenizer.s2_bits - 1, (B, T))

        print("  Tokenizer encode...")
        with torch.no_grad():
            indices = tokenizer.encode(x)
        assert indices.shape == (B, T), f"encode shape error: {indices.shape}"

        print("  Tokenizer decode...")
        with torch.no_grad():
            recon = tokenizer.decode(indices)
        assert recon.shape == (B, T, 6), f"decode shape error: {recon.shape}"

        print("  Model forward...")
        with torch.no_grad():
            s1_logits, s2_logits = model(s1_ids, s2_ids)
        assert s1_logits.shape == (B, T, 2 ** model.s1_bits), \
            f"s1_logits shape error: {s1_logits.shape}"
        assert s2_logits.shape == (B, T, 2 ** model.s2_bits), \
            f"s2_logits shape error: {s2_logits.shape}"

        print("  Verification passed!")
        return True

    except Exception as e:
        print(f"  Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Kronos pretrained weights (HuggingFace)"
    )
    parser.add_argument("--tokenizer-only", action="store_true",
                        help="Download tokenizer weights only")
    parser.add_argument("--model-only", action="store_true",
                        help="Download model weights only")
    parser.add_argument("--model-size", choices=["base", "large"],
                        default="base", help="Model size (default: base)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification after download")
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

    print("\nAll done.")
    if tokenizer_path:
        print(f"  Tokenizer: {tokenizer_path}")
    if model_path:
        print(f"  Model:     {model_path}")


if __name__ == "__main__":
    main()
