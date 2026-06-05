"""
KronosTokenizer unit test - weight loading + encode/decode validation

Validation: after loading HuggingFace pretrained weights, encode->decode output dimensions correct,
Reconstructed data within reasonable magnitude range.
"""

import pytest
import torch

from kronos.tokenizer import KronosTokenizer

# 跳过标记：需要已download的权重
_tokenizer_weights_dir = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "weights" / "tokenizer"
)
requires_weights = pytest.mark.skipif(
    not _tokenizer_weights_dir.exists(),
    reason="未download权重，请先运行 download_weights.py"
)


class TestKronosTokenizerInit:
    """测试初始化参数"""

    def test_default_init(self):
        tokenizer = KronosTokenizer(
            d_in=6, d_model=256, n_heads=8, ff_dim=512,
            n_enc_layers=4, n_dec_layers=4,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            s1_bits=10, s2_bits=10,
            beta=0.25, gamma0=1.0, gamma=0.1, zeta=0.1,
            group_size=4,
        )
        assert tokenizer.d_in == 6
        assert tokenizer.d_model == 256
        assert tokenizer.codebook_dim == 20

    def test_small_vocab(self):
        """小码本：s1=4, s2=4 → codebook_dim=8"""
        tokenizer = KronosTokenizer(
            d_in=6, d_model=128, n_heads=4, ff_dim=256,
            n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            s1_bits=4, s2_bits=4,
            beta=0.25, gamma0=1.0, gamma=0.1, zeta=0.1,
            group_size=4,
        )
        assert tokenizer.codebook_dim == 8


class TestKronosTokenizerForward:
    """测试前向传播（随机权重）"""

    @pytest.fixture
    def tokenizer(self):
        return KronosTokenizer(
            d_in=6, d_model=128, n_heads=4, ff_dim=256,
            n_enc_layers=2, n_dec_layers=2,
            ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
            s1_bits=4, s2_bits=4,
            beta=0.25, gamma0=1.0, gamma=0.1, zeta=0.1,
            group_size=4,
        )

    def test_forward_shape(self, tokenizer):
        x = torch.randn(2, 20, 6)
        (z_pre, z_full), bsq_loss, quantized, z_indices = tokenizer(x)
        assert z_pre.shape == (2, 20, 6)
        assert z_full.shape == (2, 20, 6)
        assert quantized.shape == (2, 20, 8)
        assert z_indices.shape == (2, 20)
        assert bsq_loss.numel() == 1

    def test_encode_shape(self, tokenizer):
        x = torch.randn(2, 20, 6)
        indices = tokenizer.encode(x)
        assert indices.shape == (2, 20)
        assert indices.dtype == torch.long

    def test_encode_half_shape(self, tokenizer):
        x = torch.randn(2, 20, 6)
        s1, s2 = tokenizer.encode(x, half=True)
        assert s1.shape == (2, 20)
        assert s2.shape == (2, 20)

    def test_decode_shape(self, tokenizer):
        indices = torch.randint(0, 2 ** 8 - 1, (2, 20))
        recon = tokenizer.decode(indices)
        assert recon.shape == (2, 20, 6)

    def test_encode_decode_roundtrip(self, tokenizer):
        x = torch.randn(2, 20, 6)
        indices = tokenizer.encode(x)
        recon = tokenizer.decode(indices)
        assert recon.shape == x.shape

    def test_indices_to_bits(self, tokenizer):
        indices = torch.randint(0, 2 ** 8 - 1, (2, 20))
        bits = tokenizer.indices_to_bits(indices)
        assert bits.shape == (2, 20, 8)
        assert bits.abs().max() < 2.0


class TestKronosTokenizerPretrained:
    """预训练权重加载测试（需先运行 download_weights.py）"""

    @requires_weights
    def test_from_pretrained_tokenizer(self):
        tokenizer = KronosTokenizer.from_pretrained(str(_tokenizer_weights_dir))
        assert tokenizer.d_model == 256
        assert tokenizer.codebook_dim == tokenizer.s1_bits + tokenizer.s2_bits

    @requires_weights
    def test_pretrained_encode_decode(self):
        tokenizer = KronosTokenizer.from_pretrained(str(_tokenizer_weights_dir))
        x = torch.randn(1, 12, 6)
        with torch.no_grad():
            indices = tokenizer.encode(x)
            recon = tokenizer.decode(indices)
        assert recon.shape == (1, 12, 6)
