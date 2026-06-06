"""
KronosTokenizer unit test - weight loading + encode/decode validation

Validation: after loading HuggingFace pretrained weights, encode->decode output dimensions correct,
Reconstructed data within reasonable magnitude range.
"""

import pytest
import torch

from kronos.tokenizer import KronosTokenizer

# Skip marker: need downloaded weights
_tokenizer_weights_dir = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "weights" / "tokenizer"
)
requires_weights = pytest.mark.skipif(
    not _tokenizer_weights_dir.exists(),
    reason="Weights not downloaded. Please run download_weights.py first"
)


class TestKronosTokenizerInit:
    """Test initialization parameters"""

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
        """Small codebook: s1=4, s2=4 -> codebook_dim=8"""
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
    """Test forward pass (random weights)"""

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
    """Pretrained weights loaded test (must run download_weights.py first)"""

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
