"""
Kronos Tokenizer - K-line data -> BSQ hierarchical discrete tokens

Core pipeline:
1. Linear embedding: OHLCV(6-dim) -> d_model
2. Encoder Transformer -> compress to codebook_dim
3. BSQuantizer: L2 normalization -> sign quantization -> s1 (coarse) + s2 (fine) tokens
4. Decoder Transformer -> reconstruct OHLCV

s1_bits=10, s2_bits=10 -> codebook dim=20 -> total vocab=2^20≈1M tokens
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


from typing import Tuple

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .module import BSQuantizer, TransformerBlock


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    K-line tokenizer: hybrid quantization encode-decode architecture

    Compresses continuous OHLCV data into discrete token sequences.
    s1 (first s1_bits bits) captures coarse-grained trend,
    s2 (last s2_bits bits) captures fine-grained fluctuation.

    Args:
        d_in: input dimension (default 6: open/high/low/close/volume/amount)
        d_model: hidden dimension
        n_heads: number of attention heads
        ff_dim: FFN hidden dimension
        n_enc_layers: number of encoder layers
        n_dec_layers: number of decoder layers
        s1_bits: coarse token bit count
        s2_bits: fine token bit count
        beta/gamma0/gamma/zeta: BSQ loss weights
        group_size: BSQ group size
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        n_enc_layers: int,
        n_dec_layers: int,
        ffn_dropout_p: float,
        attn_dropout_p: float,
        resid_dropout_p: float,
        s1_bits: int,
        s2_bits: int,
        beta: float,
        gamma0: float,
        gamma: float,
        zeta: float,
        group_size: int,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits  # Total codebook dimension

        # Embedding layer
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder: compress input to quantization space
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim,
                             self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])

        # Decoder: reconstruct data from quantized codes (shared weights for pre and full)
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim,
                             self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])

        # Projection layers before and after quantization
        self.quant_embed = nn.Linear(self.d_model, self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(self.s1_bits, self.d_model)
        self.post_quant_embed = nn.Linear(self.codebook_dim, self.d_model)

        # BSQ quantizer
        self.tokenizer = BSQuantizer(
            self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for training (includes reconstruction loss).

        Returns:
            (z_pre, z): two outputs (s1-only reconstruction, full-codebook reconstruction)
            bsq_loss: quantization loss
            quantized: quantized codewords
            z_indices: token indices
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z)  # [B, T, codebook_dim]

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        # Reconstruct using only s1_bits (coarse-grained)
        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # Reconstruct using full codebook (coarse + fine)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        """
        Integer indices -> binary codes {-1, +1}^D with scaling

        Args:
            x: integer indices, or (s1_idx, s2_idx) when half=True
            half: whether to use separated s1/s2 indices
        """
        if half:
            x1, x2 = x[0], x[1]
            mask = 2 ** torch.arange(self.codebook_dim // 2, device=x1.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask) != 0
            x2 = (x2.unsqueeze(-1) & mask) != 0
            x = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
            x = (x.unsqueeze(-1) & mask) != 0

        x = x.float() * 2 - 1  # boolean → {-1, +1}
        q_scale = 1.0 / (self.codebook_dim ** 0.5)
        x = x * q_scale
        return x

    @torch.no_grad()
    def encode(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        """
        Encode: continuous data -> discrete token indices

        Args:
            x: [B, T, d_in] input data
            half: if True, return (s1_indices, s2_indices)

        Returns:
            integer token indices (or dual-index tuple)
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        _bsq_loss, _quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    @torch.no_grad()
    def decode(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        """
        Decode: token indices -> reconstructed OHLCV data

        Args:
            x: token indices (or dual-index tuple)
            half: whether input uses separated s1/s2 indices

        Returns:
            [B, T, d_in] reconstructed data
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z
