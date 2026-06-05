"""
Kronos Tokenizer — K线数据 → BSQ 分层离散 token

核心流程：
1. 线性嵌入：OHLCV(6维) → d_model
2. Encoder Transformer → 压缩到 codebook_dim
3. BSQuantizer：L2归一化 → sign量化 → s1(粗粒度)+s2(细粒度) token
4. Decoder Transformer → 重建 OHLCV

s1_bits=10, s2_bits=10 → 码本维度=20 → 总词汇量=2^20≈1M token
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
    K线分词器：混合量化编码-解码架构

    将连续 OHLCV 数据压缩为离散 token 序列。
    s1 (前 s1_bits 位) 捕获粗粒度趋势，
    s2 (后 s2_bits 位) 捕获细粒度波动。

    Args:
        d_in: 输入维度（默认 6: open/high/low/close/volume/amount）
        d_model: 隐藏维度
        n_heads: 注意力头数
        ff_dim: FFN 隐藏维度
        n_enc_layers: Encoder 层数
        n_dec_layers: Decoder 层数
        s1_bits: 粗粒度 token 比特数
        s2_bits: 细粒度 token 比特数
        beta/gamma0/gamma/zeta: BSQ 损失权重
        group_size: BSQ 分组大小
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
        self.codebook_dim = s1_bits + s2_bits  # 总码本维度

        # 嵌入层
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder: 压缩输入到量化空间
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim,
                             self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])

        # Decoder: 从量化码重建数据（共享权重用于 pre 和 full）
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim,
                             self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])

        # 量化前后的投影层
        self.quant_embed = nn.Linear(self.d_model, self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(self.s1_bits, self.d_model)
        self.post_quant_embed = nn.Linear(self.codebook_dim, self.d_model)

        # BSQ 量化器
        self.tokenizer = BSQuantizer(
            self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size,
        )

    def forward(self, x: torch.Tensor):
        """
        训练用前向传播（含重建损失）。

        Returns:
            (z_pre, z): (仅用 s1 重建, 用完整码本重建) 两个输出
            bsq_loss: 量化损失
            quantized: 量化后的码字
            z_indices: token 索引
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z)  # [B, T, codebook_dim]

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        # 仅用 s1_bits 重建（粗粒度）
        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # 用完整码本重建（粗+细）
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x: torch.Tensor, half: bool = False) -> torch.Tensor:
        """
        整数索引 → 二元码 {-1, +1}^D 并缩放

        Args:
            x: 整数索引 或 (s1_idx, s2_idx) 当 half=True
            half: 是否使用分离的 s1/s2 索引
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
        编码：连续数据 → 离散 token 索引

        Args:
            x: [B, T, d_in] 输入数据
            half: True 返回 (s1_indices, s2_indices)

        Returns:
            整数 token 索引（或双索引元组）
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
        解码：token 索引 → 重建 OHLCV 数据

        Args:
            x: token 索引（或双索引元组）
            half: 输入是否为分离的 s1/s2 索引

        Returns:
            [B, T, d_in] 重建数据
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z
