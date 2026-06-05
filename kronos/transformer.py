"""
Kronos 自回归 Transformer — 分层 token 预测模型

Decoder-only 架构：
1. 输入 s1+s2 token → HierarchicalEmbedding + TemporalEmbedding
2. N 层 TransformerBlock 处理
3. decode_s1: 先预测 s1（粗粒度趋势），采样得到 s1
4. decode_s2: 用 s1 作为条件，经 DependencyAwareLayer 预测 s2（细粒度波动）

自回归生成时逐 token 采样（temperature + top-p），s1 先于 s2。
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .module import (
    DependencyAwareLayer,
    DualHead,
    HierarchicalEmbedding,
    RMSNorm,
    TemporalEmbedding,
    TransformerBlock,
)


class Kronos(nn.Module, PyTorchModelHubMixin):
    """
    Kronos 预测模型：自回归预测 K线 token 序列

    输入历史 token → Transformer → 逐 token 生成未来序列。
    s1/s2 双级解码：先定趋势 (s1) 再定细节 (s2)。

    Args:
        s1_bits: 粗粒度 token 比特数
        s2_bits: 细粒度 token 比特数
        n_layers: Transformer 层数
        d_model: 隐藏维度
        n_heads: 注意力头数
        ff_dim: FFN 隐藏维度
        learn_te: 是否学习时间嵌入（True=可学习, False=固定正弦）
    """

    def __init__(
        self,
        s1_bits: int,
        s2_bits: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        ffn_dropout_p: float,
        attn_dropout_p: float,
        resid_dropout_p: float,
        token_dropout_p: float,
        learn_te: bool,
    ):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2 ** self.s1_bits

        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)

        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim,
                             self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])

        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = False,
        s1_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练用前向传播。

        Args:
            s1_ids: [B, T] s1 token IDs
            s2_ids: [B, T] s2 token IDs
            stamp: [B, T, 5] 时间戳
            padding_mask: [B, T] padding mask (True=padding)
            use_teacher_forcing: 是否用真实 s1 作为 s2 的条件
            s1_targets: teacher forcing 时的目标 s1

        Returns:
            (s1_logits, s2_logits): 形状分别为 [B,T,2^s1_bits] 和 [B,T,2^s2_bits]
        """
        x = self.embedding([s1_ids, s2_ids])

        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding

        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        # s1 粗粒度预测
        s1_logits = self.head(x)

        # s2 条件预测：依赖 s1
        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1,
            ).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)

        return s1_logits, s2_logits

    def decode_s1(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅解码 s1：返回 s1 logits + Transformer 上下文（供后续 decode_s2 使用）

        自回归生成时，每步先用此方法预测 s1，采样后再调 decode_s2 预测 s2。
        """
        x = self.embedding([s1_ids, s2_ids])

        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding

        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(
        self,
        context: torch.Tensor,
        s1_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        基于 decode_s1 的上下文和 s1 token 预测 s2。

        Args:
            context: decode_s1 返回的 Transformer 上下文 [B, T, d_model]
            s1_ids: 已采样的 s1 token [B, T]
            padding_mask: padding mask

        Returns:
            s2_logits: [B, T, 2^s2_bits]
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)
