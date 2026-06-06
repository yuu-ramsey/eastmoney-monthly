"""
Kronos autoregressive Transformer - hierarchical token prediction model

Decoder-only architecture:
1. Input s1+s2 token -> HierarchicalEmbedding + TemporalEmbedding
2. N layer TransformerBlock processing
3. decode_s1: first predict s1 (coarse trend), sample to get s1
4. decode_s2: use s1 as condition, via DependencyAwareLayer predict s2 (fine fluctuation)

Autoregressive generation samples token by token (temperature + top-p), s1 before s2.
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
    Kronos prediction model: autoregressive K-line token sequence prediction

    Input historical tokens -> Transformer -> generate future sequence token by token.
    s1/s2 dual-level decoding: decide trend (s1) first, then details (s2).

    Args:
        s1_bits: coarse token bit count
        s2_bits: fine token bit count
        n_layers: number of Transformer layers
        d_model: hidden dimension
        n_heads: number of attention heads
        ff_dim: FFN hidden dimension
        learn_te: whether to learn temporal embeddings (True=learnable, False=fixed sinusoidal)
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
        Forward pass for training.

        Args:
            s1_ids: [B, T] s1 token IDs
            s2_ids: [B, T] s2 token IDs
            stamp: [B, T, 5] timestamps
            padding_mask: [B, T] padding mask (True=padding)
            use_teacher_forcing: whether to use ground-truth s1 as s2's condition
            s1_targets: target s1 for teacher forcing

        Returns:
            (s1_logits, s2_logits): shapes [B,T,2^s1_bits] and [B,T,2^s2_bits] respectively
        """
        x = self.embedding([s1_ids, s2_ids])

        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding

        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        # s1 coarse trend prediction
        s1_logits = self.head(x)

        # s2 conditional prediction: depends on s1
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
        Decode s1 only: returns s1 logits + Transformer context (for subsequent decode_s2 use)

        During autoregressive generation, first use this method to predict s1 at each step,
        then call decode_s2 to predict s2 after sampling.
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
        Predict s2 based on decode_s1's context and s1 tokens.

        Args:
            context: Transformer context returned by decode_s1 [B, T, d_model]
            s1_ids: sampled s1 tokens [B, T]
            padding_mask: padding mask

        Returns:
            s2_logits: [B, T, 2^s2_bits]
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)
