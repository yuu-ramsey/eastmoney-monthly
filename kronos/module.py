"""
Kronos 核心模块 — BSQ 量化器、注意力机制、Transformer 块、嵌入层

参照 Kronos 原 repo (github.com/shiyu-coder/Kronos) model/module.py 复现。
为兼容 HuggingFace 预训练权重，类名和 __init__ 签名与原 repo 完全一致。
"""

# Reproduced from Kronos (https://github.com/shiyu-coder/Kronos)
# Original author: shiyu-coder | License: MIT | Paper: arXiv:2508.02739 (AAAI 2026)
# See kronos/LICENSE for full license text.


import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.autograd import Function


# ============================================================
# 可微分熵估计 — 用于 BSQ 码本使用率正则化
# 论文: Binary Spherical Quantization (arxiv 2406.07548)
# ============================================================

class DifferentiableEntropyFunction(Function):
    """可微分码本熵损失：用直方图近似码本使用分布的熵"""

    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        zb = (zq + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)
        cnt = torch.scatter_reduce(
            torch.zeros(2 ** K, device=zq.device, dtype=zq.dtype),
            0, zi.flatten(),
            torch.ones_like(zi.flatten()).to(zq.dtype),
            'sum',
        )
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-4):
    """码本熵包装函数"""
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


# ============================================================
# Binary Spherical Quantizer (BSQ)
# 将连续向量量化为 {-1, +1}^D 的二元球面码字
# ============================================================

class BinarySphericalQuantizer(nn.Module):
    """
    二元球面量化器。

    将归一化后的连续向量 z 通过 sign(z) 量化为二元向量 z_hat ∈ {-1, +1}^D，
    训练时通过 straight-through estimator 传递梯度。

    Args:
        embed_dim: 码字维度（比特数）
        beta: commit loss 权重
        gamma0: 逐样本熵惩罚权重
        gamma: 码本熵奖励权重（鼓励码本利用率）
        zeta: 总熵惩罚缩放
        group_size: 分组大小（用于熵近似计算）
    """

    def __init__(
        self,
        embed_dim: int,
        beta: float,
        gamma0: float,
        gamma: float,
        zeta: float,
        input_format: str = 'bchw',
        soft_entropy: bool = True,
        group_size: int = 9,
        persample_entropy_compute: str = 'analytical',
        cb_entropy_compute: str = 'group',
        l2_norm: bool = True,
        inv_temperature: float = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.input_format = input_format

        assert self.embed_dim % group_size == 0, "embed_dim 必须能被 group_size 整除"
        self.num_groups = self.embed_dim // group_size
        self.group_size = group_size
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature

        # 位权基向量（用于将二元码转换为整数索引）
        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer('group_basis', 2 ** torch.arange(group_size - 1, -1, -1))

        self.num_dimensions = 2 ** embed_dim
        self.bits_per_index = embed_dim

        # 分组码本（用于近似熵计算）
        group_codes = torch.arange(2 ** self.group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer('group_codebook', group_codebook, persistent=False)

        self.soft_entropy = soft_entropy

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator: forward=sign(z), backward=identity"""
        zhat = torch.where(
            z > 0,
            torch.tensor(1, dtype=z.dtype, device=z.device),
            torch.tensor(-1, dtype=z.dtype, device=z.device),
        )
        return z + (zhat - z).detach()

    def forward(self, z: torch.Tensor, collect_metrics: bool = True):
        """前向传播：量化 + 计算损失"""
        zq = self.quantize(z)
        q_scale = 1.0 / (self.embed_dim ** 0.5) if self.l2_norm else 1.0
        zq = zq * q_scale

        if not collect_metrics:
            return zq, zq.new_zeros(()), {}

        indices = self.codes_to_indexes(zq.detach())
        group_indices = self.codes_to_group_indexes(zq.detach())
        used_codes = None if self.training else torch.unique(indices, return_counts=False)

        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb_by_sample = ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        return (
            zq,
            commit_loss + self.zeta * entropy_penalty / self.inv_temperature,
            {
                "H": cb_entropy,
                "used_codes": used_codes,
                "indices": indices,
                "group_indices": group_indices,
                "avg_prob": avg_prob,
            },
        )

    def soft_entropy_loss(self, z: torch.Tensor):
        """软分配熵估计：用分组码本近似逐样本熵和码本熵"""
        group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1)
        divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)

        distance = -2 * torch.einsum('... g c, d c ->... g d', divided_z, group_code_book)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)

        if self.persample_entropy_compute == 'analytical':
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        avg_prob = reduce(prob, '... g d ->g d', 'mean')
        codebook_entropy_val = self.get_entropy(avg_prob, dim=-1, normalize=False)

        return per_sample_entropy, codebook_entropy_val.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample: torch.Tensor) -> torch.Tensor:
        """硬分配逐样本熵（基于统计频率）"""
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = (
            -probs_per_dim * torch.log(probs_per_dim + 1e-8)
            - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
        )
        return persample_entropy.sum(-1).mean()

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """二元码 {-1,+1} → 整数索引（大端序）"""
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        """二元码 → 分组整数索引"""
        zhat_in_group = rearrange(zhat, 'b ... (g c) -> b ... g c', c=self.group_size)
        return ((zhat_in_group + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """整数索引 → 二元码 {-1,+1}"""
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self.basis), 2)
        return codes_non_centered * 2 - 1

    def group_indexes_to_codes(self, group_indices: torch.Tensor) -> torch.Tensor:
        """分组索引 → 二元码"""
        group_indices = group_indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(group_indices, self.group_basis), 2)
        codes_non_centered = rearrange(codes_non_centered, 'b ... g c -> b ... (g c)')
        return codes_non_centered * 2 - 1

    def get_entropy(self, count: torch.Tensor, dim: int = -1, eps: float = 1e-4, normalize: bool = True) -> torch.Tensor:
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        return -(probs * torch.log(probs + 1e-8)).sum(dim=dim)

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """索引 → 量化向量（含缩放）"""
        z_q = self.indexes_to_codes(indices)
        q_scale = 1.0 / (self.embed_dim ** 0.5) if self.l2_norm else 1.0
        z_q = z_q * q_scale
        return z_q

    def get_group_codebook_entry(self, group_indices: torch.Tensor) -> torch.Tensor:
        """分组索引 → 量化向量"""
        z_q = self.group_indexes_to_codes(group_indices)
        q_scale = 1.0 / (self.embed_dim ** 0.5) if self.l2_norm else 1.0
        z_q = z_q * q_scale
        return z_q


# ============================================================
# BSQuantizer — BSQ 包装器
# 将输入向量 L2 归一化后量化，输出 s1/s2 双级 token 索引
# ============================================================

class BSQuantizer(nn.Module):
    """BSQ 包装器：归一化 + BSQ 量化 → s1(粗粒度) + s2(细粒度) token"""

    def __init__(self, s1_bits: int, s2_bits: int, beta: float, gamma0: float, gamma: float,
                 zeta: float, group_size: int):
        super().__init__()
        self.codebook_dim = s1_bits + s2_bits
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.bsq = BinarySphericalQuantizer(
            self.codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size,
        )

    def bits_to_indices(self, bits: torch.Tensor) -> torch.Tensor:
        """二元码 → 整数索引"""
        bits = (bits >= 0).to(torch.long)
        indices = 2 ** torch.arange(0, bits.shape[-1], 1, dtype=torch.long, device=bits.device)
        return (bits * indices).sum(-1)

    def forward(self, z: torch.Tensor, half: bool = False, collect_metrics: bool = True):
        z = F.normalize(z, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
        if half:
            # 分离 s1（前 s1_bits）和 s2（后 s2_bits）
            q_pre = quantized[:, :, :self.s1_bits]
            q_post = quantized[:, :, self.s1_bits:]
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices


# ============================================================
# RMS 归一化
# ============================================================

class RMSNorm(torch.nn.Module):
    """RMS 归一化（比 LayerNorm 更快，去掉了均值中心化）"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ============================================================
# SwiGLU 前馈网络
# ============================================================

class FeedForward(nn.Module):
    """SwiGLU 前馈：w2(SiLU(w1(x)) * w3(x))"""

    def __init__(self, d_model: int, ff_dim: int, ffn_dropout_p: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ============================================================
# Rotary Positional Embedding (RoPE)
# ============================================================

class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码（RoPE），用于自注意力和交叉注意力"""

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached: Optional[int] = None
        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None

    def _update_cos_sin_cache(self, x: torch.Tensor, seq_len: int):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        cos, sin = self._update_cos_sin_cache(q, q.shape[-2])
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


# ============================================================
# 多头注意力（带 RoPE）
# ============================================================

class MultiHeadAttentionWithRoPE(nn.Module):
    """因果自注意力 + RoPE 位置编码"""

    def __init__(self, d_model: int, n_heads: int, attn_dropout_p: float = 0.0,
                 resid_dropout_p: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    """交叉注意力 + RoPE：query 来自当前 token，key/value 来自上下文"""

    def __init__(self, d_model: int, n_heads: int, attn_dropout_p: float = 0.0,
                 resid_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, q_len, _ = query.shape
        _, seq_len, _ = key.shape

        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        is_causal_flag = self.training

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal_flag,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


# ============================================================
# Transformer 块
# ============================================================

class TransformerBlock(nn.Module):
    """标准 Pre-LN Transformer 块：自注意力 + SwiGLU FFN"""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int = 1024,
                 ffn_dropout_p: float = 0.0, attn_dropout_p: float = 0.0,
                 resid_dropout_p: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 分层嵌入 — s1(粗粒度) + s2(细粒度) token 融合
# ============================================================

class HierarchicalEmbedding(nn.Module):
    """
    分层嵌入：s1(粗粒度趋势) + s2(细粒度波动) → 融合向量

    s1_bits 位粗 token（高位） + s2_bits 位细 token（低位）
    → 各自查 Embedding 表 → concat → Linear 融合
    """

    def __init__(self, s1_bits: int, s2_bits: int, d_model: int = 256):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits

        vocab_s1 = 2 ** s1_bits
        vocab_s2 = 2 ** s2_bits

        self.emb_s1 = nn.Embedding(vocab_s1, d_model)
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)
        self.d_model = d_model
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        nn.init.normal_(self.emb_s1.weight, mean=0, std=d_model ** -0.5)
        nn.init.normal_(self.emb_s2.weight, mean=0, std=d_model ** -0.5)

    def split_token(self, token_ids: torch.Tensor, s2_bits: int):
        """复合 token → (s1_ids, s2_ids)"""
        t = token_ids.long()
        mask = (1 << s2_bits) - 1
        s2_ids = t & mask          # 低位 = 细粒度
        s1_ids = t >> s2_bits      # 高位 = 粗粒度
        return s1_ids, s2_ids

    def forward(self, token_ids: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """输入 (s1_ids, s2_ids) 或复合 token → [B, T, d_model]"""
        if isinstance(token_ids, (tuple, list)):
            s1_ids, s2_ids = token_ids
        else:
            s1_ids, s2_ids = self.split_token(token_ids, self.s2_bits)

        s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
        s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
        return self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))


# ============================================================
# 依赖感知层 — s1 预测 → s2 条件交叉注意力
# ============================================================

class DependencyAwareLayer(nn.Module):
    """
    s1→s2 依赖感知层：用已解码的 s1 token 作为 query，
    对 Transformer 上下文做交叉注意力，实现 s2 对 s1 的条件依赖。
    """

    def __init__(self, d_model: int, n_heads: int = 4, attn_dropout_p: float = 0.0,
                 resid_dropout: float = 0.0):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states: torch.Tensor, sibling_embed: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.cross_attn(
            query=sibling_embed,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(hidden_states + attn_out)


# ============================================================
# 双头输出 — s1 粗粒度 + s2 细粒度 分类头
# ============================================================

class DualHead(nn.Module):
    """双头输出：s1_head(趋势) + s2_head(波动)"""

    def __init__(self, s1_bits: int, s2_bits: int, d_model: int):
        super().__init__()
        self.vocab_s1 = 2 ** s1_bits
        self.vocab_s2 = 2 ** s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)

    def compute_loss(self, s1_logits: torch.Tensor, s2_logits: torch.Tensor,
                     s1_targets: torch.Tensor, s2_targets: torch.Tensor,
                     padding_mask: Optional[torch.Tensor] = None):
        """交叉熵损失（s1 + s2 平均）"""
        if padding_mask is not None:
            valid_mask = (padding_mask == 0)
            s1_logits = s1_logits[valid_mask]
            s2_logits = s2_logits[valid_mask]
            s1_targets = s1_targets[valid_mask]
            s2_targets = s2_targets[valid_mask]
        ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
        ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        return (ce_s1 + ce_s2) / 2, ce_s1, ce_s2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """s1 输出头"""
        return self.proj_s1(x)

    def cond_forward(self, x2: torch.Tensor) -> torch.Tensor:
        """s2 条件输出头（在 s1 已解码后调用）"""
        return self.proj_s2(x2)


# ============================================================
# 时间戳嵌入 — 将分钟/小时/星期/日/月编码为连续向量
# ============================================================

class FixedEmbedding(nn.Module):
    """固定位置编码（正弦/余弦），不可学习"""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    时间戳嵌入：将 5 维时间特征（分钟/小时/星期/日/月）
    分别查表嵌入后求和，注入到 token 嵌入中
    """

    def __init__(self, d_model: int, learn_pe: bool):
        super().__init__()
        Embed = FixedEmbedding if not learn_pe else nn.Embedding

        self.minute_embed = Embed(60, d_model)
        self.hour_embed = Embed(24, d_model)
        self.weekday_embed = Embed(7, d_model)
        self.day_embed = Embed(32, d_model)
        self.month_embed = Embed(13, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 5] (minute, hour, weekday, day, month)"""
        x = x.long()
        return (
            self.minute_embed(x[:, :, 0])
            + self.hour_embed(x[:, :, 1])
            + self.weekday_embed(x[:, :, 2])
            + self.day_embed(x[:, :, 3])
            + self.month_embed(x[:, :, 4])
        )
