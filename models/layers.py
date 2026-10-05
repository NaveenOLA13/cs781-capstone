"""Shared building blocks for architectures #3 (MoE) and #4 (Llama-style).

RMSNorm (Zhang & Sennrich 2019) - RoPE (Su et al. 2021) - grouped-query attention (Ainslie et al. 2023)
- SwiGLU (Shazeer 2020). GQAttention is a REFERENCE implementation, to be replaced by my own A3 version.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModernConfig:
    vocab_size: int
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4            # GQA: 12 query heads share 4 K/V heads (n_kv_heads=1 -> MQA, =n_heads -> MHA)
    d_ff: int = 2048               # SwiGLU hidden size (per expert for MoE)
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    # MoE only
    n_experts: int = 8
    top_k: int = 2
    aux_loss_coef: float = 0.01


# ------------------------------------------------------------------ building blocks
class RMSNorm(nn.Module):
    """x / RMS(x) * g   (no mean subtraction, no bias)."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_cache(max_len, d_head, theta=10000.0):
    """cos/sin tables (max_len, d_head/2) for rotary position embedding."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    ang = torch.outer(torch.arange(max_len).float(), inv_freq)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, d_head). Rotates each pair (x[2i], x[2i+1]) by angle pos * theta_i."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


class GQAttention(nn.Module):
    """Causal grouped-query attention with RoPE (reference version, uses PyTorch SDPA kernel)."""
    def __init__(self, cfg: ModernConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.h, self.kv = cfg.n_heads, cfg.n_kv_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, self.h * self.dh, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.kv * self.dh, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.kv * self.dh, bias=False)
        self.out_proj = nn.Linear(self.h * self.dh, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        cos, sin = rope_cache(cfg.max_seq_len, self.dh, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)    # (B, H,  T, dh)
        k = self.k_proj(x).view(B, T, self.kv, self.dh).transpose(1, 2)   # (B, KV, T, dh)
        v = self.v_proj(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        cos, sin = self.cos[:T], self.sin[:T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kv                                           # each K/V head serves `rep` q-heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(y.transpose(1, 2).reshape(B, T, self.h * self.dh))


class SwiGLU(nn.Module):
    """W2( SiLU(W1 x) * W3 x )."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-norm block: x + Attn(RMSNorm(x)), then x + FFN(RMSNorm(x)). The FFN is passed in
    (dense SwiGLU for Llama, Mixture-of-Experts for MoE)."""
    def __init__(self, cfg: ModernConfig, ffn: nn.Module):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = GQAttention(cfg)
        self.ffn = ffn
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.ffn(self.norm2(x)))


class DecoderBackbone(nn.Module):
    """Embedding -> n_layers x Block -> RMSNorm -> tied output head. No absolute PE (RoPE in attention)."""
    def __init__(self, cfg: ModernConfig, make_ffn):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, make_ffn()) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                          # weight tying

    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm_f(x))