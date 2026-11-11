"""Architecture 1: decoder-only Transformer (multi-head attention, sinusoidal positions), from my Assignment 3."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0, attention_scale=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attention_scale = (1.0 / math.sqrt(self.d_head)) if attention_scale is None else float(attention_scale)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mat = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool().view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mat, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn_score = self.attention_scale * (q @ k.transpose(2, 3))
        attn_score = attn_score.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        attn_score_probability = F.softmax(attn_score, dim=-1)
        attn_score_probability = self.attn_dropout(attn_score_probability)
        out = attn_score_probability @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        net_out = self.resid_dropout(out)
        return net_out


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"
    attention_scale: float = None
    norm_first: bool = False


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Return a (max_len, d_model) tensor of the classic Transformer sinusoidal PE."""
    mat = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    mat[:, 0::2] = torch.sin(pos * div)
    mat[:, 1::2] = torch.cos(pos * div)
    return mat


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.dropout, attention_scale=cfg.attention_scale
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm_first = cfg.norm_first

    def forward(self, x):
        if self.norm_first:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ffn(x))
        return x


class DecoderLM(nn.Module):
    """The model class the autograder imports directly."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if cfg.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
            self.register_buffer("pos_emb", pe, persistent=False)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) LongTensor of token ids -> logits (B, T, vocab_size)."""
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.cfg.pos_emb_type == "sinusoidal":
            x = x + self.pos_emb[:T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)
