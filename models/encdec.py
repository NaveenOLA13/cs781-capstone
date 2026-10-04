"""Architecture #2: recurrent encoder-decoder with attention (from my A3 Seq2Seq-Attention code).

Same design as A3 (BiGRU encoder + tanh bridge -> decoder h0, attention over encoder states,
logits from [decoder state ; context]), adapted to language modelling:

  * one shared embedding (source and target are the same language here)
  * nn.GRU (cuDNN) instead of the Python-loop GRUCell -> same GRU equations, ~50x faster
  * Luong-style decoder: GRU runs over the whole target in parallel (teacher forcing = 1.0),
    attention is applied on top of its outputs (A3 fed the context INTO the cell, which forces
    a Python loop over every target step -> far too slow for 512-token targets)
  * dot attention is scaled by 1/sqrt(d) (2048-dim dot products saturate softmax otherwise)

Training split of each 1024-token window (see compute_loss):
  encoder <- tokens[0 : src_len]          decoder input  <- tokens[src_len-1 : 1023]
                                          decoder target <- tokens[src_len : 1024]
At test time: encoder <- last src_len prompt tokens, decoder generates the continuation.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncDecConfig:
    vocab_size: int
    emb_dim: int = 512
    hidden_size: int = 1024
    enc_layers: int = 2
    dec_layers: int = 2
    attention: str = "dot"        # "dot" (Luong, scaled) | "bahdanau" (additive; memory heavy)
    attn_dim: int = 256           # bahdanau only
    dropout: float = 0.0
    src_len: int = 512            # encoder part of each training window
    max_seq_len: int = 1024       # training window length (used by generate.py)


class Encoder(nn.Module):
    """2-layer bidirectional GRU + bridge (A3 Encoder)."""
    def __init__(self, embedding, emb_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.bridge = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src):
        outputs, h_n = self.rnn(self.embedding(src))               # [B,S,2H], [2L,B,H]
        last = torch.cat([h_n[-2], h_n[-1]], dim=-1)               # top layer: forward ; backward
        return outputs, torch.tanh(self.bridge(last))              # dec_h0: [B,H]


class DotProductAttention(nn.Module):
    """Luong multiplicative attention, batched over all decoder steps."""
    def __init__(self, dec_hidden, enc_hidden):
        super().__init__()
        self.proj = nn.Linear(dec_hidden, enc_hidden)
        self.scale = 1.0 / math.sqrt(enc_hidden)

    def forward(self, q, enc_outputs):                             # q: [B,T,Hd], enc: [B,S,He]
        scores = (self.proj(q) @ enc_outputs.transpose(1, 2)) * self.scale      # [B,T,S]
        w = scores.float().softmax(-1).to(enc_outputs.dtype)
        return w @ enc_outputs, w                                  # context [B,T,He]


class BahdanauAttention(nn.Module):
    """Additive attention v^T tanh(W q + U h). Builds a [B,T,S,A] tensor -> use small batches."""
    def __init__(self, dec_hidden, enc_hidden, attn_dim):
        super().__init__()
        self.W_a = nn.Linear(dec_hidden, attn_dim)
        self.U_a = nn.Linear(enc_hidden, attn_dim)
        self.v_a = nn.Linear(attn_dim, 1)

    def forward(self, q, enc_outputs):
        e = self.v_a(torch.tanh(self.W_a(q).unsqueeze(2) + self.U_a(enc_outputs).unsqueeze(1)))
        w = e.squeeze(-1).float().softmax(-1).to(enc_outputs.dtype)
        return w @ enc_outputs, w


class Decoder(nn.Module):
    """Unidirectional GRU decoder; logits = out([s_t ; c_t]) (A3 input-feeding output layer)."""
    def __init__(self, embedding, emb_dim, hidden_size, enc_hidden, num_layers, attention,
                 dropout, vocab_size):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.attention = attention
        self.out = nn.Linear(hidden_size + enc_hidden, vocab_size)

    def forward(self, tgt_in, h, enc_outputs):
        s, h = self.rnn(self.embedding(tgt_in), h)                 # [B,T,H]
        c, _ = self.attention(s, enc_outputs)                      # [B,T,2H]
        return self.out(torch.cat([s, c], dim=-1)), h


class EncDecLM(nn.Module):
    def __init__(self, cfg: EncDecConfig):
        super().__init__()
        self.cfg = cfg
        H, enc_h = cfg.hidden_size, 2 * cfg.hidden_size
        emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.encoder = Encoder(emb, cfg.emb_dim, H, cfg.enc_layers, cfg.dropout)
        attn = (DotProductAttention(H, enc_h) if cfg.attention == "dot"
                else BahdanauAttention(H, enc_h, cfg.attn_dim))
        self.decoder = Decoder(emb, cfg.emb_dim, H, enc_h, cfg.dec_layers, attn, cfg.dropout,
                               cfg.vocab_size)

    # --- used by generate.py (encode once, then one decoder step per new token) ---
    def encode(self, src):
        enc_out, h0 = self.encoder(src)
        h = h0.unsqueeze(0).repeat(self.cfg.dec_layers, 1, 1).contiguous()   # A3: same h0 per layer
        return enc_out, h

    def decode(self, tgt_in, h, enc_out):
        return self.decoder(tgt_in, h, enc_out)

    def forward(self, src, tgt_in):
        enc_out, h = self.encode(src)
        logits, _ = self.decode(tgt_in, h, enc_out)
        return logits

    # --- used by train.py ---
    def compute_loss(self, x, y):
        """x, y: [B,T] window and its shift-by-one. Loss only on the decoder part."""
        S = self.cfg.src_len
        logits = self(x[:, :S], x[:, S - 1:])                      # last src token starts decoding
        tgt = y[:, S - 1:]
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))