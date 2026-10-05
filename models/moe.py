"""Architecture #3: sparse Mixture-of-Experts Transformer (Mixtral-style, Jiang et al. 2024).

Same backbone as Llama (RMSNorm + RoPE + GQA) but every feed-forward is a Mixture-of-Experts:
a router sends each token to its top-k of n_experts SwiGLU experts (Shazeer et al. 2017; GShard 2020).
Load-balancing auxiliary loss from Switch Transformer (Fedus et al. 2021).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import ModernConfig, DecoderBackbone, SwiGLU


class MoE(nn.Module):
    """Sparse Mixture-of-Experts: a router sends each token to its top-k SwiGLU experts."""
    def __init__(self, cfg: ModernConfig):
        super().__init__()
        self.E, self.k = cfg.n_experts, cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts)])
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)                                             # (N, C) tokens
        probs = self.router(xf).float().softmax(-1)                       # (N, E)
        top_w, top_i = probs.topk(self.k, dim=-1)                         # (N, k)
        top_w = top_w / top_w.sum(-1, keepdim=True)                       # renormalise over chosen experts
        out = torch.zeros(xf.shape, dtype=torch.float32, device=x.device)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)               # tokens routed to expert e
            if tok.numel():
                y = expert(xf[tok]).float() * top_w[tok, slot, None]
                out.index_add_(0, tok, y)
        # Switch-style load balancing: E * sum_e (fraction of tokens to e) * (mean router prob of e)
        frac = F.one_hot(top_i, self.E).float().sum((0, 1)) / (xf.size(0) * self.k)
        self.aux_loss = self.E * (frac * probs.mean(0)).sum()
        return out.to(x.dtype).view(B, T, C)


class MoELM(DecoderBackbone):
    def __init__(self, cfg: ModernConfig):
        super().__init__(cfg, make_ffn=lambda: MoE(cfg))

    def compute_loss(self, x, y):
        logits = self(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        if self.training:                                               # val loss = pure cross-entropy
            aux = sum(b.ffn.aux_loss for b in self.blocks) / len(self.blocks)
            loss = loss + self.cfg.aux_loss_coef * aux
        return loss

    def active_params(self):
        """Parameters used per token (only top_k of n_experts experts)."""
        total = sum(p.numel() for p in self.parameters())
        expert = sum(p.numel() for b in self.blocks for p in b.ffn.experts.parameters())
        return total - expert + expert * self.cfg.top_k // self.cfg.n_experts