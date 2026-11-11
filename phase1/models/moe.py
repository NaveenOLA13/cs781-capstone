"""Architecture 3: sparse mixture-of-experts Transformer (Mixtral-style, Jiang et al. 2024)."""

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
        xf = x.reshape(-1, C)
        probs = self.router(xf).float().softmax(-1)
        top_w, top_i = probs.topk(self.k, dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)
        out = torch.zeros(xf.shape, dtype=torch.float32, device=x.device)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            if tok.numel():
                y = expert(xf[tok]).float() * top_w[tok, slot, None]
                out.index_add_(0, tok, y)
        frac = F.one_hot(top_i, self.E).float().sum((0, 1)) / (xf.size(0) * self.k)
        self.aux_loss = self.E * (frac * probs.mean(0)).sum()
        return out.to(x.dtype).view(B, T, C)


class MoELM(DecoderBackbone):
    def __init__(self, cfg: ModernConfig):
        super().__init__(cfg, make_ffn=lambda: MoE(cfg))

    def compute_loss(self, x, y):
        logits = self(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        if self.training:
            aux = sum(b.ffn.aux_loss for b in self.blocks) / len(self.blocks)
            loss = loss + self.cfg.aux_loss_coef * aux
        return loss
