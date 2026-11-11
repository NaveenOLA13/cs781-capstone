"""Parameter-efficient fine-tuning wrappers for nn.Linear: LoRA, LoRMA_+ and LoRMA_pi."""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * ((self.drop(x) @ self.A.t()) @ self.B.t())


class LoRMAPlusLinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        d = base.out_features
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h0 = self.base(x)
        return h0 + self.scale * ((self.drop(h0) @ self.A.t()) @ self.B.t())


class LoRMAPiLinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=None, dropout=0.0, init_std=0.02):
        super().__init__()
        d = base.out_features
        self.base = base
        B = torch.randn(d, r) * init_std
        B[:, 0] = 1.0
        A = torch.zeros(r, d)
        A[0, 0] = 1.0
        self.A, self.B = nn.Parameter(A), nn.Parameter(B)
        i = torch.arange(d)
        self.register_buffer("perm_idx", (i[None, :] - i[:, None]) % d, persistent=False)
        self.drop = nn.Dropout(dropout)

    def M(self):
        return torch.gather(self.B @ self.A, 1, self.perm_idx)

    def forward(self, x):
        h0 = self.base(x)
        return self.drop(h0) @ self.M().t().to(h0.dtype)


WRAPPERS = {"lora": LoRALinear, "lorma_plus": LoRMAPlusLinear, "lorma_pi": LoRMAPiLinear}

DEFAULT_TARGETS = {
    "decoder": ["q_proj", "k_proj", "v_proj", "out_proj", "net.0", "net.2"],
    "llama": ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"],
    "moe": ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"],
    "encdec": [],
}


def apply_peft(model: nn.Module, method: str, targets, r=8, alpha=16, dropout=0.0):
    """Freeze everything in `model`, then wrap every nn.Linear whose qualified name ends with one of `targets`."""
    for p in model.parameters():
        p.requires_grad = False
    Wrap = WRAPPERS[method]
    todo = []
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            full = f"{name}.{cname}" if name else cname
            if (
                isinstance(child, nn.Linear)
                and any(full.endswith(t) for t in targets)
                and not full.endswith("head")
                and "router" not in full
            ):
                todo.append((mod, cname))
    for parent, cname in todo:
        setattr(parent, cname, Wrap(getattr(parent, cname), r=r, alpha=alpha, dropout=dropout))
    return len(todo)


def count_trainable(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return tr, tot
