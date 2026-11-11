"""Token-classification model = Phase 1 backbone (+ optional PEFT) + tagging head."""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "phase1")]
from models import build_model, init_weights  # noqa: E402
from phase2.peft import apply_peft, DEFAULT_TARGETS  # noqa: E402


def feat_dim(arch, mcfg):
    return 2 * mcfg["hidden_size"] if arch == "encdec" else mcfg["d_model"]


def backbone_features(lm, arch, idx):
    if arch == "decoder":
        x = lm.tok_emb(idx)
        if lm.cfg.pos_emb_type == "sinusoidal":
            x = x + lm.pos_emb[: idx.size(1)]
        for b in lm.blocks:
            x = b(x)
        return lm.ln_f(x)
    if arch in ("llama", "moe"):
        x = lm.tok_emb(idx)
        for b in lm.blocks:
            x = b(x)
        return lm.norm_f(x)
    if arch == "encdec":
        out, _ = lm.encoder(idx)
        return out
    raise ValueError(arch)


def _gqa_forward_bidir(self, x):
    """Same as models/layers.py::GQAttention.forward but WITHOUT the causal mask."""
    from models.layers import apply_rope

    B, T, _ = x.shape
    q = self.q_proj(x).view(B, T, self.h, self.dh).transpose(1, 2)
    k = self.k_proj(x).view(B, T, self.kv, self.dh).transpose(1, 2)
    v = self.v_proj(x).view(B, T, self.kv, self.dh).transpose(1, 2)
    cos, sin = self.cos[:T], self.sin[:T]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    rep = self.h // self.kv
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout if self.training else 0.0)
    return self.out_proj(y.transpose(1, 2).reshape(B, T, self.h * self.dh))


def make_bidirectional(lm, arch):
    """LLM2Vec-style (BehnamGhader et al. 2024): drop the causal mask so every token attends to the whole window;
    fine-tuning then teaches the left-to-right pretrained weights to use right context."""
    import types

    n = 0
    for b in lm.blocks:
        if arch == "decoder":
            b.attn.causal_mask.fill_(True)
        elif arch in ("llama", "moe"):
            b.attn.forward = types.MethodType(_gqa_forward_bidir, b.attn)
        else:
            raise ValueError(f"--bidir not applicable to arch={arch}")
        n += 1
    return n


class TagHead(nn.Module):
    def __init__(self, d_in, n_tags, kind="linear", rnn_hidden=256, dropout=0.1):
        super().__init__()
        self.kind = kind
        self.drop = nn.Dropout(dropout)
        if kind == "bigru":
            self.rnn = nn.GRU(d_in, rnn_hidden, batch_first=True, bidirectional=True)
            d_in = 2 * rnn_hidden
        self.out = nn.Linear(d_in, n_tags)

    def forward(self, h):
        h = self.drop(h)
        if self.kind == "bigru":
            h, _ = self.rnn(h.float())
            h = self.drop(h)
        return self.out(h)


class NERModel(nn.Module):
    def __init__(self, lm, arch, mcfg, n_tags, head="linear", echo=False, rnn_hidden=256, dropout=0.1):
        super().__init__()
        self.lm, self.arch, self.echo = lm, arch, echo
        self.head = TagHead(feat_dim(arch, mcfg), n_tags, head, rnn_hidden, dropout)

    def forward(self, idx):
        T = idx.size(1)
        if self.echo:
            h = backbone_features(self.lm, self.arch, torch.cat([idx, idx], 1))[:, T:]
        else:
            h = backbone_features(self.lm, self.arch, idx)
        return self.head(h)


def build_ner_model(args, phase1, n_tags, load_weights=True):
    """phase1: path to a Phase 1 best.pt, or a dict {"config", "vocab_size"(, "model")}."""
    ck = torch.load(phase1, map_location="cpu", weights_only=False) if isinstance(phase1, str) else phase1
    cfg, V = ck["config"], ck["vocab_size"]
    arch, mcfg = cfg["arch"], dict(cfg["model"])
    lm = build_model(arch, mcfg, V)
    if load_weights and not args.get("scratch"):
        lm.load_state_dict(ck["model"])
    elif args.get("scratch"):
        init_weights(lm, 0.02)
    del ck
    k = args.get("keep_layers") or 0
    if k and hasattr(lm, "blocks"):
        lm.blocks = lm.blocks[:k]
    if args.get("bidir"):
        print(f"[model] bidirectional attention in {make_bidirectional(lm, arch)} layers")
    method = args.get("method", "full")
    if method in ("lora", "lorma_plus", "lorma_pi"):
        targets = args.get("targets") or DEFAULT_TARGETS[arch]
        n = apply_peft(lm, method, targets, args.get("r", 8), args.get("alpha", 16), args.get("lora_dropout", 0.0))
        assert n > 0, f"{method}: no nn.Linear matched {targets} in arch={arch} (use --method full/head)"
        print(f"[model] {method}: wrapped {n} Linear layers ({targets})")
    elif method == "top":
        k = args.get("top_k", 1)
        for p in lm.parameters():
            p.requires_grad = False
        if arch == "encdec":
            L = mcfg["enc_layers"]
            keep = [f"_l{i}" for i in range(L - k, L)]
            for n, p in lm.encoder.rnn.named_parameters():
                if any(n.endswith(t) or n.endswith(t + "_reverse") for t in keep):
                    p.requires_grad = True
        else:
            for b in lm.blocks[-k:]:
                for p in b.parameters():
                    p.requires_grad = True
            for p in (lm.ln_f if arch == "decoder" else lm.norm_f).parameters():
                p.requires_grad = True
        print(f"[model] partial fine-tuning: top {k} layer(s) trainable")
    elif method == "head":
        for p in lm.parameters():
            p.requires_grad = False
    elif method != "full":
        raise ValueError(method)
    model = NERModel(
        lm,
        arch,
        mcfg,
        n_tags,
        args.get("head", "linear"),
        args.get("echo", False),
        args.get("rnn_hidden", 256),
        args.get("dropout", 0.1),
    )
    return model, cfg, V


def load_ner_checkpoint(path, device="cpu"):
    """Rebuild a fine-tuned NER model saved by train_ner.py."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    p1 = {"config": ck["phase1_config"], "vocab_size": ck["vocab_size"]}
    model, _, _ = build_ner_model(a, p1, len(ck["tags"]), load_weights=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def kd_loss(student_logits, teacher_logits, labels, T=2.0):
    """Hinton KD: T^2 * KL(teacher_T || student_T), averaged over non-ignored tokens."""
    m = labels != -100
    s = F.log_softmax(student_logits[m].float() / T, -1)
    t = F.softmax(teacher_logits[m].float() / T, -1)
    return F.kl_div(s, t, reduction="batchmean") * T * T
