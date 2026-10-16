"""Phase 3: sentence-level classification head on top of a Phase 1 model.

Works with any of our Phase 1 architectures without editing their code:
  * decoders (GPT-style, Llama-style, MoE): captures the INPUT of the LM-head Linear,
    i.e. the final hidden states [B, T, d], and stops the forward pass there, so the
    12k-vocabulary logits are never computed.
  * encoder-decoder: captures the OUTPUT of a named submodule (e.g. "encoder"), or uses
    a `call` that returns the encoder states directly (mode="output").
  * if a model defines `hidden_states(idx)`, that method is used directly.
rr_backbone.make_backbone() picks the right option automatically.

Usage
-----
    clf = SentenceClassifier(backbone, n_labels=13, pooling="last")   # decoders
    clf = SentenceClassifier(backbone, n_labels=13, pooling="mean",
                             module_name="encoder", mode="module_out")  # enc-dec
    logits = clf(ids, mask)          # ids, mask: LongTensor [B, T]; mask=1 for real tokens
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn


class _StopForward(Exception):
    pass


# ----------------------------------------------------------------------------- helpers
def first_tensor(out):
    """The hidden-state tensor inside whatever a module returns (tensor, tuple, list, dict)."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for key in ("last_hidden_state", "hidden_states", "memory", "encoder_out", "enc_out", "output"):
            if key in out and torch.is_tensor(out[key]):
                return out[key]
        out = list(out.values())
    if isinstance(out, (tuple, list)):
        for o in out:
            t = first_tensor(o)
            if t is not None:
                return t
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return None



def find_lm_head(model: nn.Module) -> str | None:
    """Name of the Linear layer that maps hidden states to the vocabulary.
    1) a Linear whose weight is tied to a token embedding, else
    2) among Linears whose output size equals an embedding table's size, preferring those
       outside the numbered blocks: the largest output (the vocabulary), then a typical name.
    (A position table can be as large as a feed-forward layer, so size alone is not enough.)"""
    embs = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    if not embs:
        return None
    lins = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    emb_weights = {id(e.weight) for e in embs}
    tied = [n for n, m in lins.items() if id(m.weight) in emb_weights]
    if tied:
        return tied[-1]
    sizes = {e.num_embeddings for e in embs}
    cands = [n for n, m in lins.items() if m.out_features in sizes]
    outside = [n for n in cands if not any(part.isdigit() for part in n.split("."))]
    pool = outside or cands
    if not pool:
        return None
    order = list(lins)
    pref = ("lm_head", "head", "output", "out", "generator", "logits", "fc_out", "proj_out")
    return max(pool, key=lambda n: (lins[n].out_features, n.split(".")[-1] in pref, order.index(n)))


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    seen, total = set(), 0
    for p in module.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue  # tied weights are counted once
        seen.add(id(p))
        total += p.numel()
    return total


def freeze_backbone(clf: "SentenceClassifier") -> None:
    """Freeze every Phase 1 weight; pooling, position table and head stay trainable."""
    for p in clf.encoder.parameters():
        p.requires_grad_(False)
    for p in clf.head_parameters():
        p.requires_grad_(True)


def load_phase1(model: nn.Module, path: str, min_loaded: float = 0.95) -> dict:
    """Load Phase 1 weights; fails loudly if most keys did not match."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck
    if isinstance(ck, dict):
        for key in ("model", "model_state_dict", "state_dict", "model_state"):
            if isinstance(ck.get(key), dict):
                sd = ck[key]
                break
    sd = {re.sub(r"^(module\.|_orig_mod\.)+", "", k): v for k, v in sd.items()}
    own = model.state_dict()
    matched = [k for k in own if k in sd and tuple(sd[k].shape) == tuple(own[k].shape)]
    model.load_state_dict({k: sd[k] for k in matched}, strict=False)
    frac = len(matched) / max(len(own), 1)
    print(f"[load_phase1] loaded {len(matched)}/{len(own)} tensors ({frac:.1%}) from {path}")
    missing = [k for k in own if k not in matched]
    if missing:
        print("[load_phase1] not loaded (first 10):", missing[:10])
    if frac < min_loaded:
        raise RuntimeError(f"only {frac:.1%} of the model's weights were found in the checkpoint; "
                           "check the architecture/config matches the checkpoint")
    return ck


# ------------------------------------------------------------------- hidden states
class HiddenStates(nn.Module):
    """Runs a Phase 1 model and returns hidden states [B, T, d] instead of LM logits.

    mode="pre_head":   input of the LM-head Linear (auto-detected, or module_name)
    mode="module_out": output of `module_name` (first element if it returns a tuple)
    mode="output":     `call` itself returns the hidden states (e.g. m.encoder(ids))
    call(model, ids, mask): how to run the model; default model(ids)
    """

    MODES = ("pre_head", "module_out", "output")

    def __init__(self, model: nn.Module, module_name: str | None = None, mode: str = "pre_head",
                 call=None):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.model = model
        self.mode = mode
        self.call = call or (lambda m, ids, mask: m(ids))
        self.target_name = None
        self._target = []
        self.use_method = (callable(getattr(model, "hidden_states", None)) and module_name is None
                           and mode != "output" and call is None)
        if not self.use_method and mode != "output":
            name = module_name or (find_lm_head(model) if mode == "pre_head" else None)
            if name is None:
                raise ValueError("could not find the module to read hidden states from; "
                                 "pass module_name=... explicitly (e.g. 'lm_head' or 'encoder')")
            self.target_name = name
            # kept in a list so it is not registered a second time as a submodule
            self._target = [dict(model.named_modules())[name]]

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_method:
            h = first_tensor(self.model.hidden_states(ids))
        elif self.mode == "output":
            h = first_tensor(self.call(self.model, ids, mask))
            if h is None:
                raise RuntimeError("the call returned no tensor")
        else:
            box = {}
            if self.mode == "pre_head":
                def hook(mod, args):
                    box["h"] = args[0]
                    raise _StopForward
                handle = self._target[0].register_forward_pre_hook(hook)
            else:
                def hook(mod, args, out):
                    box["h"] = first_tensor(out)
                    raise _StopForward
                handle = self._target[0].register_forward_hook(hook)
            try:
                self.call(self.model, ids, mask)
            except _StopForward:
                pass
            finally:
                handle.remove()
            if "h" not in box:
                raise RuntimeError(f"module '{self.target_name}' was never called during forward")
            h = box["h"]
            if h is None:
                raise RuntimeError(f"module '{self.target_name}' returned no tensor")
        if h.dim() != 3 or h.shape[:2] != ids.shape[:2]:
            raise RuntimeError(
                f"hidden states have shape {tuple(h.shape)} but input is {tuple(ids.shape)}. "
                "The model probably applies its LM head only to the last position at inference "
                "(nanoGPT style). Add a method `hidden_states(self, idx)` to the model that returns "
                "the final normed hidden states for every position.")
        return h


# ------------------------------------------------------------------------- pooling
class Pooler(nn.Module):
    """Token vectors [B, T, d] -> one sentence vector [B, d]. Handles left or right padding."""

    KINDS = ("last", "first", "mean", "max", "attn")

    def __init__(self, d_model: int, kind: str = "last"):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"pooling must be one of {self.KINDS}")
        self.kind = kind
        if kind == "attn":
            self.score = nn.Linear(d_model, 1)
            nn.init.zeros_(self.score.weight)   # starts as a plain mean
            nn.init.zeros_(self.score.bias)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.bool()
        B, T, _ = h.shape
        rows = torch.arange(B, device=h.device)
        pos = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
        if self.kind == "last":
            idx = torch.where(m, pos, torch.full_like(pos, -1)).max(1).values.clamp(min=0)
            return h[rows, idx]
        if self.kind == "first":
            idx = torch.where(m, pos, torch.full_like(pos, T)).min(1).values.clamp(max=T - 1)
            return h[rows, idx]
        if self.kind == "mean":
            mf = m.unsqueeze(-1).to(h.dtype)
            return (h * mf).sum(1) / mf.sum(1).clamp(min=1.0)
        if self.kind == "max":
            s = h.masked_fill(~m.unsqueeze(-1), float("-inf")).max(1).values
            return torch.nan_to_num(s, neginf=0.0)
        a = self.score(h).squeeze(-1).float().masked_fill(~m, float("-inf"))
        w = torch.nan_to_num(torch.softmax(a, dim=1))
        return (w.unsqueeze(-1).to(h.dtype) * h).sum(1)


# ---------------------------------------------------------------------- classifier
class SentenceClassifier(nn.Module):
    """Phase 1 backbone -> hidden states -> pooling -> dropout -> head -> role scores.

    forward(ids, mask, pool_mask=None, pos=None)
      mask:      1 for every real token (attention / padding)
      pool_mask: 1 for the tokens to pool over (e.g. only the target sentence when
                 neighbouring sentences are given as context); defaults to mask
      pos:       LongTensor [B] of position buckets (sentence position in the document),
                 used only if pos_buckets > 0
    """

    def __init__(self, backbone: nn.Module, n_labels: int, pooling: str = "last",
                 head: str = "linear", dropout: float = 0.1, d_model: int | None = None,
                 module_name: str | None = None, mode: str = "pre_head", call=None,
                 pos_buckets: int = 0, pos_dim: int = 32):
        super().__init__()
        self.encoder = HiddenStates(backbone, module_name=module_name, mode=mode, call=call)
        self.n_labels = n_labels
        self.n_prompt = 0           # set by prompt tuning (peft_methods.add_prompt_tuning)
        self.prompt_fill_id = 1
        if d_model is None:
            d_model = self._infer_width(backbone)
        self.d_model = d_model
        self.pool = Pooler(d_model, pooling)
        self.pos_emb = nn.Embedding(pos_buckets, pos_dim) if pos_buckets > 0 else None
        self.feat_dim = d_model + (pos_dim if pos_buckets > 0 else 0)
        self.drop = nn.Dropout(dropout)
        if head == "linear":
            self.head = nn.Linear(self.feat_dim, n_labels)
        elif head == "mlp":
            self.head = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(self.feat_dim, n_labels))
        else:
            raise ValueError("head must be 'linear' or 'mlp'")

    @torch.no_grad()
    def _infer_width(self, backbone: nn.Module) -> int:
        dev = next(backbone.parameters()).device
        was_training = backbone.training
        backbone.eval()
        ids = torch.tensor([[1, 2, 3, 4]], device=dev)
        h = self.encoder(ids, torch.ones_like(ids))
        backbone.train(was_training)
        return h.size(-1)

    def head_parameters(self):
        """Parameters that are always trained, whatever the fine-tuning method."""
        mods = [self.pool, self.head] + ([self.pos_emb] if self.pos_emb is not None else [])
        return [p for m in mods for p in m.parameters()]

    def sentence_vectors(self, ids, mask, pool_mask=None):
        if pool_mask is None:
            pool_mask = mask
        if self.n_prompt:
            B, n = ids.size(0), self.n_prompt
            ids = torch.cat([torch.full((B, n), self.prompt_fill_id, dtype=ids.dtype, device=ids.device), ids], 1)
            mask = torch.cat([torch.ones(B, n, dtype=mask.dtype, device=mask.device), mask], 1)
            pool_mask = torch.cat([torch.zeros(B, n, dtype=pool_mask.dtype, device=pool_mask.device), pool_mask], 1)
        return self.pool(self.encoder(ids, mask), pool_mask)

    def features(self, ids, mask, pool_mask=None, pos=None):
        s = self.sentence_vectors(ids, mask, pool_mask)
        if self.pos_emb is not None:
            if pos is None:
                raise ValueError("pos (position buckets) is required when pos_buckets > 0")
            s = torch.cat([s, self.pos_emb(pos).to(s.dtype)], -1)
        return s

    def forward(self, ids, mask, pool_mask=None, pos=None):
        return self.head(self.drop(self.features(ids, mask, pool_mask, pos)))


class DocClassifier(nn.Module):
    """Document-level context: sentence vectors of a whole judgment -> BiGRU over the
    sentence sequence -> head per sentence. One document per forward call.

    The Phase 1 backbone is run over the document's sentences in chunks; with
    checkpointing=True activations are recomputed in the backward pass to save memory."""

    def __init__(self, sent_clf: SentenceClassifier, d_ctx: int = 256, dropout: float = 0.1,
                 chunk: int = 32, checkpointing: bool = True):
        super().__init__()
        self.sent = sent_clf
        self.n_labels = sent_clf.n_labels
        self.ctx = nn.GRU(sent_clf.feat_dim, d_ctx // 2, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.doc_head = nn.Linear(sent_clf.feat_dim + d_ctx, sent_clf.n_labels)
        self.chunk = chunk
        self.checkpointing = checkpointing

    def head_parameters(self):
        return self.sent.head_parameters() + list(self.ctx.parameters()) + list(self.doc_head.parameters())

    def forward(self, ids, mask, pool_mask=None, pos=None):      # ids: [S, T] for one document
        from torch.utils.checkpoint import checkpoint
        vecs = []
        for i in range(0, ids.size(0), self.chunk):
            sl = slice(i, i + self.chunk)
            args = (ids[sl], mask[sl], None if pool_mask is None else pool_mask[sl],
                    None if pos is None else pos[sl])
            if self.training and self.checkpointing and torch.is_grad_enabled():
                vecs.append(checkpoint(self.sent.features, *args, use_reentrant=False))
            else:
                vecs.append(self.sent.features(*args))
        x = torch.cat(vecs, 0).float()                            # [S, feat]
        c, _ = self.ctx(self.drop(x).unsqueeze(0))                 # [1, S, d_ctx]
        return self.doc_head(self.drop(torch.cat([x, c.squeeze(0)], -1)))


N_DISTANCE_BUCKETS = 15


def distance_buckets(S: int, device=None) -> torch.Tensor:
    """[S, S] bucket of (j - i): how far sentence j is from sentence i and in which direction.
    0 = same sentence; 1..7 = j after i by 1, 2, 3-4, 5-8, 9-16, 17-32, >32; 8..14 = j before i, same steps."""
    i = torch.arange(S, device=device)
    d = i[None, :] - i[:, None]
    a = d.abs()
    far = torch.log2((a - 1).clamp(min=1).float()).floor().long() + 2
    b = torch.where(a <= 2, a, far.clamp(max=7))
    return torch.where(d > 0, b, torch.where(d < 0, b + 7, torch.zeros_like(b)))


class SentenceAttention(nn.Module):
    """Self-attention over the sentences of one judgment, written out by hand (queries, keys, values,
    softmax). Every attention score also gets a learned number for the distance and direction between
    the two sentences (distance_buckets), so the attention knows the order of the sentences. The output
    projection starts at zero: the layer starts switched off and training turns it on where it helps."""

    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d % heads:
            raise ValueError(f"width {d} is not divisible by {heads} heads")
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
        self.dist = nn.Embedding(N_DISTANCE_BUCKETS, heads)
        nn.init.zeros_(self.dist.weight)
        self.drop = nn.Dropout(dropout)
        self.last_weights = None                     # [B, heads, S, S] of the last call (for inspection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # [B, S, d]
        B, S, d = x.shape
        q, k, v = (f(x).view(B, S, self.h, self.dk).transpose(1, 2) for f in (self.q, self.k, self.v))
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dk)       # [B, heads, S, S]
        att = att + self.dist(distance_buckets(S, x.device)).permute(2, 0, 1).unsqueeze(0)
        w = att.softmax(-1)
        self.last_weights = w.detach()
        return self.o((self.drop(w) @ v).transpose(1, 2).reshape(B, S, d))


class AttentionBlock(nn.Module):
    """x + attention(norm(x)), then x + feed-forward(norm(x))."""

    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SentenceAttention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ff(self.n2(x))


class ContextClassifier(nn.Module):
    """Stage B: the sentence features of one judgment [S, F] (from a frozen Stage A) -> role scores [S, L].
    arch="gru":      LayerNorm -> BiGRU over the sentence sequence -> [feature ; context] -> Linear
    arch="gru_attn": the same, plus sentence attention on top of the BiGRU output (context + attention);
                     it starts exactly as "gru" because the attention output starts at zero
    arch="attn":     no BiGRU: Linear -> `attn_layers` attention blocks (order only from the distance
                     numbers and the position feature) -> [feature ; context] -> Linear"""

    ARCHS = ("gru", "gru_attn", "attn")

    def __init__(self, feat_dim: int, n_labels: int, d_ctx: int = 256, dropout: float = 0.3,
                 layers: int = 1, arch: str = "gru", heads: int = 4, attn_layers: int = 2):
        super().__init__()
        if arch not in self.ARCHS:
            raise ValueError(f"arch must be one of {self.ARCHS}")
        self.arch = arch
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(dropout)
        if arch == "attn":
            width = d_ctx
            self.inp = nn.Linear(feat_dim, width)
            self.blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(attn_layers)])
        else:
            width = 2 * (d_ctx // 2)
            self.gru = nn.GRU(feat_dim, d_ctx // 2, num_layers=layers, batch_first=True, bidirectional=True,
                              dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(feat_dim + width, n_labels)
        if arch == "gru_attn":                 # created last, so the other weights start as in "gru"
            self.attn = SentenceAttention(width, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # [S, F] or [1, S, F]
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        x = self.drop(self.norm(x.float()))
        if self.arch == "attn":
            c = self.inp(x)
            for blk in self.blocks:
                c = blk(c)
        else:
            c, _ = self.gru(x)
            if self.arch == "gru_attn":
                c = c + self.attn(c)
        y = self.head(self.drop(torch.cat([x, c], -1)))
        return y.squeeze(0) if single else y


# ------------------------------------------------------------------------- checks
@torch.no_grad()
def padding_check(clf: SentenceClassifier, short_ids: list[int], long_ids: list[int],
                  pad_id: int = 0) -> float:
    """Max difference in the short sentence's scores alone vs. batched with a longer one.
    ~0 means padding does not leak into predictions. A BiGRU encoder without packed
    sequences will show a clear difference: then batch sentences of similar length."""
    was = clf.training
    clf.eval()
    dev = next(clf.parameters()).device
    a = torch.tensor([short_ids], device=dev)
    T = max(len(short_ids), len(long_ids))
    b = torch.full((2, T), pad_id, device=dev)
    b[0, :len(short_ids)] = a[0]
    b[1, :len(long_ids)] = torch.tensor(long_ids, device=dev)
    mb = (torch.arange(T, device=dev).unsqueeze(0) < torch.tensor([[len(short_ids)], [len(long_ids)]], device=dev)).long()
    pos = None if getattr(clf, "pos_emb", None) is None else torch.zeros(2, dtype=torch.long, device=dev)
    alone = clf(a, torch.ones_like(a), pos=None if pos is None else pos[:1])
    batched = clf(b, mb, pos=pos)[:1]
    clf.train(was)
    return (alone - batched).abs().max().item()
