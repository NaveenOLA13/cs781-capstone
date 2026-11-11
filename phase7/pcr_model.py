"""Phase 7: a Phase 1 model as a case encoder, similarity of two cases from their chunk vectors, ranking losses,
knowledge-distillation losses, the half-size student."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import p7_paths  # noqa: F401
import cj_tech as T  # noqa: E402
import cj_train as R  # noqa: E402

AGGS = ("mean", "maxsim", "maxsim_sym", "max", "first")
LOSSES = ("infonce", "listwise", "pairwise")
KD_LOSSES = ("listwise", "margin_mse")


# --- chunks
def chunk_starts(n: int, L: int, max_chunks: int) -> list[int]:
    """Start positions of chunks of exactly L tokens (a case of at most L tokens is one chunk)."""
    if n <= L:
        return [0]
    c = min(math.ceil(n / L), max(1, max_chunks))
    if c == 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n - L, c)})


def doc_chunks(tokens: np.ndarray, L: int, max_chunks: int) -> list[np.ndarray]:
    return [tokens[s : s + L] for s in chunk_starts(len(tokens), L, max_chunks)]


def _groups(lengths: list[int], batch: int, exact: bool) -> list[list[int]]:
    """Index groups of at most `batch` items; with exact=True only items of one length per group (no padding)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) == batch or (exact and lengths[i] != lengths[cur[0]])):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def _collate(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.as_tensor(np.asarray(s, dtype=np.int64))
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


# --- encoding
def embed(clf, ids, mask) -> torch.Tensor:
    """Chunk vectors of length 1 (float32)."""
    return F.normalize(clf(ids, mask).float(), dim=-1)


def embed_with_grad(
    clf, chunks: list[np.ndarray], pad_id: int, device, batch: int, exact: bool, amp: bool, checkpointing: bool = True
) -> torch.Tensor:
    """[N, dim] for training."""
    from torch.utils.checkpoint import checkpoint

    out: list = [None] * len(chunks)
    for g in _groups([len(c) for c in chunks], batch, exact):
        ids, mask = _collate([chunks[i] for i in g], pad_id, device)
        with R._autocast(device, amp):
            if checkpointing and torch.is_grad_enabled():
                v = checkpoint(embed, clf, ids, mask, use_reentrant=False)
            else:
                v = embed(clf, ids, mask)
        for k, i in enumerate(g):
            out[i] = v[k]
    return torch.stack(out)


@torch.no_grad()
def embed_all(clf, chunks: list[np.ndarray], pad_id: int, device, batch: int, exact: bool, amp: bool) -> torch.Tensor:
    """[N, dim] float16 on the CPU, model in eval mode (dropout off)."""
    was = clf.training
    clf.eval()
    out = torch.zeros(len(chunks), clf.n_labels, dtype=torch.float16)
    for g in _groups([len(c) for c in chunks], batch, exact):
        ids, mask = _collate([chunks[i] for i in g], pad_id, device)
        with R._autocast(device, amp):
            v = embed(clf, ids, mask)
        out[g] = v.to("cpu", torch.float16)
    clf.train(was)
    return out


def encode_cases(
    clf, cases: list[list[np.ndarray]], pad_id: int, device, batch: int, exact: bool, amp: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every case's chunk vectors, padded: ([n_cases, C, dim] float16 with zeros for missing chunks, [n_cases] chunk
    counts)."""
    flat = [c for cs in cases for c in cs]
    V = embed_all(clf, flat, pad_id, device, batch, exact, amp)
    counts = torch.tensor([len(cs) for cs in cases])
    C = int(counts.max()) if len(cases) else 1
    out = torch.zeros(len(cases), C, V.size(-1), dtype=torch.float16)
    i = 0
    for k, n in enumerate(counts.tolist()):
        out[k, :n] = V[i : i + n]
        i += n
    return out, counts


def mean_vectors(V: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    m = (torch.arange(V.size(1))[None, :] < counts[:, None]).float()
    return F.normalize((V.float() * m[..., None]).sum(1), dim=-1)


@torch.no_grad()
def aggregate(Qv, Qn, Pv, Pn, aggs, device, block: int = 128) -> dict:
    """Case similarities {aggregation: [n_queries, n_candidates] float32 numpy} from padded chunk vectors."""
    aggs = list(aggs)
    Q = Qv.to(device).float()
    qn = Qn.to(device)
    qm = torch.arange(Q.size(1), device=device)[None, :] < qn[:, None]
    qmean = F.normalize((Q * qm[..., None]).sum(1), dim=-1)
    out = {a: np.zeros((Q.size(0), Pv.size(0)), dtype=np.float32) for a in aggs}
    need_pairs = any(a in aggs for a in ("maxsim", "maxsim_sym", "max"))
    neg = torch.tensor(-1e4, device=device)
    for s in range(0, Pv.size(0), block):
        P = Pv[s : s + block].to(device).float()
        pn = Pn[s : s + block].to(device)
        pm = torch.arange(P.size(1), device=device)[None, :] < pn[:, None]
        e = s + P.size(0)
        if "mean" in aggs:
            pmean = F.normalize((P * pm[..., None]).sum(1), dim=-1)
            out["mean"][:, s:e] = (qmean @ pmean.T).cpu().numpy()
        if "first" in aggs:
            out["first"][:, s:e] = (Q[:, 0] @ P[:, 0].T).cpu().numpy()
        if need_pairs:
            S = torch.einsum("qid,pjd->qpij", Q, P)
            S = torch.where(pm[None, :, None, :], S, neg)
            best_j = S.max(-1).values
            q2p = (best_j * qm[:, None, :]).sum(-1) / qn[:, None].clamp(min=1)
            if "maxsim" in aggs:
                out["maxsim"][:, s:e] = q2p.cpu().numpy()
            if "max" in aggs:
                out["max"][:, s:e] = torch.where(qm[:, None, :], best_j, neg).max(-1).values.cpu().numpy()
            if "maxsim_sym" in aggs:
                best_i = torch.where(qm[:, None, :, None], S, neg).max(-2).values
                p2q = (best_i * pm[None]).sum(-1) / pn[None, :].clamp(min=1)
                out["maxsim_sym"][:, s:e] = ((q2p + p2q) / 2).cpu().numpy()
    return out


# --- losses
def ranking_loss(
    S: torch.Tensor,
    target: torch.Tensor,
    rel: torch.Tensor,
    kind: str = "infonce",
    tau: float = 0.05,
    margin: float = 0.2,
) -> torch.Tensor:
    """S [B, N] cosines of each query with every candidate case in the batch (its own positive and hard negatives and
    the other queries' ones = in-batch negatives); target [B] the column of each row's positive."""
    if kind not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}")
    B = S.size(0)
    rows = torch.arange(B, device=S.device)
    is_t = torch.zeros_like(rel)
    is_t[rows, target] = True
    if kind == "infonce":
        logits = (S / tau).masked_fill(rel & ~is_t, float("-inf"))
        return F.cross_entropy(logits, target)
    if kind == "listwise":
        logits = S / tau
        pos = logits.masked_fill(~rel, float("-inf")).logsumexp(-1)
        return (logits.logsumexp(-1) - pos).mean()
    pos = S[rows, target][:, None]
    neg = ~rel
    h = F.relu(margin - (pos - S)) * neg
    return h.sum() / neg.sum().clamp(min=1)


def kd_loss(
    S: torch.Tensor,
    Tt: torch.Tensor,
    kind: str,
    tau: float,
    tau_t: float,
    kd_T: float,
    target: torch.Tensor,
    neg_cols: torch.Tensor | None = None,
) -> torch.Tensor:
    """Teacher signal on the same batch."""
    if kind not in KD_LOSSES:
        raise ValueError(f"kd_loss must be one of {KD_LOSSES}")
    if kind == "listwise":
        lp = F.log_softmax(S / (tau * kd_T), -1)
        pt = F.softmax(Tt / (tau_t * kd_T), -1)
        return (pt * (torch.log(pt.clamp_min(1e-12)) - lp)).sum(-1).mean() * kd_T**2
    rows = torch.arange(S.size(0), device=S.device)
    sm = (S[rows, target][:, None] - S) / tau
    tm = (Tt[rows, target][:, None] - Tt) / tau_t
    m = neg_cols if neg_cols is not None else torch.ones_like(S, dtype=torch.bool)
    return ((sm - tm) ** 2 * m).sum() / m.sum().clamp(min=1)


# --- techniques, student
def used_params(clf, probe: torch.Tensor, device) -> set[str]:
    """Names of the Phase 1 parameters that get a gradient when chunk vectors are computed (for the encoder-decoder:
    the encoder only)."""
    m = clf.encoder.model
    flags = {n: p.requires_grad for n, p in clf.named_parameters()}
    for p in clf.parameters():
        p.requires_grad_(True)
    clf.train()
    clf.zero_grad(set_to_none=True)
    ids = probe.to(device)
    clf(ids, torch.ones_like(ids)).float().pow(2).sum().backward()
    used = {n for n, p in m.named_parameters() if p.grad is not None}
    clf.zero_grad(set_to_none=True)
    for n, p in clf.named_parameters():
        p.requires_grad_(flags[n])
    return used


def _spread(n: int, k: int) -> list[int]:
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def _cut_modulelist(model: nn.Module, name: str, share) -> dict:
    blocks = model.get_submodule(name)
    n = len(blocks)
    k = int(share) if float(share) >= 1 else max(1, int(round(n * float(share))))
    if not 1 <= k < n:
        raise ValueError(f"student_layers must leave between 1 and {n - 1} of the {n} blocks (got {k})")
    idx = _spread(n, k)
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, nn.ModuleList([blocks[i] for i in idx]))
    changed = R._shrink_layer_count(model, n, len(idx))
    if parent is not model:
        changed += [f"{parent_name}:{c}" for c in R._shrink_layer_count(parent, n, len(idx))]
    return dict(
        kind="blocks", blocks=name, layers_full=n, layers_student=len(idx), kept=idx, layer_count_fields=changed
    )


def _cut_rnn(model: nn.Module, name: str, share) -> dict:
    r = model.get_submodule(name)
    n = r.num_layers
    k = int(share) if float(share) >= 1 else max(1, int(round(n * float(share))))
    if not 1 <= k < n:
        raise ValueError(f"the RNN '{name}' has {n} layer(s): no smaller student possible")
    idx = _spread(n, k)
    kw = dict(
        input_size=r.input_size,
        hidden_size=r.hidden_size,
        num_layers=len(idx),
        bias=r.bias,
        batch_first=r.batch_first,
        dropout=r.dropout if len(idx) > 1 else 0.0,
        bidirectional=r.bidirectional,
    )
    new = type(r)(**kw)
    p0 = next(r.parameters())
    new = new.to(device=p0.device, dtype=p0.dtype)
    sfx = ("", "_reverse") if r.bidirectional else ("",)
    with torch.no_grad():
        for i, j in enumerate(idx):
            for s in sfx:
                for w in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    if hasattr(r, f"{w}_l{j}{s}"):
                        getattr(new, f"{w}_l{i}{s}").copy_(getattr(r, f"{w}_l{j}{s}"))
    new.train(r.training)
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, new)
    changed = R._shrink_layer_count(parent, n, len(idx)) if parent is not model else []
    return dict(
        kind="rnn layers", blocks=name, layers_full=n, layers_student=len(idx), kept=idx, layer_count_fields=changed
    )


def make_student(model: nn.Module, share, used: set[str], log=print) -> dict:
    """KD student (shrink and fine-tune, Shleifer & Rush 2020): evenly spaced blocks (first and last kept) of the
    blocks that compute the case vectors; for an RNN encoder, evenly spaced RNN layers (the first kept)."""
    before = sum(p.numel() for p in model.parameters())
    found = T.blocks_of(model, used)
    if found:
        info = _cut_modulelist(model, found[0], share)
    else:
        rnns = [(n, m) for n, m in T.rnn_modules(model, used) if m.num_layers >= 2]
        if not rnns:
            raise ValueError("no blocks or multi-layer RNN in the part of the model that computes the case vectors")
        info = _cut_rnn(model, max(rnns, key=lambda t: sum(p.numel() for p in t[1].parameters()))[0], share)
    info.update(params_full=before, params_student=sum(p.numel() for p in model.parameters()))
    log(
        f"[student] {info['kind']} {info['kept']} of {info['layers_full']} ('{info['blocks']}'): "
        f"{info['params_student'] / 1e6:.1f}M of {info['params_full'] / 1e6:.1f}M weights"
        + (f"; layer count changed in {', '.join(info['layer_count_fields'])}" if info["layer_count_fields"] else "")
    )
    return info


def count(m) -> int:
    return sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
