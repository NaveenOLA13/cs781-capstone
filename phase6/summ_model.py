"""Phase 6: a Phase 1 model as a conditional generator, fine-tuning techniques on it, KD students and teachers,
the training loss and batched beam search.

Decoder-only models (decoder-only, Llama, MoE): one stream  [eos] window " Summary:" target [eos]; the loss is on the
target tokens only. Encoder-decoder: the encoder reads the window, the decoder writes [bos] target [eos].
Generation: beam search over many windows at once (all windows of a document have exactly the same number of tokens,
so no padding is needed), repeated n-grams banned, length-normalised scores (GNMT).
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import p6_paths  # noqa: F401
import cj_tech as T  # noqa: E402
import cj_train as R  # noqa: E402
from cj_mid import is_encoder_decoder  # noqa: E402
from rr_head import first_tensor  # noqa: E402

PROMPT = " Summary:"


# ------------------------------------------------------------------------ the model
class _Enc(nn.Module):
    def __init__(self, model, encdec):
        super().__init__()
        self.model = model
        self.encdec = encdec

    def forward(self, ids, mask=None):
        return self.model(ids, ids[:, :4]) if self.encdec else self.model(ids)


class Wrap(nn.Module):
    """Holds the Phase 1 model as .encoder.model (the layout cj_tech.apply expects); adapters are added as .adapters.
    Its state_dict therefore contains everything a technique trains."""

    def __init__(self, model):
        super().__init__()
        self.encoder = _Enc(model, is_encoder_decoder(model))

    @property
    def model(self):
        return self.encoder.model

    @property
    def encdec(self):
        return self.encoder.encdec


def full_logits(model, x):
    """Next-token scores for every position of x ([B, T] -> [B, T, V]); nanoGPT-style models give them only when
    targets are passed."""
    mode = getattr(model, "_p6_full", None)
    if mode != "xy":
        lg = first_tensor(model(x))
        if lg is not None and lg.dim() == 3 and lg.shape[1] == x.shape[1]:
            model._p6_full = "x"
            return lg
    lg = first_tensor(model(x, x))
    if lg is None or lg.dim() != 3 or lg.shape[1] != x.shape[1]:
        raise RuntimeError("could not get next-token scores for every position from the model")
    model._p6_full = "xy"
    return lg


def ed_logits(model, src, seq):
    """Encoder-decoder scores aligned with seq[:, 1:] (seq starts with bos). Handles models that shift the target
    themselves (their output is one shorter than their input)."""
    shift = getattr(model, "_p6_shift", None)
    if shift is None:
        lg = first_tensor(model(src, seq[:, :-1]))
        shift = 0 if lg.shape[1] == seq.shape[1] - 1 else 1
        model._p6_shift = shift
        if shift == 0:
            return lg
    lg = first_tensor(model(src, seq if shift else seq[:, :-1]))
    if lg.shape[1] != seq.shape[1] - 1:
        raise RuntimeError(f"encoder-decoder output length {lg.shape[1]} for a target of {seq.shape[1]}")
    return lg


# -------------------------------------------------------------------------- batches
def dec_batch(pairs, prompt, eos_id, pad_id, device):
    """(x, y): x = [eos] window prompt target [eos] (right-padded), y = the same with -100 outside the target;
    scores of x[:, :-1] are compared with y[:, 1:]."""
    seqs, labs = [], []
    for p in pairs:
        ctx = [eos_id] + list(p["src"]) + list(prompt)
        tgt = list(p["tgt"]) + [eos_id]
        seqs.append(ctx + tgt)
        labs.append([-100] * len(ctx) + tgt)
    n = max(map(len, seqs))
    x = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    y = torch.full((len(seqs), n), -100, dtype=torch.long)
    for i, (s, lab) in enumerate(zip(seqs, labs)):
        x[i, :len(s)] = torch.tensor(s)
        y[i, :len(lab)] = torch.tensor(lab)
    return x.to(device), y.to(device)


def ed_batch(pairs, bos_id, eos_id, pad_id, device):
    """(src, seq, y): all windows of the batch have the same length (exact batches: a bidirectional RNN changes
    with padding); seq = [bos] target [eos] right-padded; y = seq[:, 1:] with -100 on padding."""
    src = torch.tensor([list(p["src"]) for p in pairs], dtype=torch.long)
    seqs = [[bos_id] + list(p["tgt"]) + [eos_id] for p in pairs]
    n = max(map(len, seqs))
    seq = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    y = torch.full((len(seqs), n - 1), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        seq[i, :len(s)] = torch.tensor(s)
        y[i, :len(s) - 1] = torch.tensor(s[1:])
    return src.to(device), seq.to(device), y.to(device)


def batches(pairs, batch_size, exact, rng):
    """Index batches: exact=True -> only windows of the same length together; otherwise similar total lengths
    together (less padding). The batch order is shuffled."""
    if exact:
        groups: dict[int, list[int]] = {}
        for i, p in enumerate(pairs):
            groups.setdefault(len(p["src"]), []).append(i)
        out = []
        for g in groups.values():
            rng.shuffle(g)
            out += [g[k:k + batch_size] for k in range(0, len(g), batch_size)]
    else:
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        out = []
        span = batch_size * 50
        for k in range(0, len(idx), span):
            part = sorted(idx[k:k + span], key=lambda i: len(pairs[i]["src"]) + len(pairs[i]["tgt"]))
            out += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(out)
    return out


def logits_and_targets(wrap, batch, ids, device):
    """(scores [N, V], targets [N]) over the target tokens of a batch of pairs."""
    m = wrap.model
    if wrap.encdec:
        src, seq, y = ed_batch(batch, ids["bos"], ids["eos"], ids["pad"], device)
        lg = ed_logits(m, src, seq)
    else:
        x, y = dec_batch(batch, ids["prompt"], ids["eos"], ids["pad"], device)
        lg = full_logits(m, x[:, :-1])
        y = y[:, 1:]
    mask = y != -100
    return lg[mask], y[mask]


def seq_loss(wrap, batch, ids, device, label_smoothing=0.1, teacher=None, kd_alpha=0.5, kd_T=2.0):
    """(loss, gold term, teacher term or None). Teacher term = T^2 x KL(teacher || student) on the target tokens
    (word-level knowledge distillation; all our models share the bpe12k vocabulary)."""
    lg, y = logits_and_targets(wrap, batch, ids, device)
    gold = F.cross_entropy(lg.float(), y, label_smoothing=label_smoothing)
    if teacher is None:
        return gold, gold, None
    with torch.no_grad():
        tl, _ = logits_and_targets(teacher, batch, ids, device)
    kd = F.kl_div(F.log_softmax(lg.float() / kd_T, -1), F.softmax(tl.float() / kd_T, -1),
                  reduction="batchmean") * kd_T ** 2
    return (1 - kd_alpha) * gold + kd_alpha * kd, gold, kd


# ------------------------------------------------------------------- techniques, KD
def rnn_param_names(model) -> set[str]:
    out = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.RNNBase):
            out |= {f"{n}.{p}" if n else p for p, _ in m.named_parameters(recurse=False)}
    return out


def used_params(wrap, probe, ids, device) -> set[str]:
    """Names of the Phase 1 parameters that get a gradient from the summarization loss."""
    m = wrap.model
    flags = {n: p.requires_grad for n, p in m.named_parameters()}
    for p in m.parameters():
        p.requires_grad_(True)
    m.zero_grad(set_to_none=True)
    lg, y = logits_and_targets(wrap, probe, ids, device)
    F.cross_entropy(lg.float(), y).backward()
    used = {n for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    for n, p in m.named_parameters():
        p.requires_grad_(flags[n])
    return used


def apply_technique(wrap, technique, cfg, total_steps, probe, ids, device, log=print):
    """cj_tech technique on the wrapped model. Encoder-decoder: low-rank techniques only on its linear layers
    (they failed on its RNN weights in Phase 5)."""
    used = used_params(wrap, probe, ids, device)
    if wrap.encdec and technique in T.LOW_RANK and not cfg.get("lowrank_rnn"):
        rnn = rnn_param_names(wrap.model)
        used = used - rnn
        log(f"[technique] encoder-decoder: {technique} on linear layers only ({len(rnn)} RNN tensors left fixed)")
    return T.apply(wrap, technique, used, cfg, total_steps, log)


def make_student(model, share, log=print) -> dict:
    """KD student: evenly spaced blocks of the (mid-trained) model, first and last included (shrink and fine-tune,
    Shleifer & Rush 2020)."""
    info = R.cut_blocks(model, share, pick="spread")
    log(f"[student] blocks {info['kept']} of {info['layers_full']} ('{info['blocks']}'): "
        f"{info['params_student'] / 1e6:.1f}M of {info['params_full'] / 1e6:.1f}M weights")
    return info


def load_trained(bb_model, run_cfg: dict, state: dict, probe, ids, device, log=print) -> Wrap:
    """A finished run rebuilt from its Phase 1 starting model + technique + its trained tensors (for KD teachers)."""
    m = copy.deepcopy(bb_model)
    if run_cfg.get("student_layers"):
        make_student(m, run_cfg["student_layers"], log)
    w = Wrap(m).to(device).float()
    apply_technique(w, run_cfg["technique"], run_cfg, 1, probe, ids, device, log)
    missing = [k for k in state if k not in w.state_dict()]
    if missing:
        raise RuntimeError(f"teacher state has {len(missing)} unknown tensors, e.g. {missing[:3]}")
    w.load_state_dict(state, strict=False)
    w.eval()
    for p in w.parameters():
        p.requires_grad_(False)
    return w


# ------------------------------------------------------------------------ generation
class EncoderCache:
    """While generating, the encoder of an encoder-decoder runs once per batch of windows instead of once per new
    token: its output is kept while it is called with the same input tensor."""

    def __init__(self, model):
        self.mod = None
        for name in ("encoder", "enc"):
            m = getattr(model, name, None)
            if isinstance(m, nn.Module):
                self.mod = m
                break
        self.key, self.val, self.orig, self.had = None, None, None, False

    def __enter__(self):
        if self.mod is not None:
            self.had = "forward" in self.mod.__dict__        # already patched (e.g. an RNN without cuDNN)
            self.orig = self.mod.forward

            def fwd(*a, **k):
                x = a[0] if a else None
                key = (id(x), x.data_ptr(), tuple(x.shape)) if torch.is_tensor(x) and not k else None
                if key is not None and key == self.key:
                    return self.val
                out = self.orig(*a, **k)
                self.key, self.val = key, out
                return out
            self.mod.forward = fwd
        return self

    def __exit__(self, *exc):
        if self.mod is not None:
            if self.had:
                self.mod.forward = self.orig
            else:
                del self.mod.forward
        self.key = self.val = None


def _banned(rows: list[list[int]], n: int) -> list[list[int]]:
    out = []
    for toks in rows:
        if n <= 0 or len(toks) < n - 1:
            out.append([])
            continue
        key = tuple(toks[len(toks) - n + 1:])
        out.append([toks[i + n - 1] for i in range(len(toks) - n + 1) if tuple(toks[i:i + n - 1]) == key])
    return out


@torch.no_grad()
def beam_search(step, n_prob, eos_id, beams=4, max_new=128, min_new=0, no_repeat=3, len_pen=1.0, device="cpu"):
    """Beam search for n_prob independent problems at once. step(rows [n_prob*beams, t]) -> log-probabilities
    [n_prob*beams, V] of the next token (row p*beams + b belongs to problem p). max_new / min_new: at most / at least
    this many tokens before the end (an int or one per problem). Returns one token list per problem (without eos)."""
    k = beams
    mx = max_new if isinstance(max_new, list) else [max_new] * n_prob
    mn = min_new if isinstance(min_new, list) else [min_new] * n_prob
    rows = torch.zeros(n_prob * k, 0, dtype=torch.long, device=device)
    scores = torch.full((n_prob, k), float("-inf"), device=device)
    scores[:, 0] = 0.0
    done: list[list] = [[] for _ in range(n_prob)]
    active = [True] * n_prob
    for t in range(max(mx) + 1):
        lp = step(rows).float()
        V = lp.size(-1)
        for p in range(n_prob):
            if t < mn[p]:
                lp[p * k:(p + 1) * k, eos_id] = float("-inf")
            if t >= mx[p]:                       # max_new tokens written: only the end is allowed
                keep = lp[p * k:(p + 1) * k, eos_id].clone()
                lp[p * k:(p + 1) * k] = float("-inf")
                lp[p * k:(p + 1) * k, eos_id] = keep
        if no_repeat and t >= no_repeat - 1:
            for r, ban in enumerate(_banned(rows.tolist(), no_repeat)):
                if ban:
                    lp[r, ban] = float("-inf")
        cand = (scores.view(-1, 1) + lp).view(n_prob, k * V)
        top_v, top_i = torch.topk(cand, min(2 * k, k * V), dim=1)
        top_v, top_i = top_v.tolist(), top_i.tolist()
        parents, toks, new_scores = [], [], []
        for p in range(n_prob):
            alive = []
            if active[p]:
                for sc, idx in zip(top_v[p], top_i[p]):
                    if sc == float("-inf"):
                        break
                    b, tok = divmod(idx, V)
                    r = p * k + b
                    if tok == eos_id:
                        L = t + 1
                        done[p].append((sc / (((5 + L) / 6) ** len_pen), rows[r].tolist()))
                    else:
                        alive.append((sc, r, tok))
                    if len(alive) == k:
                        break
                if len(done[p]) >= k or not alive:
                    active[p] = False
            if not alive:                         # finished problem: keep shapes, nothing to extend
                alive = [(float("-inf"), p * k, eos_id)]
            while len(alive) < k:
                alive.append((float("-inf"), alive[0][1], alive[0][2]))
            for sc, r, tok in alive:
                parents.append(r)
                toks.append(tok)
                new_scores.append(sc)
        if not any(active):
            break
        rows = torch.cat([rows[torch.tensor(parents, device=device)],
                          torch.tensor(toks, device=device).view(-1, 1)], 1)
        scores = torch.tensor(new_scores, device=device).view(n_prob, k)
    out = []
    for p in range(n_prob):
        if not done[p]:                           # nothing finished: best unfinished row
            sc = scores[p].tolist()
            b = max(range(k), key=lambda i: sc[i])
            toks = rows[p * k + b].tolist()
            done[p] = [(0.0, [x for x in toks if x != eos_id])]
        out.append(max(done[p], key=lambda d: d[0])[1])
    return out


@torch.no_grad()
def sample_search(step, n_prob, eos_id, max_new=128, min_new=0, no_repeat=3, top_p=0.9, top_k=0, temperature=0.7,
                  seed=0, device="cpu"):
    """Top-p (nucleus) sampling, Holtzman et al. 2020, for n_prob problems at once (one row each): the next token is
    drawn from the smallest set of most likely tokens whose probabilities sum to top_p, after dividing the scores by
    temperature (and keeping only the top_k tokens if top_k > 0). Repeated n-grams are banned; the random generator
    is seeded, so the output is reproducible. Returns one token list per problem (without eos)."""
    mx = max_new if isinstance(max_new, list) else [max_new] * n_prob
    mn = min_new if isinstance(min_new, list) else [min_new] * n_prob
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rows = torch.zeros(n_prob, 0, dtype=torch.long, device=device)
    ended = torch.zeros(n_prob, dtype=torch.bool, device=device)
    for t in range(max(mx) + 1):
        lp = step(rows).float()
        for p in range(n_prob):
            if t < mn[p]:
                lp[p, eos_id] = float("-inf")
            if t >= mx[p] or bool(ended[p]):
                lp[p] = float("-inf")
                lp[p, eos_id] = 0.0
        if no_repeat and t >= no_repeat - 1:
            for r, ban in enumerate(_banned(rows.tolist(), no_repeat)):
                if ban and not bool(ended[r]):
                    lp[r, ban] = float("-inf")
        z = lp / max(temperature, 1e-5)
        dead = torch.isinf(z).all(-1)                       # everything banned: end here
        if bool(dead.any()):
            z[dead, eos_id] = 0.0
        if top_k and top_k < z.size(-1):
            kth = torch.topk(z, top_k, dim=-1).values[:, -1:]
            z = z.masked_fill(z < kth, float("-inf"))
        probs = F.softmax(z, -1)
        sp, si = torch.sort(probs, descending=True, dim=-1)
        drop = sp.cumsum(-1) - sp > top_p                  # keep the smallest set reaching top_p (at least one)
        sp = sp.masked_fill(drop, 0.0)
        sp = sp / sp.sum(-1, keepdim=True).clamp(min=1e-12)
        pick = torch.multinomial(sp, 1, generator=gen)
        tok = si.gather(-1, pick).view(-1)
        tok = torch.where(ended, torch.full_like(tok, eos_id), tok)
        rows = torch.cat([rows, tok.view(-1, 1)], 1)
        ended |= tok == eos_id
        if bool(ended.all()):
            break
    out = []
    for r in rows.tolist():
        out.append(r[:r.index(eos_id)] if eos_id in r else r)
    return out


class Generator:
    """Summaries of many windows: windows with the same context length are generated together. decoding="beam"
    (beam search) or "top_p" (nucleus sampling, one sample per window)."""

    def __init__(self, wrap, ids, device, amp=True, max_len=None):
        self.w, self.ids, self.device, self.amp, self.max_len = wrap, ids, device, amp, max_len

    def _step_fn(self, ctx):
        m, ids = self.w.model, self.ids
        if self.w.encdec:
            shift = getattr(m, "_p6_shift", 0)

            def step(rows):
                src = ctx if ctx.size(0) == rows.size(0) else ctx.repeat_interleave(rows.size(0) // ctx.size(0), 0)
                seq = torch.cat([torch.full((rows.size(0), 1), ids["bos"], device=self.device), rows], 1)
                if shift:
                    seq = torch.cat([seq, torch.full((rows.size(0), 1), ids["pad"], device=self.device)], 1)
                with R._autocast(self.device, self.amp):
                    lg = first_tensor(m(src, seq))
                return F.log_softmax(lg[:, -1].float(), -1)
            return step

        def step(rows):
            x = torch.cat([ctx, rows], 1)
            with R._autocast(self.device, self.amp):
                lg = first_tensor(m(x))              # nanoGPT-style models return the last position only
            return F.log_softmax(lg[:, -1].float(), -1)
        return step

    @torch.no_grad()
    def run(self, items, beams=4, no_repeat=3, len_pen=1.0, batch=8, min_frac=0.0, log=None, decoding="beam",
            top_p=0.9, top_k=0, temperature=0.7, seed=0):
        """items: [{"src": token ids, "max_new": int}] -> generated token lists (same order)."""
        if decoding not in ("beam", "top_p"):
            raise ValueError("decoding must be 'beam' or 'top_p'")
        if decoding == "top_p":
            beams, batch = 1, batch * 2                     # one row per window: twice as many windows at once
        was = self.w.training
        self.w.eval()
        ids = self.ids
        groups: dict[int, list[int]] = {}
        for i, it in enumerate(items):
            groups.setdefault(len(it["src"]), []).append(i)
        out: list = [None] * len(items)
        n_done = 0
        for _, idx in sorted(groups.items()):
            for s in range(0, len(idx), batch):
                part = idx[s:s + batch]
                mx = [int(items[i]["max_new"]) for i in part]
                if self.w.encdec:
                    ctx = torch.tensor([items[i]["src"] for i in part], device=self.device)
                    ctx = ctx.repeat_interleave(beams, 0)
                else:
                    pre = [ids["eos"]]
                    ctx = torch.tensor([pre + list(items[i]["src"]) + list(ids["prompt"]) for i in part],
                                       device=self.device).repeat_interleave(beams, 0)
                    if self.max_len:
                        mx = [min(x, self.max_len - ctx.size(1)) for x in mx]
                mn = [int(min_frac * x) for x in mx]
                with EncoderCache(self.w.model) if self.w.encdec else _null():
                    if decoding == "beam":
                        res = beam_search(self._step_fn(ctx), len(part), ids["eos"], beams, mx, mn, no_repeat,
                                          len_pen, self.device)
                    else:
                        res = sample_search(self._step_fn(ctx), len(part), ids["eos"], mx, mn, no_repeat, top_p,
                                            top_k, temperature, seed + s, self.device)
                for i, r in zip(part, res):
                    out[i] = r
                n_done += len(part)
                if log and (n_done // batch) % 10 == 0:
                    log(f"  [generate] {n_done}/{len(items)} windows")
        self.w.train(was)
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def decode_fn(tokenizer):
    """ids -> text with the bpe12k tokenizer (or any object with .decode)."""
    if hasattr(tokenizer, "decode"):
        return lambda ids: tokenizer.decode(list(ids))
    raise TypeError("tokenizer has no decode()")


def window_budget(encdec: bool, max_len: int, prompt_len: int, tgt_tokens: int, enc_window: int = 512) -> int:
    if encdec:
        return enc_window
    return max(64, max_len - 2 - prompt_len - tgt_tokens)


def count(m) -> int:
    return sum(p.numel() for p in m.parameters())


def lr_factor_fn(plan, warmup_frac):
    def f(s):
        if s < plan["warm"]:
            return (s + 1) / max(1, plan["warm"])
        return max(0.0, (plan["total"] - s) / max(1, plan["total"] - plan["warm"]))
    return f


def math_ok(x) -> bool:
    return x is not None and math.isfinite(float(x))
