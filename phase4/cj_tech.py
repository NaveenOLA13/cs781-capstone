"""Phase 4 fine-tuning techniques, all written by us (no peft library). Each one decides which weights of the
Phase 1 model are trained; the classification head (pooling + linear layer) is always trained.

  frozen    nothing of the Phase 1 model (reference: how much its features already know)
  full      every weight of the Phase 1 model
  partial   only the top third of the blocks (+ the final norm); lower blocks and embeddings stay fixed
  unfreeze  gradual unfreezing: head only at first, then block groups from the top down, embeddings last
  bitfit    only the bias values; a model without biases gets zero biases added to its linear layers
  lorma     LoRMA+ (Bihany, Patel, Modi 2025): every weight matrix becomes (I + alpha/r * B A) W0;
            W0 stays fixed, only B (d_out x r, starts at 0) and A (r x d_out) are trained
  lora      LoRA (Hu et al. 2021): every weight matrix becomes W0 + alpha/r * B A; W0 stays fixed, only
            B (d_out x r, starts at 0) and A (r x d_in) are trained
  qlora     QLoRA (Dettmers et al. 2023): LoRA on top of a 4-bit copy of the fixed weights: W0 is stored as
            4-bit NormalFloat (NF4) codes with one scale (absolute maximum) per block of 64 values, and is
            rebuilt from them at every use (W0 itself is freed); the low-rank B A is trained in full precision.
            Not done here: quantising the scales again ("double quantisation") and paged optimiser states.
  adalora   AdaLoRA (Zhang et al. 2023): W0 + alpha/r0 * P diag(lambda) Q with a total rank budget that
            shrinks during training; rank is kept where it matters most (importance scores)
  adapter   a bottleneck layer after every block, x + up(gelu(down(x))) with d -> 64 -> d, starts as identity

"Blocks" are the repeated layers of a transformer (an nn.ModuleList); for the encoder-decoder, whose encoder
is a bidirectional RNN, the RNN layers play that role. LoRMA and AdaLoRA also change RNN weight matrices
(via torch.nn.utils.parametrize; cuDNN is switched off for those RNNs so the changed weights are used).

    hooks = apply(clf, "lorma", used, cfg, total_steps, log)
    # hooks.body_params: Phase 1 / adapter parameters to optimise (head parameters are separate)
    # hooks.on_step_start(step), hooks.extra_loss(), hooks.after_step(step, stepped), hooks.summary()
"""
from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

TECHNIQUES = ("frozen", "full", "partial", "unfreeze", "bitfit", "lorma", "adalora", "adapter", "lora", "qlora")
BODY_LR = {"frozen": None, "full": 3e-5, "partial": 3e-5, "unfreeze": 3e-5, "bitfit": 5e-4,
           "lorma": 5e-4, "adalora": 5e-4, "adapter": 1e-3, "lora": 5e-4, "qlora": 5e-4}
TECH_DEFAULTS = dict(
    partial_frac=1 / 3,          # partial: share of the blocks (from the top) that are trained
    unfreeze_stages=4,           # unfreeze: number of stages (stage 0 = head only)
    lorma_r=8, lorma_alpha=8.0,  # LoRMA rank and alpha (scale alpha / r)
    adalora_r0=12, adalora_r=8, adalora_alpha=16.0,   # AdaLoRA start rank, final average rank, alpha
    adalora_tinit=0.1, adalora_tfinal=0.2, adalora_deltaT=10, adalora_orth=0.1,
    adalora_beta=0.85,
    adapter_dim=64,
    lora_r=8, lora_alpha=16.0,   # LoRA / QLoRA rank and alpha (scale alpha / r)
    qlora_block=64,              # QLoRA: values per quantisation block (one scale each)
)
LOW_RANK = ("lorma", "adalora", "lora", "qlora")       # techniques that parametrize weight matrices
_RNN_LAYER = re.compile(r"_l(\d+)(_reverse)?$")


# ------------------------------------------------------------------------ structure
def blocks_of(model: nn.Module, used: set[str]):
    """(name of the nn.ModuleList of blocks, list of (index, block)) whose parameters are used; None if none."""
    best = None
    for mname, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 2 and len({type(x) for x in m}) == 1:
            n_used = sum(1 for n, _ in m.named_parameters() if f"{mname}.{n}" in used)
            if n_used and (best is None or n_used > best[1]):
                best = (mname, n_used, m)
    if best is None:
        return None
    mname, _, ml = best
    items = [(i, b) for i, b in enumerate(ml) if any(f"{mname}.{i}.{n}" in used for n, _ in b.named_parameters())]
    return mname, items


def rnn_modules(model: nn.Module, used: set[str]):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.RNNBase) and any(f"{n}.{p}" in used for p, _ in m.named_parameters(recurse=False))]


def layer_groups(model: nn.Module, used: set[str]):
    """Used parameter names split into (bottom, [layer 0, ..., layer L-1], top): the layers are transformer
    blocks, or else RNN layers; bottom = what comes before them (embeddings), top = what comes after."""
    names = [n for n, _ in model.named_parameters() if n in used]
    if not names:
        raise RuntimeError("no parameter of the Phase 1 model gets a gradient for this task")
    layers: list[list[str]] = []
    found = blocks_of(model, used)
    kind = "blocks"
    if found:
        mname, items = found
        layers = [[n for n in names if n.startswith(f"{mname}.{i}.")] for i, _ in items]
    else:
        kind = "RNN layers"
        for rname, _ in rnn_modules(model, used):
            groups: dict[int, list[str]] = {}
            for n in names:
                if n.startswith(rname + ".") and "." not in n[len(rname) + 1:]:
                    m = _RNN_LAYER.search(n)
                    if m:
                        groups.setdefault(int(m.group(1)), []).append(n)
            layers += [groups[k] for k in sorted(groups)]
    if not layers:
        kind = "top-level parts"
        tops: dict[str, list[str]] = {}
        for n in names:
            tops.setdefault(n.split(".")[0], []).append(n)
        layers = list(tops.values())
    in_layers = {n for g in layers for n in g}
    first = min(names.index(g[0]) for g in layers)
    bottom = [n for n in names[:first] if n not in in_layers]
    top = [n for n in names[first:] if n not in in_layers]
    return bottom, layers, top, kind


def targets(model: nn.Module, used: set[str]):
    """Weight matrices for LoRMA / AdaLoRA: (module name, module, parameter name) of every used nn.Linear weight
    and every used RNN weight matrix. Weights shared by several modules (tied) are skipped."""
    owners = Counter(id(p) for m in model.modules() for p in m.parameters(recurse=False))
    out = []
    for mname, m in model.named_modules():
        pref = f"{mname}." if mname else ""
        if isinstance(m, nn.Linear) and f"{pref}weight" in used and owners[id(m.weight)] == 1:
            out.append((mname, m, "weight"))
        elif isinstance(m, nn.RNNBase):
            for pn, p in m.named_parameters(recurse=False):
                if pn.startswith("weight_") and f"{pref}{pn}" in used and p.dim() == 2:
                    out.append((mname, m, pn))
    return out


def _param_map(model):
    return dict(model.named_parameters())


# ---------------------------------------------------------------------------- hooks
class Hooks:
    """What the training loop calls; the base class does nothing extra."""

    def __init__(self, technique: str, body_params: list, info: dict | None = None):
        self.technique = technique
        self.body_params = body_params
        self.info = info or {}

    def on_step_start(self, step: int) -> None:
        pass

    def extra_loss(self):
        return None

    def after_step(self, step: int, stepped: bool) -> None:
        pass

    def summary(self) -> dict:
        return self.info


class UnfreezeHooks(Hooks):
    """Stage s of n (equal shares of the training steps): stage 0 trains the head only; stage s trains the
    top round(s * L / (n - 1)) layers (+ the parts above them); the last stage also the embeddings."""

    def __init__(self, params, bottom, layers, top, total_steps, stages):
        super().__init__("unfreeze", [params[n] for g in [bottom] + layers + [top] for n in g])
        self.params, self.bottom, self.layers, self.top = params, bottom, layers, top
        self.total = max(1, total_steps)
        self.stages = max(2, min(stages, len(layers) + 1))
        self.stage = -1
        self.info = dict(stages=self.stages, layers=len(layers), schedule=[])

    def _set(self, stage):
        k = round(stage * len(self.layers) / (self.stages - 1))
        on = set()
        if k > 0:
            on |= {n for g in self.layers[len(self.layers) - k:] for n in g} | set(self.top)
        if stage == self.stages - 1:
            on |= set(self.bottom)
        for n, p in self.params.items():
            if n in self.bottom or n in self.top or any(n in g for g in self.layers):
                p.requires_grad_(n in on)
        return k

    def on_step_start(self, step):
        stage = min(self.stages - 1, step * self.stages // self.total)
        if stage != self.stage:
            k = self._set(stage)
            self.stage = stage
            self.info["schedule"].append(dict(step=step, stage=stage, layers_trained=k,
                                              embeddings_trained=stage == self.stages - 1))


# ---------------------------------------------------------------------------- LoRMA
class LoRMA(nn.Module):
    """Parametrization W -> (I + scale * B A) W = W + scale * B (A W), scale = alpha / r."""

    def __init__(self, d_out: int, r: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, d_out))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, W):
        return W + self.scale * (self.B @ (self.A @ W))


class LoRA(nn.Module):
    """Parametrization W -> W + scale * B A, scale = alpha / r (B starts at 0: no change at the start)."""

    def __init__(self, d_out: int, d_in: int, r: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, W):
        return W + self.scale * (self.B @ self.A)


# the 16 NormalFloat-4 values (QLoRA paper, appendix E / bitsandbytes): quantiles of a standard normal, scaled to [-1, 1]
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)


def nf4_quantize(W: torch.Tensor, block: int = 64):
    """(packed codes uint8 [two 4-bit codes per byte], absmax per block float32): every block of `block` values is
    divided by its absolute maximum and each value replaced by the nearest of the 16 NF4 values."""
    flat = W.detach().float().reshape(-1)
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    absmax = blocks.abs().amax(1).clamp(min=1e-12)
    levels = torch.tensor(NF4, device=flat.device)
    codes = torch.bucketize(blocks / absmax[:, None], (levels[1:] + levels[:-1]) / 2).reshape(-1).to(torch.uint8)
    if codes.numel() % 2:
        codes = torch.cat([codes, codes.new_zeros(1)])
    return codes[0::2] | (codes[1::2] << 4), absmax


def nf4_dequantize(packed: torch.Tensor, absmax: torch.Tensor, levels: torch.Tensor, shape, block: int = 64):
    codes = torch.stack([packed & 15, packed >> 4], 1).reshape(-1).long()
    n = math.prod(shape)
    w = levels[codes[:absmax.numel() * block]].view(-1, block) * absmax[:, None]
    return w.reshape(-1)[:n].view(*shape)


class QLoRA(nn.Module):
    """Parametrization W -> dequantize(NF4 codes of W0) + scale * B A. The incoming W is not used: apply() frees
    the full-precision original after registering, so only the 4-bit codes and the block scales are kept."""

    def __init__(self, W: torch.Tensor, r: int, alpha: float, block: int = 64):
        super().__init__()
        packed, absmax = nf4_quantize(W, block)
        self.register_buffer("packed", packed)
        self.register_buffer("absmax", absmax)
        self.register_buffer("levels", torch.tensor(NF4, device=W.device))
        self.shape = tuple(W.shape)
        self.block = block
        self.A = nn.Parameter(torch.empty(r, W.shape[1], device=W.device))
        self.B = nn.Parameter(torch.zeros(W.shape[0], r, device=W.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def weight0(self):
        return nf4_dequantize(self.packed, self.absmax, self.levels, self.shape, self.block)

    def forward(self, W):
        return self.weight0().to(self.B.dtype) + self.scale * (self.B @ self.A)


# -------------------------------------------------------------------------- AdaLoRA
class AdaLoRA(nn.Module):
    """Parametrization W -> W + scale * P diag(lambda * mask) Q. lambda starts at 0 (no change at the start);
    mask (0/1 per rank) is set by AdaLoRAHooks from importance scores."""

    def __init__(self, d_out: int, d_in: int, r: int, scale: float):
        super().__init__()
        self.P = nn.Parameter(torch.randn(d_out, r) * 0.02)
        self.lam = nn.Parameter(torch.zeros(r))
        self.Q = nn.Parameter(torch.randn(r, d_in) * 0.02)
        self.register_buffer("mask", torch.ones(r))
        self.scale = scale

    def forward(self, W):
        return W + self.scale * ((self.P * (self.lam * self.mask)) @ self.Q)

    def orth(self):
        eye = torch.eye(self.lam.numel(), device=self.P.device)
        return (torch.linalg.matrix_norm(self.P.float().T @ self.P.float() - eye)
                + torch.linalg.matrix_norm(self.Q.float() @ self.Q.float().T - eye)) / 2


class AdaLoRAHooks(Hooks):
    """Importance of each rank (singular triplet) = smoothed sensitivity |w * grad| times its uncertainty, summed
    over lambda_i, the column P[:, i] and the row Q[i, :] (means). Budget (total ranks kept): r0 per matrix until
    tinit, then a cubic decrease to r per matrix, fixed for the last tfinal share of steps. Every deltaT steps the
    ranks with the lowest importance are masked out."""

    def __init__(self, mods: list[tuple[str, AdaLoRA]], total_steps: int, cfg: dict):
        super().__init__("adalora", [p for _, m in mods for p in (m.P, m.lam, m.Q)])
        self.mods = mods
        self.total = max(1, total_steps)
        self.t_init = int(cfg["adalora_tinit"] * self.total)
        self.t_final = int(cfg["adalora_tfinal"] * self.total)
        self.r0, self.r = cfg["adalora_r0"], cfg["adalora_r"]
        self.b0, self.b_end = self.r0 * len(mods), self.r * len(mods)
        self.delta = max(1, cfg["adalora_deltaT"])
        self.beta = cfg["adalora_beta"]
        self.orth_w = cfg["adalora_orth"]
        self.ipt: dict[int, torch.Tensor] = {}
        self.unc: dict[int, torch.Tensor] = {}
        self.budget = self.b0
        self.final_fixed = False

    def extra_loss(self):
        if not self.orth_w:
            return None
        return self.orth_w * torch.stack([m.orth() for _, m in self.mods]).mean()

    def schedule(self, step):
        if step <= self.t_init:
            return self.b0, False
        if step > self.total - self.t_final:
            return self.b_end, not self.final_fixed
        frac = 1 - (step - self.t_init) / max(1, self.total - self.t_final - self.t_init)
        return int((self.b0 - self.b_end) * frac ** 3 + self.b_end), step % self.delta == 0

    @torch.no_grad()
    def _score(self, p):
        if p.grad is None:
            return torch.zeros_like(p)
        s = (p * p.grad).abs().float()
        k = id(p)
        if k not in self.ipt:
            self.ipt[k], self.unc[k] = torch.zeros_like(s), torch.zeros_like(s)
        self.ipt[k].mul_(self.beta).add_((1 - self.beta) * s)
        self.unc[k].mul_(self.beta).add_((1 - self.beta) * (s - self.ipt[k]).abs())
        return self.ipt[k] * self.unc[k]

    @torch.no_grad()
    def after_step(self, step, stepped):
        if not stepped:
            return
        scores = [self._score(m.lam) + self._score(m.P).mean(0) + self._score(m.Q).mean(1) for _, m in self.mods]
        budget, do_mask = self.schedule(step)
        self.budget = budget
        if not do_mask:
            return
        flat = torch.cat(scores)
        keep = torch.zeros_like(flat)
        keep[torch.topk(flat, min(budget, flat.numel())).indices] = 1.0
        i = 0
        for (_, m), s in zip(self.mods, scores):
            m.mask.copy_(keep[i:i + s.numel()].to(m.mask.dtype))
            i += s.numel()
        if step > self.total - self.t_final:
            self.final_fixed = True

    def summary(self):
        by_type: dict[str, list[int]] = {}
        for name, m in self.mods:
            key = re.sub(r"(^|\.)\d+(\.|$)", r"\1*\2", name)
            by_type.setdefault(key, []).append(int(m.mask.sum().item()))
        return dict(budget=self.budget, start_rank=self.r0, target_rank=self.r, matrices=len(self.mods),
                    ranks_kept=sum(sum(v) for v in by_type.values()),
                    final_rank_mean_by_layer_type={k: round(sum(v) / len(v), 2) for k, v in by_type.items()})


# ------------------------------------------------------------------------- adapters
class Adapter(nn.Module):
    """x + up(gelu(down(x))); up starts at zero, so the adapter starts as the identity."""

    def __init__(self, d: int, r: int):
        super().__init__()
        self.down = nn.Linear(d, r)
        self.act = nn.GELU()
        self.up = nn.Linear(r, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


def _adapter_hook(adapter):
    def hook(mod, args, out):
        if torch.is_tensor(out):
            return adapter(out)
        if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            return (adapter(out[0]),) + tuple(out[1:])
        if isinstance(out, list) and out and torch.is_tensor(out[0]):
            return [adapter(out[0])] + list(out[1:])
        raise RuntimeError(f"adapter: cannot handle a block output of type {type(out).__name__}")
    return hook


def _first_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    return None


def _rnn_without_cudnn(rnn: nn.RNNBase) -> None:
    """Parametrized RNN weights are recomputed every call; cuDNN would copy them into its own flat buffer, so
    it is switched off for this RNN and its weight list is refreshed before every call."""
    orig = rnn.forward

    def forward(*args, **kwargs):
        with torch.backends.cudnn.flags(enabled=False):
            rnn._init_flat_weights()
            return orig(*args, **kwargs)

    rnn.forward = forward


# ---------------------------------------------------------------------------- apply
def apply(clf: nn.Module, technique: str, used: set[str], cfg: dict, total_steps: int, log=print) -> Hooks:
    """Sets up `technique` on clf.encoder.model (the Phase 1 model inside the classifier). `used`: names (relative
    to that model) of the parameters that get a gradient for this task. Returns the hooks; every Phase 1
    parameter not in hooks.body_params is frozen."""
    if technique not in TECHNIQUES:
        raise ValueError(f"technique must be one of {TECHNIQUES}")
    cfg = {**TECH_DEFAULTS, **cfg}
    model = clf.encoder.model
    params = _param_map(model)
    for p in model.parameters():
        p.requires_grad_(False)
    bottom, layers, top, kind = layer_groups(model, used)
    hooks: Hooks

    if technique == "frozen":
        hooks = Hooks(technique, [])
    elif technique == "full":
        hooks = Hooks(technique, [params[n] for n in params if n in used])
    elif technique == "partial":
        k = max(1, math.ceil(len(layers) * cfg["partial_frac"]))
        names = [n for g in layers[len(layers) - k:] for n in g] + top
        hooks = Hooks(technique, [params[n] for n in names],
                      dict(layers_trained=k, layers=len(layers), layer_kind=kind))
        log(f"[technique] partial: top {k} of {len(layers)} {kind} (+ {len(top)} tensors above them) are trained")
    elif technique == "unfreeze":
        hooks = UnfreezeHooks(params, bottom, layers, top, total_steps, cfg["unfreeze_stages"])
        hooks.info["layer_kind"] = kind
        log(f"[technique] gradual unfreezing over {hooks.stages} stages of {len(layers)} {kind}: head only, "
            f"then more {kind} from the top, embeddings in the last stage")
    elif technique == "bitfit":
        biases = [params[n] for n in params if n in used and n.split(".")[-1].startswith("bias")]
        added = []
        for mname, m in model.named_modules():
            pref = f"{mname}." if mname else ""
            if isinstance(m, nn.Linear) and m.bias is None and f"{pref}weight" in used:
                m.bias = nn.Parameter(torch.zeros(m.out_features, device=m.weight.device, dtype=m.weight.dtype))
                added.append(m.bias)
        hooks = Hooks(technique, biases + added, dict(existing_biases=len(biases), added_biases=len(added)))
        log(f"[technique] BitFit: {len(biases)} existing bias tensors"
            + (f" + {len(added)} zero biases added to linear layers that had none" if added else ""))
        if not biases and not added:
            raise RuntimeError("BitFit: the model has no bias values and no linear layers to add them to")
    elif technique in LOW_RANK:
        tg = targets(model, used)
        if not tg:
            raise RuntimeError(f"{technique}: no weight matrices found")
        mods, rnns = [], {}
        q_err, q_bytes, full_bytes = [], 0, 0
        for mname, m, pn in tg:
            W = getattr(m, pn)
            if technique == "lorma":
                pm = LoRMA(W.shape[0], cfg["lorma_r"], cfg["lorma_alpha"])
            elif technique == "lora":
                pm = LoRA(W.shape[0], W.shape[1], cfg["lora_r"], cfg["lora_alpha"])
            elif technique == "qlora":
                pm = QLoRA(W, cfg["lora_r"], cfg["lora_alpha"], cfg["qlora_block"])
            else:
                pm = AdaLoRA(W.shape[0], W.shape[1], cfg["adalora_r0"], cfg["adalora_alpha"] / cfg["adalora_r0"])
            pm = pm.to(device=W.device, dtype=W.dtype)
            if technique == "qlora":
                with torch.no_grad():
                    q_err.append(((pm.weight0() - W).norm() / W.norm().clamp(min=1e-12)).item())
                q_bytes += pm.packed.numel() + pm.absmax.numel() * 4
                full_bytes += W.numel() * W.element_size()
            parametrize.register_parametrization(m, pn, pm)
            orig = m.parametrizations[pn].original
            orig.requires_grad_(False)
            if technique == "qlora":            # the 4-bit codes replace the full-precision original
                orig.data = torch.empty(0, device=orig.device, dtype=orig.dtype)
            mods.append((f"{mname}.{pn}" if mname else pn, pm))
            if isinstance(m, nn.RNNBase):
                rnns[id(m)] = m
        for r in rnns.values():
            _rnn_without_cudnn(r)
        n_lin = sum(1 for _, m, _ in tg if isinstance(m, nn.Linear))
        if technique in ("lora", "qlora"):
            info = dict(matrices=len(mods), linear=n_lin, rnn=len(mods) - n_lin, r=cfg["lora_r"],
                        alpha=cfg["lora_alpha"])
            if technique == "qlora":
                info.update(block=cfg["qlora_block"], quantized_params=sum(math.prod(pm.shape) for _, pm in mods),
                            frozen_mb_4bit=q_bytes / 2 ** 20,
                            frozen_mb_full=full_bytes / 2 ** 20, rel_error_mean=sum(q_err) / len(q_err))
            hooks = Hooks(technique, [p for _, pm in mods for p in (pm.A, pm.B)], info)
            log(f"[technique] {'QLoRA' if technique == 'qlora' else 'LoRA'} rank {cfg['lora_r']} (scale "
                f"{cfg['lora_alpha'] / cfg['lora_r']:g}) on {len(mods)} weight matrices ({n_lin} linear, "
                f"{len(mods) - n_lin} RNN)"
                + (f"; fixed weights in 4-bit NF4 (blocks of {cfg['qlora_block']}): {q_bytes / 2 ** 20:.1f} MB instead "
                   f"of {full_bytes / 2 ** 20:.1f} MB, mean relative error {info['rel_error_mean']:.3f}"
                   if technique == "qlora" else ""))
        elif technique == "lorma":
            hooks = Hooks(technique, [p for _, pm in mods for p in (pm.A, pm.B)],
                          dict(matrices=len(mods), linear=n_lin, rnn=len(mods) - n_lin, r=cfg["lorma_r"],
                               alpha=cfg["lorma_alpha"]))
            log(f"[technique] LoRMA+ rank {cfg['lorma_r']} (scale {cfg['lorma_alpha'] / cfg['lorma_r']:g}) on "
                f"{len(mods)} weight matrices ({n_lin} linear, {len(mods) - n_lin} RNN)")
        else:
            hooks = AdaLoRAHooks(mods, total_steps, cfg)
            log(f"[technique] AdaLoRA on {len(mods)} weight matrices ({n_lin} linear, {len(mods) - n_lin} RNN): "
                f"rank {cfg['adalora_r0']} each at the start, {cfg['adalora_r']} on average at the end")
    else:  # adapter
        found = blocks_of(model, used)
        if found:
            where = [b for _, b in found[1]]
            what = f"{len(where)} blocks"
        else:
            where = [m for _, m in rnn_modules(model, used)]
            what = f"{len(where)} RNN module(s)"
        if not where:
            raise RuntimeError("adapter: no blocks or RNN modules found")
        dims, tmp = {}, []

        def probe(i):
            def hook(mod, args, out):
                t = _first_tensor(out)
                if t is not None:
                    dims[i] = t.size(-1)
            return hook

        for i, w in enumerate(where):
            tmp.append(w.register_forward_hook(probe(i)))
        dev = next(model.parameters()).device
        was = clf.training
        clf.eval()
        with torch.no_grad():
            ids = torch.tensor([[1, 2, 3, 4]], device=dev)
            clf.encoder(ids, torch.ones_like(ids))
        clf.train(was)
        for h in tmp:
            h.remove()
        if len(dims) != len(where):
            raise RuntimeError("adapter: some blocks were not run or returned no tensor")
        adapters = nn.ModuleList([Adapter(dims[i], cfg["adapter_dim"]) for i in range(len(where))]).to(dev)
        clf.adapters = adapters
        for w, a in zip(where, adapters):
            w.register_forward_hook(_adapter_hook(a))
        hooks = Hooks(technique, list(adapters.parameters()), dict(adapters=len(where), dim=cfg["adapter_dim"],
                                                                   where=what))
        log(f"[technique] adapters (d -> {cfg['adapter_dim']} -> d, start as identity) after {what}")

    for p in hooks.body_params:
        p.requires_grad_(True)
    if isinstance(hooks, UnfreezeHooks):
        hooks.on_step_start(0)
    hooks.info.setdefault("layer_kind", kind)
    hooks.info["layers"] = len(layers)
    return hooks
