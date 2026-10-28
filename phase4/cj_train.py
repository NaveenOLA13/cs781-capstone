"""Phase 4 training: one Phase 1 model + one fine-tuning technique -> outcome (0 / 1) of each case.

    from cj_models import load_backbone
    from cj_train import run
    bb, info = load_backbone("llama")
    res = run(bb, name="llama", technique="lorma")

Input of a case: the last max_len tokens of the judgment (window="tail"; "head" and "head_tail" also exist).
Model: Phase 1 model -> case vector (last token for left-to-right models, mean for the encoder-decoder) ->
dropout -> linear layer -> 2 scores. Loss: cross-entropy with class weights (smaller class weighted more).
Every epoch is scored on dev; the epoch with the best dev macro-F1 (at the dev-tuned threshold) is kept.
Every run trains a fresh copy of the Phase 1 weights.

Outputs in out_dir/run_name/: log.txt, config.json, metrics.json, dev_probs.json and test_probs.json
({case id: P(label 1)}), submission_tuned/ and submission_050/ (submission.zip with prediction.json, using the
dev-tuned threshold or 0.5), trained.pt (only the tensors this technique trains, + head).

Knowledge distillation (cj_kd.py starts these runs): student_layers keeps only the first blocks of the model (a
smaller student); kd_teacher = a teacher's train_probs.json ({case id: P(label 1)}) adds a teacher term to the loss:
(1 - kd_alpha) x class-weighted cross-entropy + kd_alpha x T^2 x KL(teacher || student) at temperature kd_T.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cj_data as D
import cj_tech as T
from cj_models import count_params

warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")
warnings.filterwarnings("ignore", message=r".*RNN module weights are not part of single contiguous chunk.*")

DEFAULTS = dict(
    data_root=D.DEFAULT_ROOT, out_dir="/kaggle/working/cj_runs", run_name=None, device=None, seed=42,
    technique="lorma", window="tail", max_len=512,
    epochs=3, batch_size=16, eval_batch_size=32, lr_body=None, lr_head=1e-3, weight_decay=0.01, warmup=0.06,
    grad_clip=1.0, amp=True, class_weight_power=0.5, dropout=0.1, head="linear", pooling=None,
    exact_batches=None,          # None: automatic (only cases of exactly one length per batch if padding matters)
    limit_train=0, limit_dev=0,  # >0: quick check on the first k cases of each label
    train_chunks=1,              # >1: every training case gives its last k chunks of max_len tokens as examples
    chunk_stride=448,            # tokens between the starts of neighbouring chunks (overlap = max_len - stride)
    ckpt_path=None,              # the Phase 1 / mid-trained checkpoint used (recorded for later steps)
    student_layers=0,            # knowledge distillation: >0 keeps only the first blocks (a number, or a share < 1)
    kd_teacher=None,             # knowledge distillation: a teacher's train_probs.json ({case id: P(label 1)})
    kd_alpha=0.5, kd_T=2.0,      # weight of the teacher term (gold labels get 1 - kd_alpha) and its temperature
    tok_workers=4, cache_dir="/kaggle/working/cj_cache", log_every=200, fit_sample=1000,
    save_weights=True, fetch_data=True,
    **T.TECH_DEFAULTS,
)


# ------------------------------------------------------------------------------ helpers
class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg=""):
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _scaler(device, enabled):
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _peak_gb(device):
    return torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else None


def collate(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def limit(docs, k, seed=0):
    """The first k cases of each label (quick checks)."""
    if not k:
        return docs
    out = []
    for lab in (0, 1):
        out += [d for d in docs if d["label"] == lab][:k]
    random.Random(seed).shuffle(out)
    return out


def resolve_cfg(overrides: dict) -> dict:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **overrides}
    if cfg["technique"] not in T.TECHNIQUES:
        raise ValueError(f"technique must be one of {T.TECHNIQUES}")
    if cfg["window"] not in D.WINDOWS:
        raise ValueError(f"window must be one of {D.WINDOWS}")
    if cfg["lr_body"] is None:
        cfg["lr_body"] = T.BODY_LR[cfg["technique"]]
    if cfg["kd_teacher"]:
        if not os.path.exists(cfg["kd_teacher"]):
            raise FileNotFoundError(f"kd_teacher {cfg['kd_teacher']} does not exist")
        if not 0.0 <= cfg["kd_alpha"] <= 1.0 or cfg["kd_T"] <= 0:
            raise ValueError("kd_alpha must be in [0, 1] and kd_T > 0")
    return cfg


# --------------------------------------------------------------------------------- data
def prepare_tokens(encode, cfg, log=print, bos_id=None, eos_id=None):
    """(data, {split: token id lists}, case_of) with the window of every case; tokenised once and cached.
    With train_chunks = k > 1 every training case gives its last k chunks (each one example with the case's label);
    case_of[j] is the case of training example j. Dev and test always use one window per case."""
    data = D.load_data(cfg["data_root"], fetch=cfg["fetch_data"], log=log)
    data["train"] = limit(data["train"], cfg["limit_train"], cfg["seed"])
    data["dev"] = limit(data["dev"], cfg["limit_dev"], cfg["seed"])
    log(D.summary(data))
    n = cfg["max_len"] - (bos_id is not None) - (eos_id is not None)
    k = int(cfg.get("train_chunks", 1) or 1)
    if k > 1 and cfg["window"] != "tail":
        raise ValueError("train_chunks > 1 needs window='tail'")

    def wrap(x):
        return ([bos_id] if bos_id is not None else []) + (x or [0]) + ([eos_id] if eos_id is not None else [])

    seqs, case_of = {}, list(range(len(data["train"])))
    for s, ds in data.items():
        span = D.chunk_span(n, cfg["chunk_stride"], k) if (s == "train" and k > 1) else n
        ids = D.encode_docs(ds, encode, span, cfg["window"], workers=cfg["tok_workers"], cache_dir=cfg["cache_dir"],
                            log=log)
        if s == "train" and k > 1:
            seqs[s], case_of = [], []
            for i, x in enumerate(ids):
                for c in D.chunk_ids(x, n, cfg["chunk_stride"], k):
                    seqs[s].append(wrap(c))
                    case_of.append(i)
            log(f"[data] train: {len(seqs[s])} chunk examples from {len(ids)} cases (last {k} chunks of {n} tokens, "
                f"stride {cfg['chunk_stride']})")
        else:
            seqs[s] = [wrap(x) for x in ids]
            log(f"[data] {s}: " + D.length_report([len(x) for x in ids], n))
    return data, seqs, case_of


@torch.no_grad()
def predict(clf, seqs, cfg, device, pad_id) -> list[float]:
    """P(label 1) for every sequence, in the given order."""
    was = clf.training
    clf.eval()
    out = [0.0] * len(seqs)
    for b in D.batches([len(s) for s in seqs], cfg["eval_batch_size"], shuffle=False, exact=cfg["exact_batches"]):
        ids, mask = collate([seqs[i] for i in b], pad_id, device)
        with _autocast(device, cfg["amp"]):
            logits = clf(ids, mask)
        p = torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
        for i, v in zip(b, p):
            out[i] = v
    clf.train(was)
    return out


def find_used(clf, seqs, cfg, device, pad_id) -> set[str]:
    """Names (inside the Phase 1 model) of the parameters that get a gradient for this task; e.g. the decoder and
    LM head of an encoder-decoder do not."""
    model = clf.encoder.model
    for p in clf.parameters():
        p.requires_grad_(True)
    clf.train()
    clf.zero_grad(set_to_none=True)
    lengths = [len(s) for s in seqs]
    for b in D.batches(lengths, 2, shuffle=True, seed=123, exact=cfg["exact_batches"])[:2]:
        ids, mask = collate([seqs[i] for i in b], pad_id, device)
        with _autocast(device, cfg["amp"]):
            out = clf(ids, mask)
        out.float().logsumexp(-1).mean().backward()
    used = {n for n, p in model.named_parameters() if p.grad is not None}
    clf.zero_grad(set_to_none=True)
    return used


def warmup_linear(opt, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _evaluate(y, p):
    t, _ = D.tune_threshold(y, p)
    return D.report(y, p, 0.5), D.report(y, p, t)


# ------------------------------------------------------------------ knowledge distillation
LAYER_COUNT_NAMES = ("n_layers", "n_layer", "num_layers", "num_hidden_layers", "n_blocks", "depth")


def _shrink_layer_count(model: nn.Module, n: int, k: int) -> list[str]:
    """Models that loop `for i in range(config.n_layers)` need the stored layer count changed too."""
    changed = []
    holders = [("model", model)] + [(a, getattr(model, a)) for a in ("config", "cfg", "args", "params")
                                     if getattr(model, a, None) is not None]
    for hname, h in holders:
        for attr in LAYER_COUNT_NAMES:
            try:
                if isinstance(h, dict):
                    if h.get(attr) == n:
                        h[attr] = k
                        changed.append(f"{hname}[{attr!r}]")
                elif getattr(h, attr, None) == n and not isinstance(getattr(h, attr), bool):
                    if dataclasses.is_dataclass(h) and type(h).__dataclass_params__.frozen:  # type: ignore
                        object.__setattr__(h, attr, k)
                    else:
                        setattr(h, attr, k)
                    changed.append(f"{hname}.{attr}")
            except (AttributeError, TypeError):
                pass
    return changed


def cut_blocks(model: nn.Module, keep) -> dict:
    """Keeps only the first blocks of the model (in place). keep: a number of blocks, or a share (< 1).
    The blocks are the nn.ModuleList of same-class layers with the most weights."""
    best = None
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 2 and len({type(x) for x in m}) == 1:
            n_w = sum(p.numel() for p in m.parameters())
            if best is None or n_w > best[2]:
                best = (name, m, n_w)
    if best is None:
        raise ValueError("no repeated blocks (an nn.ModuleList of layers) to shorten")
    name, blocks, _ = best
    n = len(blocks)
    keep = float(keep)
    k = int(keep) if keep >= 1 else max(1, int(round(n * keep)))
    if not 1 <= k < n:
        raise ValueError(f"student_layers must leave between 1 and {n - 1} of the {n} blocks (got {k})")
    before = sum(p.numel() for p in model.parameters())
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, nn.ModuleList(list(blocks)[:k]))
    changed = _shrink_layer_count(model, n, k)
    after = sum(p.numel() for p in model.parameters())
    return dict(blocks=name, layers_full=n, layers_student=k, params_full=before, params_student=after,
                layer_count_fields=changed)


def make_model(bb, cfg: dict, log=print):
    """A fresh copy of the Phase 1 model (bb.model is not changed); a KD student keeps only its first blocks.
    Returns (model, student info or None)."""
    model = copy.deepcopy(bb.model)
    if not cfg.get("student_layers"):
        return model, None
    info = cut_blocks(model, cfg["student_layers"])
    log(f"[student] first {info['layers_student']} of {info['layers_full']} blocks ('{info['blocks']}'): "
        f"{info['params_student'] / 1e6:.1f}M of {info['params_full'] / 1e6:.1f}M weights"
        + (f"; layer count changed in {', '.join(info['layer_count_fields'])}" if info["layer_count_fields"] else ""))
    return model, info


def load_teacher(path: str, ids: list[str], labels: list[int], log=print):
    """Teacher P(label 1) for every training example (ids = case id of each example). Returns (probs, info)."""
    with open(path) as f:
        tp = json.load(f)
    cases = set(ids)
    missing = {i for i in cases if i not in tp}
    if len(missing) > 0.01 * len(cases):
        raise ValueError(f"the teacher {path} has no probability for {len(missing)} of {len(cases)} training cases "
                         "(made from other data?)")
    probs = [float(tp[i]) if i in tp else float(y) for i, y in zip(ids, labels)]
    agree = sum(int(p >= 0.5) == y for p, y in zip(probs, labels)) / max(1, len(labels))
    info = dict(teacher=path, cases=len(cases), cases_missing=len(missing), agreement_050=agree,
                mean_p=sum(probs) / max(1, len(probs)))
    log(f"[kd] teacher {path}: P(label 1) for {len(cases) - len(missing)} of {len(cases)} training cases"
        + (f" ({len(missing)} missing: gold label used)" if missing else "")
        + f"; agrees with the gold label at 0.5 on {agree:.1%} of the examples; mean P(label 1) {info['mean_p']:.3f}")
    return probs, info


def kd_loss(logits: torch.Tensor, p_teacher: torch.Tensor, T: float) -> torch.Tensor:
    """T^2 x KL(teacher || student) at temperature T, averaged over the batch. The teacher gives only
    p = P(label 1), i.e. the score difference log(p / (1 - p)); both sides are softened by dividing scores by T."""
    p = p_teacher.clamp(1e-4, 1 - 1e-4)
    q1 = torch.sigmoid((torch.log(p) - torch.log1p(-p)) / T)
    q = torch.stack([1 - q1, q1], -1)
    return (q * (torch.log(q) - F.log_softmax(logits / T, -1))).sum(-1).mean() * T * T


# ---------------------------------------------------------------------------------- run
def run(bb, name: str = "model", **overrides) -> dict:
    cfg = resolve_cfg(overrides)
    tech = cfg["technique"]
    cfg["run_name"] = cfg["run_name"] or f"{name}_{tech}"
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    kind = (" (KD student" + (" with a teacher)" if cfg["kd_teacher"] else ", no teacher)")
            if cfg["student_layers"] else (" (with a teacher)" if cfg["kd_teacher"] else ""))
    log(f"===== '{cfg['run_name']}': {name} + {tech}{kind} on {device} -> {out}")
    if cfg["exact_batches"] is None:
        cfg["exact_batches"] = bool(getattr(bb, "pad_effect", 0.0) > 1e-4)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    data, seqs, case_of = prepare_tokens(bb.encode, cfg, log, bb.bos_id, bb.eos_id)
    top = max(max(s) for ss in seqs.values() for s in ss)
    if bb.vocab is not None and top >= bb.vocab:
        raise ValueError(f"token id {top} >= the model's vocabulary {bb.vocab}: wrong tokenizer?")
    y_tr = [data["train"][c]["label"] for c in case_of]
    y_dev = [d["label"] for d in data["dev"]]
    t_tr, kd = None, None
    if cfg["kd_teacher"]:
        t_tr, kd = load_teacher(cfg["kd_teacher"], [data["train"][c]["id"] for c in case_of], y_tr, log)
        kd.update(alpha=cfg["kd_alpha"], T=cfg["kd_T"])
        log(f"[kd] loss = {1 - cfg['kd_alpha']:.2f} x gold-label cross-entropy + {cfg['kd_alpha']:.2f} x teacher "
            f"term (temperature {cfg['kd_T']})")

    # ------------------------------------------------------------------ model + technique
    model, student = make_model(bb, cfg, log)
    clf = bb.classifier(2, model=model, head=cfg["head"], dropout=cfg["dropout"],
                        pooling=cfg["pooling"]).to(device).float()
    used = find_used(clf, seqs["train"], cfg, device, bb.pad_id)
    n_model = sum(p.numel() for n, p in clf.encoder.model.named_parameters())
    n_used = sum(p.numel() for n, p in clf.encoder.model.named_parameters() if n in used)
    if n_used < n_model:
        log(f"[setup] {n_used / 1e6:.1f}M of the model's {n_model / 1e6:.1f}M weights take part in reading a case "
            "(the rest, e.g. a decoder or LM head, is never trained)")
    lengths = [len(s) for s in seqs["train"]]
    steps_per_epoch = len(D.batches(lengths, cfg["batch_size"], shuffle=True, exact=cfg["exact_batches"]))
    total = steps_per_epoch * cfg["epochs"]
    hooks = T.apply(clf, tech, used, cfg, total, log)
    head = [p for p in clf.head_parameters()]
    for p in head:
        p.requires_grad_(True)
    body = hooks.body_params
    n_trainable = sum(p.numel() for p in {id(p): p for p in head + body}.values())
    n_total = count_params(clf, trainable_only=False) + hooks.info.get("quantized_params", 0)   # QLoRA: 4-bit W0
    log(f"[setup] {tech}: {n_trainable / 1e6:.3f}M trained weights of {n_total / 1e6:.1f}M "
        f"({n_trainable / n_total:.2%}); case vector: '{clf.pool.kind}' pooling; body lr {cfg['lr_body']}, "
        f"head lr {cfg['lr_head']}")
    log("[setup] batches: " + ("only cases of exactly the same length (no padding)" if cfg["exact_batches"]
                               else "similar lengths, padded at the end"))

    groups = {}
    for p, lr in [(p, cfg["lr_head"]) for p in head] + [(p, cfg["lr_body"]) for p in body]:
        wd = cfg["weight_decay"] if p.dim() >= 2 else 0.0
        groups.setdefault((lr, wd), []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=k[0], weight_decay=k[1]) for k, v in groups.items()])
    sched = warmup_linear(opt, int(cfg["warmup"] * total), total)
    scaler = _scaler(device, cfg["amp"])
    all_params = head + body
    w = D.class_weights(y_tr, cfg["class_weight_power"])
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    log(f"[train] {len(y_tr)} examples from {len(data['train'])} cases ({sum(y_tr)} label 1), class weights "
        f"{w[0]:.2f} / {w[1]:.2f}; "
        f"{steps_per_epoch} steps per epoch x {cfg['epochs']} epochs")

    keep_ids = {id(p) for p in all_params}
    keep_names = [n for n, p in clf.named_parameters() if id(p) in keep_ids] + \
                 [n for n, _ in clf.named_buffers() if n.endswith("mask")]

    def snapshot():
        sd = clf.state_dict()
        return {n: sd[n].detach().to("cpu", copy=True) for n in keep_names if n in sd}

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    best, best_state, best_epoch, history, step = -1.0, None, 0, [], 0
    t_start = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        clf.train()
        t0, tot, tot_kd, seen, skipped = time.time(), 0.0, 0.0, 0, 0
        order = D.batches(lengths, cfg["batch_size"], shuffle=True, seed=cfg["seed"] * 1000 + ep,
                          exact=cfg["exact_batches"])
        for k, b in enumerate(order, 1):
            hooks.on_step_start(step)
            ids, mask = collate([seqs["train"][i] for i in b], bb.pad_id, device)
            y = torch.tensor([y_tr[i] for i in b], dtype=torch.long, device=device)
            with _autocast(device, cfg["amp"]):
                logits = clf(ids, mask)
            ce = F.cross_entropy(logits.float(), y, weight=weight)
            main = ce
            if t_tr is not None:
                kd_term = kd_loss(logits.float(), torch.tensor([t_tr[i] for i in b], dtype=torch.float32,
                                                               device=device), cfg["kd_T"])
                main = (1 - cfg["kd_alpha"]) * ce + cfg["kd_alpha"] * kd_term
            extra = hooks.extra_loss()
            loss = main if extra is None else main + extra
            opt.zero_grad(set_to_none=True)
            step += 1
            if not torch.isfinite(loss):
                skipped += 1
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in all_params if p.grad is not None], cfg["grad_clip"])
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            sched.step()
            hooks.after_step(step, stepped=scaler.get_scale() >= scale)
            tot += ce.item() * len(b)
            if t_tr is not None:
                tot_kd += kd_term.item() * len(b)
            seen += len(b)
            if cfg["log_every"] and k % cfg["log_every"] == 0:
                el = time.time() - t0
                log(f"  epoch {ep} step {k}/{len(order)}  loss {tot / max(1, seen):.4f}  "
                    + (f"teacher term {tot_kd / max(1, seen):.4f}  " if t_tr is not None else "")
                    + f"{el / 60:.1f} min, about {el / k * (len(order) - k) / 60:.1f} min left in this epoch")
        p_dev = predict(clf, seqs["dev"], cfg, device, bb.pad_id)
        r50, rt = _evaluate(y_dev, p_dev)
        rec = dict(epoch=ep, loss=tot / max(1, seen), seconds=time.time() - t0, skipped_steps=skipped,
                   dev_f1_050=r50["macro_f1"], dev_f1_tuned=rt["macro_f1"], threshold=rt["threshold"],
                   teacher_term=tot_kd / max(1, seen) if t_tr is not None else None)
        mark = ""
        if rt["macro_f1"] > best:
            best, best_epoch, mark = rt["macro_f1"], ep, "  <- best so far"
            best_state = snapshot()
        log(f"[train] epoch {ep}/{cfg['epochs']}  loss {rec['loss']:.4f}  dev macro-F1 {r50['macro_f1']:.4f} at 0.5, "
            f"{rt['macro_f1']:.4f} at {rt['threshold']:.2f}  {rec['seconds'] / 60:.1f} min{mark}"
            + (f"  ({skipped} steps skipped: non-finite loss)" if skipped else ""))
        history.append(rec)
    train_seconds = time.time() - t_start
    peak = _peak_gb(device)
    clf.load_state_dict(best_state, strict=False)
    log(f"[train] using epoch {best_epoch}")

    # ------------------------------------------------------------------ results
    t_inf = time.time()
    p_dev = predict(clf, seqs["dev"], cfg, device, bb.pad_id)
    infer_s = time.time() - t_inf
    p_test = predict(clf, seqs["test"], cfg, device, bb.pad_id)
    r50, rt = _evaluate(y_dev, p_dev)
    rng = random.Random(cfg["seed"])
    fit_idx = sorted(rng.sample(range(len(y_tr)), min(cfg["fit_sample"], len(y_tr))))
    p_fit = predict(clf, [seqs["train"][i] for i in fit_idx], cfg, device, bb.pad_id)
    r_fit = D.report([y_tr[i] for i in fit_idx], p_fit, rt["threshold"])
    log(D.format_report(r50, "[dev] at threshold 0.5:   "))
    log(D.format_report(rt, "[dev] at tuned threshold: "))
    log(f"[fit] on {len(fit_idx)} of its own training cases: macro-F1 {r_fit['macro_f1']:.4f} "
        f"(dev {rt['macro_f1']:.4f}; a big gap = memorising)")

    ids_dev = [d["id"] for d in data["dev"]]
    ids_test = [d["id"] for d in data["test"]]
    with open(os.path.join(out, "dev_probs.json"), "w") as f:
        json.dump(dict(zip(ids_dev, p_dev)), f)
    with open(os.path.join(out, "test_probs.json"), "w") as f:
        json.dump(dict(zip(ids_test, p_test)), f)
    subs = {}
    for key, t in (("tuned", rt["threshold"]), ("050", 0.5)):
        preds = {i: int(p >= t) for i, p in zip(ids_test, p_test)}
        subs[key] = D.write_submission(data["test"], preds, os.path.join(out, f"submission_{key}"))
    log(f"[test] predicted label 1 for {sum(int(p >= rt['threshold']) for p in p_test)} of {len(p_test)} cases "
        f"(tuned threshold {rt['threshold']:.2f}); dev has {sum(y_dev)} of {len(y_dev)}")
    saved_mb = None
    if cfg["save_weights"]:
        path = os.path.join(out, "trained.pt")
        torch.save({"state": snapshot(), "config": cfg, "technique": tech, "model": name}, path)
        saved_mb = os.path.getsize(path) / 2 ** 20

    metrics = dict(run_name=cfg["run_name"], model=name, technique=tech, route=bb.route, kind=bb.kind,
                   pooling=clf.pool.kind, window=cfg["window"], max_len=cfg["max_len"],
                   data={s: dict(cases=len(ds), label1=sum(1 for d in ds if d["label"] == 1))
                         for s, ds in data.items()},
                   class_weights=w, history=history, best_epoch=best_epoch,
                   dev_050=r50, dev_tuned=rt, threshold=rt["threshold"], train_fit=r_fit,
                   trainable_params=n_trainable, total_params=n_total, used_params=n_used,
                   trainable_share=n_trainable / n_total, peak_gpu_gb=peak, train_seconds=train_seconds,
                   dev_infer_seconds=infer_s, dev_cases_per_second=len(p_dev) / max(infer_s, 1e-9),
                   saved_mb=saved_mb, technique_info=hooks.summary(), submissions=subs, config=cfg,
                   student=student, kd=kd)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=str)
    log(f"[done] dev macro-F1 {rt['macro_f1']:.4f} (tuned threshold {rt['threshold']:.2f}), {r50['macro_f1']:.4f} "
        f"at 0.5 | submit {subs['tuned']}")
    clf.cpu()
    opt.state.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(out_dir=out, metrics=metrics)


# ---------------------------------------------------------------------------- comparison
def load_metrics(out_dir: str, run_name: str) -> dict | None:
    p = os.path.join(out_dir, run_name, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def comparison(rows: list[dict]) -> str:
    """Table of finished runs (metrics dicts): quality, cost and what each technique trains."""
    hdr = (f"{'run':22s} {'trained':>9s} {'share':>7s} {'dev F1@0.5':>11s} {'dev F1@t':>9s} {'t':>5s} "
           f"{'P0':>6s} {'R0':>6s} {'P1':>6s} {'R1':>6s} {'fit F1':>7s} {'ep':>3s} {'train':>7s} {'GPU GB':>7s} "
           f"{'dev s':>6s} {'MB':>7s}")
    lines = [hdr, "-" * len(hdr)]
    for m in sorted(rows, key=lambda m: -m["dev_tuned"]["macro_f1"]):
        t = m["dev_tuned"]
        gb = "-" if m.get("peak_gpu_gb") is None else f"{m['peak_gpu_gb']:.2f}"
        mb = "-" if m.get("saved_mb") is None else f"{m['saved_mb']:.1f}"
        lines.append(
            f"{m['run_name']:22s} {m['trainable_params'] / 1e6:8.2f}M {m['trainable_share']:7.2%} "
            f"{m['dev_050']['macro_f1']:11.4f} {t['macro_f1']:9.4f} {t['threshold']:5.2f} "
            f"{t['class0']['precision']:6.3f} {t['class0']['recall']:6.3f} {t['class1']['precision']:6.3f} "
            f"{t['class1']['recall']:6.3f} {m['train_fit']['macro_f1']:7.4f} {m['best_epoch']:3d} "
            f"{m['train_seconds'] / 60:6.1f}m {gb:>7s} {m['dev_infer_seconds']:6.1f} {mb:>7s}")
    lines.append("dev F1@t: macro-F1 at the threshold tuned on dev (t); P/R: precision / recall of class 0 and 1 "
                 "at t; fit F1: on a sample of the run's own training cases; ep: chosen epoch; dev s: seconds to "
                 "predict all dev cases; MB: size of the saved trained tensors.")
    return "\n".join(lines)
