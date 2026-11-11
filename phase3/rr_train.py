"""Phase 3 training and submission."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

import rr_data as D
import rr_peft as P
from rr_head import ContextClassifier, count_params, freeze_backbone

warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")

DEFAULTS = dict(
    data_root=D.DEFAULT_ROOT,
    out_dir="/kaggle/working/rr_runs",
    run_name="run",
    device=None,
    seed=42,
    limit_train_docs=0,
    max_len=128,
    pos_buckets=10,
    pos_dim=32,
    context=False,
    context_tokens=64,
    context_sep_id=None,
    method="full",
    head="linear",
    dropout=0.1,
    epochs=3,
    batch_size=64,
    eval_batch_size=128,
    lr_backbone=5e-5,
    lr_head=1e-3,
    weight_decay=0.01,
    warmup=0.06,
    grad_clip=1.0,
    amp=True,
    weight_cap=10.0,
    weight_power=0.5,
    log_every=100,
    exact_batches=None,
    lr_peft=None,
    peft_r=8,
    peft_alpha=16,
    peft_dropout=0.05,
    peft_targets=None,
    ia3_ff=None,
    adalora_init_r=12,
    adalora_tinit=0.1,
    adalora_tfinal=0.2,
    adalora_deltaT=10,
    adalora_orth=0.5,
    kd_teacher=None,
    kd_alpha=0.5,
    kd_T=2.0,
    kd_stage_b=True,
    b_epochs=40,
    b_patience=10,
    b_lr=1e-3,
    b_weight_decay=0.01,
    b_hidden=256,
    b_dropout=0.3,
    b_layers=1,
    b_arch="gru",
    b_heads=4,
    b_attn_layers=2,
    b_variants=None,
    save_model=True,
    select_metric="macro_f1_all13",
)

METRIC_NAMES = {"macro_f1_all13": "13-role avg", "macro_f1": "macro-F1"}


# --- helpers
class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg=""):
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def set_seed(seed: int) -> None:
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


def collate(seqs, pad_id, device):
    """Right-padded ids [B, T] and mask [B, T] (1 = real token)."""
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


def flat(docs, buckets):
    """(doc index, sentence index, position bucket) for every sentence, in document order."""
    out = []
    for di, d in enumerate(docs):
        n = len(d["ids"])
        out += [(di, si, D.pos_bucket(si, n, buckets)) for si in range(n)]
    return out


def item_seq(docs, item, cfg):
    """Token ids for one sentence and how many of them are context: the sentence alone (context=False) or [end of
    previous sentence] <sep> [start of next sentence] <sep> [sentence] (context=True)."""
    sents = docs[item[0]]["ids"]
    si = item[1]
    tgt = list(sents[si])
    if not cfg.get("context"):
        return tgt, 0
    k, sep = cfg["context_tokens"], cfg["context_sep_id"]
    prev = list(sents[si - 1][-k:]) if si > 0 and k > 0 else []
    nxt = list(sents[si + 1][:k]) if si + 1 < len(sents) and k > 0 else []
    ctx = prev + [sep] + nxt + [sep]
    return ctx + tgt, len(ctx)


def item_lengths(docs, items, cfg):
    if not cfg.get("context"):
        return [len(docs[di]["ids"][si]) for di, si, _ in items]
    return [len(item_seq(docs, it, cfg)[0]) for it in items]


def _batch(docs, items, b, pad_id, device, cfg=None):
    """ids, mask, pool mask (None = pool over all tokens; with context only the sentence's own tokens), pos."""
    if cfg is not None and cfg.get("context"):
        seqs = [item_seq(docs, items[k], cfg) for k in b]
        ids, mask = collate([q for q, _ in seqs], pad_id, device)
        pool = mask.clone()
        for j, (_, n_ctx) in enumerate(seqs):
            pool[j, :n_ctx] = 0
    else:
        ids, mask = collate([docs[items[k][0]]["ids"][items[k][1]] for k in b], pad_id, device)
        pool = None
    pos = torch.tensor([items[k][2] for k in b], dtype=torch.long, device=device)
    return ids, mask, pool, pos


def to_preds(docs, outputs, labels):
    """Per-document score tensors -> {doc_id: [role per sentence]}."""
    return {d["id"]: ([] if o is None else [labels[i] for i in o.argmax(-1).tolist()]) for d, o in zip(docs, outputs)}


def _gb(device):
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


# --- Stage A
@torch.no_grad()
def run_sentences(clf, docs, cfg, device, pad_id, what="logits"):
    """Stage A over every sentence of `docs`: role scores ("logits") or sentence features ("features", the input of
    head A)."""
    was = clf.training
    clf.eval()
    items = flat(docs, cfg["pos_buckets"])
    lengths = item_lengths(docs, items, cfg)
    out = [None] * len(items)
    for b in D.length_batches(lengths, cfg["eval_batch_size"], shuffle=False, exact=cfg["exact_batches"]):
        ids, mask, pool, pos = _batch(docs, items, b, pad_id, device, cfg)
        with _autocast(device, cfg["amp"]):
            y = (
                clf(ids, mask, pool_mask=pool, pos=pos)
                if what == "logits"
                else clf.features(ids, mask, pool_mask=pool, pos=pos)
            )
        y = y.float().cpu()
        for j, k in enumerate(b):
            out[k] = y[j]
    clf.train(was)
    res, k = [], 0
    for d in docs:
        n = len(d["ids"])
        res.append(torch.stack(out[k : k + n]) if n else None)
        k += n
    return res


def kd_loss(logits, teacher_logp, T):
    """T^2 * KL(teacher_T || student_T), averaged over sentences."""
    p = F.softmax(teacher_logp.float() / T, dim=-1)
    return F.kl_div(F.log_softmax(logits.float() / T, dim=-1), p, reduction="batchmean") * (T * T)


def teacher_targets(path, docs, labels):
    """The teacher's log-probabilities [S, L] for each judgment in `docs` (same order), from teacher.pt."""
    t = torch.load(path, map_location="cpu", weights_only=False)
    if list(t["labels"]) != list(labels):
        raise ValueError("the teacher was trained with a different label list")
    out, missing = [], []
    for d in docs:
        lp = t["train"].get(d["id"])
        if lp is None:
            missing.append(d["id"])
            continue
        if lp.shape != (len(d["labels"]), len(labels)):
            raise ValueError(f"{d['id']}: teacher has {tuple(lp.shape)} for {len(d['labels'])} sentences")
        out.append(lp.float())
    if missing:
        raise ValueError(f"the teacher has no probabilities for {len(missing)} training judgments, e.g. {missing[:3]}")
    return out


def make_optimizer(named_params, head_ids, cfg):
    body_lr = cfg["lr_peft"] if cfg["method"] in P.PEFT_METHODS else cfg["lr_backbone"]
    groups = {}
    for n, p in named_params:
        if not p.requires_grad:
            continue
        lr = cfg["lr_head"] if id(p) in head_ids else body_lr
        wd = cfg["weight_decay"] if p.dim() >= 2 and "ia3_" not in n else 0.0
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([dict(params=v, lr=k[0], weight_decay=k[1]) for k, v in groups.items()])


def warmup_linear(opt, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def stage_a_steps(docs, cfg) -> int:
    """Optimizer steps per epoch of Stage A (the same for every epoch and seed)."""
    items = flat(docs, cfg["pos_buckets"])
    lengths = item_lengths(docs, items, cfg)
    return len(D.length_batches(lengths, cfg["batch_size"], shuffle=True, exact=cfg["exact_batches"]))


def freeze_unused(clf, docs, cfg, device, pad_id, log=print, n_batches=3):
    """Parameters that never get a gradient for role prediction (e.g. the decoder and LM head of an encoder-decoder)
    are frozen so they are not counted or stored as trainable."""
    clf.train()
    clf.zero_grad(set_to_none=True)
    items = flat(docs, cfg["pos_buckets"])
    lengths = item_lengths(docs, items, cfg)
    for b in D.length_batches(lengths, cfg["batch_size"], shuffle=True, seed=123, exact=cfg["exact_batches"])[
        :n_batches
    ]:
        ids, mask, pool, pos = _batch(docs, items, b, pad_id, device, cfg)
        with _autocast(device, cfg["amp"]):
            out = clf(ids, mask, pool_mask=pool, pos=pos)
        out.float().logsumexp(-1).mean().backward()
    unused = [(n, p) for n, p in clf.named_parameters() if p.requires_grad and p.grad is None]
    for _, p in unused:
        p.requires_grad_(False)
    clf.zero_grad(set_to_none=True)
    if unused:
        parts = Counter()
        for n, p in unused:
            n = n[len("encoder.model.") :] if n.startswith("encoder.model.") else n
            parts[n.split(".")[0]] += p.numel()
        log(
            "[setup] not used for role prediction (left unchanged): "
            + ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in parts.most_common())
        )


def train_stage_a(clf, train_docs, dev_docs, labels, cfg, device, pad_id, log=print, soft=None):
    """Fine-tune Stage A."""
    sel, mname = cfg["select_metric"], METRIC_NAMES[cfg["select_metric"]]
    li = {l: i for i, l in enumerate(labels)}
    items = flat(train_docs, cfg["pos_buckets"])
    y_all = [li[train_docs[di]["labels"][si]] for di, si, _ in items]
    lengths = item_lengths(train_docs, items, cfg)
    w, counts = D.class_weights(y_all, len(labels), cfg["weight_cap"], cfg["weight_power"])
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    log("[stage A] class weights: " + ", ".join(f"{l} {x:.2f} (n={c})" for l, x, c in zip(labels, w, counts)))

    head_ids = {id(p) for p in clf.head_parameters()}
    opt = make_optimizer(clf.named_parameters(), head_ids, cfg)
    steps = stage_a_steps(train_docs, cfg)
    hooks = getattr(clf, "peft_hooks", None)
    a_kd, T_kd = cfg["kd_alpha"], cfg["kd_T"]
    if soft is not None:
        log(
            f"[stage A] knowledge distillation: {a_kd} x gold-label loss + {1 - a_kd:.2f} x teacher term "
            f"(temperature {T_kd})"
        )
    total = steps * cfg["epochs"]
    sched = warmup_linear(opt, int(cfg["warmup"] * total), total)
    scaler = _scaler(device, cfg["amp"])
    trainable = [p for p in clf.parameters() if p.requires_grad]
    log(f"[stage A] {len(items)} sentences, {steps} steps per epoch, {cfg['epochs']} epochs")

    best, best_state, best_epoch, history = -1.0, None, 0, []
    t_start = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        clf.train()
        t0, tot, tot_kd, seen, skipped = time.time(), 0.0, 0.0, 0, 0
        batches = D.length_batches(
            lengths, cfg["batch_size"], shuffle=True, seed=cfg["seed"] * 1000 + ep, exact=cfg["exact_batches"]
        )
        for step, b in enumerate(batches, 1):
            ids, mask, pool, pos = _batch(train_docs, items, b, pad_id, device, cfg)
            y = torch.tensor([y_all[k] for k in b], dtype=torch.long, device=device)
            with _autocast(device, cfg["amp"]):
                logits = clf(ids, mask, pool_mask=pool, pos=pos)
            ce = F.cross_entropy(logits.float(), y, weight=weight)
            loss = ce
            if soft is not None:
                t_lp = torch.stack([soft[items[k][0]][items[k][1]] for k in b]).to(device)
                kd = kd_loss(logits, t_lp, T_kd)
                loss = a_kd * ce + (1 - a_kd) * kd
            extra = hooks.extra_loss() if hooks is not None else None
            loss = loss if extra is None else loss + extra
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            scale = scaler.get_scale() if hooks is not None else 0.0
            scaler.step(opt)
            scaler.update()
            sched.step()
            if hooks is not None:
                hooks.after_step(stepped=scaler.get_scale() >= scale)
            tot += ce.item() * len(b)
            if soft is not None:
                tot_kd += kd.item() * len(b)
            seen += len(b)
            if cfg["log_every"] and step % cfg["log_every"] == 0:
                log(f"  epoch {ep} step {step}/{steps}  loss {tot / max(1, seen):.4f}")
        rec = dict(epoch=ep, loss=tot / max(1, seen), seconds=time.time() - t0, skipped_steps=skipped)
        extra = f"  ({skipped} steps skipped: non-finite loss)" if skipped else ""
        if soft is not None:
            rec["kd_loss"] = tot_kd / max(1, seen)
            extra = f"  teacher term {rec['kd_loss']:.4f}" + extra
        if dev_docs:
            dev_out = run_sentences(clf, dev_docs, cfg, device, pad_id)
            rep = D.evaluate(dev_docs, to_preds(dev_docs, dev_out, labels), labels)
            f = rep["all"][sel]
            rec.update(dev=f, **{f"dev_{k}": v[sel] for k, v in rep.items() if k != "all"})
            mark = ""
            if f > best:
                best, best_epoch, mark = f, ep, "  <- best so far"
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in clf.state_dict().items()}
            doms = ", ".join(f"{k} {v[sel]:.4f}" for k, v in rep.items() if k != "all")
            log(
                f"[stage A] epoch {ep}/{cfg['epochs']}  loss {rec['loss']:.4f}  dev {mname} {f:.4f} ({doms})"
                f"  {rec['seconds']:.0f}s{mark}{extra}"
            )
        else:
            best_epoch = ep
            log(f"[stage A] epoch {ep}/{cfg['epochs']}  loss {rec['loss']:.4f}  {rec['seconds']:.0f}s{extra}")
        history.append(rec)
    if best_state is not None:
        clf.load_state_dict(best_state)
        log(f"[stage A] using epoch {best_epoch} (dev {mname} {best:.4f})")
    else:
        log(f"[stage A] using the last epoch ({best_epoch}); dev was not looked at")
    info = dict(
        history=history,
        best_epoch=best_epoch,
        best_dev_score=best if best_state is not None else None,
        select_metric=sel,
        class_weights=w,
        train_counts=dict(zip(labels, counts)),
        train_seconds=time.time() - t_start,
        weight_tensor=weight,
    )
    if hooks is not None:
        info["peft"] = hooks.summary()
        if "final_rank_mean_by_layer_type" in info["peft"]:
            log(
                "[peft] AdaLoRA final average rank by layer type: "
                + ", ".join(f"{k} {v}" for k, v in info["peft"]["final_rank_mean_by_layer_type"].items())
            )
    return info


# --- Stage B
@torch.no_grad()
def predict_stage_b(model, feats, device):
    model.eval()
    return [None if f is None else model(f.to(device)).float().cpu() for f in feats]


def train_stage_b(feats, ys, dev_feats, dev_docs, labels, cfg, device, weight, log=print, soft=None):
    """Train the BiGRU over judgments; the best epoch on dev_docs (early stopping) is kept."""
    sel, mname = cfg["select_metric"], METRIC_NAMES[cfg["select_metric"]]
    torch.manual_seed(cfg["seed"])
    feat_dim = next(f for f in feats if f is not None).size(-1)
    model = ContextClassifier(
        feat_dim,
        len(labels),
        cfg["b_hidden"],
        cfg["b_dropout"],
        cfg["b_layers"],
        arch=cfg["b_arch"],
        heads=cfg["b_heads"],
        attn_layers=cfg["b_attn_layers"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["b_lr"], weight_decay=cfg["b_weight_decay"])
    rng = random.Random(cfg["seed"])
    order = [i for i, f in enumerate(feats) if f is not None]
    arch = {
        "gru": f"BiGRU {cfg['b_hidden']}",
        "gru_attn": f"BiGRU {cfg['b_hidden']} + attention",
        "attn": f"{cfg['b_attn_layers']} attention blocks {cfg['b_hidden']}",
    }[cfg["b_arch"]]
    log(
        f"[stage B] {len(order)} judgments, input {feat_dim} -> {arch} -> {len(labels)} roles, "
        f"{count_params(model) / 1e6:.2f}M parameters"
    )
    best, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t_start = time.time()
    for ep in range(1, cfg["b_epochs"] + 1):
        model.train()
        rng.shuffle(order)
        tot, n = 0.0, 0
        for i in order:
            x, y = feats[i].to(device), ys[i].to(device)
            out = model(x)
            loss = F.cross_entropy(out, y, weight=weight)
            if soft is not None:
                loss = cfg["kd_alpha"] * loss + (1 - cfg["kd_alpha"]) * kd_loss(out, soft[i].to(device), cfg["kd_T"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        rep = D.evaluate(dev_docs, to_preds(dev_docs, predict_stage_b(model, dev_feats, device), labels), labels)
        f = rep["all"][sel]
        history.append(
            dict(epoch=ep, loss=tot / max(1, n), dev=f, **{f"dev_{k}": v[sel] for k, v in rep.items() if k != "all"})
        )
        mark = ""
        if f > best:
            best, best_epoch, bad, mark = f, ep, 0, "  <- best so far"
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
        if mark or ep == 1 or ep % 5 == 0:
            doms = ", ".join(f"{k} {v[sel]:.4f}" for k, v in rep.items() if k != "all")
            log(
                f"[stage B] epoch {ep}/{cfg['b_epochs']}  loss {tot / max(1, n):.4f}  dev {mname} {f:.4f} ({doms}){mark}"
            )
        if bad >= cfg["b_patience"]:
            log(f"[stage B] no improvement for {bad} epochs, stopping")
            break
    model.load_state_dict(best_state)
    log(f"[stage B] using epoch {best_epoch} (dev {mname} {best:.4f})")
    return model, dict(
        history=history,
        best_epoch=best_epoch,
        best_dev_score=best,
        select_metric=sel,
        train_seconds=time.time() - t_start,
    )


# --- shared run steps
def resolve_cfg(bb, overrides, defaults=None) -> dict:
    defaults = DEFAULTS if defaults is None else defaults
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}; known: {sorted(defaults)}")
    cfg = {**defaults, **overrides}
    if cfg["exact_batches"] is None:
        cfg["exact_batches"] = bool(getattr(bb, "pad_effect", 0.0) > 1e-4)
    if cfg["select_metric"] not in METRIC_NAMES:
        raise ValueError(f"select_metric must be one of {sorted(METRIC_NAMES)}")
    if cfg["method"] not in P.ALL_METHODS:
        raise ValueError(f"method must be one of {P.ALL_METHODS}")
    if cfg["method"] in P.PEFT_METHODS and cfg["lr_peft"] is None:
        cfg["lr_peft"] = P.DEFAULT_LR[cfg["method"]]
    if not 0.0 <= cfg["kd_alpha"] <= 1.0 or cfg["kd_T"] <= 0:
        raise ValueError("kd_alpha must be in [0, 1] and kd_T > 0")
    if cfg["kd_teacher"] is not None and not os.path.exists(cfg["kd_teacher"]):
        raise ValueError(f"kd_teacher file not found: {cfg['kd_teacher']}")
    from rr_head import ContextClassifier as _CC

    for a in [cfg["b_arch"]] + list(cfg["b_variants"] or []):
        if a not in _CC.ARCHS:
            raise ValueError(f"Stage B version {a!r} unknown; choose from {_CC.ARCHS}")
    if cfg["context"]:
        if cfg["context_sep_id"] is None:
            cfg["context_sep_id"] = getattr(bb, "sep_id", None)
        if cfg["context_sep_id"] is None:
            raise ValueError("context=True needs a separator token: pass context_sep_id=...")
        cfg["context_sep_id"] = int(cfg["context_sep_id"])
    return cfg


def check_input_length(bb, n: int) -> None:
    """Runs the model once on n tokens, on a CPU copy (a too-long input on the GPU would break the session)."""
    m = copy.deepcopy(bb.model).cpu().float().eval()
    clf = bb.classifier(2, model=m)
    tok = 5 if (bb.vocab or 0) > 5 else 0
    ids = torch.full((1, n), tok, dtype=torch.long)
    try:
        with torch.no_grad():
            clf.encoder(ids, torch.ones_like(ids))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"the model cannot take {n} tokens ({type(e).__name__}: {e}); lower context_tokens or max_len"
        ) from e
    finally:
        del m, clf


def start_run(cfg, kind="run"):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"===== {kind} '{cfg['run_name']}' on {device} -> {out}")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    return device, out, log


def prepare(bb, cfg, log=print):
    """Labels and train/dev/test judgments, tokenised (d["ids"]) and checked against the model."""
    labels = D.load_labels(cfg["data_root"])
    docs = D.load_docs(cfg["data_root"])
    D.check_labels(docs, labels)
    train = [d for d in docs if d["split"] == "train"]
    dev = [d for d in docs if d["split"] == "dev"]
    test = [d for d in docs if d["split"] == "test"]
    if cfg["limit_train_docs"]:
        k = cfg["limit_train_docs"]
        train = [
            d for dom in sorted({x["domain"] for x in train}) for d in [x for x in train if x["domain"] == dom][:k]
        ]
        log(f"[data] quick check: only {len(train)} training judgments")
    log(D.summary(train + dev + test))
    raw = D.encode_docs(
        train + dev + test, bb.encode, cfg["max_len"], bos_id=bb.bos_id, eos_id=bb.eos_id, fallback_id=bb.pad_id
    )
    log("[data] " + D.length_report(raw, cfg["max_len"]))
    top = max((max(s) for d in train + dev + test for s in d["ids"]), default=0)
    if bb.vocab is not None and top >= bb.vocab:
        raise ValueError(f"token id {top} is >= the model's vocabulary size {bb.vocab}; wrong tokenizer?")
    if cfg["context"]:
        docs_all = train + dev + test
        lens = item_lengths(docs_all, flat(docs_all, cfg["pos_buckets"]), cfg)
        n_max = max(lens, default=0)
        check_input_length(bb, n_max)
        lens.sort()
        log(
            f"[data] with neighbour context (up to {cfg['context_tokens']} tokens of each neighbour, separator id "
            f"{cfg['context_sep_id']}): tokens per input median {lens[len(lens) // 2]}, longest {n_max} (fits)"
        )
    return labels, train, dev, test


def build_stage_a(bb, labels, cfg, device, train_docs, log=print):
    """A fresh copy of the Phase 1 model + head, with the chosen method; returns (clf, trainable, total)."""
    try:
        fresh = copy.deepcopy(bb.model)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"could not copy the model ({e}); build it again before each run") from e
    hooks = None
    if cfg["method"] in P.PEFT_METHODS:
        total = stage_a_steps(train_docs, cfg) * cfg["epochs"]
        hooks = P.apply_peft(fresh, cfg, bb.d_model, bb.vocab, total, log)
    clf = bb.classifier(
        len(labels),
        model=fresh,
        pos_buckets=cfg["pos_buckets"],
        pos_dim=cfg["pos_dim"],
        head=cfg["head"],
        dropout=cfg["dropout"],
    )
    clf = clf.to(device).float()
    if hooks is not None:
        P.attach(clf, hooks)
    log(
        f"[setup] fresh copy of the Phase 1 weights; hidden states: {bb.route}; pooling '{clf.pool.kind}'; "
        f"position buckets {cfg['pos_buckets']}; head {cfg['head']}"
    )
    log(
        "[setup] batches: "
        + (
            "only sentences of exactly the same length (no padding)"
            if cfg["exact_batches"]
            else "similar lengths, padded at the end"
        )
    )
    if cfg["method"] == "frozen":
        freeze_backbone(clf)
    freeze_unused(clf, train_docs, cfg, device, bb.pad_id, log)
    n_trainable, n_total = count_params(clf), count_params(clf, trainable_only=False)
    log(
        f"[setup] method '{cfg['method']}': {n_trainable / 1e6:.2f}M trainable of {n_total / 1e6:.2f}M "
        f"parameters ({n_trainable / n_total:.1%})"
    )
    return clf, n_trainable, n_total


def _save_stage(out, name, dev_docs, dev_preds, test_docs, test_preds, rep, labels):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    D.write_submission(test_docs, test_preds, labels, d)
    with open(os.path.join(d, "dev_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(dev_preds, f)
    D.confusion_csv(rep["all"]["confusion"], labels, os.path.join(d, "dev_confusion.csv"))


def _fit_summary(rep):
    return {
        k: dict(accuracy=v["accuracy"], macro_f1_all13=v["macro_f1_all13"], macro_f1=v["macro_f1"])
        for k, v in rep.items()
    }


# --- run
def run(bb, **overrides) -> dict:
    cfg = resolve_cfg(bb, overrides)
    sel = cfg["select_metric"]
    device, out, log = start_run(cfg)
    labels, train, dev, test = prepare(bb, cfg, log)
    soft = teacher_targets(cfg["kd_teacher"], train, labels) if cfg["kd_teacher"] else None
    if soft is not None:
        log(f"[kd] teacher probabilities for all {len(train)} training judgments from {cfg['kd_teacher']}")
    clf, n_trainable, n_total = build_stage_a(bb, labels, cfg, device, train, log)

    # --- Stage A
    _reset_peak(device)
    info_a = train_stage_a(clf, train, dev, labels, cfg, device, bb.pad_id, log, soft=soft)
    peak_a = _gb(device)
    weight = info_a.pop("weight_tensor")
    t_inf = time.time()
    out_a = {"dev": run_sentences(clf, dev, cfg, device, bb.pad_id)}
    infer_s = time.time() - t_inf
    out_a["test"] = run_sentences(clf, test, cfg, device, bb.pad_id)
    preds_a = {s: to_preds(ds, out_a[s], labels) for s, ds in (("dev", dev), ("test", test))}
    rep_a = D.evaluate(dev, preds_a["dev"], labels)
    rep_fit = D.evaluate(train, to_preds(train, run_sentences(clf, train, cfg, device, bb.pad_id), labels), labels)
    log(
        f"[stage A] on its own training sentences: accuracy {rep_fit['all']['accuracy']:.3f}, 13-role avg "
        f"{rep_fit['all']['macro_f1_all13']:.3f} | on dev: accuracy {rep_a['all']['accuracy']:.3f}, 13-role avg "
        f"{rep_a['all']['macro_f1_all13']:.3f}  (a big gap = memorising)"
    )
    _save_stage(out, "stageA", dev, preds_a["dev"], test, preds_a["test"], rep_a, labels)
    saved_mb = None
    if cfg["save_model"]:
        if cfg["method"] in P.PEFT_METHODS:
            path = os.path.join(out, "stageA_adapter.pt")
            P.save_adapter(clf, path, labels, cfg)
        else:
            path = os.path.join(out, "stageA_model.pt")
            torch.save({"state_dict": clf.state_dict(), "labels": labels, "config": cfg}, path)
        saved_mb = os.path.getsize(path) / 2**20

    # --- Stage B
    feats = {
        s: run_sentences(clf, ds, cfg, device, bb.pad_id, what="features")
        for s, ds in (("train", train), ("dev", dev), ("test", test))
    }
    torch.save(
        {
            "feats": feats,
            "doc_ids": {s: [d["id"] for d in ds] for s, ds in (("train", train), ("dev", dev), ("test", test))},
        },
        os.path.join(out, "features.pt"),
    )
    clf.cpu()
    _reset_peak(device)
    li = {l: i for i, l in enumerate(labels)}
    ys = [torch.tensor([li[l] for l in d["labels"]], dtype=torch.long) for d in train]
    model_b, info_b = train_stage_b(
        feats["train"],
        ys,
        feats["dev"],
        dev,
        labels,
        cfg,
        device,
        weight,
        log,
        soft=soft if cfg["kd_stage_b"] else None,
    )
    peak_b = _gb(device)
    out_b = {
        "dev": predict_stage_b(model_b, feats["dev"], device),
        "test": predict_stage_b(model_b, feats["test"], device),
    }
    preds_b = {s: to_preds(ds, out_b[s], labels) for s, ds in (("dev", dev), ("test", test))}
    rep_b = D.evaluate(dev, preds_b["dev"], labels)
    _save_stage(out, "stageB", dev, preds_b["dev"], test, preds_b["test"], rep_b, labels)
    if cfg["save_model"]:
        torch.save(
            {"state_dict": model_b.state_dict(), "labels": labels, "config": cfg}, os.path.join(out, "stageB_model.pt")
        )

    # --- report
    log("\n" + "=" * 78)
    log(D.format_report({"Stage A": rep_a, "Stage B": rep_b}, labels))
    n_b = count_params(model_b)
    fmt_gb = lambda x: "-" if x is None else f"{x:.2f} GB"  # noqa: E731
    log(f"\n{'':10s}{'trainable params':>18s}{'peak GPU memory':>17s}{'train time':>12s}{'best epoch':>12s}")
    log(
        f"{'Stage A':10s}{n_trainable / 1e6:>17.2f}M{fmt_gb(peak_a):>17s}{info_a['train_seconds'] / 60:>10.1f}m"
        f"{info_a['best_epoch']:>12d}"
    )
    log(
        f"{'Stage B':10s}{n_b / 1e6:>17.2f}M{fmt_gb(peak_b):>17s}{info_b['train_seconds'] / 60:>10.1f}m"
        f"{info_b['best_epoch']:>12d}"
    )
    best = "stageB" if rep_b["all"][sel] >= rep_a["all"][sel] else "stageA"
    log(f"\nbest on dev ({METRIC_NAMES[sel]}): {best} -> submit {os.path.join(out, best, 'submission.zip')}")

    metrics = dict(
        run_name=cfg["run_name"],
        route=bb.route,
        kind=bb.kind,
        pooling=clf.pool.kind,
        data={
            s: dict(docs=len(ds), sentences=sum(len(d["sents"]) for d in ds))
            for s, ds in (("train", train), ("dev", dev), ("test", test))
        },
        stage_a=dict(
            dev=rep_a,
            train_fit=_fit_summary(rep_fit),
            trainable_params=n_trainable,
            total_params=n_total,
            peak_gpu_gb=peak_a,
            saved_mb=saved_mb,
            method=cfg["method"],
            dev_infer_seconds=infer_s,
            **info_a,
        ),
        stage_b=dict(dev=rep_b, trainable_params=n_b, peak_gpu_gb=peak_b, **info_b),
        best=best,
        config=cfg,
    )
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return dict(out_dir=out, best=best, metrics=metrics, stage_a=clf, stage_b=model_b, labels=labels)
