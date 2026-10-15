"""Phase 3 training and submission.

Stage A  each sentence alone: Phase 1 model -> pooling -> + position -> head  (fine-tuned)
Stage B  whole judgment: frozen Stage A features x_1..x_S -> BiGRU -> head
Both are scored on dev (overall, CL, IT, per role) and both get a Codabench zip.

    from rr_backbone import make_backbone
    from rr_train import run
    bb = make_backbone(model, tokenizer)
    res = run(bb, run_name="encdec_full")

Every run trains a fresh copy of the Phase 1 weights, so runs never affect each other.
Outputs in out_dir/run_name/: log.txt, metrics.json, stageA/ and stageB/ (submission.zip,
prediction.json, dev_predictions.json, dev_confusion.csv), stageA_model.pt, stageB_model.pt,
features.pt.
"""
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
from rr_head import ContextClassifier, count_params, freeze_backbone

# the fp16 loss scaler skips its first step(s) on purpose; the scheduler warning about it is harmless
warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")

DEFAULTS = dict(
    data_root=D.DEFAULT_ROOT, out_dir="/kaggle/working/rr_runs", run_name="run", device=None, seed=42,
    limit_train_docs=0,          # >0: only the first k training judgments per domain (quick check)
    # inputs
    max_len=128, pos_buckets=10, pos_dim=32,
    # Stage A
    method="full",               # "full" fine-tuning or "frozen" backbone (head only)
    head="linear", dropout=0.1, epochs=5, batch_size=64, eval_batch_size=128,
    lr_backbone=5e-5, lr_head=1e-3, weight_decay=0.01, warmup=0.06, grad_clip=1.0, amp=True,
    weight_cap=10.0, weight_power=0.5, log_every=100,
    exact_batches=None,          # None: automatic (True if padding changes the model's outputs)
    # Stage B
    b_epochs=40, b_patience=10, b_lr=1e-3, b_weight_decay=0.01, b_hidden=256, b_dropout=0.3, b_layers=1,
    save_model=True,
)


# --------------------------------------------------------------------------- helpers
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
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def flat(docs, buckets):
    """(doc index, sentence index, position bucket) for every sentence, in document order."""
    out = []
    for di, d in enumerate(docs):
        n = len(d["ids"])
        out += [(di, si, D.pos_bucket(si, n, buckets)) for si in range(n)]
    return out


def _batch(docs, items, b, pad_id, device):
    ids, mask = collate([docs[items[k][0]]["ids"][items[k][1]] for k in b], pad_id, device)
    pos = torch.tensor([items[k][2] for k in b], dtype=torch.long, device=device)
    return ids, mask, pos


def to_preds(docs, outputs, labels):
    """Per-document score tensors -> {doc_id: [role per sentence]}."""
    return {d["id"]: ([] if o is None else [labels[i] for i in o.argmax(-1).tolist()])
            for d, o in zip(docs, outputs)}


def _gb(device):
    return torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else None


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


# --------------------------------------------------------------------------- Stage A
@torch.no_grad()
def run_sentences(clf, docs, cfg, device, pad_id, what="logits"):
    """Stage A over every sentence of `docs`: role scores ("logits") or sentence features
    ("features", the input of head A). Returns one tensor [S, *] per document."""
    was = clf.training
    clf.eval()
    items = flat(docs, cfg["pos_buckets"])
    lengths = [len(docs[di]["ids"][si]) for di, si, _ in items]
    out = [None] * len(items)
    for b in D.length_batches(lengths, cfg["eval_batch_size"], shuffle=False, exact=cfg["exact_batches"]):
        ids, mask, pos = _batch(docs, items, b, pad_id, device)
        with _autocast(device, cfg["amp"]):
            y = clf(ids, mask, pos=pos) if what == "logits" else clf.features(ids, mask, pos=pos)
        y = y.float().cpu()
        for j, k in enumerate(b):
            out[k] = y[j]
    clf.train(was)
    res, k = [], 0
    for d in docs:
        n = len(d["ids"])
        res.append(torch.stack(out[k:k + n]) if n else None)
        k += n
    return res


def make_optimizer(named_params, head_ids, cfg):
    groups = {}
    for _, p in named_params:
        if not p.requires_grad:
            continue
        lr = cfg["lr_head"] if id(p) in head_ids else cfg["lr_backbone"]
        wd = cfg["weight_decay"] if p.dim() >= 2 else 0.0
        groups.setdefault((lr, wd), []).append(p)
    return torch.optim.AdamW([dict(params=v, lr=k[0], weight_decay=k[1]) for k, v in groups.items()])


def warmup_linear(opt, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def freeze_unused(clf, docs, cfg, device, pad_id, log=print, n_batches=3):
    """Parameters that never get a gradient for role prediction (e.g. the decoder and LM head
    of an encoder-decoder) are frozen so they are not counted or stored as trainable."""
    clf.train()
    clf.zero_grad(set_to_none=True)
    items = flat(docs, cfg["pos_buckets"])
    lengths = [len(docs[di]["ids"][si]) for di, si, _ in items]
    for b in D.length_batches(lengths, cfg["batch_size"], shuffle=True, seed=123,
                              exact=cfg["exact_batches"])[:n_batches]:
        ids, mask, pos = _batch(docs, items, b, pad_id, device)
        with _autocast(device, cfg["amp"]):
            out = clf(ids, mask, pos=pos)
        out.float().logsumexp(-1).mean().backward()
    unused = [(n, p) for n, p in clf.named_parameters() if p.requires_grad and p.grad is None]
    for _, p in unused:
        p.requires_grad_(False)
    clf.zero_grad(set_to_none=True)
    if unused:
        parts = Counter()
        for n, p in unused:
            n = n[len("encoder.model."):] if n.startswith("encoder.model.") else n
            parts[n.split(".")[0]] += p.numel()
        log("[setup] not used for role prediction (left unchanged): " +
            ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in parts.most_common()))


def train_stage_a(clf, train_docs, dev_docs, labels, cfg, device, pad_id, log=print):
    li = {l: i for i, l in enumerate(labels)}
    items = flat(train_docs, cfg["pos_buckets"])
    y_all = [li[train_docs[di]["labels"][si]] for di, si, _ in items]
    lengths = [len(train_docs[di]["ids"][si]) for di, si, _ in items]
    w, counts = D.class_weights(y_all, len(labels), cfg["weight_cap"], cfg["weight_power"])
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    log("[stage A] class weights: " + ", ".join(f"{l} {x:.2f} (n={c})" for l, x, c in zip(labels, w, counts)))

    head_ids = {id(p) for p in clf.head_parameters()}
    opt = make_optimizer(clf.named_parameters(), head_ids, cfg)
    steps = len(D.length_batches(lengths, cfg["batch_size"], shuffle=True, exact=cfg["exact_batches"]))
    total = steps * cfg["epochs"]
    sched = warmup_linear(opt, int(cfg["warmup"] * total), total)
    scaler = _scaler(device, cfg["amp"])
    trainable = [p for p in clf.parameters() if p.requires_grad]
    log(f"[stage A] {len(items)} sentences, {steps} steps per epoch, {cfg['epochs']} epochs")

    best, best_state, best_epoch, history = -1.0, None, 0, []
    t_start = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        clf.train()
        t0, tot, seen, skipped = time.time(), 0.0, 0, 0
        batches = D.length_batches(lengths, cfg["batch_size"], shuffle=True, seed=cfg["seed"] * 1000 + ep,
                                   exact=cfg["exact_batches"])
        for step, b in enumerate(batches, 1):
            ids, mask, pos = _batch(train_docs, items, b, pad_id, device)
            y = torch.tensor([y_all[k] for k in b], dtype=torch.long, device=device)
            with _autocast(device, cfg["amp"]):
                logits = clf(ids, mask, pos=pos)
            loss = F.cross_entropy(logits.float(), y, weight=weight)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                sched.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(b)
            seen += len(b)
            if cfg["log_every"] and step % cfg["log_every"] == 0:
                log(f"  epoch {ep} step {step}/{steps}  loss {tot / max(1, seen):.4f}")
        dev_out = run_sentences(clf, dev_docs, cfg, device, pad_id)
        rep = D.evaluate(dev_docs, to_preds(dev_docs, dev_out, labels), labels)
        f = rep["all"]["macro_f1"]
        rec = dict(epoch=ep, loss=tot / max(1, seen), dev_macro_f1=f, seconds=time.time() - t0,
                   skipped_steps=skipped, **{f"dev_{k}": v["macro_f1"] for k, v in rep.items() if k != "all"})
        history.append(rec)
        mark = ""
        if f > best:
            best, best_epoch, mark = f, ep, "  <- best so far"
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in clf.state_dict().items()}
        doms = ", ".join(f"{k} {v['macro_f1']:.4f}" for k, v in rep.items() if k != "all")
        extra = f"  ({skipped} steps skipped: non-finite loss)" if skipped else ""
        log(f"[stage A] epoch {ep}/{cfg['epochs']}  loss {rec['loss']:.4f}  dev macro-F1 {f:.4f} ({doms})"
            f"  {rec['seconds']:.0f}s{mark}{extra}")
    clf.load_state_dict(best_state)
    log(f"[stage A] using epoch {best_epoch} (dev macro-F1 {best:.4f})")
    return dict(history=history, best_epoch=best_epoch, best_dev_macro_f1=best, class_weights=w,
                train_counts=dict(zip(labels, counts)), train_seconds=time.time() - t_start,
                weight_tensor=weight)


# --------------------------------------------------------------------------- Stage B
@torch.no_grad()
def predict_stage_b(model, feats, device):
    model.eval()
    return [None if f is None else model(f.to(device)).float().cpu() for f in feats]


def train_stage_b(feats, ys, dev_feats, dev_docs, labels, cfg, device, weight, log=print):
    torch.manual_seed(cfg["seed"])
    feat_dim = next(f for f in feats if f is not None).size(-1)
    model = ContextClassifier(feat_dim, len(labels), cfg["b_hidden"], cfg["b_dropout"], cfg["b_layers"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["b_lr"], weight_decay=cfg["b_weight_decay"])
    rng = random.Random(cfg["seed"])
    order = [i for i, f in enumerate(feats) if f is not None]
    log(f"[stage B] {len(order)} judgments, input {feat_dim} -> BiGRU {cfg['b_hidden']} -> {len(labels)} roles, "
        f"{count_params(model) / 1e6:.2f}M parameters")
    best, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t_start = time.time()
    for ep in range(1, cfg["b_epochs"] + 1):
        model.train()
        rng.shuffle(order)
        tot, n = 0.0, 0
        for i in order:
            x, y = feats[i].to(device), ys[i].to(device)
            loss = F.cross_entropy(model(x), y, weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        rep = D.evaluate(dev_docs, to_preds(dev_docs, predict_stage_b(model, dev_feats, device), labels), labels)
        f = rep["all"]["macro_f1"]
        history.append(dict(epoch=ep, loss=tot / max(1, n), dev_macro_f1=f,
                            **{f"dev_{k}": v["macro_f1"] for k, v in rep.items() if k != "all"}))
        mark = ""
        if f > best:
            best, best_epoch, bad, mark = f, ep, 0, "  <- best so far"
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
        if mark or ep == 1 or ep % 5 == 0:
            doms = ", ".join(f"{k} {v['macro_f1']:.4f}" for k, v in rep.items() if k != "all")
            log(f"[stage B] epoch {ep}/{cfg['b_epochs']}  loss {tot / max(1, n):.4f}  dev macro-F1 {f:.4f} ({doms}){mark}")
        if bad >= cfg["b_patience"]:
            log(f"[stage B] no improvement for {bad} epochs, stopping")
            break
    model.load_state_dict(best_state)
    log(f"[stage B] using epoch {best_epoch} (dev macro-F1 {best:.4f})")
    return model, dict(history=history, best_epoch=best_epoch, best_dev_macro_f1=best,
                       train_seconds=time.time() - t_start)


# ------------------------------------------------------------------------------- run
def _save_stage(out, name, dev_docs, dev_preds, test_docs, test_preds, rep, labels):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    D.write_submission(test_docs, test_preds, labels, d)
    with open(os.path.join(d, "dev_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(dev_preds, f)
    D.confusion_csv(rep["all"]["confusion"], labels, os.path.join(d, "dev_confusion.csv"))


def run(bb, **overrides) -> dict:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    cfg = {**DEFAULTS, **overrides}
    if cfg["exact_batches"] is None:
        cfg["exact_batches"] = bool(getattr(bb, "pad_effect", 0.0) > 1e-4)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    log = Logger(os.path.join(out, "log.txt"))
    log(f"===== run '{cfg['run_name']}' on {device} -> {out}")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    # ---------------------------------------------------------------- data
    labels = D.load_labels(cfg["data_root"])
    docs = D.load_docs(cfg["data_root"])
    D.check_labels(docs, labels)
    train = [d for d in docs if d["split"] == "train"]
    dev = [d for d in docs if d["split"] == "dev"]
    test = [d for d in docs if d["split"] == "test"]
    if cfg["limit_train_docs"]:
        k = cfg["limit_train_docs"]
        train = [d for dom in sorted({x["domain"] for x in train})
                 for d in [x for x in train if x["domain"] == dom][:k]]
        log(f"[data] quick check: only {len(train)} training judgments")
    log(D.summary(train + dev + test))
    raw = D.encode_docs(train + dev + test, bb.encode, cfg["max_len"], bos_id=bb.bos_id, eos_id=bb.eos_id,
                        fallback_id=bb.pad_id)
    log("[data] " + D.length_report(raw, cfg["max_len"]))
    top = max((max(s) for d in train + dev + test for s in d["ids"]), default=0)
    if bb.vocab is not None and top >= bb.vocab:
        raise ValueError(f"token id {top} is >= the model's vocabulary size {bb.vocab}; wrong tokenizer?")

    # ---------------------------------------------------------------- model
    try:
        fresh = copy.deepcopy(bb.model)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"could not copy the model ({e}); build it again before each run") from e
    clf = bb.classifier(len(labels), model=fresh, pos_buckets=cfg["pos_buckets"],
                        pos_dim=cfg["pos_dim"], head=cfg["head"], dropout=cfg["dropout"])
    clf = clf.to(device).float()
    log(f"[setup] fresh copy of the Phase 1 weights; hidden states: {bb.route}; pooling '{clf.pool.kind}'; "
        f"position buckets {cfg['pos_buckets']}; head {cfg['head']}")
    log("[setup] batches: " + ("only sentences of exactly the same length (no padding)" if cfg["exact_batches"]
                               else "similar lengths, padded at the end"))
    if cfg["method"] == "frozen":
        freeze_backbone(clf)
    elif cfg["method"] != "full":
        raise ValueError("method must be 'full' or 'frozen' (PEFT methods come next)")
    freeze_unused(clf, train, cfg, device, bb.pad_id, log)
    n_trainable, n_total = count_params(clf), count_params(clf, trainable_only=False)
    log(f"[setup] method '{cfg['method']}': {n_trainable / 1e6:.2f}M trainable of {n_total / 1e6:.2f}M "
        f"parameters ({n_trainable / n_total:.1%})")

    # ---------------------------------------------------------------- Stage A
    _reset_peak(device)
    info_a = train_stage_a(clf, train, dev, labels, cfg, device, bb.pad_id, log)
    peak_a = _gb(device)
    weight = info_a.pop("weight_tensor")
    out_a = {s: run_sentences(clf, ds, cfg, device, bb.pad_id) for s, ds in (("dev", dev), ("test", test))}
    preds_a = {s: to_preds(ds, out_a[s], labels) for s, ds in (("dev", dev), ("test", test))}
    rep_a = D.evaluate(dev, preds_a["dev"], labels)
    _save_stage(out, "stageA", dev, preds_a["dev"], test, preds_a["test"], rep_a, labels)
    if cfg["save_model"]:
        torch.save({"state_dict": clf.state_dict(), "labels": labels, "config": cfg},
                   os.path.join(out, "stageA_model.pt"))

    # ---------------------------------------------------------------- Stage B
    feats = {s: run_sentences(clf, ds, cfg, device, bb.pad_id, what="features")
             for s, ds in (("train", train), ("dev", dev), ("test", test))}
    torch.save({"feats": feats, "doc_ids": {s: [d["id"] for d in ds] for s, ds in
                                             (("train", train), ("dev", dev), ("test", test))}},
               os.path.join(out, "features.pt"))
    clf.cpu()                                               # free the GPU for Stage B
    _reset_peak(device)
    li = {l: i for i, l in enumerate(labels)}
    ys = [torch.tensor([li[l] for l in d["labels"]], dtype=torch.long) for d in train]
    model_b, info_b = train_stage_b(feats["train"], ys, feats["dev"], dev, labels, cfg, device, weight, log)
    peak_b = _gb(device)
    out_b = {"dev": predict_stage_b(model_b, feats["dev"], device),
             "test": predict_stage_b(model_b, feats["test"], device)}
    preds_b = {s: to_preds(ds, out_b[s], labels) for s, ds in (("dev", dev), ("test", test))}
    rep_b = D.evaluate(dev, preds_b["dev"], labels)
    _save_stage(out, "stageB", dev, preds_b["dev"], test, preds_b["test"], rep_b, labels)
    if cfg["save_model"]:
        torch.save({"state_dict": model_b.state_dict(), "labels": labels, "config": cfg},
                   os.path.join(out, "stageB_model.pt"))

    # ---------------------------------------------------------------- report
    log("\n" + "=" * 78)
    log(D.format_report({"Stage A": rep_a, "Stage B": rep_b}, labels))
    n_b = count_params(model_b)
    fmt_gb = lambda x: "-" if x is None else f"{x:.2f} GB"  # noqa: E731
    log(f"\n{'':10s}{'trainable params':>18s}{'peak GPU memory':>17s}{'train time':>12s}{'best epoch':>12s}")
    log(f"{'Stage A':10s}{n_trainable / 1e6:>17.2f}M{fmt_gb(peak_a):>17s}{info_a['train_seconds'] / 60:>10.1f}m"
        f"{info_a['best_epoch']:>12d}")
    log(f"{'Stage B':10s}{n_b / 1e6:>17.2f}M{fmt_gb(peak_b):>17s}{info_b['train_seconds'] / 60:>10.1f}m"
        f"{info_b['best_epoch']:>12d}")
    best = "stageB" if rep_b["all"]["macro_f1"] >= rep_a["all"]["macro_f1"] else "stageA"
    log(f"\nbest on dev: {best} -> submit {os.path.join(out, best, 'submission.zip')}")

    metrics = dict(run_name=cfg["run_name"], route=bb.route, kind=bb.kind, pooling=clf.pool.kind,
                   data={s: dict(docs=len(ds), sentences=sum(len(d["sents"]) for d in ds))
                         for s, ds in (("train", train), ("dev", dev), ("test", test))},
                   stage_a=dict(dev=rep_a, trainable_params=n_trainable, total_params=n_total,
                                peak_gpu_gb=peak_a, **info_a),
                   stage_b=dict(dev=rep_b, trainable_params=n_b, peak_gpu_gb=peak_b, **info_b),
                   best=best, config=cfg)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return dict(out_dir=out, best=best, metrics=metrics, stage_a=clf, stage_b=model_b, labels=labels)