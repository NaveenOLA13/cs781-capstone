"""Phase 5 training: one Phase 1 model + one fine-tuning technique -> the set of statute ids of each document.

    from cj_models import load_backbone
    from lsi_train import run
    bb, info = load_backbone("llama")
    res = run(bb, name="llama", technique="lorma")

Long documents: the first doc_tokens tokens of a document are cut into chunks of max_len tokens (no overlap).
Training reads up to train_chunks chunks per document (a random choice spread over the document, new every epoch),
prediction reads up to eval_chunks chunks evenly spread over it (always the first and the last).
Model: Phase 1 model -> chunk vector (same pooling as Phase 4) -> dropout -> linear layer -> one score per statute;
the document's score for a statute = the maximum over its chunks (chunk_pool="max"; "mean" and "lse" also exist),
so a statute only needs evidence in one part of the document.
Loss: binary cross-entropy per statute with pos_weight = (negatives / positives) ** pos_weight_power (capped at
pos_weight_max), so rare statutes are not ignored. The head's bias starts at each statute's training frequency.
Every epoch is scored on dev (macro-F1 at the best global threshold); the best epoch is kept. Thresholds for the
submission: global or per statute, chosen by cross-fitting on dev (lsi_data.choose_thresholds).
Knowledge distillation: student_layers keeps only the first blocks (a smaller student); kd_teacher = a teacher run's
train_probs.json (+ its labels.json): loss = (1 - kd_alpha) x gold BCE + kd_alpha x T^2 x BCE with the teacher's
probabilities softened at temperature kd_T.

Outputs in out_dir/run_name/: log.txt, config.json, labels.json, metrics.json, per_label.json (dev F1 and support
of every statute), dev_probs.json, test_probs.json (+ train_probs.json with save_train_probs), submission/
(chosen thresholds) and submission_global/ (one global threshold), trained.pt.
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cj_data as C
import cj_tech as T
import cj_train as R
import lsi_data as L
from cj_models import count_params

DEFAULTS = dict(
    data_root=L.DEFAULT_ROOT, out_dir="/kaggle/working/lsi_runs", run_name=None, device=None, seed=42,
    technique="lorma", max_len=512, doc_tokens=8192, train_chunks=4, eval_chunks=8, chunk_pool="max",
    epochs=3, batch_size=8, eval_batch_size=32, lr_body=None, lr_head=1e-3, weight_decay=0.01, warmup=0.06,
    grad_clip=1.0, amp=True, pos_weight_power=0.5, pos_weight_max=30.0, dropout=0.1, head="linear", pooling=None,
    min_pos=3, limit_train=0, fit_sample=500, ckpt_path=None, tok_workers=4, cache_dir="/kaggle/working/lsi_cache",
    log_every=200, save_weights=True, save_train_probs=False, fetch_data=True,
    student_layers=0, kd_teacher=None, kd_alpha=0.5, kd_T=2.0,
    **T.TECH_DEFAULTS,
)
POOLS = ("max", "mean", "lse")


def resolve_cfg(overrides: dict) -> dict:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **overrides}
    if cfg["technique"] not in T.TECHNIQUES:
        raise ValueError(f"technique must be one of {T.TECHNIQUES}")
    if cfg["chunk_pool"] not in POOLS:
        raise ValueError(f"chunk_pool must be one of {POOLS}")
    if cfg["lr_body"] is None:
        cfg["lr_body"] = T.BODY_LR[cfg["technique"]]
    if cfg["kd_teacher"] and not os.path.exists(cfg["kd_teacher"]):
        raise FileNotFoundError(f"kd_teacher {cfg['kd_teacher']} does not exist")
    return cfg


# --------------------------------------------------------------------------------- data
def doc_chunks(ids: list[int], n: int) -> list[list[int]]:
    """Chunks of n tokens; a short last piece (< n / 2 tokens) is replaced by the document's last n tokens
    (overlapping the previous chunk), so no text is lost and chunks stay long."""
    ch = [ids[i:i + n] for i in range(0, len(ids), n)] or [[]]
    if len(ch) > 1 and len(ch[-1]) < n // 2:
        ch[-1] = ids[-n:]
    return ch


def spread(n_chunks: int, k: int) -> list[int]:
    """k chunk indices evenly spread over the document (first and last included), or all of them."""
    if n_chunks <= k:
        return list(range(n_chunks))
    return sorted({int(round(x)) for x in np.linspace(0, n_chunks - 1, k)})


def prepare(encode, cfg, log=print, bos_id=None, eos_id=None):
    """(data, labels, {split: [[chunk ids, ...] per document]}); tokenised once and cached."""
    data = L.load_data(cfg["data_root"], fetch=cfg["fetch_data"], log=log)
    if cfg["limit_train"]:
        data["train"] = data["train"][:cfg["limit_train"]]
    labels = L.label_space(data)
    log(L.summary(data))
    n = cfg["max_len"] - (bos_id is not None) - (eos_id is not None)

    def wrap(x):
        return ([bos_id] if bos_id is not None else []) + (x or [0]) + ([eos_id] if eos_id is not None else [])

    docs = {}
    for s, ds in data.items():
        ids = C.encode_docs(ds, encode, cfg["doc_tokens"], "head", workers=cfg["tok_workers"],
                            cache_dir=cfg["cache_dir"], log=log)
        docs[s] = [[wrap(c) for c in doc_chunks(x, n)] for x in ids]
        nc = [len(c) for c in docs[s]]
        log(f"[data] {s}: chunks of {n} tokens per document (first {cfg['doc_tokens']} tokens): median "
            f"{sorted(nc)[len(nc) // 2]}, max {max(nc)}; {sum(c == 1 for c in nc) / len(nc):.0%} fit in one chunk")
    return data, labels, docs


# -------------------------------------------------------------------------------- model
def chunk_logits(clf, seqs, pad_id, device, exact: bool, amp: bool) -> torch.Tensor:
    """Scores of every chunk; with exact=True (models whose output changes with padding) only chunks of the same
    length are run together."""
    if not exact:
        ids, mask = R.collate(seqs, pad_id, device)
        with R._autocast(device, amp):
            return clf(ids, mask).float()
    out: list = [None] * len(seqs)
    groups: dict[int, list[int]] = {}
    for i, s in enumerate(seqs):
        groups.setdefault(len(s), []).append(i)
    for idx in groups.values():
        ids, mask = R.collate([seqs[i] for i in idx], pad_id, device)
        with R._autocast(device, amp):
            z = clf(ids, mask).float()
        for r, i in enumerate(idx):
            out[i] = z[r]
    return torch.stack(out)


def aggregate(z: torch.Tensor, counts: list[int], how: str) -> torch.Tensor:
    """Document scores from the scores of its chunks (chunks of a document are next to each other)."""
    out, k = [], 0
    for c in counts:
        part = z[k:k + c]
        if how == "max":
            out.append(part.max(0).values)
        elif how == "mean":
            out.append(part.mean(0))
        else:
            out.append(torch.logsumexp(part, 0) - float(np.log(c)))
        k += c
    return torch.stack(out)


def doc_batches(counts: list[int], max_chunks: int) -> list[list[int]]:
    """Consecutive documents grouped so that a group has at most max_chunks chunks (at least one document)."""
    out, cur, n = [], [], 0
    for i, c in enumerate(counts):
        if cur and n + c > max_chunks:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += c
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def predict(clf, docs, cfg, device, pad_id, exact) -> np.ndarray:
    """P(statute) for every document: eval_chunks chunks spread over the document, max (or mean / lse) over them."""
    was = clf.training
    clf.eval()
    sel = [[d[j] for j in spread(len(d), cfg["eval_chunks"])] for d in docs]
    counts = [len(s) for s in sel]
    out = []
    for b in doc_batches(counts, cfg["eval_batch_size"]):
        z = chunk_logits(clf, [c for i in b for c in sel[i]], pad_id, device, exact, cfg["amp"])
        out.append(torch.sigmoid(aggregate(z, [counts[i] for i in b], cfg["chunk_pool"])).cpu().numpy())
    clf.train(was)
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def _head_linear(clf, n_out):
    lins = [m for m in clf.modules() if isinstance(m, nn.Linear) and m.out_features == n_out]
    head_ids = {id(p) for p in clf.head_parameters()}
    lins = [m for m in lins if id(m.weight) in head_ids]
    return lins[-1] if lins else None


def load_teacher(path: str, train_ids: list[str], labels: list[int], log=print) -> tuple[np.ndarray, dict]:
    """Teacher probabilities of the training documents, in this run's statute order."""
    lab_path = os.path.join(os.path.dirname(path), "labels.json")
    with open(lab_path) as f:
        t_labels = json.load(f)
    P = L.load_probs(path, train_ids)
    pos = {x: j for j, x in enumerate(t_labels)}
    out = np.zeros((len(train_ids), len(labels)), dtype=np.float32)
    shared = [j for j, x in enumerate(labels) if x in pos]
    out[:, shared] = P[:, [pos[labels[j]] for j in shared]]
    info = dict(teacher=path, statutes_shared=len(shared), statutes=len(labels), mean_p=float(out.mean()))
    log(f"[kd] teacher {path}: probabilities for {len(train_ids)} training documents, {len(shared)} of "
        f"{len(labels)} statutes")
    return out, info


# ---------------------------------------------------------------------------------- run
def run(bb, name: str = "model", **overrides) -> dict:
    cfg = resolve_cfg(overrides)
    tech = cfg["technique"]
    cfg["run_name"] = cfg["run_name"] or f"{name}_{tech}"
    R.set_seed(cfg["seed"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    log = R.Logger(os.path.join(out, "log.txt"))
    kind = (" (KD student" + (" with a teacher)" if cfg["kd_teacher"] else ", no teacher)")
            if cfg["student_layers"] else (" (with a teacher)" if cfg["kd_teacher"] else ""))
    log(f"===== '{cfg['run_name']}': {name} + {tech}{kind} on {device} -> {out}")
    exact = bool(getattr(bb, "pad_effect", 0.0) > 1e-4)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    data, labels, docs = prepare(bb.encode, cfg, log, bb.bos_id, bb.eos_id)
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(labels, f)
    top = max(max(c) for ds in docs.values() for d in ds for c in d)
    if bb.vocab is not None and top >= bb.vocab:
        raise ValueError(f"token id {top} >= the model's vocabulary {bb.vocab}: wrong tokenizer?")
    Y_tr = L.multi_hot(data["train"], labels)
    Y_dev = L.multi_hot(data["dev"], labels)
    n_lab = len(labels)
    teacher, kd = None, None
    if cfg["kd_teacher"]:
        teacher, kd = load_teacher(cfg["kd_teacher"], [d["id"] for d in data["train"]], labels, log)
        kd.update(alpha=cfg["kd_alpha"], T=cfg["kd_T"])
        log(f"[kd] loss = {1 - cfg['kd_alpha']:.2f} x gold BCE + {cfg['kd_alpha']:.2f} x teacher term "
            f"(temperature {cfg['kd_T']})")

    # ------------------------------------------------------------------ model + technique
    model, student = R.make_model(bb, cfg, log)
    clf = bb.classifier(n_lab, model=model, head=cfg["head"], dropout=cfg["dropout"],
                        pooling=cfg["pooling"]).to(device).float()
    probe = [c for d in docs["train"][:8] for c in d[:1]]
    used = R.find_used(clf, probe, dict(cfg, exact_batches=exact), device, bb.pad_id)
    n_steps = (len(docs["train"]) + cfg["batch_size"] - 1) // cfg["batch_size"]
    total = n_steps * cfg["epochs"]
    hooks = T.apply(clf, tech, used, cfg, total, log)
    head = list(clf.head_parameters())
    for p in head:
        p.requires_grad_(True)
    body = hooks.body_params
    prior = (Y_tr.sum(0) + 0.5) / (len(Y_tr) + 1.0)
    lin = _head_linear(clf, n_lab)
    if lin is not None and lin.bias is not None:
        with torch.no_grad():
            lin.bias.copy_(torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32))
    pos = Y_tr.sum(0)
    pw = np.minimum(((len(Y_tr) - pos) / np.maximum(pos, 1)) ** cfg["pos_weight_power"], cfg["pos_weight_max"])
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    n_trainable = sum(p.numel() for p in {id(p): p for p in head + body}.values())
    n_total = count_params(clf, trainable_only=False)
    log(f"[setup] {tech}: {n_trainable / 1e6:.3f}M trained weights of {n_total / 1e6:.1f}M "
        f"({n_trainable / n_total:.2%}); {n_lab} statutes; chunk vector '{clf.pool.kind}' pooling, document score = "
        f"{cfg['chunk_pool']} over chunks; body lr {cfg['lr_body']}, head lr {cfg['lr_head']}; pos_weight "
        f"{pw.min():.1f}-{pw.max():.1f}" + ("; same-length chunk groups (padding changes this model)" if exact else ""))

    groups = {}
    for p, lr in [(p, cfg["lr_head"]) for p in head] + [(p, cfg["lr_body"]) for p in body]:
        wd = cfg["weight_decay"] if p.dim() >= 2 else 0.0
        groups.setdefault((lr, wd), []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=k[0], weight_decay=k[1]) for k, v in groups.items()])
    sched = R.warmup_linear(opt, int(cfg["warmup"] * total), total)
    scaler = R._scaler(device, cfg["amp"])
    all_params = head + body
    keep_ids = {id(p) for p in all_params}
    keep_names = [n for n, p in clf.named_parameters() if id(p) in keep_ids] + \
                 [n for n, _ in clf.named_buffers() if n.endswith("mask")]

    def snapshot():
        sd = clf.state_dict()
        return {n: sd[n].detach().to("cpu", copy=True) for n in keep_names if n in sd}

    log(f"[train] {len(docs['train'])} documents, up to {cfg['train_chunks']} chunks each per step; "
        f"{n_steps} steps per epoch x {cfg['epochs']} epochs")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    best, best_state, best_epoch, history, step = -1.0, None, 0, [], 0
    t_start = time.time()
    rng = random.Random(cfg["seed"])
    for ep in range(1, cfg["epochs"] + 1):
        clf.train()
        t0, tot, tot_kd, seen, skipped = time.time(), 0.0, 0.0, 0, 0
        order = list(range(len(docs["train"])))
        rng.shuffle(order)
        for k in range(n_steps):
            b = order[k * cfg["batch_size"]:(k + 1) * cfg["batch_size"]]
            if not b:
                continue
            hooks.on_step_start(step)
            sel = []
            for i in b:
                d = docs["train"][i]
                j = sorted(rng.sample(range(len(d)), cfg["train_chunks"])) if len(d) > cfg["train_chunks"] \
                    else list(range(len(d)))
                sel.append([d[x] for x in j])
            z = chunk_logits(clf, [c for s in sel for c in s], bb.pad_id, device, exact, cfg["amp"])
            zd = aggregate(z, [len(s) for s in sel], cfg["chunk_pool"])
            y = torch.tensor(Y_tr[b], dtype=torch.float32, device=device)
            gold = F.binary_cross_entropy_with_logits(zd, y, pos_weight=pos_weight)
            main = gold
            if teacher is not None:
                q = torch.tensor(teacher[b], dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)
                qT = torch.sigmoid((torch.log(q) - torch.log1p(-q)) / cfg["kd_T"])
                kd_term = F.binary_cross_entropy_with_logits(zd / cfg["kd_T"], qT) * cfg["kd_T"] ** 2
                main = (1 - cfg["kd_alpha"]) * gold + cfg["kd_alpha"] * kd_term
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
            tot += gold.item() * len(b)
            if teacher is not None:
                tot_kd += kd_term.item() * len(b)
            seen += len(b)
            if cfg["log_every"] and (k + 1) % cfg["log_every"] == 0:
                el = time.time() - t0
                log(f"  epoch {ep} step {k + 1}/{n_steps}  loss {tot / max(1, seen):.4f}  "
                    + (f"teacher term {tot_kd / max(1, seen):.4f}  " if teacher is not None else "")
                    + f"{el / 60:.1f} min, about {el / (k + 1) * (n_steps - k - 1) / 60:.1f} min left in this epoch")
        P_dev = predict(clf, docs["dev"], cfg, device, bb.pad_id, exact)
        t_g, m_g = L.tune_global(Y_dev, P_dev)
        rec = dict(epoch=ep, loss=tot / max(1, seen), seconds=time.time() - t0, skipped_steps=skipped,
                   dev_macro_f1=m_g, threshold=t_g, teacher_term=tot_kd / max(1, seen) if teacher is not None else None)
        mark = ""
        if m_g > best:
            best, best_epoch, mark = m_g, ep, "  <- best so far"
            best_state = snapshot()
        log(f"[train] epoch {ep}/{cfg['epochs']}  loss {rec['loss']:.4f}  dev macro-F1 {m_g:.4f} at threshold "
            f"{t_g:.2f}  {rec['seconds'] / 60:.1f} min{mark}" + (f"  ({skipped} steps skipped)" if skipped else ""))
        history.append(rec)
    train_seconds = time.time() - t_start
    peak = R._peak_gb(device)
    clf.load_state_dict(best_state, strict=False)
    log(f"[train] using epoch {best_epoch}")

    # ------------------------------------------------------------------ results
    t_inf = time.time()
    P_dev = predict(clf, docs["dev"], cfg, device, bb.pad_id, exact)
    infer_s = time.time() - t_inf
    P_test = predict(clf, docs["test"], cfg, device, bb.pad_id, exact)
    th = L.choose_thresholds(Y_dev, P_dev, cfg["min_pos"], cfg["seed"])
    fit_idx = sorted(random.Random(cfg["seed"]).sample(range(len(Y_tr)), min(cfg["fit_sample"], len(Y_tr))))
    P_fit = predict(clf, [docs["train"][i] for i in fit_idx], cfg, device, bb.pad_id, exact)
    fit = L.scores(Y_tr[fit_idx], L.apply(P_fit, th["t_global"]))
    log(L.fmt(th["dev_global"], f"[dev] global threshold {th['t_global']:.2f}: "))
    log(L.fmt(th["dev_chosen"], f"[dev] chosen ({th['kind']}):      "))
    log(f"[dev] cross-fitted macro-F1 (thresholds tuned on the other half of dev): global {th['cv_global']:.4f}, "
        f"per statute {th['cv_per_label']:.4f} -> using {th['kind']} thresholds")
    log(f"[fit] on {len(fit_idx)} of its own training documents: macro-F1 {fit['macro_f1']:.4f} "
        f"(dev {th['dev_global']['macro_f1']:.4f}; a big gap = memorising)")

    ids = {s: [d["id"] for d in data[s]] for s in data}
    L.save_probs(os.path.join(out, "dev_probs.json"), ids["dev"], P_dev)
    L.save_probs(os.path.join(out, "test_probs.json"), ids["test"], P_test)
    if cfg["save_train_probs"]:
        P_tr = predict(clf, docs["train"], cfg, device, bb.pad_id, exact)
        L.save_probs(os.path.join(out, "train_probs.json"), ids["train"], P_tr)
        log(f"[kd] saved train_probs.json ({len(P_tr)} training documents) for knowledge distillation")
    subs = dict(chosen=L.write_submission(data["test"], labels, P_test, th["thresholds"],
                                          os.path.join(out, "submission")),
                global_=L.write_submission(data["test"], labels, P_test, th["t_global"],
                                           os.path.join(out, "submission_global")))
    with open(os.path.join(out, "per_label.json"), "w") as f:
        json.dump(dict(labels=labels, dev_f1=th["dev_chosen"]["per_label_f1"], dev_support=th["dev_chosen"]["support"],
                       train_support=Y_tr.sum(0).tolist(), thresholds=[float(x) for x in th["thresholds"]]), f)
    n_test_pred = L.apply(P_test, th["thresholds"]).sum(1)
    log(f"[test] statutes per document: mean {n_test_pred.mean():.2f} (dev gold {Y_dev.sum(1).mean():.2f})")
    saved_mb = None
    if cfg["save_weights"]:
        path = os.path.join(out, "trained.pt")
        torch.save({"state": snapshot(), "config": cfg, "technique": tech, "model": name, "labels": labels}, path)
        saved_mb = os.path.getsize(path) / 2 ** 20
    metrics = dict(run_name=cfg["run_name"], model=name, technique=tech, kind=bb.kind, pooling=clf.pool.kind,
                   chunk_pool=cfg["chunk_pool"], max_len=cfg["max_len"], statutes=n_lab,
                   data={s: len(ds) for s, ds in data.items()}, history=history, best_epoch=best_epoch,
                   threshold_kind=th["kind"], t_global=th["t_global"], cv_global=th["cv_global"],
                   cv_per_label=th["cv_per_label"], dev_global=L.short(th["dev_global"]),
                   dev_chosen=L.short(th["dev_chosen"]), train_fit=L.short(fit), trainable_params=n_trainable,
                   total_params=n_total, trainable_share=n_trainable / n_total, peak_gpu_gb=peak,
                   train_seconds=train_seconds, dev_infer_seconds=infer_s, saved_mb=saved_mb,
                   technique_info=hooks.summary(), submissions=subs, student=student, kd=kd, config=cfg)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=str)
    log(f"[done] dev macro-F1 {th['dev_chosen']['macro_f1']:.4f} ({th['kind']} thresholds; cross-fitted "
        f"{max(th['cv_global'], th['cv_per_label']):.4f}) | submit {subs['chosen']}")
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
    hdr = (f"{'run':26s} {'trained':>9s} {'share':>7s} {'macro@t':>8s} {'macro':>7s} {'cv':>7s} {'thr':>9s} "
           f"{'micro':>7s} {'subset':>7s} {'ep':>3s} {'train':>7s} {'GPU GB':>7s} {'dev s':>6s}")
    lines = [hdr, "-" * len(hdr)]
    for m in sorted(rows, key=lambda m: -max(m["cv_global"], m["cv_per_label"])):
        gb = "-" if m.get("peak_gpu_gb") is None else f"{m['peak_gpu_gb']:.2f}"
        thr = f"g {m['t_global']:.2f}" if m["threshold_kind"] == "global" else "per-stat"
        lines.append(
            f"{m['run_name']:26s} {m['trainable_params'] / 1e6:8.2f}M {m['trainable_share']:7.2%} "
            f"{m['dev_global']['macro_f1']:8.4f} {m['dev_chosen']['macro_f1']:7.4f} "
            f"{max(m['cv_global'], m['cv_per_label']):7.4f} {thr:>9s} {m['dev_chosen']['micro_f1']:7.4f} "
            f"{m['dev_chosen']['subset_acc']:7.4f} {m['best_epoch']:3d} {m['train_seconds'] / 60:6.1f}m {gb:>7s} "
            f"{m['dev_infer_seconds']:6.1f}")
    lines.append("macro@t: dev macro-F1 at one global threshold; macro: with the chosen thresholds (tuned on dev, "
                 "optimistic); cv: cross-fitted (thresholds tuned on the other half of dev: the honest estimate, "
                 "used for ranking); thr: g = global threshold, per-stat = one per statute; micro / subset: with "
                 "the chosen thresholds; dev s: seconds to predict all dev documents.")
    return "\n".join(lines)
