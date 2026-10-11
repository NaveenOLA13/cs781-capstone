"""Fine-tune a pretrained encoder for Phase_2_LNER (BIO token classification).

Kaggle (GPU T4 x2 or P100, Internet ON), from the folder holding lner_core.py:

  # 1) smoke test, ~5 min: checks data loading, offsets, training, output
  !python train_lner.py --data_dir <data folder> --model microsoft/deberta-v3-small \
        --limit_docs 40 --epochs 1 --out runs/smoke

  # 2) main run: train on 6/7 of the training docs, validate on 1/7 (~14%)
  !python train_lner.py --data_dir <data folder> --model microsoft/deberta-v3-large \
        --grad_ckpt --out runs/dl_f0

Outputs in --out: model/ (best checkpoint), val_probs.pkl, test_probs.pkl,
postproc.json (tuned per-label thresholds + post-processing), val_report.json,
submission.jsonl.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lner_core as C  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--train_files", nargs="*")
    ap.add_argument("--test_files", nargs="*")
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--out", default="runs/run")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384, help="step between windows (overlap = max_len - stride)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="default: 1.5e-5 for large models, 3e-5 otherwise")
    ap.add_argument("--head_lr", type=float, default=None, help="default: 10 x lr")
    ap.add_argument("--llrd", type=float, default=0.9, help="layer-wise lr decay per layer from the top")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--neg_keep", type=float, default=0.3, help="fraction of entity-free windows kept per epoch")
    ap.add_argument("--ema", type=float, default=0.995, help="0 disables")
    ap.add_argument("--fgm", type=float, default=0.0, help="adversarial epsilon on embeddings, e.g. 0.5; 0 disables")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (needed for large on 16 GB)")
    ap.add_argument("--n_folds", type=int, default=7)
    ap.add_argument("--fold", type=int, default=0, help="-1 = train on ALL docs (no validation)")
    ap.add_argument("--fold_seed", type=int, default=42, help="keep fixed so folds match across runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--postproc", help="postproc.json to use for test decoding when --fold -1")
    ap.add_argument("--limit_docs", type=int, default=0)
    ap.add_argument("--no_test", action="store_true")
    return ap.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def special_affixes(tok):
    """Special tokens the model expects around a sequence, found empirically so it works
    for any tokenizer (BERT/DeBERTa [CLS]..[SEP], RoBERTa <s>..</s>, Llama <s>...)."""
    w = tok("hello world", add_special_tokens=True)["input_ids"]
    wo = tok("hello world", add_special_tokens=False)["input_ids"]
    for i in range(len(w) - len(wo) + 1):
        if w[i:i + len(wo)] == wo:
            return w[:i], w[i + len(wo):]
    raise ValueError("could not locate special tokens")


def encode_docs(tok, docs, scheme, L, stride, with_labels):
    out = []
    for d in docs:
        e = tok(d.text, add_special_tokens=False, return_offsets_mapping=True)
        offs = C.normalize_offsets(d.text, e["offset_mapping"])
        lab = C.spans_to_bio(offs, d.spans, scheme) if with_labels else None
        out.append({"ids": e["input_ids"], "offs": offs, "labels": lab,
                    "wins": C.make_windows(len(e["input_ids"]), L, stride)})
    return out


def make_collate(encs, prefix, suffix, pad_id, with_labels):
    def collate(items):
        seqs, labs = [], []
        for di, s, e in items:
            seqs.append(prefix + list(encs[di]["ids"][s:e]) + suffix)
            if with_labels:
                labs.append([C.IGNORE] * len(prefix) + encs[di]["labels"][s:e].tolist() + [C.IGNORE] * len(suffix))
        T = max(map(len, seqs))
        ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        lab = torch.full((len(seqs), T), C.IGNORE, dtype=torch.long)
        for k, x in enumerate(seqs):
            ids[k, :len(x)] = torch.tensor(x)
            att[k, :len(x)] = 1
            if with_labels:
                lab[k, :len(x)] = torch.tensor(labs[k])
        return {"input_ids": ids, "attention_mask": att, "labels": lab}
    return collate


def param_groups(model, lr, head_lr, decay, wd):
    n = getattr(model.config, "num_hidden_layers", 12)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"\.layers?\.(\d+)\.", name)
        if name.startswith(("classifier", "head", "score")):
            plr = head_lr
        elif m:
            plr = lr * decay ** (n - 1 - int(m.group(1)))
        elif "embed" in name:
            plr = lr * decay ** n
        else:
            plr = lr
        no_decay = p.ndim < 2 or "norm" in name.lower() or name.endswith(".bias")
        groups.setdefault((plr, 0.0 if no_decay else wd), []).append(p)
    return [{"params": ps, "lr": l, "weight_decay": w} for (l, w), ps in groups.items()]


class EMA:
    def __init__(self, model, decay):
        self.decay, self.step = decay, 0
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def swap(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                tmp = p.detach().clone()
                p.copy_(self.shadow[n].to(p.dtype))
                self.shadow[n].copy_(tmp.float())


class FGM:
    """Fast Gradient Method: perturb word embeddings along the gradient, train on both."""

    def __init__(self, model, eps):
        self.w = model.get_input_embeddings().weight
        self.eps, self.backup = eps, None

    def attack(self):
        g = self.w.grad
        if g is None:
            return False
        norm = torch.norm(g)
        if norm == 0 or not torch.isfinite(norm):
            return False
        self.backup = self.w.data.clone()
        self.w.data.add_(self.eps * g / norm)
        return True

    def restore(self):
        if self.backup is not None:
            self.w.data.copy_(self.backup)
            self.backup = None


@torch.inference_mode()
def predict(model, encs, collate, n_pre, n_labels, batch, dev, amp):
    model.eval()
    flat = [(di, wi, s, e) for di, en in enumerate(encs) for wi, (s, e) in enumerate(en["wins"])]
    outs = [[None] * len(en["wins"]) for en in encs]
    for k in range(0, len(flat), batch):
        chunk = flat[k:k + batch]
        b = collate([(di, s, e) for di, _, s, e in chunk])
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).logits
        p = torch.softmax(logits.float(), -1).cpu().numpy()
        for r, (di, wi, s, e) in enumerate(chunk):
            outs[di][wi] = p[r, n_pre:n_pre + (e - s)]
    return [C.merge_window_probs(len(en["ids"]), en["wins"], outs[di]) if en["wins"]
            else np.zeros((0, n_labels), np.float32) for di, en in enumerate(encs)]


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    set_seed(a.seed)
    t0 = time.time()

    # ---------------------------------------------------------------- data
    train_docs, test_docs, files = C.load_data(a.data_dir, a.train_files, a.test_files)
    print("files:", json.dumps(files, indent=1))
    if a.limit_docs:
        train_docs, test_docs = train_docs[:a.limit_docs], test_docs[:max(1, a.limit_docs // 4)]
    if a.no_test:
        test_docs = []
    C.data_report(train_docs, "train")
    C.data_report(test_docs, "test")
    scheme = C.LabelScheme(C.label_set(train_docs))
    if a.fold >= 0:
        folds = C.assign_folds(train_docs, a.n_folds, a.fold_seed)
        tr = [d for d in train_docs if folds[d.id] != a.fold]
        va = [d for d in train_docs if folds[d.id] == a.fold]
    else:
        folds, tr, va = None, train_docs, []
    print(f"labels: {scheme.types}\ntrain docs {len(tr)}, val docs {len(va)}, test docs {len(test_docs)}")
    rules = C.learn_trim_rules(tr)
    lr = a.lr or (1.5e-5 if "large" in a.model.lower() else 3e-5)
    head_lr = a.head_lr or 10 * lr
    json.dump({"args": vars(a), "labels": scheme.labels, "types": scheme.types, "lr": lr,
               "val_ids": [d.id for d in va], "trim_rules": rules},
              open(os.path.join(a.out, "config.json"), "w"), indent=1)

    # ----------------------------------------------------------- tokenizer
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    assert tok.is_fast, "need a fast tokenizer (offset mapping)"
    tok.model_max_length = int(1e9)  # we window ourselves; silences length warnings
    prefix, suffix = special_affixes(tok)
    L = a.max_len - len(prefix) - len(suffix)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    tr_enc = encode_docs(tok, tr, scheme, L, a.stride, True)
    va_enc = encode_docs(tok, va, scheme, L, a.stride, True)
    te_enc = encode_docs(tok, test_docs, scheme, L, a.stride, False)
    ntok = sorted(len(e["ids"]) for e in tr_enc)
    print(f"tokens/doc: median {ntok[len(ntok) // 2]}, max {ntok[-1]}; windows: train "
          f"{sum(len(e['wins']) for e in tr_enc)}, val {sum(len(e['wins']) for e in va_enc)}")

    # key test: gold -> BIO -> decode with THIS tokenizer must give 1.0
    ceil, _ = C.ceiling_check(tr, {d.id: e["offs"] for d, e in zip(tr, tr_enc)}, scheme, rules)
    C.print_scores(ceil, "CEILING (gold through pipeline; must be ~1.0)")

    # -------------------------------------------------------------- model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(scheme), id2label=dict(enumerate(scheme.labels)), label2id=scheme.id)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(dev)
    core = model
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"using {torch.cuda.device_count()} GPUs")
    opt = torch.optim.AdamW(param_groups(core, lr, head_lr, a.llrd, a.wd), lr=lr, eps=1e-6)

    items = [(di, s, e, bool((en["labels"][s:e] > 0).any())) for di, en in enumerate(tr_enc) for s, e in en["wins"]]
    pos = [it[:3] for it in items if it[3]]
    neg = [it[:3] for it in items if not it[3]]
    per_epoch = len(pos) + int(round(a.neg_keep * len(neg)))
    micro_per_epoch = math.ceil(per_epoch / a.batch)
    updates = a.epochs * math.ceil(micro_per_epoch / a.grad_accum)
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup * updates), updates)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ema = EMA(core, a.ema) if a.ema > 0 else None
    fgm = FGM(core, a.fgm) if a.fgm > 0 else None
    tr_collate = make_collate(tr_enc, prefix, suffix, pad_id, True)
    va_collate = make_collate(va_enc, prefix, suffix, pad_id, False)
    te_collate = make_collate(te_enc, prefix, suffix, pad_id, False)
    gold_va = {d.id: d.spans for d in va}
    print(f"windows/epoch {per_epoch} (pos {len(pos)}, neg kept {per_epoch - len(pos)}/{len(neg)}), "
          f"updates {updates}, lr {lr:g}, head_lr {head_lr:g}")

    best = {"f1": -1.0}

    def evaluate(tag):
        if not va:
            return
        if ema:
            ema.swap(core)
        probs = predict(model, va_enc, va_collate, len(prefix), len(scheme), a.eval_batch, dev, amp)
        prep = {d.id: C.Prepared(d.text, e["offs"], p, scheme) for d, e, p in zip(va, va_enc, probs)}
        res = C.macro_f1(gold_va, C.predict_spans(prep, scheme, rules), scheme.types)
        C.print_scores(res, f"[{tag}] val")
        if res["macro_f1"] > best["f1"]:
            best["f1"] = res["macro_f1"]
            core.save_pretrained(os.path.join(a.out, "model"))
            tok.save_pretrained(os.path.join(a.out, "model"))
            C.save_probs(os.path.join(a.out, "val_probs.pkl"),
                         {d.id: (e["offs"], p) for d, e, p in zip(va, va_enc, probs)})
            print(f"   saved new best ({best['f1']:.4f})")
        if ema:
            ema.swap(core)
        model.train()

    # ---------------------------------------------------------- training
    rng = random.Random(a.seed)
    step = 0
    for ep in range(a.epochs):
        epoch_items = pos + rng.sample(neg, int(round(a.neg_keep * len(neg))))
        rng.shuffle(epoch_items)
        batches = [epoch_items[k:k + a.batch] for k in range(0, len(epoch_items), a.batch)]
        marks = {max(0, int(len(batches) * j / a.evals_per_epoch) - 1) for j in range(1, a.evals_per_epoch + 1)}
        model.train()
        run_loss, n_loss = 0.0, 0
        for bi, items_b in enumerate(batches):
            b = {k: v.to(dev) for k, v in tr_collate(items_b).items()}

            def loss_fn():
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), b["labels"].reshape(-1),
                                       ignore_index=C.IGNORE)

            loss = loss_fn()
            scaler.scale(loss / a.grad_accum).backward()
            if fgm and fgm.attack():
                scaler.scale(loss_fn() / a.grad_accum).backward()
                fgm.restore()
            run_loss += loss.item()
            n_loss += 1
            if (bi + 1) % a.grad_accum == 0 or bi + 1 == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if ema:
                    ema.update(core)
                if step % 50 == 0:
                    print(f"ep {ep + 1} step {step}/{updates} loss {run_loss / n_loss:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} min", flush=True)
                    run_loss, n_loss = 0.0, 0
            if bi in marks:
                evaluate(f"epoch {ep + 1} {bi + 1}/{len(batches)}")
    if not va:  # full-data training: save final (EMA) weights
        if ema:
            ema.swap(core)
        core.save_pretrained(os.path.join(a.out, "model"))
        tok.save_pretrained(os.path.join(a.out, "model"))
    del opt, ema
    torch.cuda.empty_cache()

    # ------------------------------------------------- tune on validation
    pp = {"biases": {}, "flags": dict(C.DEFAULT_FLAGS)}
    if va:
        vp = C.load_probs(os.path.join(a.out, "val_probs.pkl"))
        prep = {d.id: C.Prepared(d.text, *vp[d.id], scheme) for d in va}
        pp = C.tune(prep, gold_va, scheme, rules)
        pred = C.predict_spans(prep, scheme, rules, pp)
        res = C.macro_f1(gold_va, pred, scheme.types)
        C.print_scores(res, "VAL after threshold tuning + post-processing")
        rep = C.error_report(gold_va, pred)
        print(json.dumps(rep, indent=1))
        json.dump({"scores": res, "errors": rep}, open(os.path.join(a.out, "val_report.json"), "w"), indent=1)
    elif a.postproc:
        pp = json.load(open(a.postproc))
    json.dump(pp, open(os.path.join(a.out, "postproc.json"), "w"), indent=1)

    # --------------------------------------------------------------- test
    if test_docs:
        if va:  # reload the best checkpoint
            del model, core
            torch.cuda.empty_cache()
            model = AutoModelForTokenClassification.from_pretrained(os.path.join(a.out, "model")).to(dev)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        probs = predict(model, te_enc, te_collate, len(prefix), len(scheme), a.eval_batch, dev, amp)
        C.save_probs(os.path.join(a.out, "test_probs.pkl"),
                     {d.id: (e["offs"], p) for d, e, p in zip(test_docs, te_enc, probs)})
        prep = {d.id: C.Prepared(d.text, e["offs"], p, scheme) for d, e, p in zip(test_docs, te_enc, probs)}
        C.write_predictions(os.path.join(a.out, "submission.jsonl"), test_docs,
                            C.predict_spans(prep, scheme, rules, pp))
        print(f"wrote {os.path.join(a.out, 'submission.jsonl')}")
    print(f"done in {(time.time() - t0) / 60:.1f} min; best val macro-F1 (before tuning) {best['f1']:.4f}")


if __name__ == "__main__":
    main()