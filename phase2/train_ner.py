"""Phase 2 L-NER: fine-tune a Phase 1 backbone for token tagging, evaluate strict macro-F1 on a
held-out dev split, and write the fold3 submission.

Run from the repo root (so `models/`, `bpe.py`, `tok/` resolve):

  python phase2/train_ner.py --phase1 /kaggle/input/<p1>/runs/llama_85m_v1/best.pt \
        --method lorma_plus --run llama_lorma --out_dir runs_ner

  # knowledge distillation: fine-tuned teacher (from a previous run) -> smaller student
  python phase2/train_ner.py --phase1 .../llama_85m_v1/best.pt --keep_layers 6 --method full \
        --teacher runs_ner/moe_full/best_ner.pt --kd_alpha 0.5 --kd_T 2 --run kd_moe_to_llama6

  # only predict with a trained NER checkpoint
  python phase2/train_ner.py --predict_only runs_ner/llama_lorma/best_ner.pt --run llama_lorma

Outputs in <out_dir>/<run>/: args.json, log.jsonl, best_ner.pt, dev_scores.json,
dev_pred.json, sub/prediction.json, sub/submission.zip
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bpe import BPETokenizer                                              # noqa: E402
from phase2 import ner_data as D                                          # noqa: E402
from phase2.ner_model import build_ner_model, load_ner_checkpoint, kd_loss  # noqa: E402
from phase2.peft import count_trainable                                   # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", help="Phase 1 checkpoint (best.pt)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out_dir", default="runs_ner")
    ap.add_argument("--data_dir", default=None, help="local copy of Phase_2_LNER (else HF stream)")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "tok/bpe12k/tokenizer.json"))
    # model / adaptation
    ap.add_argument("--method", default="full", choices=["full", "head", "lora", "lorma_plus", "lorma_pi"])
    ap.add_argument("--head", default="linear", choices=["linear", "bigru"])
    ap.add_argument("--echo", type=int, default=0, help="1 = echo input (decoder-only archs)")
    ap.add_argument("--bidir", type=int, default=0, help="1 = remove causal mask (decoder/llama/moe)")
    ap.add_argument("--keep_layers", type=int, default=0, help="truncate backbone to first k blocks")
    ap.add_argument("--scratch", type=int, default=0, help="1 = random init backbone (ablation)")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--rnn_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    # data
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--dev_frac", type=float, default=0.15)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--train_on_all", type=int, default=0, help="1 = also train on dev docs (final model)")
    # optimisation
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone / adapter LR")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.06)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--o_weight", type=float, default=1.0, help="CE weight of the O tag")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    # distillation
    ap.add_argument("--teacher", default=None, help="fine-tuned NER checkpoint (best_ner.pt)")
    ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_T", type=float, default=2.0)
    # predict
    ap.add_argument("--boundary_fix", type=int, default=1, help="learned in-token boundary repair")
    ap.add_argument("--predict_only", default=None, help="NER checkpoint: skip training")
    ap.add_argument("--no_test", action="store_true", help="skip fold3 prediction")
    ap.add_argument("--max_docs", type=int, default=0, help="use only N labelled docs (smoke test)")
    return ap.parse_args()


# ------------------------------------------------------------------ batching / inference
def collate(samples, pad_id, device):
    L = max(len(s[2]) for s in samples)
    x = torch.full((len(samples), L), pad_id, dtype=torch.long)
    y = torch.full((len(samples), L), D.IGNORE, dtype=torch.long)
    for i, s in enumerate(samples):
        x[i, : len(s[2])] = torch.tensor(s[2])
        if s[3] is not None:
            y[i, : len(s[3])] = torch.tensor(s[3])
    return x.to(device), y.to(device)


@torch.no_grad()
def predict_docs(model, enc, doc_ids, n_tags, L, stride, pad_id, device, amp, bs=16):
    """Sliding windows; every token's tag distribution is the average over the windows that
    contain it, weighted by distance to the window edge (tokens near an edge have little
    context). Returns {doc: (probs [n, n_tags] np.float32)}."""
    model.eval()
    jobs = []
    for k in doc_ids:
        n = len(enc[k]["ids"])
        for s in window_starts_cached(n, L, stride):
            jobs.append((k, s, enc[k]["ids"][s:s + L], None))
    acc = {k: np.zeros((len(enc[k]["ids"]), n_tags), np.float32) for k in doc_ids}
    wsum = {k: np.zeros((len(enc[k]["ids"]), 1), np.float32) for k in doc_ids}
    for b in range(0, len(jobs), bs):
        chunk = jobs[b:b + bs]
        x, _ = collate(chunk, pad_id, device)
        with amp:
            logits = model(x)
        p = logits.float().softmax(-1).cpu().numpy()
        for i, (k, s, ids, _) in enumerate(chunk):
            m = len(ids)
            pos = np.arange(m)
            w = np.minimum(pos + 1, m - pos).astype(np.float32)[:, None]
            acc[k][s:s + m] += p[i, :m] * w
            wsum[k][s:s + m] += w
    return {k: acc[k] / np.maximum(wsum[k], 1e-6) for k in doc_ids}


def window_starts_cached(n, L, stride):
    return D.window_starts(n, L, stride)


def probs_to_spans(probs, enc, id2tag, texts=None, table=None):
    return {k: D.decode_spans(probs[k].argmax(-1).tolist(), enc[k]["spans"], id2tag,
                              texts[k] if texts else None, table) for k in probs}


# ------------------------------------------------------------------ main
def main():
    a = get_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(a.amp) and device == "cuda"
    amp = torch.autocast("cuda", dtype=torch.float16) if use_amp else torch.autocast("cpu", enabled=False)
    out = os.path.join(a.out_dir, a.run)
    os.makedirs(out, exist_ok=True)

    tok = BPETokenizer.load(a.tokenizer)
    pad_id = tok.token_to_id("<pad>")
    labels = D.load_labels(a.data_dir)
    tags, tag2id = D.make_tagset(labels)
    id2tag = {i: t for t, i in tag2id.items()}

    # ---------------- model
    if a.predict_only:
        model, ck = load_ner_checkpoint(a.predict_only, device)
        sa = ck["args"]
        a.window, a.stride, a.split_seed, a.dev_frac = sa["window"], sa["stride"], sa["split_seed"], sa["dev_frac"]
    else:
        margs = vars(a).copy()
        model, p1cfg, V = build_ner_model(margs, a.phase1, len(tags))
        assert V == tok.get_vocab_size(), "tokenizer does not match the Phase 1 checkpoint"
        model.to(device)
        maxlen = p1cfg["model"].get("max_seq_len", 10 ** 9)
        assert a.window * (2 if a.echo else 1) <= maxlen or p1cfg["arch"] == "encdec", \
            f"window{'*2 (echo)' if a.echo else ''} exceeds the backbone max_seq_len={maxlen}"
        tr, tot = count_trainable(model)
        print(f"[ner] arch={p1cfg['arch']} method={a.method} head={a.head} echo={a.echo} bidir={a.bidir} "
              f"trainable={tr/1e6:.3f}M / total={tot/1e6:.2f}M ({100*tr/tot:.2f}%) device={device}")
        json.dump({**vars(a), "trainable_params": tr, "total_params": tot},
                  open(os.path.join(out, "args.json"), "w"), indent=1)

    # ---------------- data
    t0 = time.time()
    texts, gold = D.load_train(a.data_dir)
    if a.max_docs:
        keep = sorted(texts)[: a.max_docs]
        texts, gold = {k: texts[k] for k in keep}, {k: gold[k] for k in keep}
    enc = D.encode_docs(tok, texts, gold, tag2id)
    train_ids, dev_ids = D.split_docs(list(texts), a.dev_frac, a.split_seed)
    if a.train_on_all:
        train_ids = train_ids + dev_ids
    print(f"[ner] {len(texts)} labelled docs -> train {len(train_ids)} / dev {len(dev_ids)} "
          f"(tokenized in {time.time()-t0:.0f}s)")
    dev_gold = {k: gold[k] for k in dev_ids}
    table = D.learn_boundary_table(enc, texts, gold, train_ids) if a.boundary_fix else None
    if table:
        print(f"[ner] boundary table learned on train docs: {len(table['start'])} start / "
              f"{len(table['end'])} end rules")

    def evaluate():
        probs = predict_docs(model, enc, dev_ids, len(tags), a.window, a.stride, pad_id, device,
                             amp, a.eval_batch_size)
        pred = probs_to_spans(probs, enc, id2tag, texts, table)
        return D.strict_scores(pred, dev_gold, labels), pred

    if not a.predict_only:
        train(a, model, enc, train_ids, evaluate, tags, tag2id, labels, pad_id, device, amp, use_amp,
              out, p1cfg, V)
        model, _ = load_ner_checkpoint(os.path.join(out, "best_ner.pt"), device)

    sc, pred = evaluate()
    print(f"[ner] DEV strict macro-F1={sc['macro_F1']:.4f} P={sc['macro_P']:.4f} R={sc['macro_R']:.4f}")
    for lab, v in sc["per_label"].items():
        print(f"    {lab:10s} F1={v['F1']:.3f} P={v['P']:.3f} R={v['R']:.3f} (tp={v['tp']} fp={v['fp']} fn={v['fn']})")
    json.dump(sc, open(os.path.join(out, "dev_scores.json"), "w"), indent=1)
    json.dump({k: [list(x) for x in v] for k, v in pred.items()},
              open(os.path.join(out, "dev_pred.json"), "w"))

    if not a.no_test:
        test = D.load_test(a.data_dir)
        tenc = D.encode_docs(tok, test)
        probs = predict_docs(model, tenc, list(test), len(tags), a.window, a.stride, pad_id, device,
                             amp, a.eval_batch_size)
        tpred = probs_to_spans(probs, tenc, id2tag, test, table)
        pj = D.write_submission(tpred, test, os.path.join(out, "sub"))
        n = sum(len(v) for v in tpred.values())
        print(f"[ner] wrote {pj} + submission.zip | {len(test)} test docs, {n} entities")


def train(a, model, enc, train_ids, evaluate, tags, tag2id, labels, pad_id, device, amp, use_amp,
          out, p1cfg, V):
    windows = D.make_windows(enc, train_ids, a.window, a.stride)
    teacher = None
    if a.teacher:
        teacher, tck = load_ner_checkpoint(a.teacher, device)
        assert tck["tags"] == tags, "teacher tag set differs"
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"[ner] KD teacher {a.teacher} ({sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params)"
              f" alpha={a.kd_alpha} T={a.kd_T}")

    head_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    body_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    nd = lambda ps: [p for p in ps if p.dim() < 2]
    wd = lambda ps: [p for p in ps if p.dim() >= 2]
    groups = [g for g in (
        {"params": wd(body_p), "lr": a.lr, "weight_decay": a.weight_decay},
        {"params": nd(body_p), "lr": a.lr, "weight_decay": 0.0},
        {"params": wd(head_p), "lr": a.head_lr, "weight_decay": a.weight_decay},
        {"params": nd(head_p), "lr": a.head_lr, "weight_decay": 0.0}) if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = math.ceil(len(windows) / (a.batch_size * a.grad_accum))
    total = steps_per_epoch * a.epochs
    warm = max(1, int(a.warmup_frac * total))
    cw = torch.ones(len(tags), device=device)
    cw[tag2id["O"]] = a.o_weight
    print(f"[ner] {len(windows)} train windows (L={a.window}, stride={a.stride}), "
          f"{steps_per_epoch} steps/epoch, {total} steps total")

    log = open(os.path.join(out, "log.jsonl"), "a")
    best, step = -1.0, 0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        random.shuffle(windows)
        run_loss, n_b = 0.0, 0
        for bi in range(0, len(windows), a.batch_size):
            x, y = collate(windows[bi:bi + a.batch_size], pad_id, device)
            with amp:
                logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, len(tags)), y.view(-1),
                                   weight=cw, ignore_index=D.IGNORE)
            if teacher is not None:
                with torch.no_grad(), amp:
                    tl = teacher(x)
                loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd_loss(logits, tl, y, a.kd_T)
            scaler.scale(loss / a.grad_accum).backward()
            run_loss += loss.item(); n_b += 1
            if n_b % a.grad_accum == 0 or bi + a.batch_size >= len(windows):
                lr_mult = (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * lr_mult
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], a.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    rec = {"epoch": ep, "step": step, "loss": round(run_loss / n_b, 4),
                           "lr_mult": round(lr_mult, 4), "elapsed_min": round((time.time() - t0) / 60, 1)}
                    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        sc, _ = evaluate()
        rec = {"epoch": ep, "step": step, "train_loss": round(run_loss / max(1, n_b), 4),
               "dev_macro_F1": round(sc["macro_F1"], 4), "dev_P": round(sc["macro_P"], 4),
               "dev_R": round(sc["macro_R"], 4)}
        print("[eval]", json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if sc["macro_F1"] > best or a.train_on_all:            # train_on_all: no real dev -> keep last
            best = sc["macro_F1"]
            torch.save({"model": model.state_dict(), "args": vars(a), "tags": tags,
                        "phase1_config": p1cfg, "vocab_size": V, "epoch": ep, "dev": sc},
                       os.path.join(out, "best_ner.pt"))
    print(f"[ner] best dev macro-F1 {best:.4f}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()