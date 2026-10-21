"""Mid-training (continued pretraining) of a left-to-right Phase 1 model on the CJPE judgments, before fine-tuning.

    from cj_mid import mid_train
    path = mid_train("llama", tokens=50_000_000)       # -> /kaggle/working/mid/llama_mid/best.pt
    # then fine-tune it like any Phase 1 model:  run_grid("llama", ckpt=path, out_dir="/kaggle/working/cj_runs_mid")

Objective: the model's own Phase 1 objective, next-token prediction (decoder-only, Llama, MoE). Labels are never
used. Text: the training judgments and the 994 genuine dev judgments; test text is never used. A held-out set of
training judgments (never trained on) measures the next-token loss before and after: how far the Phase 1 model was
from this kind of text, and how much mid-training closed that gap.

Each judgment is tokenised whole and followed by the end-of-text id; the stream is cut into windows of seq_len + 1
tokens (input = first seq_len, target = the same shifted by one), the windows are shuffled and `tokens` worth are
used once. AdamW, warmup + cosine decay to min_lr_frac * lr, fp16; with 2 GPUs the batch is split over both.
The saved file has the same layout as a Phase 1 best.pt (config, vocab_size, model) plus "mid_training" (what was
done), so every Phase 4 tool loads it with path=... / ckpt=....
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cj_data as D
from cj_models import load_model, load_tokenizer
from rr_backbone import make_encode, special_ids
from rr_head import first_tensor

MID_DEFAULTS = dict(
    data_root=D.DEFAULT_ROOT, out_dir="/kaggle/working/mid", fetch_data=True, use_dev_text=True,
    tokens=50_000_000,           # training tokens (one pass over this many, no window repeated)
    seq_len=512, batch_size=16,  # batch_size per GPU
    lr=1e-4, min_lr_frac=0.1, warmup_steps=100, weight_decay=0.01, beta2=0.95, grad_clip=1.0, amp=True,
    heldout_docs=300, heldout_windows=256, eval_every=500, log_every=100,
    data_parallel=True, tok_workers=4, seed=42, device=None,
)

_ENC = None
_DTYPE = np.int32


def _tok(text):
    return np.asarray(_ENC(text), dtype=_DTYPE)


def tokenize(texts, encode, workers=1, dtype=np.int32) -> list[np.ndarray]:
    global _ENC, _DTYPE
    _ENC, _DTYPE = encode, dtype
    if workers > 1 and len(texts) >= 4 * workers and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(_tok, texts, chunksize=max(1, len(texts) // (workers * 8)))
    return [_tok(t) for t in texts]


def windows(arrays: list[np.ndarray], eos_id: int | None, seq_len: int) -> np.ndarray:
    """Judgments joined into one stream (end-of-text id after each) and cut into [N, seq_len + 1] windows."""
    parts = []
    for a in arrays:
        parts.append(a)
        if eos_id is not None:
            parts.append(np.asarray([eos_id], dtype=a.dtype))
    stream = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    n = len(stream) // (seq_len + 1)
    return stream[:n * (seq_len + 1)].reshape(n, seq_len + 1)


def lm_loss(model, x, y):
    """Mean next-token cross-entropy. Works with forward(ids) -> scores for every position, and with nanoGPT-style
    forward(ids, targets) that returns (scores, loss)."""
    out = model(x)
    logits = first_tensor(out)
    if logits is None or logits.dim() != 3 or logits.shape[1] != x.shape[1]:
        out = model(x, y)
        if isinstance(out, (tuple, list)):
            for o in out:
                if torch.is_tensor(o) and o.dim() == 0:
                    return o.float()
        logits = first_tensor(out)
        if logits is None or logits.dim() != 3 or logits.shape[1] != x.shape[1]:
            raise RuntimeError("could not get next-token scores for every position from the model")
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    import contextlib
    return contextlib.nullcontext()


@torch.no_grad()
def heldout_loss(model, wins: np.ndarray, device, amp=True, batch=16) -> float:
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(wins), batch):
        w = torch.from_numpy(wins[i:i + batch].astype(np.int64)).to(device)
        with _autocast(device, amp):
            loss = lm_loss(model, w[:, :-1], w[:, 1:])
        tot += loss.item() * len(w)
        n += len(w)
    model.train(was)
    return tot / max(1, n)


def is_encoder_decoder(model: nn.Module) -> bool:
    names = {n.split(".")[-1].lower() for n, _ in model.named_children()}
    return bool({"encoder", "enc"} & names) and bool({"decoder", "dec"} & names)


def mid_train_model(model, encode, eos_id, train_texts, heldout_texts, cfg, log=print) -> dict:
    """Trains `model` in place on next-token prediction; returns what was done (losses, tokens, time)."""
    cfg = {**MID_DEFAULTS, **cfg}
    if is_encoder_decoder(model):
        raise NotImplementedError("encoder-decoder: its Phase 1 objective is not next-token prediction over one "
                                  "stream; mid-training is only done for the left-to-right models here")
    rng = random.Random(cfg["seed"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    L = cfg["seq_len"]

    held = windows(tokenize(heldout_texts, encode, cfg["tok_workers"]), eos_id, L)
    held = held[np.random.RandomState(cfg["seed"]).permutation(len(held))[:cfg["heldout_windows"]]]
    texts = list(train_texts)
    rng.shuffle(texts)
    need = int(cfg["tokens"] * 1.05) + L
    arrays, got, i, t0 = [], 0, 0, time.time()
    while got < need and i < len(texts):
        chunk = texts[i:i + 2000]
        i += len(chunk)
        new = tokenize(chunk, encode, cfg["tok_workers"])
        arrays += new
        got += sum(len(a) for a in new)
    wins = windows(arrays, eos_id, L)
    wins = wins[np.random.RandomState(cfg["seed"] + 1).permutation(len(wins))]
    log(f"[mid] tokenised {i} judgments ({got / 1e6:.1f}M tokens) in {(time.time() - t0) / 60:.1f} min -> "
        f"{len(wins)} windows of {L} tokens; held-out: {len(held)} windows from {len(heldout_texts)} judgments")

    model.to(device).float().train()
    net = model
    if cfg["data_parallel"] and n_gpu > 1:
        net = nn.DataParallel(model)
        log(f"[mid] using {n_gpu} GPUs (the batch is split between them)")
    bs = cfg["batch_size"] * max(1, n_gpu if net is not model else 1)
    steps = min(len(wins) // bs, max(1, cfg["tokens"] // (bs * L)))
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    other = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["weight_decay"]),
                             dict(params=other, weight_decay=0.0)], lr=cfg["lr"], betas=(0.9, cfg["beta2"]))

    def lr_at(s):
        if s < cfg["warmup_steps"]:
            return (s + 1) / cfg["warmup_steps"]
        prog = (s - cfg["warmup_steps"]) / max(1, steps - cfg["warmup_steps"])
        return cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and device.type == "cuda")

    before = heldout_loss(net, held, device, cfg["amp"])
    log(f"[mid] held-out next-token loss before: {before:.4f} (perplexity {math.exp(min(before, 50)):.1f}); "
        f"{steps} steps x {bs} windows x {L} tokens = {steps * bs * L / 1e6:.1f}M tokens, lr {cfg['lr']}")
    history = [dict(step=0, heldout=before)]
    t0, tot, n, skipped = time.time(), 0.0, 0, 0
    for s in range(steps):
        w = torch.from_numpy(wins[s * bs:(s + 1) * bs].astype(np.int64)).to(device)
        with _autocast(device, cfg["amp"]):
            loss = lm_loss(net, w[:, :-1], w[:, 1:])
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped += 1
            sched.step()
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item()
        n += 1
        if cfg["log_every"] and (s + 1) % cfg["log_every"] == 0:
            el = time.time() - t0
            log(f"  step {s + 1}/{steps}  loss {tot / max(1, n):.4f}  {(s + 1) * bs * L / el / 1e3:.1f}k tokens/s  "
                f"{el / 60:.1f} min, about {el / (s + 1) * (steps - s - 1) / 60:.1f} min left")
            tot, n = 0.0, 0
        if cfg["eval_every"] and (s + 1) % cfg["eval_every"] == 0 and s + 1 < steps:
            h = heldout_loss(net, held, device, cfg["amp"])
            history.append(dict(step=s + 1, heldout=h))
            log(f"[mid] step {s + 1}: held-out loss {h:.4f}")
    after = heldout_loss(net, held, device, cfg["amp"])
    history.append(dict(step=steps, heldout=after))
    minutes = (time.time() - t0) / 60
    log(f"[mid] held-out next-token loss: {before:.4f} -> {after:.4f} (perplexity {math.exp(min(before, 50)):.1f} -> "
        f"{math.exp(min(after, 50)):.1f}) after {steps * bs * L / 1e6:.1f}M tokens in {minutes:.1f} min")
    model.eval()
    return dict(objective="next-token prediction", tokens=steps * bs * L, steps=steps, batch=bs, seq_len=L,
                lr=cfg["lr"], judgments_tokenised=i, heldout_before=before, heldout_after=after, history=history,
                minutes=minutes, skipped_steps=skipped, gpus=max(1, n_gpu if net is not model else 1),
                text="CJPE training judgments + genuine dev judgments (labels not used; test text never used)")


def mid_train(name: str, path: str | None = None, **overrides) -> str:
    """Loads the Phase 1 model `name` (or the checkpoint at `path`), mid-trains it, saves out_dir/<name>_mid/best.pt
    in the Phase 1 layout and returns that path. Finished runs are reused (delete the folder to redo)."""
    cfg = {**MID_DEFAULTS, **overrides}
    unknown = set(overrides) - set(MID_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    out = os.path.join(cfg["out_dir"], f"{name}_mid")
    final = os.path.join(out, "best.pt")
    if os.path.exists(final):
        print(f"[mid] {final} exists: reusing it")
        return final
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    model, ck, src, info = load_model(name, path, log)
    tok = load_tokenizer(log)
    encode = make_encode(tok)
    eos = special_ids(tok)["eos"]
    data = D.load_data(cfg["data_root"], fetch=cfg["fetch_data"], log=log)
    train = [d["text"] for d in data["train"]]
    rng = random.Random(cfg["seed"])
    order = list(range(len(train)))
    rng.shuffle(order)
    k = min(cfg["heldout_docs"], len(train) // 10)
    heldout = [train[j] for j in order[:k]]
    pool = [train[j] for j in order[k:]] + ([d["text"] for d in data["dev"]] if cfg["use_dev_text"] else [])
    log(f"[mid] {name} from {src}: {info['arch']}, {info['params'] / 1e6:.1f}M weights; end-of-text id {eos}; "
        f"{len(pool)} judgments to train on, {len(heldout)} held out")
    res = mid_train_model(model, encode, eos, pool, heldout, cfg, log)
    res.update(source=src, model=name)
    torch.save({"config": ck.get("config"), "vocab_size": ck.get("vocab_size") or info["vocab"],
                "model": {k_: v.detach().cpu() for k_, v in model.state_dict().items()}, "mid_training": res},
               final + ".tmp")
    os.replace(final + ".tmp", final)
    with open(os.path.join(out, "mid_info.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"[mid] saved {final} ({os.path.getsize(final) / 2 ** 20:.0f} MB)")
    logf.close()
    return final
