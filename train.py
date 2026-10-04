"""Pretrain a language model from scratch on the token files from prepare_data.py.

    python train.py --config configs/decoder_80m.yaml
    python train.py --config configs/decoder_80m.yaml --set train.max_steps=50 train.eval_every=25   # quick test
    python train.py --config configs/decoder_80m.yaml --resume /kaggle/input/<prev-run>/runs/<name>/last.pt

Everything that defines an experiment is in the YAML config (copied into the run folder).
Outputs in runs/<run_name>/: config.yaml, log.jsonl, last.pt (resume), best.pt (weights only).
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data import TokenData
from models import build_model, count_params, init_weights


# ---------------------------------------------------------------------- config helpers
def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for o in overrides:                                   # e.g. train.lr=3e-4
        key, val = o.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(val)
    return cfg


def lr_at(step, t):
    """Linear warmup, then cosine decay from lr to min_lr over max_steps."""
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    prog = min(1.0, (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"]))
    return t["min_lr"] + 0.5 * (t["lr"] - t["min_lr"]) * (1 + math.cos(math.pi * prog))


def make_optimizer(model, t):
    """AdamW; no weight decay on biases / LayerNorm / embeddings-free 1D params."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": t["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=t["lr"], betas=tuple(t["betas"]), eps=1e-8,
                             **({"fused": True} if fused else {}))


def lm_loss(model, x, y):
    """Models with their own objective (e.g. encoder-decoder) define compute_loss."""
    if hasattr(model, "compute_loss"):
        return model.compute_loss(x, y)
    logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def evaluate(model, data, t, device, amp_ctx):
    """Mean val loss over a FIXED set of val batches (same every eval -> comparable)."""
    model.eval()
    saved = data.rng
    data.rng = np.random.default_rng(12345)
    losses = []
    for _ in range(t["eval_batches"]):
        x, y = data.get_batch("val", t["batch_size"], device)
        with amp_ctx:
            losses.append(lm_loss(model, x, y).item())
    data.rng = saved
    model.train()
    return float(np.mean(losses))


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config keys: train.lr=3e-4")
    ap.add_argument("--resume", default=None, help="path to a last.pt to continue from")
    ap.add_argument("--data_dir", default=None, help="override data.dir")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    if a.data_dir:
        cfg["data"]["dir"] = a.data_dir
    t, d = cfg["train"], cfg["data"]
    torch.manual_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(cfg.get("out_dir", "runs"), cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w"), sort_keys=False)

    # ---- data + model
    data = TokenData(d["dir"], d["block_size"], d["weights"], seed=t["seed"])
    model = build_model(cfg["arch"], cfg["model"], data.vocab_size)
    if t.get("init_std"):
        init_weights(model, t["init_std"])
    model = model.to(device)
    n_params = count_params(model)
    print(f"[train] {cfg['arch']} params={n_params/1e6:.2f}M device={device}")
    print(f"[train] data mix: {data.describe()}")
    assert 50e6 <= n_params <= 300e6 or cfg.get("allow_any_size"), "capstone rule: 50M-300M params"

    opt = make_optimizer(model, t)
    use_amp = device == "cuda" and t.get("amp", True)
    amp_dtype = torch.bfloat16 if t.get("amp_dtype") == "bf16" else torch.float16
    amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step, best_val, tokens_seen, elapsed_prev = 0, float("inf"), 0, 0.0
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val, tokens_seen = ck["step"], ck["best_val"], ck["tokens_seen"]
        elapsed_prev = ck.get("elapsed", 0.0)
        data.rng = np.random.default_rng(t["seed"] + step)   # new windows after resume
        print(f"[train] resumed from {a.resume} at step {step}")

    tokens_per_step = t["batch_size"] * t["grad_accum"] * d["block_size"]
    print(f"[train] tokens/step={tokens_per_step:,}  max_steps={t['max_steps']:,}  "
          f"=> {tokens_per_step * t['max_steps'] / 1e9:.3f}B tokens planned")
    log = open(os.path.join(out, "log.jsonl"), "a")

    def save(path, full=True):
        state = {"model": model.state_dict(), "config": cfg, "step": step, "best_val": best_val,
                 "tokens_seen": tokens_seen, "vocab_size": data.vocab_size,
                 "elapsed": elapsed_prev + time.time() - t_start}
        if full:
            state.update(opt=opt.state_dict(), scaler=scaler.state_dict())
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)                       # never leaves a half-written file

    # ---- loop
    model.train()
    t_start = t_log = time.time()
    tok_since_log = 0
    time_limit = t.get("time_limit_hours", 11.0) * 3600
    while step < t["max_steps"]:
        lr = lr_at(step, t)
        for g in opt.param_groups:
            g["lr"] = lr

        loss_acc = 0.0
        for _ in range(t["grad_accum"]):
            x, y = data.get_batch("train", t["batch_size"], device)
            with amp_ctx:
                loss = lm_loss(model, x, y)
            scaler.scale(loss / t["grad_accum"]).backward()
            loss_acc += loss.item() / t["grad_accum"]
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += tokens_per_step
        tok_since_log += tokens_per_step

        if step % t["log_every"] == 0:
            now = time.time()
            rec = {"step": step, "tokens": tokens_seen, "loss": round(loss_acc, 4), "lr": lr,
                   "gnorm": round(gnorm, 3), "tok_per_s": int(tok_since_log / (now - t_log)),
                   "elapsed_h": round((elapsed_prev + now - t_start) / 3600, 3)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            t_log, tok_since_log = now, 0

        stop = (time.time() - t_start) > time_limit
        if step % t["eval_every"] == 0 or step == t["max_steps"] or stop:
            vl = evaluate(model, data, t, device, amp_ctx)
            rec = {"step": step, "tokens": tokens_seen, "val_loss": round(vl, 4),
                   "val_ppl": round(math.exp(vl), 2)}
            print("[eval]", json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if vl < best_val:
                best_val = vl
                save(os.path.join(out, "best.pt"), full=False)
            save(os.path.join(out, "last.pt"))
        if stop:
            print(f"[train] time limit reached at step {step}; resume with --resume {out}/last.pt")
            break

    print(f"[train] done. step={step} tokens={tokens_seen/1e9:.3f}B best_val={best_val:.4f}")


if __name__ == "__main__":
    main()