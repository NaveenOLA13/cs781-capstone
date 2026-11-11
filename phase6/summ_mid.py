"""Phase 6 mid-training (domain adaptation) of a left-to-right Phase 1 model before fine-tuning: next-token prediction."""

from __future__ import annotations

import json
import os
import random

import p6_paths  # noqa: F401
import torch  # noqa: E402

import cj_mid as M  # noqa: E402
import summ_data as SD  # noqa: E402
from cj_models import load_model, load_tokenizer  # noqa: E402

MID_DEFAULTS = {
    **M.MID_DEFAULTS,
    "data_root": SD.DEFAULT_ROOT,
    "out_dir": "/kaggle/working/mid6",
    "tokens": 30_000_000,
    "heldout_frac": 0.05,
}


def split_texts(data: dict, cfg: dict) -> tuple[list[str], list[str]]:
    """(texts to train on: training documents + training summaries, shuffled; held-out texts: documents of our held-out
    part, for the loss before / after)."""
    pool = [d["document"] for d in data["train"]] + [d["summary"] for d in data["train"]]
    random.Random(cfg["seed"]).shuffle(pool)
    held = [d["document"] for d in data["held"][: cfg["heldout_docs"]]]
    return pool, held


def mid_train(name: str, path: str | None = None, **overrides) -> str:
    unknown = set(overrides) - set(MID_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = {**MID_DEFAULTS, **overrides}
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
    encode = M.make_encode(tok)
    eos = M.special_ids(tok)["eos"]
    data = SD.load_data(
        cfg["data_root"], fetch=cfg["fetch_data"], heldout_frac=cfg["heldout_frac"], seed=cfg["seed"], log=log
    )
    pool, held = split_texts(data, cfg)
    log(
        f"[mid] {name} from {src}: {info['arch']}, {info['params'] / 1e6:.1f}M weights; {len(pool)} texts "
        f"(training documents + summaries; no held-out or test text), {len(held)} held-out documents"
    )
    mcfg = {k: v for k, v in cfg.items() if k in M.MID_DEFAULTS}
    res = M.mid_train_model(model, encode, eos, pool, held, mcfg, log)
    res.update(source=src, model=name, data="Phase_6_Summ (train documents + summaries)")
    torch.save(
        {
            "config": ck.get("config"),
            "vocab_size": ck.get("vocab_size") or info["vocab"],
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "mid_training": res,
        },
        final + ".tmp",
    )
    os.replace(final + ".tmp", final)
    with open(os.path.join(out, "mid_info.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    log(f"[mid] saved {final} ({os.path.getsize(final) / 2**20:.0f} MB)")
    logf.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final
