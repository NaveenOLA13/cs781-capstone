"""Phase 5 mid-training (continued pretraining) of a left-to-right Phase 1 model on the LSI documents, before
fine-tuning."""

from __future__ import annotations

import json
import os
import random

import p5_paths  # noqa: F401  (Phase 4 building blocks)
import torch  # noqa: E402

import cj_mid as M  # noqa: E402
import lsi_data as L  # noqa: E402
from cj_models import load_model, load_tokenizer  # noqa: E402

MID_DEFAULTS = {**M.MID_DEFAULTS, "data_root": L.DEFAULT_ROOT, "out_dir": "/kaggle/working/mid5", "tokens": 40_000_000}


def split_texts(data: dict, cfg: dict) -> tuple[list[str], list[str]]:
    """(texts to train on, held-out texts): training documents minus a held-out sample, plus the dev documents if
    use_dev_text; test documents never."""
    train = [d["text"] for d in data["train"]]
    order = list(range(len(train)))
    random.Random(cfg["seed"]).shuffle(order)
    k = min(cfg["heldout_docs"], len(train) // 10)
    held = [train[j] for j in order[:k]]
    pool = [train[j] for j in order[k:]] + ([d["text"] for d in data["dev"]] if cfg["use_dev_text"] else [])
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
    data = L.load_data(cfg["data_root"], fetch=cfg["fetch_data"], log=log)
    pool, held = split_texts(data, cfg)
    log(
        f"[mid] {name} from {src}: {info['arch']}, {info['params'] / 1e6:.1f}M weights; end-of-text id {eos}; "
        f"{len(pool)} LSI documents to train on (train + dev text, no labels, no test), {len(held)} held out"
    )
    res = M.mid_train_model(model, encode, eos, pool, held, cfg, log)
    res.update(source=src, model=name, data="Phase_5_LSI")
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
        json.dump(res, f, indent=1)
    log(f"[mid] saved {final} ({os.path.getsize(final) / 2**20:.0f} MB)")
    logf.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final
