"""Phase 7 mid-training (domain adaptation) of a left-to-right Phase 1 model before fine-tuning: next-token prediction
(the model's own Phase 1 objective) on the case texts of the training and dev splits (queries and candidate pools;
labels are not used). Test cases are never used. A held-out set of training-split cases (not trained on) measures the
next-token loss before and after. finish_by (UNIX time): fewer steps if time is short. The encoder-decoder is not
mid-trained (its Phase 1 objective is not one-stream next-token prediction).

    from pcr_mid import mid_train
    path = mid_train("llama", path=".../best.pt", tokens=20_000_000, finish_by=...)   # -> /kaggle/working/mid7/...
"""
from __future__ import annotations

import json
import os
import random

import p7_paths  # noqa: F401
import torch  # noqa: E402

import cj_mid as M  # noqa: E402
import pcr_data as PD  # noqa: E402
from cj_models import load_model, load_tokenizer  # noqa: E402

MID_DEFAULTS = {**M.MID_DEFAULTS, "data_root": PD.DEFAULT_ROOT, "out_dir": "/kaggle/working/mid7", "tokens": 20_000_000,
                "heldout_docs": 100}


def split_texts(data: dict, cfg: dict) -> tuple[list[str], list[str]]:
    """(texts to train on, held-out texts): every distinct case of the training and dev splits (queries + pools),
    shuffled; heldout_docs of them are kept out of training. No test case is used."""
    texts: dict = {}
    for s in ("train", "dev"):
        q, p = data[s]
        for d in q + p:
            texts.setdefault(d["id"], d["text"])
    test_ids = {d["id"] for d in data["test"][0] + data["test"][1]}
    ids = sorted(i for i in texts if i not in test_ids)        # a case that is also in a test pool stays out
    random.Random(cfg["seed"]).shuffle(ids)
    k = min(cfg["heldout_docs"], len(ids) // 10)
    return [texts[i] for i in ids[k:]], [texts[i] for i in ids[:k]]


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
    data = PD.load_data(cfg["data_root"], fetch=cfg["fetch_data"], log=log)
    pool, held = split_texts(data, cfg)
    log(f"[mid] {name} from {src}: {info['arch']}, {info['params'] / 1e6:.1f}M weights; {len(pool)} cases of the "
        f"training and dev splits (no test case), {len(held)} held out")
    mcfg = {k: v for k, v in cfg.items() if k in M.MID_DEFAULTS}
    res = M.mid_train_model(model, encode, eos, pool, held, mcfg, log)
    res.update(source=src, model=name, data="Phase_7_PCR (train + dev case texts)")
    torch.save({"config": ck.get("config"), "vocab_size": ck.get("vocab_size") or info["vocab"],
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "mid_training": res},
               final + ".tmp")
    os.replace(final + ".tmp", final)
    with open(os.path.join(out, "mid_info.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    log(f"[mid] saved {final} ({os.path.getsize(final) / 2 ** 20:.0f} MB)")
    logf.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final
