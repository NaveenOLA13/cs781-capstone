"""Load one of our Phase 1 models and our bpe12k tokenizer, and connect them to the Phase 4 code.

    from cj_models import load_backbone
    bb, info = load_backbone("llama")        # "decoder", "encdec", "llama", "moe", or a path to best.pt

Where each checkpoint comes from (Kaggle):
  decoder  output of notebook enteredinyourbrain/cs781-train, version 1 (try1)  runs/decoder_80m_v1/best.pt
  encdec   same notebook, version 2 (arch2)                                      runs/encdec_86m_v1/best.pt
  llama    same notebook, version 4 (llama1, the latest)                         runs/llama_85m_v1/best.pt
  moe      on a friend's Kaggle account (set SOURCES["moe"] or pass the path)
A notebook output already attached as input is used if it holds the model; otherwise kagglehub attaches the
right version. Only one version of cs781-train can be attached per Kaggle session, so load one model per session.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import tempfile
import zipfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CS781_REPO") or os.path.dirname(HERE)
for p in (os.path.join(REPO, "phase3"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from rr_backbone import make_backbone  # noqa: E402
from rr_head import count_params  # noqa: E402

SOURCES = {
    "decoder": "enteredinyourbrain/cs781-train/versions/1",
    "encdec": "enteredinyourbrain/cs781-train/versions/2",
    "llama": "enteredinyourbrain/cs781-train/versions/4",
    "moe": None,
}
RUN_PREFIX = {"decoder": "decoder_80m", "encdec": "encdec_86m", "llama": "llama_85m", "moe": "moe_214m"}
MODELS = tuple(SOURCES)
MAX_POS_KEYS = ("max_seq_len", "block_size", "max_len", "context_length", "max_position_embeddings",
                "n_positions", "src_len", "seq_len")


def _find_in(root: str, prefix: str) -> list[str]:
    hits = glob.glob(os.path.join(root, "**", "runs", f"{prefix}*", "best.pt"), recursive=True)
    return sorted(hits)


def find_checkpoint(name: str, log=print) -> str:
    """Path of the model's best.pt (a file, or a folder if Kaggle unpacked it)."""
    if os.path.exists(name):
        return name
    if name not in SOURCES:
        raise ValueError(f"model must be one of {MODELS} or a path to best.pt, got {name!r}")
    prefix = RUN_PREFIX[name]
    hits = _find_in("/kaggle/input", prefix) if os.path.isdir("/kaggle/input") else []
    if hits:
        log(f"[model] {name}: found attached checkpoint {hits[0]}")
        return hits[0]
    handle = SOURCES[name]
    if handle is None:
        raise ValueError(f"no source known for {name!r}: pass the path to its best.pt instead")
    import kagglehub
    log(f"[model] {name}: attaching {handle} with kagglehub ...")
    root = kagglehub.notebook_output_download(handle)
    hits = _find_in(root, prefix)
    if not hits:
        found = [os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "**", "best.pt"), recursive=True)]
        raise RuntimeError(f"{handle} has no runs/{prefix}*/best.pt (found: {found}). If this Kaggle session already "
                           "attached another version of cs781-train, restart the session and load this model first.")
    log(f"[model] {name}: {hits[0]}")
    return hits[0]


def _load_ckpt(path: str) -> dict:
    """torch.load; a checkpoint that Kaggle unpacked into a folder (data.pkl, data/, ...) is zipped back first."""
    if os.path.isdir(path):
        pkl = glob.glob(os.path.join(path, "**", "data.pkl"), recursive=True)
        if not pkl:
            raise ValueError(f"{path} is a folder without data.pkl: not an unpacked checkpoint")
        base = os.path.dirname(pkl[0])
        # one file per process: two grid workers may rebuild the same checkpoint at the same time
        tmp = os.path.join(tempfile.gettempdir(), f"ckpt_{os.getpid()}_" + re.sub(r"\W", "_", path)[-60:] + ".pt")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for root, _, files in os.walk(base):
                for f in files:
                    full = os.path.join(root, f)
                    z.write(full, os.path.join("archive", os.path.relpath(full, base)))
        path = tmp
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_weights(model, ck: dict, min_loaded: float = 0.95) -> float:
    sd = ck
    for key in ("model", "model_state_dict", "state_dict", "model_state"):
        if isinstance(ck.get(key), dict):
            sd = ck[key]
            break
    sd = {re.sub(r"^(module\.|_orig_mod\.)+", "", k): v for k, v in sd.items()}
    own = model.state_dict()
    matched = [k for k in own if k in sd and tuple(sd[k].shape) == tuple(own[k].shape)]
    model.load_state_dict({k: sd[k] for k in matched}, strict=False)
    frac = len(matched) / max(len(own), 1)
    if frac < min_loaded:
        missing = [k for k in own if k not in matched][:10]
        raise RuntimeError(f"only {frac:.1%} of the model's weights are in the checkpoint (missing e.g. {missing})")
    return frac


def tokenizer_candidates(repo: str = REPO) -> list[str]:
    """Paths to try with BPETokenizer.load: tok/bpe12k, every file under tok/, and those files without extension."""
    files = sorted(p for p in glob.glob(os.path.join(repo, "tok", "**", "*"), recursive=True) if os.path.isfile(p))
    first = [p for p in files if "bpe12k" in p]
    rest = [p for p in files if "bpe12k" not in p]
    stems = sorted({os.path.splitext(p)[0] for p in files} - set(files))
    out = []
    for c in [os.path.join(repo, "tok", "bpe12k")] + first + rest + stems:
        if c not in out:
            out.append(c)
    return out


def load_tokenizer(log=print):
    from bpe import BPETokenizer
    errors = []
    for c in tokenizer_candidates():
        try:
            tok = BPETokenizer.load(c)
            if not tok.encode("The appeal is allowed."):
                raise ValueError("encodes a sentence to nothing")
            log(f"[model] tokenizer: {c}")
            return tok
        except Exception as e:  # noqa: BLE001
            errors.append(f"{os.path.relpath(c, REPO)}: {type(e).__name__}")
    raise RuntimeError("could not load the bpe12k tokenizer; tried " + "; ".join(errors))


def model_max_len(cfg_model: dict) -> int | None:
    vals = [int(cfg_model[k]) for k in MAX_POS_KEYS if isinstance(cfg_model.get(k), int)]
    return min(vals) if vals else None


def load_backbone(name: str, max_len: int = 512, path: str | None = None, log=print):
    """(Backbone, info) for a Phase 1 model; the model is in float32 on the CPU."""
    from models import build_model
    path = path or find_checkpoint(name, log)
    ck = _load_ckpt(path)
    cfg = ck.get("config") or {}
    cfg_model = dict(cfg.get("model") or {})
    arch = cfg.get("arch")
    vocab = ck.get("vocab_size") or cfg_model.get("vocab_size")
    if arch is None or vocab is None:
        raise RuntimeError(f"{path}: checkpoint has no config['arch'] / vocab_size")
    model = build_model(arch, cfg_model, vocab)
    frac = _load_weights(model, ck)
    model.eval()
    limit = model_max_len(cfg_model)
    if limit is not None and max_len > limit:
        log(f"[model] {name}: max_len {max_len} -> {limit} (the model's limit)")
        max_len = limit
    tok = load_tokenizer(log)
    bb = make_backbone(model, tok, max_len=max_len, verbose=True)
    info = dict(name=name, path=path, arch=arch, vocab=vocab, params=count_params(model, False),
                loaded=frac, max_len=max_len, model_limit=limit, route=bb.route, kind=bb.kind, pooling=bb.pooling,
                config={k: v for k, v in cfg_model.items() if isinstance(v, (int, float, str, bool))})
    log(f"[model] {name}: {arch}, {info['params'] / 1e6:.1f}M weights ({frac:.0%} loaded), input up to "
        f"{max_len} tokens, {bb.kind} -> pooling '{bb.pooling}'")
    return bb, info
