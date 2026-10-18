"""Explore the Phase 4 (CJPE) data: files and fields, cases and 0/1 balance per split, judgment length in words
and in our bpe12k tokens, duplicates across splits, tokenizer speed, the start and end of one case of each label.

Kaggle:  %run cj_explore.py        (after the setup cell that clones the repo)
"""
import collections
import glob
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

REPO = os.environ.get("CS781_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = "/kaggle/working/data/Phase_4_CJPE"
if not os.path.isdir(DATA):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="Exploration-Lab/CS-781-Capstone", repo_type="dataset",
                      local_dir=os.path.dirname(DATA), allow_patterns=["Phase_4_CJPE/*"])
print("files:")
for p in sorted(glob.glob(f"{DATA}/**/*", recursive=True)):
    if os.path.isfile(p):
        print(f"  {os.path.getsize(p) / 1e6:8.1f} MB  {p}")


def records(obj):
    if isinstance(obj, dict):
        for k in ("data", "records", "cases", "examples"):
            if isinstance(obj.get(k), list):
                return obj[k]
        return [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in obj.items()]
    return obj


def pick(keys, names):
    return next((n for n in names if n in keys), None)


splits = {}
for p in sorted(glob.glob(f"{DATA}/**/*.json", recursive=True)):
    name = os.path.basename(p).lower()
    split = "test" if "test" in name else "dev" if ("dev" in name or "val" in name) else "train"
    with open(p, encoding="utf-8") as f:
        obj = json.load(f)
    recs = records(obj)
    first = recs[0] if recs else {}
    print(f"\n{p}: top level {type(obj).__name__}, {len(recs)} records; keys of the first: {list(first)}")
    for k, v in first.items():
        s = repr(v)
        print(f"    {k}: {type(v).__name__} {s[:150]}{' ...' if len(s) > 150 else ''}")
    splits.setdefault(split, []).extend(recs)

TEXT_KEYS = ("text", "document", "case_text", "content", "judgment", "facts", "body")
LABEL_KEYS = ("label", "labels", "y", "target", "outcome", "decision", "judgment_label", "class")
docs = {}
for split, recs in splits.items():
    keys = set().union(*(r.keys() for r in recs[:50]))
    tk, lk = pick(keys, TEXT_KEYS), pick(keys, LABEL_KEYS)
    out = []
    for r in recs:
        t = r.get(tk, "")
        t = " ".join(t) if isinstance(t, list) else str(t)
        out.append(dict(id=str(r.get("id", r.get("case_id", r.get("name")))), text=t,
                        label=None if lk is None else r.get(lk)))
    docs[split] = out
    print(f"\n[{split}] text key {tk!r}, label key {lk!r}, label values: "
          f"{dict(collections.Counter(map(repr, (d['label'] for d in out))).most_common(6))}")

print(f"\n{'split':6s} {'cases':>7s} {'label 0':>9s} {'label 1':>9s} {'words: median':>14s} {'p90':>7s} "
      f"{'p99':>7s} {'max':>8s}")
for split, ds in docs.items():
    w = np.array([len(d["text"].split()) for d in ds])
    c = collections.Counter(str(d["label"]) for d in ds)
    print(f"{split:6s} {len(ds):7d} {c.get('0', 0):9d} {c.get('1', 0):9d} {int(np.median(w)):14d} "
          f"{int(np.percentile(w, 90)):7d} {int(np.percentile(w, 99)):7d} {int(w.max()):8d}")
    for lab in ("0", "1"):
        wl = [len(d["text"].split()) for d in ds if str(d["label"]) == lab]
        if wl:
            print(f"        label {lab}: median words {int(np.median(wl))}")
    short = sum(1 for d in ds if len(d["text"].split()) < 50)
    if short:
        print(f"        {short} cases with fewer than 50 words")

ids = {s: [d["id"] for d in ds] for s, ds in docs.items()}
hashes = {s: [hashlib.md5(d["text"].encode()).hexdigest() for d in ds] for s, ds in docs.items()}
for s in docs:
    print(f"[{s}] duplicate ids: {len(ids[s]) - len(set(ids[s]))}, "
          f"duplicate texts: {len(hashes[s]) - len(set(hashes[s]))}")
names = list(docs)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        print(f"[{a} vs {b}] shared ids: {len(set(ids[a]) & set(ids[b]))}, "
              f"shared texts: {len(set(hashes[a]) & set(hashes[b]))}")
tr = docs.get("train", [])
conflict = collections.defaultdict(set)
for d, h in zip(tr, hashes.get("train", [])):
    conflict[h].add(str(d["label"]))
print("train texts that appear with both labels:", sum(1 for v in conflict.values() if len(v) > 1))

for lab in ("0", "1"):
    d = next((x for x in tr if str(x["label"]) == lab), None)
    if d:
        print(f"\n===== a label-{lab} case ({d['id']}): START\n{d['text'][:500]}\n----- END\n{d['text'][-900:]}")

# ---------------------------------------------------------------- our bpe12k tokenizer
try:
    sys.path.insert(0, REPO)
    from bpe import BPETokenizer
    tok = None
    files = sorted(p for p in glob.glob(f"{REPO}/tok/**/*", recursive=True) if os.path.isfile(p))
    cands = [f"{REPO}/tok/bpe12k"] + files + sorted({os.path.splitext(p)[0] for p in files} - set(files))
    for c in cands:
        try:
            tok = BPETokenizer.load(c)
            if not tok.encode("The appeal is allowed."):
                raise ValueError("empty")
            print(f"\ntokenizer loaded from {c}")
            break
        except Exception as e:  # noqa: BLE001
            tok = None
            print(f"  tokenizer not loadable from {c}: {type(e).__name__}")
    rng = random.Random(0)
    sample = rng.sample(tr, min(300, len(tr)))
    t0 = time.time()
    n_tok = [len(tok.encode(d["text"])) for d in sample[:30]]
    sec = (time.time() - t0) / 30
    if sec > 0.5:
        sample = sample[:100]          # slow tokenizer: a smaller sample
    n_tok += [len(tok.encode(d["text"])) for d in sample[30:]]
    n_tok = np.array(n_tok)
    words = np.array([len(d["text"].split()) for d in sample])
    print(f"tokenizing: {sec:.2f} s per case on one CPU core -> all {sum(len(v) for v in docs.values())} cases "
          f"~ {sec * sum(len(v) for v in docs.values()) / 60:.0f} min on one core")
    print(f"bpe12k tokens per case ({len(sample)} random training cases): median {int(np.median(n_tok))}, "
          f"p90 {int(np.percentile(n_tok, 90))}, p99 {int(np.percentile(n_tok, 99))}, max {int(n_tok.max())}; "
          f"tokens per word {n_tok.sum() / max(1, words.sum()):.2f}")
    for L in (512, 1024, 2048, 4096, 8192):
        print(f"  cases that fit in {L:5d} tokens: {np.mean(n_tok <= L):6.1%}   "
              f"(a {L}-token window covers on average {np.mean(np.minimum(1, L / n_tok)):.0%} of a case)")
except Exception as e:  # noqa: BLE001
    print("tokenizer step failed:", type(e).__name__, e)
