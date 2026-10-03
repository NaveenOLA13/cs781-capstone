"""Tokenize the Phase-1 corpus ONCE into flat uint16 token files (one pair per source).

All data decisions live in one place: the --plan string.
    source=BUDGET:STRIDE,source=BUDGET:STRIDE,...
    BUDGET = number of TRAIN tokens to keep (e.g. 300M, 1.5B) or "all" (read until file end)
    STRIDE = keep every k-th doc of the file (1 = every doc)

Example (current choice):
    python prepare_data.py --out data/v1 \
        --plan "acts=all:1,supreme_court=all:1,high_courts=300M:5,tribunals=200M:5"

To try a different pool later, just change --plan and --out (e.g. data/v2). Nothing else.
Note: the MIX used during training is NOT set here -- it is a weight in data.py/config.
This only decides what is stored on disk.

Output folder (attach it to every training notebook):
    {source}_train.bin, {source}_val.bin   uint16 token ids, docs separated by <eos>
    tokenizer.json                         copy of the tokenizer used (ids must match!)
    val_docs.jsonl                         raw held-out docs -> local test prompts later
    meta.json                              plan, token counts, vocab, eos id, timings
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import time

import numpy as np

SOURCE_ORDER = ["acts", "supreme_court", "high_courts", "tribunals"]


# ---------------------------------------------------------------------- plan parsing
def parse_size(s):
    s = s.strip().lower()
    if s == "all":
        return None
    mult = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9}.get(s[-1])
    return int(float(s[:-1]) * mult) if mult else int(float(s))


def parse_plan(plan):
    """'acts=all:1,high_courts=300M:5' -> {'acts': (None, 1), 'high_courts': (300000000, 5)}"""
    out = {}
    for item in plan.split(","):
        name, spec = item.split("=")
        budget, _, stride = spec.partition(":")
        out[name.strip()] = (parse_size(budget), int(stride or 1))
    return out


# ---------------------------------------------------------------------- writer
class BinWriter:
    """Buffers ids in memory and appends them to a .bin file in big chunks."""

    def __init__(self, path, flush_every=8_000_000):
        self.f = open(path, "wb")
        self.buf, self.n, self.flush_every = [], 0, flush_every

    def add(self, ids):
        self.buf.extend(ids)
        self.n += len(ids)
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if self.buf:
            np.asarray(self.buf, dtype=np.uint16).tofile(self.f)
            self.buf = []

    def close(self):
        self.flush()
        self.f.close()


# ---------------------------------------------------------------------- one source (runs in its own process)
def process_source(job):
    import corpus
    from bpe import BPETokenizer

    src, budget, stride, a = job
    tok = BPETokenizer.load(a.tokenizer)
    eos = tok.token_to_id("<eos>")
    rng = random.Random(f"{a.seed}-{src}")          # reproducible, independent per source

    tr = BinWriter(os.path.join(a.out, f"{src}_train.bin"))
    va = BinWriter(os.path.join(a.out, f"{src}_val.bin"))
    val_docs = []
    n_docs = n_val_docs = n_words = 0
    t0 = last = time.time()

    for text in corpus.iter_docs(src, local_dir=a.local_dir, min_chars=a.min_chars, stride=stride):
        ids = tok.encode(text)
        ids.append(eos)                              # mark document end
        n_docs += 1
        if rng.random() < a.val_frac:                # whole doc -> validation (no leakage)
            va.add(ids)
            n_val_docs += 1
            if len(val_docs) < a.val_docs_keep:
                val_docs.append(text)
        else:
            tr.add(ids)
            n_words += len(text.split())
        if time.time() - last > 30:
            last = time.time()
            print(f"[{src}] docs={n_docs:,} train={tr.n/1e6:.1f}M val={va.n/1e6:.2f}M "
                  f"({last - t0:.0f}s)", flush=True)
        if budget is not None and tr.n >= budget:
            break

    tr.close()
    va.close()
    stats = {"budget": budget, "stride": stride, "docs": n_docs, "val_docs": n_val_docs,
             "train_tokens": tr.n, "val_tokens": va.n,
             "tokens_per_word": round(tr.n / max(n_words, 1), 4),
             "reached_end_of_file": budget is None or tr.n < budget,
             "seconds": round(time.time() - t0)}
    print(f"[{src}] DONE {stats}", flush=True)
    return src, stats, val_docs


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help='e.g. "acts=all:1,high_courts=300M:5"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="tok/bpe12k/tokenizer.json")
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--val_docs_keep", type=int, default=100, help="raw val docs saved per source")
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4, help="sources processed in parallel")
    ap.add_argument("--local_dir", default=None, help="read local json files instead of HF")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    plan = parse_plan(a.plan)
    for s in plan:
        assert s in SOURCE_ORDER, f"unknown source {s!r}"
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(a.tokenizer, os.path.join(a.out, "tokenizer.json"))

    jobs = [(s, b, k, a) for s, (b, k) in plan.items()]
    t0 = time.time()
    ctx = mp.get_context("spawn")                    # fresh processes: safe with network file handles
    with ctx.Pool(min(a.workers, len(jobs))) as pool:
        results = pool.map(process_source, jobs)

    stats = {}
    with open(os.path.join(a.out, "val_docs.jsonl"), "w") as f:
        for src, st, docs in results:
            stats[src] = st
            for d in docs:
                f.write(json.dumps({"source": src, "text": d}) + "\n")

    from bpe import BPETokenizer
    tok = BPETokenizer.load(a.tokenizer)
    meta = {"plan": a.plan, "vocab_size": tok.get_vocab_size(), "eos_id": tok.token_to_id("<eos>"),
            "dtype": "uint16", "val_frac": a.val_frac, "seed": a.seed,
            "total_train_tokens": sum(s["train_tokens"] for s in stats.values()),
            "total_val_tokens": sum(s["val_tokens"] for s in stats.values()),
            "seconds": round(time.time() - t0), "sources": stats}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()