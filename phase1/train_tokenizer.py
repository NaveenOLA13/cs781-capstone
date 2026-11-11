"""Train the from-scratch byte-level BPE (bpe.py) on a balanced sample of the Phase-1 corpus."""

import argparse
import os
import time

import corpus
from bpe import BPETokenizer


def sample_texts(docs_per_source, stride):
    """Balanced sample so the vocab covers acts, SC, HC and tribunal styles."""
    texts = []
    for src in corpus.sources:
        n = 0
        for doc in corpus.iter_docs(src, stride=stride, max_docs=docs_per_source):
            texts.append(doc)
            n += 1
        print(f"[tok] {src}: {n} docs", flush=True)
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--docs_per_source", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--out", default="tok")
    a = ap.parse_args()

    t0 = time.time()
    texts = sample_texts(a.docs_per_source, a.stride)
    print(f"[tok] sampled {len(texts)} docs in {time.time() - t0:.0f}s", flush=True)

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(texts, a.vocab_size, min_frequency=a.min_frequency)
    print(f"[tok] trained in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "tokenizer.json")
    tok.save(path)
    print(f"[tok] saved {path}  vocab={tok.get_vocab_size()}  eos_id={tok.token_to_id('<eos>')}")


if __name__ == "__main__":
    main()
