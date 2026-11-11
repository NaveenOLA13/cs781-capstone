"""Ensemble several fine-tuned NER runs by averaging their token tag probabilities."""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "phase1")]
from bpe import BPETokenizer  # noqa: E402
from phase2 import ner_data as D  # noqa: E402
from phase2.finalize import finalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out_dir", default="runs_ner")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "phase1", "tok", "bpe12k", "tokenizer.json"))
    ap.add_argument("--postproc", type=int, default=1)
    a = ap.parse_args()

    w = a.weights or [1.0] * len(a.runs)
    split = set()
    for r in a.runs:
        ra = json.load(open(os.path.join(r, "args.json")))
        assert not ra.get("train_on_all"), f"{r} was trained on dev docs; cannot ensemble-select on dev"
        split.add((ra.get("split_seed", 0), ra.get("dev_frac", 0.15)))
    assert len(split) == 1, f"runs use different dev splits: {split}"
    seed, frac = split.pop()

    tok = BPETokenizer.load(a.tokenizer)
    labels = D.load_labels(a.data_dir)
    tags, tag2id = D.make_tagset(labels)
    id2tag = {i: t for t, i in tag2id.items()}
    texts, gold = D.load_train(a.data_dir)
    enc = D.encode_docs(tok, texts, gold, tag2id)
    train_ids, dev_ids = D.split_docs(list(texts), frac, seed)
    table = D.learn_boundary_table(enc, texts, gold, train_ids)
    test = D.load_test(a.data_dir)
    tenc = D.encode_docs(tok, test)

    dev_p = {k: 0 for k in dev_ids}
    test_p = {k: 0 for k in test}
    for r, wi in zip(a.runs, w):
        z = np.load(os.path.join(r, "probs.npz"))
        for k in dev_ids:
            dev_p[k] = dev_p[k] + wi * z[f"dev/{k}"].astype(np.float32)
        for k in test:
            test_p[k] = test_p[k] + wi * z[f"test/{k}"].astype(np.float32)
        print(f"[ens] + {r} (w={wi})")
    out = os.path.join(a.out_dir, a.run)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    finalize(
        dev_p,
        test_p,
        enc,
        tenc,
        texts,
        gold,
        test,
        train_ids,
        dev_ids,
        labels,
        tags,
        id2tag,
        table,
        out,
        postproc=a.postproc,
    )


if __name__ == "__main__":
    main()
