"""Average several runs, tune per-label thresholds on their validation predictions,
decode the test set.

  python ensemble.py --data_dir <data folder> --runs runs/dl_f0 runs/dl_f0_seed2 runs/mb_f0 --out ens

Runs whose tokenizers agree are averaged token by token; otherwise every run is first
projected onto shared word units (automatic). Runs trained on different folds give
out-of-fold validation predictions, so tuning sees each training doc once per model.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lner_core as C  # noqa: E402


def reorder(probs, run_labels, labels):
    if run_labels == labels:
        return probs
    idx = [run_labels.index(l) for l in labels]
    return probs[:, idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir")
    ap.add_argument("--train_files", nargs="*")
    ap.add_argument("--test_files", nargs="*")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float)
    ap.add_argument("--out", default="ens")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train_docs, test_docs, _ = C.load_data(a.data_dir, a.train_files, a.test_files)
    by_id = {d.id: d for d in train_docs + test_docs}
    scheme = C.LabelScheme(C.label_set(train_docs))
    rules = C.learn_trim_rules(train_docs)
    w = a.weights or [1.0] * len(a.runs)

    runs = []
    for r in a.runs:
        cfg = json.load(open(os.path.join(r, "config.json")))
        val = C.load_probs(os.path.join(r, "val_probs.pkl")) if os.path.exists(os.path.join(r, "val_probs.pkl")) else {}
        test = C.load_probs(os.path.join(r, "test_probs.pkl")) if os.path.exists(os.path.join(r, "test_probs.pkl")) else {}
        runs.append({"labels": cfg["labels"], "val": val, "test": test, "model": cfg["args"]["model"]})
        print(f"{r}: {cfg['args']['model']}, val docs {len(val)}, test docs {len(test)}")

    # same tokenizer everywhere? -> token-level averaging, else word units
    def same_offsets(split):
        ids = set.intersection(*[set(r[split]) for r in runs if r[split]]) if any(r[split] for r in runs) else set()
        for i in list(ids)[:50]:
            ref = runs[0][split][i][0]
            if any(not np.array_equal(r[split][i][0], ref) for r in runs if i in r[split]):
                return False
        return True
    mode = "token" if same_offsets("test") and same_offsets("val") else "unit"
    print(f"averaging mode: {mode}")

    def combine(split):
        acc = defaultdict(list)
        for r, wt in zip(runs, w):
            for i, (offs, p) in r[split].items():
                p = reorder(np.asarray(p, np.float32), r["labels"], scheme.labels)
                if mode == "unit":
                    offs, p = C.project_to_units(by_id[i].text, offs, p, scheme)
                acc[i].append((offs, p, wt))
        return {i: C.Prepared(by_id[i].text, lst[0][0], sum(p * wt for _, p, wt in lst) / sum(wt for *_, wt in lst), scheme)
                for i, lst in acc.items()}

    val = combine("val")
    pp = {"biases": {}, "flags": dict(C.DEFAULT_FLAGS)}
    if val:
        gold = {i: by_id[i].spans for i in val}
        if mode == "unit":
            ceil, _ = C.ceiling_check([by_id[i] for i in val], {i: C.word_units(by_id[i].text) for i in val}, scheme, rules)
            C.print_scores(ceil, "word-unit CEILING (must be ~1.0)")
        C.print_scores(C.macro_f1(gold, C.predict_spans(val, scheme, rules), scheme.types), "ENSEMBLE val, untuned")
        pp = C.tune(val, gold, scheme, rules)
        pred = C.predict_spans(val, scheme, rules, pp)
        C.print_scores(C.macro_f1(gold, pred, scheme.types), "ENSEMBLE val, tuned")
        print(json.dumps(C.error_report(gold, pred), indent=1))
    json.dump(pp, open(os.path.join(a.out, "postproc.json"), "w"), indent=1)

    test = combine("test")
    if test:
        docs = [d for d in test_docs if d.id in test]
        C.write_predictions(os.path.join(a.out, "submission.jsonl"), docs, C.predict_spans(test, scheme, rules, pp))
        print(f"wrote {os.path.join(a.out, 'submission.jsonl')} ({len(docs)} docs)")


if __name__ == "__main__":
    main()