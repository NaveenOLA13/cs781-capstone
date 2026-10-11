"""Sanity check BEFORE training: gold spans -> BIO tokens -> decoded spans must give back the gold
spans exactly (strict F1 ~ 1.0). Anything below 1.0 is the ceiling the tokenizer imposes;
the printed mismatches show which boundaries fall inside a token. Also prints data statistics.

    python phase2/check_offsets.py [--data_dir path/to/Phase_2_LNER]
"""
import argparse
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bpe import BPETokenizer                       # noqa: E402
from phase2 import ner_data as D                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "tok/bpe12k/tokenizer.json"))
    ap.add_argument("--show", type=int, default=25)
    a = ap.parse_args()

    tok = BPETokenizer.load(a.tokenizer)
    labels = D.load_labels(a.data_dir)
    tags, tag2id = D.make_tagset(labels)
    id2tag = {i: t for t, i in tag2id.items()}
    texts, gold = D.load_train(a.data_dir)

    # 1) gold text field agrees with offsets?
    bad_txt = 0
    for k, ents in gold.items():
        for s, e, l in ents:
            if not (0 <= s < e <= len(texts[k])):
                bad_txt += 1
    print(f"[check] {len(texts)} docs, {sum(map(len, gold.values()))} entities, "
          f"{bad_txt} out-of-range spans")

    # 2) round trip
    enc = D.encode_docs(tok, texts, gold, tag2id)
    tr_ids, dev_ids = D.split_docs(list(texts), 0.15, 0)
    table = D.learn_boundary_table(enc, texts, gold, tr_ids)       # learned on train part only
    print(f"[check] boundary table: start={table['start']} end={table['end']}")
    for name, tb in (("no repair", None), ("with boundary table", table)):
        pred = {k: D.decode_spans([t if t != D.IGNORE else 0 for t in enc[k]["y"]], enc[k]["spans"],
                                  id2tag, texts[k], tb) for k in texts}
        sc = D.strict_scores(pred, gold, labels)
        scd = D.strict_scores({k: pred[k] for k in dev_ids}, {k: gold[k] for k in dev_ids}, labels)
        print(f"[check] ORACLE round-trip ({name}): strict macro-F1 all={sc['macro_F1']:.4f} "
              f"dev-docs={scd['macro_F1']:.4f}  (labels absent from gold score 0 in the macro)")
    for lab, v in sc["per_label"].items():
        print(f"    {lab:10s} F1={v['F1']:.4f}  fn={v['fn']} fp={v['fp']}")
    shown = 0
    for k in texts:
        g = set(gold[k]); p = set(pred[k])
        for s, e, l in sorted(g - p):
            if shown >= a.show:
                break
            near = [x for x in p if x[2] == l and x[0] < e and s < x[1]]
            print(f"  MISS {k} {l:9s} gold={texts[k][s:e]!r} [{s},{e}]  "
                  f"decoded={[(texts[k][x[0]:x[1]], x[0], x[1]) for x in near]}")
            shown += 1

    # 3) statistics (for the data-analysis part of the report)
    cnt = Counter(l for v in gold.values() for _, _, l in v)
    print("[stats] entities per label:", dict(cnt.most_common()))
    lens = sorted(len(enc[k]["ids"]) for k in texts)
    print(f"[stats] doc length in tokens: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
    ent_tok = Counter()
    for k in texts:
        for s, e, l in gold[k]:
            ent_tok[l] += sum(1 for (a_, b_) in enc[k]["spans"] if a_ < b_ and b_ > s and a_ < e)
    print("[stats] mean tokens per entity:", {l: round(ent_tok[l] / cnt[l], 1) for l in cnt})
    overl = sum(1 for v in gold.values() for i, x in enumerate(sorted(v)) for y in sorted(v)[i+1:]
                if y[0] < x[1])
    print(f"[stats] overlapping gold entity pairs: {overl}")


if __name__ == "__main__":
    main()
