"""Shared tail of train_ner.py and ensemble.py (torch-free): dev scoring, dev-selected post-processing
(phase2/postproc.py) and the fold3 submission."""

import json
import os

from phase2 import ner_data as D


def finalize(
    dev_probs,
    test_probs,
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
    postproc=1,
):
    """dev scoring + post-processing selection on dev + test submission."""
    from phase2 import postproc as PP

    dev_gold = {k: gold[k] for k in dev_ids}

    def decode(probs, encd, txts, viterbi):
        res = {}
        for k, p in probs.items():
            t = PP.viterbi_tags(p, encd[k]["spans"], tags) if viterbi else p.argmax(-1)
            res[k] = D.decode_spans(list(t), encd[k]["spans"], id2tag, txts[k], table)
        return res

    pred = decode(dev_probs, enc, texts, False)
    sc = D.strict_scores(pred, dev_gold, labels)
    report = {"argmax": sc["macro_F1"]}
    use_vit = False
    if postproc:
        pv = decode(dev_probs, enc, texts, True)
        scv = D.strict_scores(pv, dev_gold, labels)
        report["viterbi"] = scv["macro_F1"]
        if scv["macro_F1"] >= sc["macro_F1"]:
            use_vit, pred, sc = True, pv, scv
    choice, ctx = None, None
    if postproc:
        gaz = PP.learn_gazetteer(texts, gold, train_ids)
        hdr, hdr_hits, n_hdr = PP.learn_header_policy(texts, gold, train_ids)
        print(f"[post] header rule: {n_hdr} train headers, hits per variant {hdr_hits} -> policy {hdr}")
        ctx = {"gaz": gaz, "hdr": hdr}
        dev_texts = {k: texts[k] for k in dev_ids}
        choice, sc, var_scores = PP.choose_per_label(pred, dev_texts, dev_gold, ctx, labels, D.strict_scores)
        for v, s in var_scores.items():
            report[f"variant:{v}"] = s["macro_F1"]
        pred = PP.combine({v: PP.run_variant(v, pred, dev_texts, ctx) for v in set(choice.values())}, choice)
        report["final(per-label choice)"] = sc["macro_F1"]
        print(f"[post] gazetteer: {len(ctx['gaz'])} strings | viterbi={use_vit} | per-label choice: {choice}")
    print("[post] dev macro-F1 by step:", json.dumps({k: round(v, 4) for k, v in report.items()}))
    print(f"[ner] DEV strict macro-F1={sc['macro_F1']:.4f} P={sc['macro_P']:.4f} R={sc['macro_R']:.4f}")
    for lab, v in sc["per_label"].items():
        print(f"    {lab:10s} F1={v['F1']:.3f} P={v['P']:.3f} R={v['R']:.3f} (tp={v['tp']} fp={v['fp']} fn={v['fn']})")
    sc["postproc"] = {"report": report, "viterbi": use_vit, "choice": choice}
    json.dump(sc, open(os.path.join(out, "dev_scores.json"), "w"), indent=1)
    json.dump({k: [list(x) for x in v] for k, v in pred.items()}, open(os.path.join(out, "dev_pred.json"), "w"))

    if test_probs is not None:
        tpred = decode(test_probs, tenc, test, use_vit)
        if choice:
            tpred = PP.apply_choice(tpred, test, ctx, choice)
        pj = D.write_submission(tpred, test, os.path.join(out, "sub"))
        n = sum(len(v) for v in tpred.values())
        print(f"[ner] wrote {pj} + submission.zip | {len(test)} test docs, {n} entities")
