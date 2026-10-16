"""Phase 3 final submission: cross-fitted ensemble on train + dev, then role-sequence smoothing.

1. Ensemble: our best pipeline (Phase 1 Llama + LoRA Stage A + BiGRU Stage B) trained K times. The 80
   training and 10 dev judgments are pooled and split into K groups (CL/IT balanced); each copy trains on
   K-1 groups and picks its epochs on the group it did not see. Test prediction = average of the K copies.
   Every labelled judgment is also predicted by the copy that did not see it (out-of-fold): an honest score
   over all 90 judgments (slightly optimistic, because the same group chose that copy's epochs).
2. Smoothing: Viterbi decoding of each judgment with role-change log-probabilities counted from the gold
   labels (per domain). Its weight w is chosen on the out-of-fold predictions, with the change counts for
   each group taken from the other groups only. w = 0 means no change.

    from rr_final import run_final
    res = run_final(bb_llama, prefix="final")

Outputs in out_dir: <prefix>_ensemble/ (stageB/submission.zip = plain average, teacher.pt, metrics.json,
ensemble_fold<k>.pt), <prefix>_smoothed/ (submission.zip, smoothing.json, oof_predictions.json) and
<prefix>_summary.txt.
"""
from __future__ import annotations

import json
import os

import torch

import rr_data as D
import rr_kd as KD
import rr_train as T

WEIGHTS = (0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


def _np(t):
    return None if t is None else t.float().numpy()


def _row(rep):
    return "  ".join(f"{k} {v['macro_f1_all13']:.4f}" for k, v in rep.items())


def smooth_submission(teacher_path: str, data_root: str, out: str, weights=WEIGHTS, log=print) -> dict:
    """Chooses the smoothing weight on the out-of-fold predictions in teacher.pt and writes the smoothed
    test submission to out/submission.zip."""
    t = torch.load(teacher_path, map_location="cpu", weights_only=False)
    labels = list(t["labels"])
    docs = D.load_docs(data_root)
    by_id = {d["id"]: d for d in docs}
    pool = [by_id[i] for f in t["folds"] for i in f]
    test = [d for d in docs if d["split"] == "test"]
    fold_of = {i: k for k, f in enumerate(t["folds"]) for i in f}
    K = len(t["folds"])
    domains = sorted({d["domain"] for d in pool})

    def counts(ds):
        return D.transition_logprobs([d["labels"] for d in ds], labels)

    per_fold = {(k, dm): counts([d for d in pool if fold_of[d["id"]] != k and d["domain"] == dm])
                for k in range(K) for dm in domains}
    full = {dm: counts([d for d in pool if d["domain"] == dm]) for dm in domains}
    oof = [_np(t["train"][d["id"]]) for d in pool]
    best, table = D.tune_smoothing(pool, oof, labels, lambda d: per_fold[(fold_of[d["id"]], d["domain"])], weights)

    log(f"\n[smoothing] out-of-fold 13-role avg over {len(pool)} labelled judgments for each weight "
        "(0 = no smoothing):")
    for w, rep in table.items():
        log(f"  w {w:<5}  {_row(rep)}" + ("   <- chosen" if w == best else ""))
    log(D.format_report({"plain": table[0], f"smoothed (w={best})": table[best]}, labels))

    test_lp = [_np(t["test"].get(d["id"])) for d in test]
    preds = D.smoothed_preds(test, test_lp, labels, lambda d: full[d["domain"]], best)
    os.makedirs(out, exist_ok=True)
    D.write_submission(test, preds, labels, out)
    oof_preds = D.smoothed_preds(pool, oof, labels, lambda d: per_fold[(fold_of[d["id"]], d["domain"])], best)
    with open(os.path.join(out, "oof_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(oof_preds, f)
    res = dict(chosen=best, weights={str(w): {k: v["macro_f1_all13"] for k, v in rep.items()}
                                     for w, rep in table.items()},
               oof_plain=table[0], oof_smoothed=table[best], submission=os.path.join(out, "submission.zip"),
               teacher=teacher_path)
    with open(os.path.join(out, "smoothing.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def run_final(bb, prefix: str = "final", folds: int = 5, method: str = "lora", weights=WEIGHTS,
              skip_done: bool = True, **settings) -> dict:
    """Ensemble on train + dev (K copies, out-of-fold) -> smoothing -> two submissions and a summary.
    A finished ensemble (teacher.pt + metrics.json) is reused."""
    for bad in ("run_name", "method", "folds", "train_on_dev", "kd_teacher"):
        if bad in settings:
            raise ValueError(f"run_final sets {bad} itself")
    cfg0 = T.resolve_cfg(bb, {**settings, "method": method})
    out_dir = cfg0["out_dir"]
    ename = f"{prefix}_ensemble"
    edir = os.path.join(out_dir, ename)
    tpath, mpath = os.path.join(edir, "teacher.pt"), os.path.join(edir, "metrics.json")
    if skip_done and os.path.exists(tpath) and os.path.exists(mpath):
        with open(mpath) as f:
            metrics = json.load(f)
        print(f"===== {ename}: already done, using {tpath}")
    else:
        metrics = KD.teacher_oof(bb, run_name=ename, method=method, folds=folds, train_on_dev=True,
                                 **settings)["metrics"]
    sm = smooth_submission(tpath, cfg0["data_root"], os.path.join(out_dir, f"{prefix}_smoothed"), weights)

    K = metrics["copies"]
    n_fit = metrics["folds"][0]["fit_judgments"]
    plain = os.path.join(edir, "stageB", "submission.zip")
    lines = [
        f"FINAL ENSEMBLE '{prefix}': {K} copies of Llama + {method} + BiGRU, each trained on {n_fit} of the "
        f"{metrics['data']['train']['docs'] + metrics['data']['dev']['docs']} train + dev judgments",
        "out-of-fold 13-role avg over all train + dev judgments (each scored by the copy that did not see it):",
        f"  Stage A (sentence alone): {_row(metrics['stage_a']['dev'])}",
        f"  Stage B, plain average:   {_row(sm['oof_plain'])}",
        f"  Stage B, smoothed (w={sm['chosen']}): {_row(sm['oof_smoothed'])}",
        "submissions:",
        f"  plain average: {plain}",
        f"  smoothed:      {sm['submission']}" + ("   (w = 0: same predictions as plain)" if sm["chosen"] == 0 else ""),
        "(the dev judgments were used for training here, so there is no separate dev score)",
    ]
    text = "\n".join(lines)
    print("\n" + "=" * 78 + "\n" + text)
    with open(os.path.join(out_dir, f"{prefix}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return dict(metrics=metrics, smoothing=sm, submissions=dict(plain=plain, smoothed=sm["submission"]))