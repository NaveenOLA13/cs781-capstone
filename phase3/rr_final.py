"""Phase 3 final submission: cross-fitted ensemble on train + dev, then role-sequence smoothing."""

from __future__ import annotations

import gzip
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


def _by_domain(docs, logps, labels, trans_for, w_dom):
    preds = {}
    for dom, w in w_dom.items():
        idx = [i for i, d in enumerate(docs) if d["domain"] == dom]
        preds.update(D.smoothed_preds([docs[i] for i in idx], [logps[i] for i in idx], labels, trans_for, w))
    return preds


def smooth_submission(
    teacher_path: str, data_root: str, out: str, weights=WEIGHTS, log=print, export_path: str | None = None
) -> dict:
    """For every Stage B version in teacher.pt: chooses smoothing weights on the out-of-fold predictions and writes
    <out>/<version>/plain/submission.zip and <out>/<version>/smoothed/submission.zip."""
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

    per_fold = {
        (k, dm): counts([d for d in pool if fold_of[d["id"]] != k and d["domain"] == dm])
        for k in range(K)
        for dm in domains
    }
    full = {dm: counts([d for d in pool if d["domain"] == dm]) for dm in domains}

    def trans_oof(d):
        return per_fold[(fold_of[d["id"]], d["domain"])]

    def trans_test(d):
        return full[d["domain"]]

    versions = t.get("versions") or {"main": {"train": t["train"], "test": t["test"]}}
    res: dict = {"versions": {}, "domains": domains}
    export = {
        "labels": labels,
        "folds": t["folds"],
        "versions": list(versions),
        "pool": [
            dict(id=d["id"], domain=d["domain"], split=d["split"], sents=d["sents"], gold=d["labels"]) for d in pool
        ],
        "test": [dict(id=d["id"], domain=d["domain"], sents=d["sents"]) for d in test],
    }
    log(f"\n[smoothing] out-of-fold 13-role avg over {len(pool)} labelled judgments (0 = no smoothing)")
    for v, vt in versions.items():
        oof = [_np(vt["train"][d["id"]]) for d in pool]
        test_lp = [_np(vt["test"].get(d["id"])) for d in test]
        w_shared, table = D.tune_smoothing(pool, oof, labels, trans_oof, weights)
        w_dom, rep_dom = D.tune_smoothing_per_domain(pool, oof, labels, trans_oof, weights)
        log(
            f"  {v}: plain {_row(table[0])} | one weight (w={w_shared}) {_row(table[w_shared])} | "
            f"per domain ({', '.join(f'{k} {w}' for k, w in w_dom.items())}) {_row(rep_dom)}"
        )
        vdir = os.path.join(out, v)
        D.write_submission(
            test, D.smoothed_preds(test, test_lp, labels, trans_test, 0), labels, os.path.join(vdir, "plain")
        )
        D.write_submission(
            test, _by_domain(test, test_lp, labels, trans_test, w_dom), labels, os.path.join(vdir, "smoothed")
        )
        with open(os.path.join(vdir, "oof_predictions_smoothed.json"), "w", encoding="utf-8") as f:
            json.dump(_by_domain(pool, oof, labels, trans_oof, w_dom), f)
        res["versions"][v] = dict(
            plain={k: r["macro_f1_all13"] for k, r in table[0].items()},
            one_weight={k: r["macro_f1_all13"] for k, r in table[w_shared].items()},
            w_shared=w_shared,
            per_domain={k: r["macro_f1_all13"] for k, r in rep_dom.items()},
            w_domain=w_dom,
            weights={str(w): {k: r["macro_f1_all13"] for k, r in rep.items()} for w, rep in table.items()},
            plain_submission=os.path.join(vdir, "plain", "submission.zip"),
            smoothed_submission=os.path.join(vdir, "smoothed", "submission.zip"),
            oof_plain_report=table[0],
            oof_smoothed_report=rep_dom,
        )
        for i in range(len(pool)):
            export["pool"][i].setdefault("probs", {})[v] = [] if oof[i] is None else oof[i].round(4).tolist()
        for i in range(len(test)):
            export["test"][i].setdefault("probs", {})[v] = [] if test_lp[i] is None else test_lp[i].round(4).tolist()
    best = max(res["versions"], key=lambda v: res["versions"][v]["per_domain"]["all"])
    res["best"] = best
    b = res["versions"][best]
    log(f"\n[smoothing] best version: {best} (per-domain weights {b['w_domain']})")
    log(D.format_report({f"{best} plain": b["oof_plain_report"], f"{best} smoothed": b["oof_smoothed_report"]}, labels))
    with open(os.path.join(out, "smoothing.json"), "w") as f:
        json.dump(res, f, indent=1)
    if export_path:
        with gzip.open(export_path, "wt", encoding="utf-8") as f:
            json.dump(export, f)
        res["export"] = export_path
    return res


def run_final(
    bb, prefix: str = "final", folds: int = 5, method: str = "lora", weights=WEIGHTS, skip_done: bool = True, **settings
) -> dict:
    """Ensemble on train + dev (K copies, out-of-fold) -> smoothing for every Stage B version -> submissions and a
    summary."""
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
        metrics = KD.teacher_oof(bb, run_name=ename, method=method, folds=folds, train_on_dev=True, **settings)[
            "metrics"
        ]
    sm = smooth_submission(
        tpath,
        cfg0["data_root"],
        os.path.join(out_dir, f"{prefix}_subs"),
        weights,
        export_path=os.path.join(out_dir, f"{prefix}_probs.json.gz"),
    )

    K = metrics["copies"]
    n_fit = metrics["folds"][0]["fit_judgments"]
    cfg = metrics["config"]
    ctx = (
        f"with the previous and next sentence (up to {cfg['context_tokens']} tokens each)"
        if cfg.get("context")
        else "one sentence at a time"
    )
    w = max(len(v) for v in sm["versions"]) + 2
    lines = [
        f"FINAL ENSEMBLE '{prefix}': {K} copies of Llama + {method}, each trained on {n_fit} of the "
        f"{metrics['data']['train']['docs'] + metrics['data']['dev']['docs']} train + dev judgments; Stage A reads {ctx}",
        "out-of-fold 13-role avg over all train + dev judgments (each scored by the copy that did not see it):",
        f"  Stage A (sentence model alone): {_row(metrics['stage_a']['dev'])}",
        "  Stage B versions:",
    ]
    for v, r in sm["versions"].items():
        fmt = lambda x: "  ".join(f"{k} {val:.4f}" for k, val in x.items())  # noqa: E731
        lines.append(f"    {v:{w}s} plain: {fmt(r['plain'])}")
        lines.append(f"    {'':{w}s} smoothed, one weight {r['w_shared']}: {fmt(r['one_weight'])}")
        lines.append(f"    {'':{w}s} smoothed per domain {r['w_domain']}: {fmt(r['per_domain'])}")
    best = sm["best"]
    lines += [
        f"best (out-of-fold, smoothed per domain): {best}",
        f"submit: {sm['versions'][best]['smoothed_submission']}",
        "other files: "
        + ", ".join(f"{v}/plain, {v}/smoothed" for v in sm["versions"])
        + f" in {os.path.join(out_dir, prefix + '_subs')}",
        f"probabilities for analysis: {sm.get('export')}",
        "(the dev judgments were used for training here, so there is no separate dev score)",
    ]
    text = "\n".join(lines)
    print("\n" + "=" * 78 + "\n" + text)
    with open(os.path.join(out_dir, f"{prefix}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return dict(metrics=metrics, smoothing=sm, best=best, submission=sm["versions"][best]["smoothed_submission"])
