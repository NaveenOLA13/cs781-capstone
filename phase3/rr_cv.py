"""Phase 3: out-of-fold training (stacking).

1. Stage A is trained k times (default 5). Each time it learns from (k-1)/k of the training
   judgments and predicts the 1/k it did not see, so every training judgment gets role
   probabilities from a Stage A that never saw it ("out-of-fold"). Every Stage A also predicts
   dev and test; their probabilities are averaged (a k-model ensemble).
2. Stage B (BiGRU over a judgment) reads, for every sentence, Stage A's 13 log-probabilities +
   its position bucket + the domain, and learns to correct them from the neighbouring sentences.
   It is trained k times on the same groups, which gives out-of-fold Stage B predictions for
   every training judgment; dev/test use the average of the k Stage B models.
3. Per-role biases are tuned on the out-of-fold Stage B predictions to maximise the 13-role
   macro-F1 (one bias set for all, and one per domain).
Dev is never used for training or for any choice, so its score is an honest estimate.

    from rr_cv import run_cv
    res = run_cv(bb, run_name="cv_base")

Outputs in out_dir/run_name/: log.txt, config.json, metrics.json, oof.pt, one folder per variant
(A_ens, B, B_bias, B_bias_dom) with submission.zip, prediction.json, dev_predictions.json and
dev_confusion.csv, stageA_fold<k>.pt (only the trained tensors) and stageB_fold<k>.pt.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import rr_data as D
import rr_train as T

CV_DEFAULTS = {**T.DEFAULTS, "run_name": "cv", "folds": 5, "bias_passes": 3}
VARIANTS = ("A_ens", "B", "B_bias", "B_bias_dom")
SHORT_NAMES = {"A_ens": "A ens", "B": "B", "B_bias": "B+bias", "B_bias_dom": "B+bias/dom"}


def _quiet(*args, **kwargs):
    pass


def _log_softmax_all(outs):
    return [None if o is None else F.log_softmax(o.float(), -1) for o in outs]


def _avg_logprobs(per_fold):
    """per_fold: [fold][doc] -> [S, L] log-probs. Returns per doc the log of the mean probability."""
    out = []
    for runs in zip(*per_fold):
        out.append(None if runs[0] is None else torch.logsumexp(torch.stack(runs), 0) - math.log(len(runs)))
    return out


def _stage_b_inputs(doc, logp, buckets, domains):
    """[S, L + buckets + n_domains]: Stage A log-probs, position bucket (one-hot), domain (one-hot)."""
    if logp is None:
        return None
    n = len(doc["sents"])
    parts = [logp.float()]
    if buckets > 0:
        pos = torch.zeros(n, buckets)
        pos[torch.arange(n), torch.tensor([D.pos_bucket(i, n, buckets) for i in range(n)])] = 1.0
        parts.append(pos)
    dom = torch.zeros(n, len(domains))
    if doc["domain"] in domains:
        dom[:, domains.index(doc["domain"])] = 1.0
    parts.append(dom)
    return torch.cat(parts, -1)


def _predict_b(model, xs, device):
    return _log_softmax_all(T.predict_stage_b(model, xs, device))


def _trainable_state(clf):
    names = {n for n, p in clf.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in clf.state_dict().items() if k in names}


def _short(rep):
    doms = ", ".join(f"{k} {v['macro_f1_all13']:.4f}" for k, v in rep.items() if k != "all")
    return f"13-role avg {rep['all']['macro_f1_all13']:.4f} ({doms}), accuracy {rep['all']['accuracy']:.3f}"


def run_cv(bb, **overrides) -> dict:
    cfg = T.resolve_cfg(bb, overrides, CV_DEFAULTS)
    if cfg["folds"] < 2:
        raise ValueError("folds must be at least 2")
    device, out, log = T.start_run(cfg, kind="out-of-fold run")
    labels, train, dev, test = T.prepare(bb, cfg, log)
    L = len(labels)
    li = {l: i for i, l in enumerate(labels)}
    domains = sorted({d["domain"] for d in train + dev + test})
    K = cfg["folds"]
    folds = D.make_folds(train, K, cfg["seed"])
    if any(not f for f in folds):
        raise ValueError(f"need at least {K} training judgments for {K} folds")
    log(f"[folds] {K} groups of training judgments: " + " | ".join(
        "+".join(f"{sum(train[i]['domain'] == dm for i in f)} {dm}" for dm in domains) for f in folds))

    # ------------------------------------------------------------- Stage A, K times
    t_a = time.time()
    T._reset_peak(device)
    oof_a = [None] * len(train)
    dev_a, test_a, fold_a = [], [], []
    n_trainable = n_total = 0
    for k, held in enumerate(folds):
        held_set = set(held)
        fit_docs = [d for i, d in enumerate(train) if i not in held_set]
        held_docs = [train[i] for i in held]
        T.set_seed(cfg["seed"] + k)
        clf, n_trainable, n_total = T.build_stage_a(bb, labels, cfg, device, fit_docs, log if k == 0 else _quiet)
        t0 = time.time()
        info = T.train_stage_a(clf, fit_docs, None, labels, cfg, device, bb.pad_id, _quiet)
        held_lp = _log_softmax_all(T.run_sentences(clf, held_docs, cfg, device, bb.pad_id))
        for i, o in zip(held, held_lp):
            oof_a[i] = o
        dev_a.append(_log_softmax_all(T.run_sentences(clf, dev, cfg, device, bb.pad_id)))
        test_a.append(_log_softmax_all(T.run_sentences(clf, test, cfg, device, bb.pad_id)))
        rep = D.evaluate(held_docs, T.to_preds(held_docs, held_lp, labels), labels)
        minutes = (time.time() - t0) / 60
        fold_a.append(dict(fold=k + 1, fit_judgments=len(fit_docs), held_out=[d["id"] for d in held_docs],
                           minutes=minutes, last_loss=info["history"][-1]["loss"],
                           held_out_13avg=rep["all"]["macro_f1_all13"], held_out_accuracy=rep["all"]["accuracy"]))
        log(f"[stage A {k + 1}/{K}] trained on {len(fit_docs)} judgments in {minutes:.1f} min "
            f"(last loss {info['history'][-1]['loss']:.3f}); its {len(held_docs)} unseen judgments: {_short(rep)}")
        if cfg["save_model"]:
            torch.save({"trained_weights": _trainable_state(clf), "labels": labels, "config": cfg, "fold": k + 1,
                        "note": "only the trained tensors; all other weights equal the Phase 1 checkpoint"},
                       os.path.join(out, f"stageA_fold{k + 1}.pt"))
        del clf
        if device.type == "cuda":
            torch.cuda.empty_cache()
    peak_a, minutes_a = T._gb(device), (time.time() - t_a) / 60
    dev_a_avg, test_a_avg = _avg_logprobs(dev_a), _avg_logprobs(test_a)
    rep_oof_a = D.evaluate(train, T.to_preds(train, oof_a, labels), labels)
    rep_dev_a = D.evaluate(dev, T.to_preds(dev, dev_a_avg, labels), labels)
    log(f"[stage A] out-of-fold, all {len(train)} training judgments: {_short(rep_oof_a)}")
    log(f"[stage A] dev, average of the {K} models:          {_short(rep_dev_a)}")
    log("          (these two should be close now: Stage B learns from the first and is used on the second)")

    # ------------------------------------------------------------- Stage B, K times
    nb = cfg["pos_buckets"]
    xb_train = [_stage_b_inputs(d, o, nb, domains) for d, o in zip(train, oof_a)]
    xb_dev = [_stage_b_inputs(d, o, nb, domains) for d, o in zip(dev, dev_a_avg)]
    xb_test = [_stage_b_inputs(d, o, nb, domains) for d, o in zip(test, test_a_avg)]
    ys = [torch.tensor([li[l] for l in d["labels"]], dtype=torch.long) for d in train]
    w, _ = D.class_weights([li[l] for d in train for l in d["labels"]], L, cfg["weight_cap"], cfg["weight_power"])
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    t_b = time.time()
    T._reset_peak(device)
    oof_b = [None] * len(train)
    dev_b, test_b, b_epochs = [], [], []
    n_b = 0
    for k, held in enumerate(folds):
        held_set = set(held)
        fit = [i for i in range(len(train)) if i not in held_set]
        T.set_seed(cfg["seed"] + 100 + k)
        model, info = T.train_stage_b([xb_train[i] for i in fit], [ys[i] for i in fit],
                                      [xb_train[i] for i in held], [train[i] for i in held],
                                      labels, cfg, device, weight, _quiet)
        n_b = T.count_params(model)
        for i, o in zip(held, _predict_b(model, [xb_train[i] for i in held], device)):
            oof_b[i] = o
        dev_b.append(_predict_b(model, xb_dev, device))
        test_b.append(_predict_b(model, xb_test, device))
        b_epochs.append(info["best_epoch"])
        log(f"[stage B {k + 1}/{K}] best epoch {info['best_epoch']}; its unseen judgments: "
            f"13-role avg {info['best_dev_score']:.4f}")
        if cfg["save_model"]:
            torch.save({"state_dict": model.state_dict(), "labels": labels, "config": cfg, "fold": k + 1,
                        "domains": domains, "input": "Stage A log-probs + position one-hot + domain one-hot"},
                       os.path.join(out, f"stageB_fold{k + 1}.pt"))
        del model
    peak_b, minutes_b = T._gb(device), (time.time() - t_b) / 60
    dev_b_avg, test_b_avg = _avg_logprobs(dev_b), _avg_logprobs(test_b)

    # ------------------------------------------------------------- per-role biases
    keep = [i for i, d in enumerate(train) if d["labels"]]
    gold = np.concatenate([[li[l] for l in train[i]["labels"]] for i in keep])
    lp = np.concatenate([oof_b[i].numpy() for i in keep])
    grp = np.concatenate([[domains.index(train[i]["domain"])] * len(train[i]["labels"]) for i in keep])
    zero = np.zeros((len(domains), L))
    b_one, s_none, s_one = D.tune_biases(lp, gold, L, passes=cfg["bias_passes"])
    b_one = np.repeat(b_one, len(domains), 0)
    b_dom, _, s_dom = D.tune_biases(lp, gold, L, groups=grp, n_groups=len(domains), passes=cfg["bias_passes"])
    log(f"[bias] out-of-fold 13-role avg on training judgments: no bias {s_none:.4f} | one set {s_one:.4f} | "
        f"per domain {s_dom:.4f}")
    log(f"{'[bias] role':28s}{'one set':>9s}" + "".join(f"{dm:>8s}" for dm in domains))
    for c, l in enumerate(labels):
        log(f"       {l:21s}{b_one[0, c]:9.2f}" + "".join(f"{b_dom[g, c]:8.2f}" for g in range(len(domains))))

    def group_of(d):
        return domains.index(d["domain"])

    biases = {"B": zero, "B_bias": b_one, "B_bias_dom": b_dom}
    preds = {"A_ens": {"dev": T.to_preds(dev, dev_a_avg, labels), "test": T.to_preds(test, test_a_avg, labels),
                       "train_oof": T.to_preds(train, oof_a, labels)}}
    for v, b in biases.items():
        preds[v] = {"dev": D.biased_preds(dev, dev_b_avg, labels, b, group_of),
                    "test": D.biased_preds(test, test_b_avg, labels, b, group_of),
                    "train_oof": D.biased_preds(train, oof_b, labels, b, group_of)}
    rep_dev = {v: D.evaluate(dev, preds[v]["dev"], labels) for v in VARIANTS}
    rep_oof = {v: D.evaluate(train, preds[v]["train_oof"], labels) for v in VARIANTS}
    for v in VARIANTS:
        T._save_stage(out, v, dev, preds[v]["dev"], test, preds[v]["test"], rep_dev[v], labels)
    chosen = max(VARIANTS, key=lambda v: rep_oof[v]["all"]["macro_f1_all13"])   # chosen without looking at dev

    # ------------------------------------------------------------- report
    log("\n" + "=" * 92)
    log("OUT-OF-FOLD on the training judgments (what the choices are based on):")
    for v in VARIANTS:
        log(f"  {SHORT_NAMES[v]:12s} {_short(rep_oof[v])}")
    log("\nDEV (never used for any choice, an honest estimate):")
    log(D.format_report({SHORT_NAMES[v]: rep_dev[v] for v in VARIANTS}, labels))
    fmt_gb = lambda x: "-" if x is None else f"{x:.2f} GB"  # noqa: E731
    log(f"\n{'':10s}{'trainable params':>18s}{'peak GPU memory':>17s}{'total time':>12s}")
    log(f"{'Stage A':10s}{n_trainable / 1e6:>17.2f}M{fmt_gb(peak_a):>17s}{minutes_a:>10.1f}m   ({K} models)")
    log(f"{'Stage B':10s}{n_b / 1e6:>17.2f}M{fmt_gb(peak_b):>17s}{minutes_b:>10.1f}m   ({K} models, "
        f"best epochs {b_epochs})")
    log(f"\nchosen on the training judgments only: {chosen} (out-of-fold 13-role avg "
        f"{rep_oof[chosen]['all']['macro_f1_all13']:.4f}; dev {rep_dev[chosen]['all']['macro_f1_all13']:.4f})")
    log(f"submit: {os.path.join(out, chosen, 'submission.zip')}")

    torch.save({"labels": labels, "domains": domains, "folds": folds,
                "doc_ids": {s: [d["id"] for d in ds] for s, ds in (("train", train), ("dev", dev), ("test", test))},
                "oof_a": oof_a, "dev_a": dev_a_avg, "test_a": test_a_avg,
                "oof_b": oof_b, "dev_b": dev_b_avg, "test_b": test_b_avg,
                "bias_one": b_one, "bias_domain": b_dom}, os.path.join(out, "oof.pt"))
    metrics = dict(run_name=cfg["run_name"], kind="out-of-fold", route=bb.route, config=cfg,
                   folds=[[train[i]["id"] for i in f] for f in folds],
                   stage_a=dict(per_fold=fold_a, oof_train=rep_oof_a, dev=rep_dev_a, trainable_params=n_trainable,
                                total_params=n_total, peak_gpu_gb=peak_a, minutes=minutes_a),
                   stage_b=dict(best_epochs=b_epochs, trainable_params=n_b, peak_gpu_gb=peak_b, minutes=minutes_b),
                   bias=dict(oof_13avg=dict(none=s_none, one_set=s_one, per_domain=s_dom),
                             one_set=b_one[0].tolist(), per_domain={dm: b_dom[g].tolist() for g, dm in enumerate(domains)}),
                   oof_train={v: rep_oof[v] for v in VARIANTS}, dev={v: rep_dev[v] for v in VARIANTS},
                   chosen=chosen)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return dict(out_dir=out, chosen=chosen, metrics=metrics, labels=labels)
