"""Ensemble of finished runs: average the dev / test probabilities of several runs (any model, technique, mid-trained
or not, whole-case versions) and tune one threshold on dev. No torch needed: works on the saved dev_probs.json /
test_probs.json files, e.g. after downloading the results zips.

    from cj_ensemble import find_runs, ensemble
    runs = find_runs(["/kaggle/working/cj_runs", "/kaggle/working/cj_runs_mid"])
    ensemble(runs, out_dir="/kaggle/working/cj_ensemble")

Members are chosen greedily on dev: start with the best run, then repeatedly add the run that raises the dev
macro-F1 of the average the most, while it rises by at least min_gain (at most max_members runs). The dev score of
the result is a little optimistic (members and threshold are chosen on the same 994 cases); the list of members and
the gain of each step are printed so the choice can be judged.
"""
from __future__ import annotations

import json
import os

import cj_data as D


def find_runs(base_dirs) -> dict[str, str]:
    """{run name: folder} for every folder under base_dirs that has dev_probs.json and test_probs.json."""
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    out = {}
    for base in base_dirs:
        for root, _, files in os.walk(base):
            if "dev_probs.json" in files and "test_probs.json" in files:
                name = os.path.relpath(root, base).replace(os.sep, "/")
                key = name if name not in out else f"{os.path.basename(base.rstrip('/'))}/{name}"
                out[key] = root
    return dict(sorted(out.items()))


def _load(folder):
    with open(os.path.join(folder, "dev_probs.json")) as f:
        dev = json.load(f)
    with open(os.path.join(folder, "test_probs.json")) as f:
        test = json.load(f)
    return dev, test


def _avg(probs: list[dict], ids):
    return [sum(p[i] for p in probs) / len(probs) for i in ids]


def ensemble(runs: dict[str, str], out_dir: str, data_root: str = D.DEFAULT_ROOT, max_members: int = 5,
             min_gain: float = 0.001, fetch: bool = True, log=print) -> dict:
    data = D.load_data(data_root, fetch=fetch)
    y = {d["id"]: d["label"] for d in data["dev"]}
    dev_ids = [d["id"] for d in data["dev"]]
    test_ids = [d["id"] for d in data["test"]]
    loaded, scores = {}, {}
    for name, folder in runs.items():
        dev, test = _load(folder)
        if not set(dev_ids) <= set(dev) or not set(test_ids) <= set(test):
            log(f"[ensemble] skipped {name}: its dev / test cases differ (a quick-check run?)")
            continue
        loaded[name] = (dev, test)
        scores[name] = D.tune_threshold([y[i] for i in dev_ids], [dev[i] for i in dev_ids])[1]
    if not loaded:
        raise ValueError("no usable runs")
    ranked = sorted(scores, key=lambda n: -scores[n])
    log("[ensemble] runs by dev macro-F1 (tuned threshold):")
    for n in ranked:
        log(f"  {scores[n]:.4f}  {n}")
    yd = [y[i] for i in dev_ids]
    members = [ranked[0]]
    cur = scores[ranked[0]]
    steps = [dict(added=ranked[0], dev_f1=cur)]
    while len(members) < max_members:
        best = None
        for n in ranked:
            if n in members:
                continue
            p = _avg([loaded[m][0] for m in members + [n]], dev_ids)
            f = D.tune_threshold(yd, p)[1]
            if best is None or f > best[1]:
                best = (n, f)
        if best is None or best[1] < cur + min_gain:
            break
        members.append(best[0])
        cur = best[1]
        steps.append(dict(added=best[0], dev_f1=cur))
    p_dev = _avg([loaded[m][0] for m in members], dev_ids)
    p_test = _avg([loaded[m][1] for m in members], test_ids)
    t, _ = D.tune_threshold(yd, p_dev)
    r50, rt = D.report(yd, p_dev, 0.5), D.report(yd, p_dev, t)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dev_probs.json"), "w") as f:
        json.dump(dict(zip(dev_ids, p_dev)), f)
    with open(os.path.join(out_dir, "test_probs.json"), "w") as f:
        json.dump(dict(zip(test_ids, p_test)), f)
    sub = D.write_submission(data["test"], {i: int(p >= t) for i, p in zip(test_ids, p_test)},
                             os.path.join(out_dir, "submission_tuned"))
    res = dict(members=members, steps=steps, threshold=t, dev_050=r50, dev_tuned=rt, single_best=ranked[0],
               single_best_dev_f1=scores[ranked[0]], submission=sub)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("[ensemble] members: " + " + ".join(f"{s['added']} ({s['dev_f1']:.4f})" for s in steps))
    log(D.format_report(rt, f"[ensemble] dev, {len(members)} runs averaged: "))
    log(f"[ensemble] best single run {ranked[0]} {scores[ranked[0]]:.4f} -> ensemble {rt['macro_f1']:.4f}; "
        f"submit {sub}")
    return res
