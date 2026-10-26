"""Phase 5 (Legal Statute Identification, multi-label) data: loading, label space, metrics, thresholds, submission.
No torch needed.

Data: Phase_5_LSI/Train/train.json, Train/dev.json, Test/test.json from the course's Hugging Face dataset; records
{"id", "text", "labels": [statute ids]}; test has no labels (and any label field there is never read).
Metric (brief 11.4): macro-F1 over statute ids (primary), micro-F1, subset accuracy. Here macro-F1 averages the F1 of
the statutes that occur in the gold labels of the evaluated split (the evaluator derives the label dimension from
the reference data); predicted statutes that never occur in the gold count only in micro-F1.
Thresholds: one global threshold, or one per statute (statutes with fewer than min_pos dev positives keep the global
one); which of the two is used is decided by 2-fold cross-fitting on dev (tune on one half, score on the other), so
the choice is not judged on the same cases it was tuned on. Every document gets at least its top statute.
"""
from __future__ import annotations

import json
import os
import random
import zipfile
from collections import Counter

import numpy as np

import cj_data as C

DEFAULT_ROOT = "/kaggle/working/data/Phase_5_LSI"
LABEL_KEYS = ("labels", "label", "statutes", "statute_ids", "statute", "y", "target")
GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)


# ------------------------------------------------------------------------------ loading
def to_labels(v) -> list[int]:
    """Sorted unique statute ids from a list (or a single id) of ints / integer strings / integer floats."""
    if v is None:
        return []
    if not isinstance(v, (list, tuple)):
        v = [v]
    out = []
    for x in v:
        if isinstance(x, bool):
            raise ValueError(f"statute id {x!r} is a bool")
        if isinstance(x, str):
            x = x.strip()
            if not x:
                continue
            x = float(x)
        if isinstance(x, float):
            if not x.is_integer():
                raise ValueError(f"statute id {x!r} is not an integer")
            x = int(x)
        out.append(int(x))
    return sorted(set(out))


def load_split(path: str, need_labels: bool, read_labels: bool = True) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        recs = C._records(json.load(f))
    if not recs:
        raise ValueError(f"{path}: no documents")
    keys = set().union(*(r.keys() for r in recs[:100]))
    tk, lk, ik = C._pick(keys, C.TEXT_KEYS), C._pick(keys, LABEL_KEYS), C._pick(keys, C.ID_KEYS)
    if tk is None or ik is None:
        raise ValueError(f"{path}: no text field ({C.TEXT_KEYS}) or id field ({C.ID_KEYS}) in {sorted(keys)}")
    if need_labels and lk is None:
        raise ValueError(f"{path}: no label field ({LABEL_KEYS}) in {sorted(keys)}")
    out = []
    for r in recs:
        t = r.get(tk, "")
        t = " ".join(map(str, t)) if isinstance(t, list) else ("" if t is None else str(t))
        labs = to_labels(r.get(lk)) if (read_labels and lk is not None) else None
        if need_labels and not labs:
            raise ValueError(f"{path}: document {r.get(ik)!r} has no statute id")
        out.append(dict(id=str(r[ik]), text=t, labels=labs))
    if need_labels:
        lens = Counter(len(r[lk]) for r in recs if isinstance(r.get(lk), list))
        vals = set().union(*({x for x in r[lk]} for r in recs[:200] if isinstance(r.get(lk), list)))
        if len(lens) == 1 and next(iter(lens)) > 2 and vals <= {0, 1}:
            raise ValueError(f"{path}: labels look like 0/1 vectors, not lists of statute ids")
    return out


def load_data(root: str = DEFAULT_ROOT, fetch: bool = True, clean: bool = True, log=None) -> dict[str, list[dict]]:
    """{"train", "dev", "test"} lists of {"id", "text", "labels"}; test labels are never read. clean=True removes
    test documents (same id or text) from train / dev, dev documents also in train, and repeated texts."""
    if fetch:
        C.download(root)
    files = C.find_files(root)
    data = {s: load_split(files[s], need_labels=(s != "test"), read_labels=(s != "test"))
            for s in ("train", "dev", "test")}
    dup = [k for k, c in Counter(d["id"] for d in data["test"]).items() if c > 1]
    if dup:
        raise ValueError(f"test.json has repeated ids, e.g. {dup[:3]}")
    if clean:
        rep = C.clean_splits(data)
        if log is not None:
            for s, r in rep.items():
                log(f"[data] {s}: {r['before']} -> {r['after']} documents (removed {r['test_cases_removed']} test "
                    f"documents, {r['repeats_removed']} repeats"
                    + (f", {r['train_cases_removed']} also in train" if s == "dev" else "") + ")")
    return data


def label_space(data: dict) -> list[int]:
    """Every statute id seen in train or dev (the output units of the model), sorted."""
    return sorted({x for s in ("train", "dev") for d in data[s] for x in d["labels"]})


def multi_hot(docs: list[dict], labels: list[int]) -> np.ndarray:
    idx = {x: j for j, x in enumerate(labels)}
    y = np.zeros((len(docs), len(labels)), dtype=bool)
    for i, d in enumerate(docs):
        for x in d["labels"] or []:
            if x in idx:
                y[i, idx[x]] = True
    return y


def summary(data: dict) -> str:
    labels = label_space(data)
    lines = [f"{'split':6s} {'docs':>7s} {'statutes':>9s} {'per doc':>8s} {'words: median':>14s} {'p90':>7s} "
             f"{'max':>8s}"]
    for s, ds in data.items():
        w = sorted(len(d["text"].split()) for d in ds) or [0]
        per = np.mean([len(d["labels"]) for d in ds]) if s != "test" else float("nan")
        n_lab = len({x for d in ds for x in (d["labels"] or [])})
        lines.append(f"{s:6s} {len(ds):7d} {n_lab:9d} {per:8.2f} {w[len(w) // 2]:14d} {w[int(len(w) * .9)]:7d} "
                     f"{w[-1]:8d}")
    cnt = Counter(x for d in data["train"] for x in d["labels"])
    dev_only = sorted({x for d in data["dev"] for x in d["labels"]} - set(cnt))
    freq = sorted(cnt.values())
    lines.append(f"statute ids (train + dev): {len(labels)}; training documents per statute: min {freq[0]}, median "
                 f"{freq[len(freq) // 2]}, max {freq[-1]}; statutes with < 10 training documents: "
                 f"{sum(c < 10 for c in freq)}; only in dev: {len(dev_only)} {dev_only[:10]}")
    lines.append("most frequent: " + ", ".join(f"{x} ({c})" for x, c in cnt.most_common(8)))
    return "\n".join(lines)


# ------------------------------------------------------------------------------ metrics
def scores(Y: np.ndarray, pred: np.ndarray) -> dict:
    """Macro-F1 over statutes present in the gold labels, micro-F1, subset accuracy (+ per-statute F1)."""
    Y, pred = Y.astype(bool), pred.astype(bool)
    tp = (Y & pred).sum(0)
    fp = (~Y & pred).sum(0)
    fn = (Y & ~pred).sum(0)
    support = Y.sum(0)
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    present = support > 0
    macro = float(f1[present].mean()) if present.any() else 0.0
    micro = float(2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum()))
    p_micro = float(tp.sum() / max(1, tp.sum() + fp.sum()))
    r_micro = float(tp.sum() / max(1, tp.sum() + fn.sum()))
    return dict(macro_f1=macro, micro_f1=micro, subset_acc=float((Y == pred).all(1).mean()) if len(Y) else 0.0,
                micro_precision=p_micro, micro_recall=r_micro, statutes_scored=int(present.sum()),
                mean_pred_per_doc=float(pred.sum(1).mean()) if len(Y) else 0.0,
                mean_gold_per_doc=float(Y.sum(1).mean()) if len(Y) else 0.0,
                per_label_f1=f1.tolist(), support=support.tolist())


def short(s: dict) -> dict:
    return {k: v for k, v in s.items() if k not in ("per_label_f1", "support")}


def fmt(s: dict, title: str = "") -> str:
    return (f"{title}macro-F1 {s['macro_f1']:.4f} | micro-F1 {s['micro_f1']:.4f} (P {s['micro_precision']:.3f} "
            f"R {s['micro_recall']:.3f}) | subset acc {s['subset_acc']:.4f} | statutes per doc: predicted "
            f"{s['mean_pred_per_doc']:.2f}, gold {s['mean_gold_per_doc']:.2f}")


# ------------------------------------------------------------------------------ thresholds
def apply(P: np.ndarray, t, min_one: bool = True) -> np.ndarray:
    """Predicted statutes (bool matrix) from probabilities P and a threshold (scalar or one per statute); with
    min_one, a document with nothing above its thresholds gets its highest-scoring statute."""
    pred = P >= np.asarray(t)
    if min_one and len(P):
        empty = ~pred.any(1)
        if empty.any():
            pred[np.flatnonzero(empty), P[empty].argmax(1)] = True
    return pred


def tune_global(Y: np.ndarray, P: np.ndarray, min_one: bool = True) -> tuple[float, float]:
    """(threshold, macro-F1): best macro-F1 over GRID; ties go to the threshold nearest 0.5."""
    best = None
    for t in GRID:
        m = scores(Y, apply(P, t, min_one))["macro_f1"]
        if best is None or m > best[1] + 1e-12 or (abs(m - best[1]) <= 1e-12 and abs(t - .5) < abs(best[0] - .5)):
            best = (float(t), m)
    return best


def tune_per_label(Y: np.ndarray, P: np.ndarray, t_global: float, min_pos: int = 3) -> np.ndarray:
    """One threshold per statute (best F1 of that statute on GRID; ties nearest t_global); statutes with fewer
    than min_pos positives keep t_global."""
    Y = Y.astype(bool)
    t = np.full(P.shape[1], float(t_global))
    for j in range(P.shape[1]):
        y = Y[:, j]
        pos = int(y.sum())
        if pos < min_pos:
            continue
        pr = P[None, :, j] >= GRID[:, None]
        tp = (pr & y).sum(1)
        fp = (pr & ~y).sum(1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + (pos - tp), 1)
        cand = GRID[f1 >= f1.max() - 1e-12]
        t[j] = float(cand[np.argmin(np.abs(cand - t_global))])
    return t


def choose_thresholds(Y: np.ndarray, P: np.ndarray, min_pos: int = 3, seed: int = 0, min_one: bool = True) -> dict:
    """Global vs per-statute thresholds, chosen by 2-fold cross-fitting on dev; final thresholds tuned on all dev."""
    n = len(Y)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    halves = [np.array(sorted(idx[: n // 2])), np.array(sorted(idx[n // 2:]))]
    cv = {"global": [], "per_label": []}
    for a, b in ((0, 1), (1, 0)):
        fa, ev = halves[a], halves[b]
        tg, _ = tune_global(Y[fa], P[fa], min_one)
        tl = tune_per_label(Y[fa], P[fa], tg, min_pos)
        cv["global"].append(scores(Y[ev], apply(P[ev], tg, min_one))["macro_f1"])
        cv["per_label"].append(scores(Y[ev], apply(P[ev], tl, min_one))["macro_f1"])
    cvm = {k: float(np.mean(v)) for k, v in cv.items()}
    tg, _ = tune_global(Y, P, min_one)
    tl = tune_per_label(Y, P, tg, min_pos)
    kind = "per_label" if cvm["per_label"] > cvm["global"] + 0.002 else "global"
    thr = tl if kind == "per_label" else np.full(P.shape[1], tg)
    return dict(kind=kind, thresholds=thr, t_global=tg, cv_global=cvm["global"], cv_per_label=cvm["per_label"],
                dev_global=scores(Y, apply(P, tg, min_one)), dev_chosen=scores(Y, apply(P, thr, min_one)))


# ------------------------------------------------------------------------------ files
def save_probs(path: str, ids: list[str], P: np.ndarray) -> None:
    with open(path, "w") as f:
        json.dump({i: [round(float(x), 5) for x in row] for i, row in zip(ids, P)}, f)


def load_probs(path: str, ids: list[str]) -> np.ndarray:
    with open(path) as f:
        d = json.load(f)
    missing = [i for i in ids if i not in d]
    if missing:
        raise ValueError(f"{path} has no probabilities for {len(missing)} documents, e.g. {missing[:3]}")
    return np.array([d[i] for i in ids], dtype=np.float32)


def write_submission(test_docs: list[dict], labels: list[int], P: np.ndarray, thresholds, out_dir: str,
                     min_one: bool = True) -> str:
    """prediction.json = [{"id": ..., "pred": [statute ids]}, ...] in test.json order (integer, unique, sorted
    ids; at least one per document with min_one) and submission.zip holding it; the zip is read back and checked."""
    if len(P) != len(test_docs) or P.shape[1] != len(labels):
        raise ValueError(f"probabilities {P.shape} do not match {len(test_docs)} documents x {len(labels)} statutes")
    pred = apply(P, thresholds, min_one)
    rows = [{"id": d["id"], "pred": sorted(int(labels[j]) for j in np.flatnonzero(pred[i]))}
            for i, d in enumerate(test_docs)]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prediction.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    zpath = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "prediction.json")
    with zipfile.ZipFile(zpath) as z:
        back = json.loads(z.read("prediction.json"))
    valid = set(int(x) for x in labels)
    ok = (len(back) == len(test_docs) and [r["id"] for r in back] == [d["id"] for d in test_docs]
          and all(isinstance(r["pred"], list) and all(type(x) is int and x in valid for x in r["pred"])
                  and len(set(r["pred"])) == len(r["pred"]) for r in back))
    if not ok:
        raise RuntimeError("submission.zip check failed")
    return zpath
