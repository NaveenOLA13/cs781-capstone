"""Phase 3 (rhetorical roles): data loading, tokenisation, batching, class weights,
metrics and the submission writer. No torch in this file.

Data layout (Exploration-Lab/CS-781-Capstone, Phase_3_RR):
    Train/{CL,IT}_train.json, Train/{CL,IT}_dev.json : [{"id", "text": [sent, ...], "labels": [role, ...]}]
    Test/{CL,IT}_test.json                           : {doc_id: [sent, ...]}
    Train/label_vocab.json                           : the 13 roles
Submission: submission.zip containing prediction.json = {doc_id: [one role per sentence, in order]}
"""
from __future__ import annotations

import glob
import json
import os
import random
import zipfile
from collections import Counter

DEFAULT_ROOT = "/kaggle/working/data/Phase_3_RR"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- loading
def load_labels(root: str = DEFAULT_ROOT) -> list[str]:
    """The role names in label_vocab.json order (accepts a list or a dict in either direction)."""
    paths = sorted(glob.glob(os.path.join(root, "**", "label_vocab.json"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"label_vocab.json not found under {root}")
    v = _read(paths[0])
    if isinstance(v, dict):
        for key in ("label2id", "labels", "id2label"):
            if key in v:
                v = v[key]
                break
    if isinstance(v, list):
        labels = [str(x) for x in v]
    elif isinstance(v, dict) and all(isinstance(x, int) for x in v.values()):
        labels = sorted(v, key=v.get)                               # {"Fact": 0, ...}
    elif isinstance(v, dict) and all(str(k).lstrip("-").isdigit() for k in v):
        labels = [str(v[k]) for k in sorted(v, key=int)]            # {"0": "Fact", ...}
    else:
        raise ValueError(f"unrecognised label_vocab.json format: {str(v)[:200]}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate labels in label_vocab.json: {labels}")
    return labels


def load_docs(root: str = DEFAULT_ROOT) -> list[dict]:
    """Every judgment as {id, domain, split, sents, labels (None for test), file}."""
    docs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        name = os.path.basename(path)
        if "label_vocab" in name:
            continue
        low = name.lower()
        split = "test" if "test" in low else ("dev" if "dev" in low else "train")
        domain = name.split("_")[0].upper()
        data = _read(path)
        if isinstance(data, dict):
            recs = [dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v} for k, v in data.items()]
        else:
            recs = data
        for r in recs:
            sents = r.get("text")
            if sents is None:
                sents = r.get("sentences", [])
            labels = r.get("labels")
            if labels is not None and len(labels) != len(sents):
                raise ValueError(f"{name} / {r.get('id')}: {len(sents)} sentences but {len(labels)} labels")
            docs.append(dict(id=str(r["id"]), domain=domain, split=split, file=os.path.relpath(path, root),
                             sents=[str(s) for s in sents],
                             labels=None if labels is None else [str(l) for l in labels]))
    if not docs:
        raise FileNotFoundError(f"no data files found under {root}")
    for split in ("train", "dev", "test"):
        ids = [d["id"] for d in docs if d["split"] == split]
        dup = [k for k, c in Counter(ids).items() if c > 1]
        if dup:
            raise ValueError(f"duplicate document ids in {split}: {dup[:5]}")
    return docs


def summary(docs: list[dict]) -> str:
    lines = [f"{'split':6s} {'domain':6s} {'docs':>5s} {'sentences':>10s}"]
    for split in ("train", "dev", "test"):
        for dom in sorted({d["domain"] for d in docs}):
            ds = [d for d in docs if d["split"] == split and d["domain"] == dom]
            if ds:
                lines.append(f"{split:6s} {dom:6s} {len(ds):5d} {sum(len(d['sents']) for d in ds):10d}")
    return "\n".join(lines)


def check_labels(docs: list[dict], labels: list[str]) -> None:
    known = set(labels)
    bad = Counter(l for d in docs if d["labels"] for l in d["labels"] if l not in known)
    if bad:
        raise ValueError(f"labels not in label_vocab.json: {dict(bad)}")


# ------------------------------------------------------------------------ positions
def pos_bucket(i: int, n: int, buckets: int = 10) -> int:
    """Relative position of sentence i (0-based) in a judgment of n sentences -> 0..buckets-1."""
    if buckets <= 0 or n <= 0:
        return 0
    return min(buckets - 1, (i * buckets) // n)


# ----------------------------------------------------------------------- tokenising
def truncate(ids: list[int], budget: int, tail: int) -> list[int]:
    """Keep the first (budget - tail) and the last `tail` tokens of a long sentence."""
    if len(ids) <= budget:
        return ids
    tail = max(0, min(tail, budget - 1))
    return ids[:budget - tail] + (ids[-tail:] if tail else [])


def encode_docs(docs: list[dict], encode, max_len: int = 128, tail: int | None = None,
                bos_id: int | None = None, eos_id: int | None = None, fallback_id: int = 0) -> list[int]:
    """Tokenise every sentence once; stores d["ids"] (list of token-id lists) on each doc.
    Returns the untruncated token count of every sentence (for the length report)."""
    budget = max_len - (bos_id is not None) - (eos_id is not None)
    if budget < 2:
        raise ValueError("max_len too small")
    tail = budget // 4 if tail is None else tail
    cache: dict[str, list[int]] = {}
    raw_lengths = []
    for d in docs:
        out = []
        for s in d["sents"]:
            ids = cache.get(s)
            if ids is None:
                ids = [int(x) for x in encode(s)]
                cache[s] = ids
            raw_lengths.append(len(ids))
            t = truncate(ids, budget, tail)
            if bos_id is not None:
                t = [bos_id] + t
            if eos_id is not None:
                t = t + [eos_id]
            out.append(t if t else [fallback_id])     # an empty sentence still gets one token
        d["ids"] = out
    return raw_lengths


def length_report(raw_lengths: list[int], max_len: int) -> str:
    if not raw_lengths:
        return "no sentences"
    s = sorted(raw_lengths)
    pct = {p: s[min(len(s) - 1, int(p / 100 * len(s)))] for p in (50, 90, 95, 99)}
    cut = sum(1 for x in s if x > max_len)
    return (f"tokens per sentence: median {pct[50]}, p90 {pct[90]}, p95 {pct[95]}, p99 {pct[99]}, "
            f"max {s[-1]} | longer than max_len={max_len}: {cut} ({cut / len(s):.1%}), kept start + end")


# ------------------------------------------------------------------------- batching
def length_batches(lengths: list[int], batch_size: int, shuffle: bool, seed: int = 0,
                   pool: int = 50) -> list[list[int]]:
    """Batches of indices with similar lengths (little padding).
    shuffle=False: sorted by length (prediction). shuffle=True: random pools of
    batch_size*pool items are sorted by length and cut into batches; batch order is shuffled."""
    idx = list(range(len(lengths)))
    if not shuffle:
        idx.sort(key=lambda i: lengths[i])
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    rng = random.Random(seed)
    rng.shuffle(idx)
    batches = []
    chunk = batch_size * pool
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: lengths[i])
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


# -------------------------------------------------------------------- class weights
def class_weights(y: list[int], n_labels: int, cap: float = 10.0, power: float = 0.5):
    """w_c = min(cap, (count of most common role / count_c) ** power); 1/sqrt(frequency) by default."""
    cnt = Counter(y)
    top = max(cnt.values())
    w = [min(cap, (top / cnt[c]) ** power) if cnt.get(c) else cap for c in range(n_labels)]
    return w, [cnt.get(c, 0) for c in range(n_labels)]


# -------------------------------------------------------------------------- metrics
def scores(gold: list[int], pred: list[int], labels: list[str]) -> dict:
    """Per-role precision/recall/F1, confusion matrix and macro-F1.
    macro_f1:       average over roles that occur in gold or predictions (sklearn's default)
    macro_f1_all13: average over all roles (a role absent from both counts as 0)"""
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for g, p in zip(gold, pred):
        cm[g][p] += 1
    per, f_present, f_all = {}, [], []
    for c in range(n):
        tp = cm[c][c]
        support = sum(cm[c])
        predicted = sum(cm[r][c] for r in range(n))
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[labels[c]] = dict(precision=p, recall=r, f1=f, support=support, predicted=predicted)
        f_all.append(f)
        if support or predicted:
            f_present.append(f)
    return dict(macro_f1=sum(f_present) / max(1, len(f_present)), macro_f1_all13=sum(f_all) / n,
                accuracy=sum(cm[c][c] for c in range(n)) / max(1, len(gold)), n=len(gold),
                per_label=per, confusion=cm)


def evaluate(docs: list[dict], preds: dict[str, list[str]], labels: list[str]) -> dict:
    """Scores over all docs and per domain. preds: {doc_id: [role per sentence]}."""
    idx = {l: i for i, l in enumerate(labels)}
    groups = {"all": docs}
    for dom in sorted({d["domain"] for d in docs}):
        groups[dom] = [d for d in docs if d["domain"] == dom]
    out = {}
    for name, ds in groups.items():
        g, p = [], []
        for d in ds:
            pr = preds[d["id"]]
            if len(pr) != len(d["labels"]):
                raise ValueError(f"{d['id']}: {len(pr)} predictions for {len(d['labels'])} sentences")
            g += [idx[l] for l in d["labels"]]
            p += [idx[l] for l in pr]
        out[name] = scores(g, p, labels)
    return out


def top_confusions(cm: list[list[int]], labels: list[str], k: int = 10):
    pairs = [(cm[g][p], labels[g], labels[p]) for g in range(len(labels)) for p in range(len(labels))
             if g != p and cm[g][p]]
    return sorted(pairs, reverse=True)[:k]


def format_report(results: dict[str, dict], labels: list[str]) -> str:
    """results: {"Stage A": evaluate(...), "Stage B": evaluate(...)} -> text tables."""
    names = list(results)
    doms = [k for k in next(iter(results.values())) if k != "all"]
    lines = [f"{'':10s}" + "".join(f"{n:>14s}" for n in names)]
    for key in ["all"] + doms:
        lines.append(f"{('macro-F1 ' + key):10s}" + "".join(f"{results[n][key]['macro_f1']:14.4f}" for n in names))
    lines.append(f"{'accuracy':10s}" + "".join(f"{results[n]['all']['accuracy']:14.4f}" for n in names))
    lines.append("")
    lines.append(f"{'role':24s}{'support':>8s}" + "".join(f"{('F1 ' + n):>14s}" for n in names))
    first = results[names[0]]["all"]["per_label"]
    for l in sorted(labels, key=lambda l: -first[l]["support"]):
        lines.append(f"{l:24s}{first[l]['support']:8d}" +
                     "".join(f"{results[n]['all']['per_label'][l]['f1']:14.3f}" for n in names))
    last = names[-1]
    lines.append(f"\nmost common mistakes ({last}): gold -> predicted")
    for c, g, p in top_confusions(results[last]["all"]["confusion"], labels):
        lines.append(f"  {c:5d}  {g} -> {p}")
    return "\n".join(lines)


def confusion_csv(cm: list[list[int]], labels: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("gold\\pred," + ",".join(labels) + "\n")
        for l, row in zip(labels, cm):
            f.write(l + "," + ",".join(map(str, row)) + "\n")


# ----------------------------------------------------------------------- submission
def write_submission(test_docs: list[dict], preds: dict[str, list[str]], labels: list[str],
                     out_dir: str) -> str:
    """Checks every judgment and sentence count, writes prediction.json and submission.zip."""
    allowed = set(labels)
    problems, sub = [], {}
    for d in test_docs:
        p = preds.get(d["id"])
        if p is None:
            problems.append(f"{d['id']}: no prediction")
            continue
        if len(p) != len(d["sents"]):
            problems.append(f"{d['id']}: {len(p)} labels for {len(d['sents'])} sentences")
        bad = sorted({x for x in p if x not in allowed})
        if bad:
            problems.append(f"{d['id']}: unknown labels {bad}")
        sub[d["id"]] = [str(x) for x in p]
    if problems:
        raise ValueError("submission check failed:\n  " + "\n  ".join(problems[:20]))
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "prediction.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    zpath = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jpath, arcname="prediction.json")
    with zipfile.ZipFile(zpath) as z:                      # read back what Codabench will read
        names = z.namelist()
        back = json.loads(z.read("prediction.json"))
    assert names == ["prediction.json"] and back == sub, "zip read-back mismatch"
    n = sum(len(v) for v in sub.values())
    dist = Counter(l for v in sub.values() for l in v)
    print(f"[submission] {zpath}: {len(sub)} judgments, {n} sentences, all counts match")
    print("             predicted roles:", ", ".join(f"{l} {c}" for l, c in dist.most_common()))
    return zpath