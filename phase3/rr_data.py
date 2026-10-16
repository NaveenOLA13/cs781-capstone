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
                   pool: int = 50, exact: bool = False) -> list[list[int]]:
    """Batches of indices with similar lengths (little padding).
    shuffle=False: sorted by length (prediction). shuffle=True: random pools of
    batch_size*pool items are sorted by length and cut into batches; batch order is shuffled.
    exact=True: every batch holds sentences of exactly one length (no padding at all), for
    models whose outputs change with padding (e.g. a bidirectional RNN encoder)."""
    idx = list(range(len(lengths)))
    if exact:
        rng = random.Random(seed)
        groups: dict[int, list[int]] = {}
        for i in idx:
            groups.setdefault(lengths[i], []).append(i)
        batches = []
        for L in sorted(groups):
            g = groups[L]
            if shuffle:
                rng.shuffle(g)
            batches += [g[i:i + batch_size] for i in range(0, len(g), batch_size)]
        if shuffle:
            rng.shuffle(batches)
        return batches
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


def format_report(results: dict[str, dict], labels: list[str], k_mistakes: int = 8) -> str:
    """results: {"Stage A": evaluate(...), "Stage B": evaluate(...), ...} -> text tables:
    13-role average and macro-F1 (overall, CL, IT), accuracy, per-role F1 for each domain,
    and the most common mistakes of the last result in each domain."""
    names = list(results)
    doms = [k for k in next(iter(results.values())) if k != "all"]
    w = max(10, max(len(n) for n in names) + 2)
    lines = [f"{'':20s}" + "".join(f"{n:>{w}s}" for n in names)]
    for key in ["all"] + doms:
        lines.append(f"{('13-role avg ' + key):20s}" +
                     "".join(f"{results[n][key]['macro_f1_all13']:{w}.4f}" for n in names))
    for key in ["all"] + doms:
        lines.append(f"{('macro-F1 ' + key):20s}" + "".join(f"{results[n][key]['macro_f1']:{w}.4f}" for n in names))
    lines.append(f"{'accuracy':20s}" + "".join(f"{results[n]['all']['accuracy']:{w}.4f}" for n in names))
    lines.append("   (13-role avg = every role counts, as on Codabench; macro-F1 = only roles present)")
    lines.append("")
    cw = max(9, max(len(n) for n in names) + 5)
    lines.append(f"{'F1 per role':24s}" + "".join(f"{('n ' + dm):>7s}" for dm in doms) +
                 "".join(f"{(n + ' ' + dm):>{cw}s}" for n in names for dm in doms))
    first = results[names[0]]
    order = sorted(labels, key=lambda l: -first["all"]["per_label"][l]["support"])
    for l in order:
        lines.append(f"{l:24s}" + "".join(f"{first[dm]['per_label'][l]['support']:7d}" for dm in doms) +
                     "".join(f"{results[n][dm]['per_label'][l]['f1']:{cw}.3f}" for n in names for dm in doms))
    last = names[-1]
    for dm in doms:
        lines.append(f"\nmost common mistakes ({last}, {dm}): gold -> predicted")
        for c, g, p in top_confusions(results[last][dm]["confusion"], labels, k_mistakes):
            lines.append(f"  {c:5d}  {g} -> {p}")
    return "\n".join(lines)


def confusion_csv(cm: list[list[int]], labels: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("gold\\pred," + ",".join(labels) + "\n")
        for l, row in zip(labels, cm):
            f.write(l + "," + ",".join(map(str, row)) + "\n")


# ------------------------------------------------------------------- comparing runs
# Roles that are easy to mix up: similar wording, or the same part of the judgment.
# The first four include the most common mistakes of encdec_full (PrecRelied->Ratio, ArgResp->ArgPet).
CLOSE_PAIRS = [
    ("Pet/Resp", "ArgumentPetitioner", "ArgumentRespondent"),
    ("Relied/Ratio", "PrecedentReliedUpon", "RatioOfTheDecision"),
    ("Relied/NotRel", "PrecedentReliedUpon", "PrecedentNotReliedUpon"),
    ("Fact/Ratio", "Fact", "RatioOfTheDecision"),
    ("Lower/Fact", "RulingByLowerCourt", "Fact"),
    ("Lower/Present", "RulingByLowerCourt", "RulingByPresentCourt"),
    ("Ratio/Present", "RatioOfTheDecision", "RulingByPresentCourt"),
]


def pair_confusions(rep: dict, labels: list[str], pairs=CLOSE_PAIRS) -> dict:
    """For each pair (a, b) and domain: of the sentences whose gold role is a or b, the share that
    got the other role of the pair (rate; lower = the two roles are kept apart better).
    rep: evaluate(...) output. Returns {short name: {domain: {rate, a_to_b, b_to_a, n}}}."""
    idx = {l: i for i, l in enumerate(labels)}
    out = {}
    for short, a, b in pairs:
        if a not in idx or b not in idx:
            continue
        ia, ib = idx[a], idx[b]
        out[short] = {}
        for dom, s in rep.items():
            cm = s["confusion"]
            n = sum(cm[ia]) + sum(cm[ib])
            ab, ba = cm[ia][ib], cm[ib][ia]
            out[short][dom] = dict(rate=(ab + ba) / n if n else None, a_to_b=ab, b_to_a=ba, n=n)
    return out


def _fmt(x, spec: str, width: int) -> str:
    return f"{'-':>{width}s}" if x is None else f"{x:>{width}{spec}}"


def format_comparison(runs: dict[str, dict | None], pairs=CLOSE_PAIRS) -> str:
    """runs: {method name: metrics dict saved by rr_train.run (None = the run failed)} -> tables:
    cost, dev 13-role average (all / CL / IT) of Stage A and Stage B, memorising (train vs dev
    accuracy of Stage A), and how often closely related roles are mixed up, per domain."""
    ok = {k: v for k, v in runs.items() if v}
    failed = [k for k, v in runs.items() if not v]
    if not ok:
        return "no finished runs" + (f" (failed: {', '.join(failed)})" if failed else "")
    first = next(iter(ok.values()))
    labels = list(first["stage_a"]["dev"]["all"]["per_label"])
    doms = [k for k in first["stage_a"]["dev"] if k != "all"]
    w = max(10, max(len(k) for k in ok) + 2)
    L = []

    L.append("COST (Stage A)")
    L.append(f"{'method':{w}s}{'model size':>12s}{'trainable':>12s}{'% of all':>11s}{'saved file':>13s}"
             f"{'peak GPU':>11s}{'time':>9s}{'s/epoch':>9s}{'best epoch':>12s}{'dev pass':>10s}")
    for k, m in ok.items():
        a = m["stage_a"]
        hist = a.get("history") or []
        per_ep = sum(h["seconds"] for h in hist) / len(hist) if hist else None
        tot = a.get("total_params")
        pct = 100 * a["trainable_params"] / tot if tot else None
        L.append(f"{k:{w}s}{_fmt(None if tot is None else tot / 1e6, '.2f', 11)}M"
                 f"{a['trainable_params'] / 1e6:>11.3f}M{_fmt(pct, '.2f', 10)}%"
                 f"{_fmt(a.get('saved_mb'), '.1f', 10)} MB{_fmt(a.get('peak_gpu_gb'), '.2f', 8)} GB"
                 f"{a['train_seconds'] / 60:>8.1f}m{_fmt(per_ep, '.0f', 8)}s{a['best_epoch']:>9d}/{len(hist):<2d}"
                 f"{_fmt(a.get('dev_infer_seconds'), '.1f', 9)}s")
    L.append("   (model size = all weights of one model; dev pass = Stage A time to read every dev sentence once)")

    L.append("\nDEV 13-ROLE AVERAGE (the Codabench metric)")
    cols = [(st, dm) for st in ("A", "B") for dm in ["all"] + doms]
    L.append(f"{'method':{w}s}" + "".join(f"{(st + ' ' + dm):>9s}" for st, dm in cols) + f"{'B acc':>9s}")
    for k, m in ok.items():
        st_key = {"A": "stage_a", "B": "stage_b"}
        L.append(f"{k:{w}s}" + "".join(f"{m[st_key[st]]['dev'][dm]['macro_f1_all13']:>9.4f}" for st, dm in cols)
                 + f"{m['stage_b']['dev']['all']['accuracy']:>9.4f}")

    L.append("\nMEMORISING (Stage A accuracy on its own training sentences vs dev; big gap = memorising)")
    L.append(f"{'method':{w}s}{'train acc':>11s}{'dev acc':>9s}{'gap':>8s}")
    for k, m in ok.items():
        fit = (m["stage_a"].get("train_fit") or {}).get("all", {}).get("accuracy")
        dev = m["stage_a"]["dev"]["all"]["accuracy"]
        L.append(f"{k:{w}s}{_fmt(fit, '.3f', 11)}{dev:>9.3f}{_fmt(None if fit is None else fit - dev, '.3f', 8)}")

    names = [p[0] for p in pairs if p[1] in labels and p[2] in labels]
    if names:
        pw = max(9, max(len(n) for n in names) + 2)
        L.append("\nCLOSELY RELATED ROLES: % of the two roles' dev sentences given the other role of the pair "
                 "(lower = kept apart better)")
        L += [f"  {s:14s} {a} / {b}" for s, a, b in pairs if s in names]
        for st, st_key in (("Stage A", "stage_a"), ("Stage B", "stage_b")):
            pc = {k: pair_confusions(m[st_key]["dev"], labels, pairs) for k, m in ok.items()}
            for dm in ["all"] + doms:
                L.append(f"\n{st}, {dm}")
                L.append(f"{'method':{w}s}" + "".join(f"{n:>{pw}s}" for n in names))
                for k in ok:
                    L.append(f"{k:{w}s}" + "".join(
                        _fmt(None if pc[k][n][dm]["rate"] is None else 100 * pc[k][n][dm]["rate"], ".1f", pw)
                        for n in names))
                ref = next(iter(pc.values()))
                L.append(f"{'(sentences)':{w}s}" + "".join(f"{ref[n][dm]['n']:>{pw}d}" for n in names))
    if failed:
        L.append(f"\nfailed runs: {', '.join(failed)}")
    return "\n".join(L)


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


# ------------------------------------------------------------------ cross-validation
def make_folds(docs: list[dict], k: int = 5, seed: int = 0) -> list[list[int]]:
    """k groups of document indices with the same share of each domain (CL, IT) in every group."""
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    j = 0
    for dom in sorted({d["domain"] for d in docs}):
        idx = [i for i, d in enumerate(docs) if d["domain"] == dom]
        rng.shuffle(idx)
        for i in idx:
            folds[j % k].append(i)
            j += 1
    return [sorted(f) for f in folds]


def macro_f1_np(gold, pred, n_labels: int, present_only: bool = False) -> float:
    """Fast macro-F1 with numpy. Default: average over all n_labels roles (Codabench style)."""
    import numpy as np
    gold, pred = np.asarray(gold), np.asarray(pred)
    cm = np.bincount(gold * n_labels + pred, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    p = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    r = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    if present_only:
        m = (support + predicted) > 0
        return float(f[m].mean()) if m.any() else 0.0
    return float(f.mean())


def tune_biases(logp, gold, n_labels: int, groups=None, n_groups: int = 1, grid=None, passes: int = 3):
    """Per-role numbers added to the log-probabilities before choosing the role, set to
    maximise the 13-role macro-F1 (a role with a positive bias is chosen more readily).
    groups: optional group id per sentence (e.g. 0 = CL, 1 = IT) -> one bias vector per group.
    Coordinate ascent: for each (group, role) try every grid value, keep the best; repeat.
    Returns (biases [n_groups, n_labels], score before, score after)."""
    import numpy as np
    logp = np.asarray(logp, dtype=np.float64)
    gold = np.asarray(gold)
    groups = np.zeros(len(gold), dtype=int) if groups is None else np.asarray(groups)
    grid = np.round(np.arange(-3.0, 3.0001, 0.25), 4) if grid is None else np.asarray(grid)
    b = np.zeros((n_groups, n_labels))

    def score():
        return macro_f1_np(gold, (logp + b[groups]).argmax(1), n_labels)

    start = best = score()
    for _ in range(passes):
        improved = False
        for g in range(n_groups):
            if not (groups == g).any():
                continue
            for c in range(n_labels):
                keep = b[g, c]
                for v in grid:
                    b[g, c] = v
                    s = score()
                    if s > best + 1e-9:
                        best, keep, improved = s, v, True
                b[g, c] = keep
        if not improved:
            break
    return b, start, best


def biased_preds(docs: list[dict], logps, labels: list[str], bias, group_of) -> dict[str, list[str]]:
    """{doc_id: [role per sentence]} after adding bias[group_of(doc)] to each sentence's log-probs."""
    import numpy as np
    out = {}
    for d, lp in zip(docs, logps):
        if lp is None or len(lp) == 0:
            out[d["id"]] = []
            continue
        out[d["id"]] = [labels[i] for i in (np.asarray(lp, dtype=np.float64) + bias[group_of(d)]).argmax(1)]
    return out