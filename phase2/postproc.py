"""Post-processing on top of the token tag probabilities (no retraining needed).

1. viterbi_spans     : constrained BIO decoding (CRF-style): I-X may only follow B-X / I-X, so the
                       argmax can never produce a broken entity (strict F1 punishes those twice).
2. propagate         : document-level consistency. A string predicted as e.g. APP somewhere in a
                       document is also tagged at its other (unlabelled) mentions in that document.
3. Gazetteer         : strings that the TRAIN gold labels consistently (e.g. "Indian Penal Code" ->
                       STAT in >= min_prec of its occurrences) are tagged wherever they appear.
4. choose_per_label  : every option is kept only for the labels where it improves DEV strict F1
                       (the metric is per label, so labels can use different variants).
Pure numpy / python.
"""
import re
from collections import Counter, defaultdict

import numpy as np


# ------------------------------------------------------------------ 1. constrained decoding
def _allowed(tags):
    """allowed[i, j] = transition tag i -> tag j is valid BIO; start[j] = may start a sequence."""
    T = len(tags)
    allowed = np.ones((T, T), bool)
    start = np.ones(T, bool)
    for j, t in enumerate(tags):
        if t.startswith("I-"):
            lab = t[2:]
            start[j] = False
            for i, s in enumerate(tags):
                allowed[i, j] = s in (f"B-{lab}", f"I-{lab}")
    return allowed, start


def viterbi_tags(probs, spans, tags):
    """probs [n, T] -> tag ids [n] (whitespace tokens get O and are skipped in the chain)."""
    allowed, start = _allowed(tags)
    trans = np.where(allowed, 0.0, -1e9)
    idx = [i for i, (s, e) in enumerate(spans) if s < e]
    out = np.zeros(len(spans), dtype=np.int64)
    if not idx:
        return out
    lp = np.log(np.clip(probs[idx], 1e-9, 1.0))
    n, T = lp.shape
    score = lp[0] + np.where(start, 0.0, -1e9)
    back = np.zeros((n, T), dtype=np.int32)
    for t in range(1, n):
        cand = score[:, None] + trans                      # [from, to]
        back[t] = cand.argmax(0)
        score = cand.max(0) + lp[t]
    best = np.empty(n, dtype=np.int64)
    best[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        best[t - 1] = back[t, best[t]]
    out[idx] = best
    return out


# ------------------------------------------------------------------ helpers
def _occurrences(text, s):
    """All word-bounded occurrences of string s in text."""
    out, i = [], text.find(s)
    while i != -1:
        j = i + len(s)
        okl = i == 0 or not (text[i - 1].isalnum() and s[0].isalnum())
        okr = j == len(text) or not (text[j].isalnum() and s[-1].isalnum())
        if okl and okr:
            out.append((i, j))
        i = text.find(s, i + 1)
    return out


def _add_nonoverlapping(spans, extra):
    """Add (s, e, l) from `extra` that do not overlap any span already present."""
    occ = sorted((s, e) for s, e, _ in spans)
    starts = [s for s, _ in occ]
    import bisect
    res = list(spans)
    for s, e, l in sorted(set(extra), key=lambda x: -(x[1] - x[0])):   # longest first
        k = bisect.bisect_left(starts, e)
        if any(not (e2 <= s or e <= s2) for s2, e2 in occ[max(0, k - 50):k]):
            continue
        k2 = bisect.bisect_left(occ, (s, e))
        occ.insert(k2, (s, e)); starts.insert(k2, s)
        res.append((s, e, l))
    return res


# ------------------------------------------------------------------ 2. propagation
def propagate(spans, text, labels=None, min_len=3):
    votes = defaultdict(Counter)
    for s, e, l in spans:
        if labels is None or l in labels:
            st = text[s:e]
            if len(st.strip()) >= min_len:
                votes[st][l] += 1
    extra = []
    for st, c in votes.items():
        lab = c.most_common(1)[0][0]
        for s, e in _occurrences(text, st):
            extra.append((s, e, lab))
    return _add_nonoverlapping(spans, extra)


# ------------------------------------------------------------------ 3. gazetteer from train gold
def learn_gazetteer(texts, gold, doc_ids, min_count=2, min_prec=0.7, min_len=4):
    """-> {string: label} for strings labelled consistently in the training documents."""
    lab_count = defaultdict(Counter)
    for k in doc_ids:
        for s, e, l in gold.get(k, []):
            st = texts[k][s:e]
            if len(st.strip()) >= min_len:
                lab_count[st][l] += 1
    gaz = {}
    for st, c in lab_count.items():
        lab, n_lab = c.most_common(1)[0]
        if n_lab < min_count:
            continue
        total = sum(len(_occurrences(texts[k], st)) for k in doc_ids)
        if total and n_lab / total >= min_prec:
            gaz[st] = lab
    return gaz


def apply_gazetteer(spans, text, gaz, labels=None):
    extra = []
    for st, lab in gaz.items():
        if labels is not None and lab not in labels:
            continue
        if st in text:
            extra.extend((s, e, lab) for s, e in _occurrences(text, st))
    return _add_nonoverlapping(spans, extra)


# ------------------------------------------------------------------ 4. per-label selection on dev
VARIANTS = ["base", "prop", "gaz", "prop+gaz"]


def run_variant(name, base_spans, texts, gaz):
    out = {}
    for k, sp in base_spans.items():
        x = list(sp)
        if "prop" in name:
            x = propagate(x, texts[k])
        if "gaz" in name:
            x = apply_gazetteer(x, texts[k], gaz)
        out[k] = x
    return out


def choose_per_label(dev_base, dev_texts, dev_gold, gaz, labels, scorer):
    """-> ({label: variant}, combined dev score dict)."""
    per = {}
    preds = {v: run_variant(v, dev_base, dev_texts, gaz) for v in VARIANTS}
    scores = {v: scorer(preds[v], dev_gold, labels) for v in VARIANTS}
    choice = {}
    for lab in labels:
        best = max(VARIANTS, key=lambda v: (round(scores[v]["per_label"][lab]["F1"], 6), -VARIANTS.index(v)))
        choice[lab] = best
    comb = combine(preds, choice)
    return choice, scorer(comb, dev_gold, labels), scores


def combine(preds_by_variant, choice):
    docs = next(iter(preds_by_variant.values())).keys()
    return {k: [x for lab, v in choice.items() for x in preds_by_variant[v][k] if x[2] == lab] for k in docs}


def apply_choice(base, texts, gaz, choice):
    used = set(choice.values())
    preds = {v: run_variant(v, base, texts, gaz) for v in used}
    return combine(preds, choice)
