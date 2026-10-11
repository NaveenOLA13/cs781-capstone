"""lner_core.py - everything except the neural network (pure Python + NumPy).

Data loading, BIO labels, sliding windows + merging, BIO-constrained Viterbi decoding,
per-label threshold tuning, post-processing, strict macro-F1, stratified folds,
cross-tokenizer projection for ensembling. Used by train_lner.py and ensemble.py.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

IGNORE = -100


# =============================================================================== data
@dataclass(frozen=True)
class Span:
    start: int  # inclusive char offset
    end: int    # exclusive char offset
    label: str


@dataclass
class Doc:
    id: str
    text: str
    spans: list = field(default_factory=list)
    fold: str | None = None
    entity_texts: list = field(default_factory=list)


def _get(r, *keys, default=None):
    for k in keys:
        if isinstance(r, dict) and k in r and r[k] is not None:
            return r[k]
    return default


def _read_records(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".csv", ".tsv"):
        import pandas as pd
        df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
        return df.to_dict("records")
    raw = open(path, encoding="utf-8").read().strip()
    if not raw:
        return []
    if raw[0] == "[":
        return json.loads(raw)
    if raw[0] == "{":
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:  # JSONL
            return [json.loads(l) for l in raw.splitlines() if l.strip()]
        for k in ("data", "documents", "docs", "examples", "records"):
            if isinstance(obj.get(k), list):
                return obj[k]
        if obj and all(isinstance(v, dict) for v in obj.values()):  # {doc_id: record}
            return [dict(v, id=v.get("id", k)) for k, v in obj.items()]
        return [obj]
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def _parse_entities(ents):
    if ents is None:
        return []
    if isinstance(ents, str):
        ents = json.loads(ents) if ents.strip() else []
    if hasattr(ents, "tolist"):
        ents = ents.tolist()
    out = []
    if isinstance(ents, dict):  # {"APP": [[s, e], ...], ...}
        for lab, lst in ents.items():
            for e in lst:
                out.append((int(e[0]), int(e[1]), str(lab), e[2] if len(e) > 2 else None))
        return out
    for e in ents:
        if isinstance(e, (list, tuple)):
            out.append((int(e[0]), int(e[1]), str(e[2]), e[3] if len(e) > 3 else None))
            continue
        e = dict(e)
        if isinstance(e.get("value"), dict):  # Label Studio style
            e = e["value"]
        s = _get(e, "start", "start_offset", "begin", "start_char")
        en = _get(e, "end", "end_offset", "end_char")
        lab = _get(e, "label", "type", "tag", "entity", "entity_type", "labels")
        if isinstance(lab, (list, tuple)):
            lab = lab[0]
        out.append((int(s), int(en), str(lab), _get(e, "text", "entity_text", "span", "word")))
    return out


def load_file(path, fold=None):
    docs = []
    for n, r in enumerate(_read_records(path)):
        text = _get(r, "text", "document", "content", "judgement", "judgment", "data", "doc")
        if isinstance(text, dict):
            text = _get(text, "text", "document", "content")
        if text is None:
            raise KeyError(f"{path}: cannot find the text field; keys = {list(r)}")
        ents = _parse_entities(_get(r, "spans", "entities", "annotations", "labels", "ner", "label"))
        doc_id = str(_get(r, "id", "doc_id", "uid", "document_id", "file_name", "filename",
                          default=f"{os.path.basename(path)}#{n}"))
        docs.append(Doc(doc_id, str(text), [Span(s, e, l) for s, e, l, _ in ents], fold,
                        [t for *_, t in ents]))
    return docs


_EXTS = (".json", ".jsonl", ".parquet", ".csv", ".tsv")


def find_files(data_dir):
    """Classifies data files by name: test / fold1 / fold2 / train / other."""
    found = defaultdict(list)
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.lower().endswith(_EXTS):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, data_dir).lower()
            if re.search(r"(^|[^a-z])test", rel):
                found["test"].append(p)
            elif re.search(r"fold[ _\-]?1", rel):
                found["fold1"].append(p)
            elif re.search(r"fold[ _\-]?2", rel):
                found["fold2"].append(p)
            elif re.search(r"train|dev|val", rel):
                found["train"].append(p)
            else:
                found["other"].append(p)
    return dict(found)


def load_data(data_dir=None, train_files=None, test_files=None):
    """Returns (train_docs, test_docs, files). Explicit file lists override data_dir."""
    files = find_files(data_dir) if data_dir else {}
    if train_files:
        files = {k: v for k, v in files.items() if k == "test"}
        files["train"] = list(train_files)
    if test_files:
        files["test"] = list(test_files)
    train = []
    for key in ("fold1", "fold2", "train"):
        for p in files.get(key, []):
            train += load_file(p, key if key != "train" else None)
    test = [d for p in files.get("test", []) for d in load_file(p, "test")]
    if not train:
        raise FileNotFoundError(f"No training files found. Detected: {files}. Use --train_files.")
    for docs in (train, test):  # make ids unique
        c = Counter(d.id for d in docs)
        if any(v > 1 for v in c.values()):
            for d in docs:
                d.id = f"{d.fold}:{d.id}"
    return train, test, files


def label_set(docs):
    return sorted({s.label for d in docs for s in d.spans})


def check_entity_text(docs, show=5):
    ok = bad = inclusive = missing = 0
    ex = []
    for d in docs:
        for sp, et in zip(d.spans, d.entity_texts or [None] * len(d.spans)):
            if et is None:
                missing += 1
            elif d.text[sp.start:sp.end] == et:
                ok += 1
            else:
                bad += 1
                inclusive += d.text[sp.start:sp.end + 1] == et
                if len(ex) < show:
                    ex.append((d.id, sp.start, sp.end, et, d.text[sp.start:sp.end]))
    return {"match": ok, "mismatch": bad, "mismatch_fixed_if_end_inclusive": inclusive,
            "no_entity_text": missing, "examples": ex}


def data_report(docs, name="data"):
    lens = sorted(len(d.text) for d in docs) or [0]
    overl = sum(1 for d in docs for a, b in zip(sorted(d.spans, key=lambda s: s.start),
                                                sorted(d.spans, key=lambda s: s.start)[1:]) if b.start < a.end)
    print(f"[{name}] docs={len(docs)} folds={dict(Counter(d.fold for d in docs))}")
    print(f"[{name}] chars: median={lens[len(lens) // 2]} max={lens[-1]}")
    print(f"[{name}] entities: {dict(Counter(s.label for d in docs for s in d.spans).most_common())}")
    if any(d.spans for d in docs):
        print(f"[{name}] text[start:end]==entity_text: {check_entity_text(docs)}")
        print(f"[{name}] overlapping gold pairs (BIO can't represent): {overl}")


# ========================================================================= labels
class LabelScheme:
    def __init__(self, types):
        self.types = list(types)
        self.labels = ["O"] + [f"{p}-{t}" for t in self.types for p in ("B", "I")]
        self.id = {l: i for i, l in enumerate(self.labels)}
        self.O = 0
        self.B = np.array([self.id[f"B-{t}"] for t in self.types], dtype=np.int64)
        self.I = np.array([self.id[f"I-{t}"] for t in self.types], dtype=np.int64)
        C = len(self.labels)
        allowed = np.ones((C, C), bool)  # allowed[prev, cur]
        for b, i in zip(self.B, self.I):
            allowed[:, i] = False
            allowed[b, i] = allowed[i, i] = True
        self.trans = np.where(allowed, 0.0, -1e4)
        self.start = np.zeros(C)
        self.start[self.I] = -1e4
        self.is_b = np.zeros(C, bool)
        self.is_b[self.B] = True
        self.type_of = {}
        for t, b, i in zip(self.types, self.B, self.I):
            self.type_of[int(b)] = t
            self.type_of[int(i)] = t

    def __len__(self):
        return len(self.labels)

    def bias_vector(self, biases):
        v = np.zeros(len(self.labels), np.float32)
        for t, b in (biases or {}).items():
            if t in self.types:
                k = self.types.index(t)
                v[self.B[k]] += b
                v[self.I[k]] += b
        return v


def normalize_offsets(text, offsets):
    """Strip whitespace inside token offsets (SentencePiece '▁'); whitespace-only and
    special tokens become (-1, -1)."""
    out = []
    for o in offsets:
        s, e = (o[0], o[1]) if o is not None else (None, None)
        if s is None or e is None or e <= s:
            out.append((-1, -1))
            continue
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        out.append((s, e) if e > s else (-1, -1))
    return out


def spans_to_bio(offsets, spans, scheme):
    """Token gets the label of any gold span it overlaps; first such token is B."""
    lab = np.array([IGNORE if s < 0 else scheme.O for s, _ in offsets], dtype=np.int64)
    if not len(offsets):
        return lab
    starts = np.array([s for s, _ in offsets])
    ends = np.array([e for _, e in offsets])
    for sp in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        k = scheme.types.index(sp.label)
        idx = np.nonzero((starts >= 0) & (ends > sp.start) & (starts < sp.end))[0]
        idx = [i for i in idx if lab[i] == scheme.O]
        if idx:
            lab[idx[0]] = scheme.B[k]
            lab[idx[1:]] = scheme.I[k]
    return lab


# ========================================================================= windows
def make_windows(n, L, stride):
    """Windows of <= L content tokens, starting every `stride` tokens; last one is full."""
    if n == 0:
        return []
    if n <= L:
        return [(0, n)]
    out, s = [], 0
    while True:
        e = min(s + L, n)
        if e == n:
            out.append((n - L, n))
            return out
        out.append((s, e))
        s += stride


def _centrality(n, floor=0.05):
    if n == 1:
        return np.ones(1, np.float32)
    pos = np.arange(n)
    return np.maximum(1.0 - np.abs(pos - (n - 1) / 2) / ((n - 1) / 2), floor).astype(np.float32)


def merge_window_probs(n, windows, probs):
    """Weighted average of overlapping window probabilities (tokens near a window's
    centre - more context - count more)."""
    C = probs[0].shape[1]
    acc = np.zeros((n, C), np.float32)
    wsum = np.zeros(n, np.float32)
    for (s, e), p in zip(windows, probs):
        w = _centrality(e - s)
        acc[s:e] += p * w[:, None]
        wsum[s:e] += w
    return acc / np.maximum(wsum, 1e-9)[:, None]


# ======================================================================== decoding
def viterbi(logp, scheme):
    T, C = logp.shape
    score = logp[0] + scheme.start
    back = np.empty((T, C), np.int32)
    cols = np.arange(C)
    for k in range(1, T):
        cand = score[:, None] + scheme.trans
        back[k] = cand.argmax(0)
        score = cand[back[k], cols] + logp[k]
    path = np.empty(T, np.int32)
    path[-1] = int(score.argmax())
    for k in range(T - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


class Prepared:
    """One document's probabilities, ready for repeated decoding with different biases.
    Only regions where some entity has probability > `floor` are decoded (exactly O
    elsewhere), which makes threshold tuning fast."""

    def __init__(self, text, offsets, probs, scheme, floor=1e-5, pad=2):
        offs = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
        valid = np.nonzero(offs[:, 0] >= 0)[0]
        self.text = text
        self.offs = offs[valid]
        P = np.asarray(probs, dtype=np.float32).reshape(-1, len(scheme))[valid]
        self.ent = 1.0 - P[:, scheme.O]
        self.logp = np.log(np.clip(P, 1e-9, 1.0))
        keep = self.ent > floor
        if pad and keep.any():
            keep = np.convolve(keep.astype(np.int32), np.ones(2 * pad + 1, np.int32), "same") > 0
        d = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
        self.runs = list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))

    def decode(self, scheme, bias_vec=None):
        """Returns [(Span, confidence)]."""
        out = []
        for a, b in self.runs:
            lp = self.logp[a:b] if bias_vec is None else self.logp[a:b] + bias_vec
            path = viterbi(lp, scheme)
            cur = None
            for k, lab in enumerate(path):
                lab = int(lab)
                if lab == scheme.O:
                    if cur:
                        out.append(cur)
                        cur = None
                    continue
                t = scheme.type_of[lab]
                if scheme.is_b[lab] or cur is None or cur[0] != t:
                    if cur:
                        out.append(cur)
                    cur = [t, a + k, a + k]
                else:
                    cur[2] = a + k
            if cur:
                out.append(cur)
        return [(Span(int(self.offs[i, 0]), int(self.offs[j, 1]), t), float(self.ent[i:j + 1].mean()))
                for t, i, j in out]


# ================================================================= post-processing
TRIM_CHARS = set(" \t\n\r,;:.!?\"'`()[]{}-–—/\\|*")
DEFAULT_FLAGS = {"trim": True, "consistency": False, "propagate": False}


def learn_trim_rules(docs):
    """Edge 'words' that gold spans legitimately end/start with (e.g. 'Ltd.', 'U.P.',
    'A.G.A.'). Punctuation is only trimmed if the result isn't one of these."""
    kr, kl = Counter(), Counter()
    for d in docs:
        for s in d.spans:
            t = d.text[s.start:s.end]
            if t and t[-1] in TRIM_CHARS and not t[-1].isspace():
                m = re.search(r"\S+$", t)
                kr[m.group()] += 1
            if t and t[0] in TRIM_CHARS and not t[0].isspace():
                kl[re.match(r"\S+", t).group()] += 1
    return {"keep_right": sorted(kr), "keep_left": sorted(kl)}


def trim_span(text, sp, rules):
    kr, kl = rules["_kr"], rules["_kl"]
    s, e = sp.start, sp.end
    changed = True
    while changed and e > s:
        changed = False
        while e > s and text[e - 1].isspace():
            e -= 1
        while s < e and text[s].isspace():
            s += 1
        if e > s and text[e - 1] in TRIM_CHARS and re.search(r"\S+$", text[s:e]).group() not in kr:
            e -= 1
            changed = True
        if e > s and text[s] in TRIM_CHARS and re.match(r"\S+", text[s:e]).group() not in kl:
            s += 1
            changed = True
    return Span(s, e, sp.label) if e > s else None


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def make_consistent(text, preds):
    """Same surface string in one document -> one label (confidence-weighted vote)."""
    votes = defaultdict(Counter)
    for sp, c in preds:
        votes[_norm(text[sp.start:sp.end])][sp.label] += c
    return [(Span(sp.start, sp.end, votes[_norm(text[sp.start:sp.end])].most_common(1)[0][0]), c)
            for sp, c in preds]


def propagate(text, preds, min_len=4, min_conf=0.6):
    """Label other exact mentions of a confidently predicted name in the same document."""
    occ = np.zeros(len(text) + 1, bool)
    for sp, _ in preds:
        occ[sp.start:sp.end] = True
    new, seen = [], set()
    for sp, c in sorted(preds, key=lambda x: -x[1]):
        surf = text[sp.start:sp.end]
        if c < min_conf or len(surf) < min_len or surf in seen or not re.search(r"[A-Za-z]", surf):
            continue
        seen.add(surf)
        for m in re.finditer(r"(?<!\w)" + re.escape(surf) + r"(?!\w)", text):
            if not occ[m.start():m.end()].any():
                new.append((Span(m.start(), m.end(), sp.label), c * 0.9))
                occ[m.start():m.end()] = True
    return preds + new


def postprocess(text, preds, rules, flags=DEFAULT_FLAGS):
    if "_kr" not in rules:
        rules["_kr"], rules["_kl"] = set(rules["keep_right"]), set(rules["keep_left"])
    if flags.get("consistency"):
        preds = make_consistent(text, preds)
    if flags.get("propagate"):
        preds = propagate(text, preds)
    out = set()
    for sp, _ in preds:
        if flags.get("trim", True):
            sp = trim_span(text, sp, rules)
        if sp is not None:
            out.add(sp)
    return sorted(out, key=lambda s: (s.start, s.end, s.label))


# ========================================================================= scoring
def macro_f1(gold, pred, labels=None):
    """Strict: exact start, end and label. Labels with no gold and no predictions are
    skipped (check the official scorer does the same)."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for did in set(gold) | set(pred):
        G, P = set(gold.get(did, [])), set(pred.get(did, []))
        for s in P:
            (tp if s in G else fp)[s.label] += 1
        for s in G - P:
            fn[s.label] += 1
    labels = labels or sorted(set(tp) | set(fp) | set(fn))
    per = {}
    for l in labels:
        t, a, b = tp[l], fp[l], fn[l]
        if t + a + b == 0:
            continue
        p = t / (t + a) if t + a else 0.0
        r = t / (t + b) if t + b else 0.0
        per[l] = {"f1": 2 * p * r / (p + r) if p + r else 0.0, "p": p, "r": r, "tp": t, "fp": a, "fn": b}
    return {"macro_f1": float(np.mean([v["f1"] for v in per.values()])) if per else 0.0, "per_class": per}


def error_report(gold, pred):
    per, conf = defaultdict(Counter), Counter()
    for did in set(gold) | set(pred):
        G, P = set(gold.get(did, [])), set(pred.get(did, []))
        for p in P - G:
            ov = [g for g in G if g.start < p.end and p.start < g.end]
            same = [g for g in ov if (g.start, g.end) == (p.start, p.end)]
            if same:
                per[p.label]["wrong_type"] += 1
                conf[f"{same[0].label} -> {p.label}"] += 1
            elif any(g.label == p.label for g in ov):
                per[p.label]["boundary"] += 1
            elif ov:
                per[p.label]["other_type_and_boundary"] += 1
                conf[f"{ov[0].label} -> {p.label} (+boundary)"] += 1
            else:
                per[p.label]["spurious"] += 1
        for g in G - P:
            if not any(p.start < g.end and g.start < p.end for p in P):
                per[g.label]["missed"] += 1
    return {"per_label": {k: dict(v) for k, v in per.items()}, "confusions (gold -> pred)": dict(conf.most_common())}


def print_scores(res, title):
    print(f"{title}: macro-F1 = {res['macro_f1']:.4f}")
    for l, v in sorted(res["per_class"].items()):
        print(f"   {l:14s} F1 {v['f1']:.4f}  P {v['p']:.3f}  R {v['r']:.3f}  (gold {v['tp'] + v['fn']})")


# =========================================================================== folds
def assign_folds(docs, k, seed=42):
    """Document-level folds balanced per label (rare labels placed first)."""
    rng = random.Random(seed)
    total = Counter(s.label for d in docs for s in d.spans)
    order = docs[:]
    rng.shuffle(order)
    order.sort(key=lambda d: sum(1.0 / total[s.label] for s in d.spans), reverse=True)
    tgt = {l: n / k for l, n in total.items()}
    tdocs = len(docs) / k
    cur, nd, fold = [Counter() for _ in range(k)], [0] * k, {}
    for d in order:
        c = Counter(s.label for s in d.spans)
        sc = lambda f: (sum(n * (tgt[l] - cur[f][l]) / tgt[l] for l, n in c.items())
                        + (tdocs - nd[f]) / tdocs, rng.random())
        f = max(range(k), key=sc)
        fold[d.id] = f
        cur[f] += c
        nd[f] += 1
    return fold


# ========================================================================== tuning
BIAS_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def tune(prepared, gold, scheme, rules, grid=BIAS_GRID, min_gain=0.003, log=print):
    """Per-label log-prob bias (= per-label threshold) by coordinate ascent on macro-F1,
    then post-processing switches, each kept only if it gains >= min_gain."""
    ids = list(prepared)
    g = {i: gold.get(i, []) for i in ids}

    def score(biases, flags):
        bv = scheme.bias_vector(biases)
        pred = {i: postprocess(prepared[i].text, prepared[i].decode(scheme, bv), rules, flags) for i in ids}
        return macro_f1(g, pred, scheme.types)["macro_f1"]

    flags, biases = dict(DEFAULT_FLAGS), {t: 0.0 for t in scheme.types}
    base = best = score(biases, flags)
    log(f"[tune] start macro-F1 {base:.4f}")
    for t in scheme.types:
        for b in grid:
            if b == biases[t]:
                continue
            trial = dict(biases)
            trial[t] = b
            f = score(trial, flags)
            if f > best + 1e-9:
                best, biases = f, trial
        log(f"[tune] bias {t:14s} = {biases[t]:+.1f}  -> {best:.4f}")
    after_bias = best
    for name in ("trim", "consistency", "propagate"):
        trial = dict(flags)
        trial[name] = not flags[name]
        f = score(biases, trial)
        log(f"[tune] {name}={trial[name]}: {f:.4f} (current {best:.4f})")
        if f > best + min_gain:
            best, flags = f, trial
    return {"biases": biases, "flags": flags, "f1_start": base, "f1_after_bias": after_bias, "f1_final": best}


def predict_spans(prepared, scheme, rules, pp=None):
    pp = pp or {"biases": {}, "flags": DEFAULT_FLAGS}
    bv = scheme.bias_vector(pp["biases"])
    return {i: postprocess(p.text, p.decode(scheme, bv), rules, pp["flags"]) for i, p in prepared.items()}


def ceiling_check(docs, offsets_by_id, scheme, rules):
    """Gold -> BIO -> decode -> F1 with perfect one-hot probabilities. Must be ~1.0."""
    gold, prep = {}, {}
    for d in docs:
        offs = offsets_by_id[d.id]
        lab = spans_to_bio(offs, d.spans, scheme)
        oh = np.zeros((len(lab), len(scheme)), np.float32)
        oh[np.arange(len(lab)), np.where(lab == IGNORE, scheme.O, lab)] = 1.0
        prep[d.id] = Prepared(d.text, offs, oh, scheme)
        gold[d.id] = d.spans
    return macro_f1(gold, predict_spans(prep, scheme, rules), scheme.types), gold


# ==================================================================== ensembling
UNIT_RE = re.compile(r"\w+|[^\w\s]")


def word_units(text):
    return np.array([m.span() for m in UNIT_RE.finditer(text)], dtype=np.int64).reshape(-1, 2)


def project_to_units(text, offsets, probs, scheme):
    """Token probabilities -> shared word-unit probabilities, so models with different
    tokenizers can be averaged. A unit takes its first overlapping token's probs; if
    that token began in an earlier unit, its B mass is moved to I."""
    units = word_units(text)
    offs = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    P = np.asarray(probs, dtype=np.float32)
    idx = np.nonzero(offs[:, 0] >= 0)[0]
    st, en = offs[idx, 0], offs[idx, 1]
    C = len(scheme)
    b2i = np.arange(C)
    b2i[scheme.B] = scheme.I
    U = np.zeros((len(units), C), np.float32)
    U[:, scheme.O] = 1.0
    j = 0
    for u, (us, ue) in enumerate(units):
        while j < len(idx) and en[j] <= us:
            j += 1
        if j < len(idx) and st[j] < ue:
            p = P[idx[j]]
            if st[j] < us:
                q = np.zeros(C, np.float32)
                np.add.at(q, b2i, p)
                p = q
            U[u] = p
    return units, U


# ============================================================================== io
def save_probs(path, store):
    """store: {doc_id: (offsets, probs)} -> pickle with float16 probs."""
    with open(path, "wb") as f:
        pickle.dump({k: (np.asarray(o, np.int32), np.asarray(p, np.float16)) for k, (o, p) in store.items()}, f)


def load_probs(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_predictions(path, docs, preds):
    """JSONL: {"id", "entities": [{"start","end","label","text"}]}. Adapt this one
    function to the exact Codabench submission format."""
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            ents = [{"start": s.start, "end": s.end, "label": s.label, "text": d.text[s.start:s.end]}
                    for s in preds.get(d.id, [])]
            f.write(json.dumps({"id": d.id, "entities": ents}, ensure_ascii=False) + "\n")