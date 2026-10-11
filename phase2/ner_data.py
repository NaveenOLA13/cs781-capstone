"""Phase 2 (L-NER) data: loading, char<->token offsets, BIO tags, sliding windows, span decoding.

Pure Python (no torch) so the offset logic can be unit-tested anywhere (see check_offsets.py).

Gold format (HF Exploration-Lab/CS-781-Capstone/Phase_2_LNER):
    fold*-text.json : {doc_id: "full document text"}
    fold*-gold.json : {doc_id: [[start_char, end_char, LABEL, entity_text], ...]}   end is EXCLUSIVE
Submission       : {doc_id: [[start_char, end_char, LABEL(, text)], ...]}

Offsets: my BPE tokenizer splits text into pre-tokens with PRETOKEN_RE, then BPE inside each
pre-token. Every token is a byte string, so I walk the bytes of each pre-token and convert
byte offsets -> char offsets (non-ASCII chars are several bytes). Pre-tokens carry a LEADING
SPACE (" STATE"), so each token's char span is stripped of whitespace; whitespace-only tokens
get an empty span and are ignored in the loss and in decoding.
"""
import json
import os
import random
import re

HF_ROOT = "datasets/Exploration-Lab/CS-781-Capstone/Phase_2_LNER"
DEFAULT_LABELS = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT",
                  "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
IGNORE = -100


# ------------------------------------------------------------------ loading
def _open(rel, data_dir=None):
    """data_dir -> local copy (same sub-paths as HF: Train/..., Test/...); else stream from HF."""
    if data_dir:
        return open(os.path.join(data_dir, rel), "rb")
    from huggingface_hub import HfFileSystem
    return HfFileSystem().open(f"{HF_ROOT}/{rel}", "rb")


def load_json(rel, data_dir=None):
    with _open(rel, data_dir) as f:
        return json.load(f)


def load_labels(data_dir=None):
    try:
        with _open("Train/labels.txt", data_dir) as f:
            labs = [x for x in re.split(r"[\s,]+", f.read().decode("utf-8")) if x]
        return labs or DEFAULT_LABELS
    except Exception as e:                                    # offline fallback
        print(f"[data] labels.txt not readable ({e}); using built-in list")
        return DEFAULT_LABELS


def load_train(data_dir=None, folds=("fold1", "fold2")):
    """-> texts {id: str}, gold {id: [(s, e, label)]}"""
    texts, gold = {}, {}
    for fold in folds:
        t = load_json(f"Train/{fold}-text.json", data_dir)
        g = load_json(f"Train/{fold}-gold.json", data_dir)
        for k, v in t.items():
            texts[k] = v
            gold[k] = [(int(a[0]), int(a[1]), a[2]) for a in g.get(k, [])]
    return texts, gold


def load_test(data_dir=None):
    return load_json("Test/fold3-text.json", data_dir)


def split_docs(ids, dev_frac=0.15, seed=0):
    """Held-out split BY DOCUMENT (no window of a dev doc is ever trained on)."""
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n_dev = max(1, int(round(len(ids) * dev_frac)))
    return ids[n_dev:], ids[:n_dev]


# ------------------------------------------------------------------ tags
def make_tagset(labels):
    tags = ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]
    return tags, {t: i for i, t in enumerate(tags)}


# ------------------------------------------------------------------ tokenization with offsets
def encode_with_offsets(tok, text):
    """-> ids [n], spans [(cs, ce)] char span of each token, whitespace-stripped
    (cs == ce for whitespace-only tokens)."""
    from bpe import PRETOKEN_RE
    ids, spans = [], []
    for m in PRETOKEN_RE.finditer(text):
        w, base = m.group(), m.start()
        wb = w.encode("utf-8")
        # byte index -> char index inside this pre-token
        b2c = []
        for ci, ch in enumerate(w):
            b2c.extend([ci] * len(ch.encode("utf-8")))
        b2c.append(len(w))                                     # sentinel for end
        bpos = 0
        for t in tok._encode_word(w):
            nb = len(tok.vocab[t])
            cs = b2c[bpos]
            last_char = b2c[bpos + nb - 1]                     # char holding the token's last byte
            ce = last_char + 1
            bpos += nb
            # strip whitespace inside the token span
            while cs < ce and w[cs].isspace():
                cs += 1
            while ce > cs and w[ce - 1].isspace():
                ce -= 1
            ids.append(t)
            spans.append((base + cs, base + ce))
        assert bpos == len(wb)
    return ids, spans


def bio_labels(spans, entities, tag2id):
    """Token-level BIO ids from char-level gold entities. A token belongs to an entity if its
    (non-empty) char span overlaps the entity; the first such token gets B-. Nested/overlapping
    gold entities: the longer one wins (it is assigned first)."""
    y = [tag2id["O"] if s < e else IGNORE for (s, e) in spans]
    taken = [False] * len(spans)
    starts = [s for s, _ in spans]
    import bisect
    for (es, ee, lab) in sorted(entities, key=lambda x: -(x[1] - x[0])):
        if f"B-{lab}" not in tag2id:
            continue
        i = max(0, bisect.bisect_right(starts, es) - 1)
        toks = []
        while i < len(spans) and spans[i][0] < ee:
            s, e = spans[i]
            if s < e and e > es and s < ee:
                toks.append(i)
            i += 1
        if not toks or any(taken[j] for j in toks):
            continue
        for k, j in enumerate(toks):
            y[j] = tag2id[("B-" if k == 0 else "I-") + lab]
            taken[j] = True
    return y


def learn_boundary_table(enc, texts, gold, doc_ids):
    """Entity boundaries that fall INSIDE a token (e.g. 'Ltd.' + ',' -> one BPE token '.,'),
    learned from training docs: for the token holding an entity's first/last char, how many
    chars of it (from the token start) are cut off / kept. Only keys whose usual cut differs
    from the default (start: 0, end: whole token) are kept. -> {"start": {tok: cut}, "end": {...}}"""
    from collections import Counter, defaultdict
    st, en = defaultdict(Counter), defaultdict(Counter)
    for k in doc_ids:
        sp, text = enc[k]["spans"], texts[k]
        starts = [a for a, _ in sp]
        import bisect
        for es, ee, _ in gold.get(k, []):
            i = max(0, bisect.bisect_right(starts, es) - 1)
            toks = []
            while i < len(sp) and sp[i][0] < ee:
                if sp[i][0] < sp[i][1] and sp[i][1] > es:
                    toks.append(i)
                i += 1
            if not toks:
                continue
            fs, fe = sp[toks[0]]; ls, le = sp[toks[-1]]
            st[text[fs:fe]][max(0, es - fs)] += 1
            en[text[ls:le]][min(le, ee) - ls] += 1
    table = {"start": {}, "end": {}}
    for t, c in st.items():
        cut = c.most_common(1)[0][0]
        if cut != 0:
            table["start"][t] = cut
    for t, c in en.items():
        cut = c.most_common(1)[0][0]
        if cut != len(t):
            table["end"][t] = cut
    return table


def decode_spans(tag_ids, spans, id2tag, text=None, table=None):
    """Token tag ids -> [(s, e, label)]. Lenient BIO repair: an I-X that does not continue an
    open X entity starts a new one. Whitespace-only tokens are skipped (they neither open nor
    close entities). With text+table, boundaries inside a token are fixed (learn_boundary_table)."""
    out = _decode(tag_ids, spans, id2tag)
    if table is None or text is None:
        return out
    starts = {s: e for s, e in spans if s < e}
    ends = {e: s for s, e in spans if s < e}
    fixed = []
    for s, e, lab in out:
        s2 = s + table["start"].get(text[s:starts[s]], 0) if s in starts else s
        if e in ends:
            ts = ends[e]
            e2 = ts + table["end"].get(text[ts:e], e - ts)
        else:
            e2 = e
        if s2 < e2:
            fixed.append((s2, e2, lab))
    return fixed


def _decode(tag_ids, spans, id2tag):
    """Token tag ids -> [(s, e, label)]. Lenient BIO repair: an I-X that does not continue an
    open X entity starts a new one. Whitespace-only tokens are skipped (they neither open nor
    close entities)."""
    out, cur = [], None                                       # cur = [s, e, label]
    for t, (s, e) in zip(tag_ids, spans):
        if s >= e:
            continue
        tag = id2tag[t]
        if tag == "O":
            if cur: out.append(tuple(cur)); cur = None
            continue
        p, lab = tag.split("-", 1)
        if p == "I" and cur and cur[2] == lab:
            cur[1] = e
        else:
            if cur: out.append(tuple(cur))
            cur = [s, e, lab]
    if cur:
        out.append(tuple(cur))
    return out


# ------------------------------------------------------------------ windows
def window_starts(n, L, stride):
    """Start offsets of length-L windows covering n tokens; the last window is aligned to the
    end so only docs shorter than L ever need padding."""
    if n <= L:
        return [0]
    st = list(range(0, n - L, stride))
    st.append(n - L)
    return sorted(set(st))


def encode_docs(tok, texts, gold=None, tag2id=None):
    """-> {id: {"ids", "spans", "y"(optional)}}"""
    enc = {}
    for k, text in texts.items():
        ids, spans = encode_with_offsets(tok, text)
        d = {"ids": ids, "spans": spans}
        if gold is not None:
            d["y"] = bio_labels(spans, gold.get(k, []), tag2id)
        enc[k] = d
    return enc


def make_windows(enc, doc_ids, L, stride):
    """Training samples: (doc_id, start, ids[L'], y[L'])."""
    out = []
    for k in doc_ids:
        d = enc[k]
        for s in window_starts(len(d["ids"]), L, stride):
            out.append((k, s, d["ids"][s:s + L], d["y"][s:s + L]))
    return out


# ------------------------------------------------------------------ scoring (strict, char-level)
def strict_scores(pred, gold, labels):
    """pred/gold: {doc: [(s, e, label), ...]}. Strict = exact (s, e, label) match.
    Macro = mean over labels of per-label P / R / F1."""
    per = {}
    for lab in labels:
        tp = fp = fn = 0
        for k in gold:
            g = {(s, e) for s, e, l in gold[k] if l == lab}
            p = {(int(x[0]), int(x[1])) for x in pred.get(k, []) if x[2] == lab}
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        per[lab] = {"P": P, "R": R, "F1": F, "tp": tp, "fp": fp, "fn": fn}
    n = len(labels)
    return {"macro_F1": sum(v["F1"] for v in per.values()) / n,
            "macro_P": sum(v["P"] for v in per.values()) / n,
            "macro_R": sum(v["R"] for v in per.values()) / n,
            "per_label": per}


def write_submission(pred, texts, out_dir):
    """prediction.json (every test id, [s, e, label, text]) + submission.zip with it at the root."""
    import zipfile
    os.makedirs(out_dir, exist_ok=True)
    sub = {k: [[int(s), int(e), l, texts[k][s:e]] for (s, e, l) in sorted(pred.get(k, []))]
           for k in texts}
    pj = os.path.join(out_dir, "prediction.json")
    with open(pj, "w") as f:
        json.dump(sub, f, ensure_ascii=False)
    with zipfile.ZipFile(os.path.join(out_dir, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    return pj
