"""Phase 6 (abstractive summarization of legal documents): data, chunking, alignment, length model, scores,
post-processing and the submission file."""

from __future__ import annotations

import json
import math
import os
import random
import re
import zipfile
from collections import Counter

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
DEFAULT_ROOT = "/kaggle/working/data/Phase_6_Summ"
ID_KEYS = ("id", "doc_id", "document_id", "case_id", "uid", "name")
DOC_KEYS = ("document", "text", "judgment", "judgement", "source", "doc", "article", "content")
SUM_KEYS = ("summary", "abstract", "headnote", "target", "reference", "highlights")
STOP = set(
    """a an the of to in and or for on at by with from as is are was were be been being that this these those
it its it's he she they them his her their we our you your i not no shall will would should may might can could has have
had do does did which who whom whose what when where there here than then so such any all also into upon under over
said same other said whether if but about against between before after during without within thus hence therefore
however further also very only just""".split()
)
ABBREV = (
    "no",
    "nos",
    "vs",
    "v",
    "ltd",
    "co",
    "mr",
    "mrs",
    "ms",
    "dr",
    "sr",
    "jr",
    "art",
    "arts",
    "sec",
    "secs",
    "s",
    "ss",
    "cl",
    "para",
    "paras",
    "p",
    "pp",
    "viz",
    "i.e",
    "e.g",
    "etc",
    "hon",
    "j",
    "jj",
    "cr",
    "crl",
    "u",
    "r",
    "o",
    "a",
    "w",
    "st",
    "govt",
    "dt",
    "ors",
    "anr",
    "inc",
    "corpn",
    "admn",
    "supp",
    "sc",
    "scc",
    "air",
)


# --- data
def download(root: str = DEFAULT_ROOT) -> str:
    if not os.path.isdir(root) or not any(os.scandir(root)):
        from huggingface_hub import snapshot_download

        name = os.path.basename(root.rstrip("/"))
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=os.path.dirname(root.rstrip("/")),
            allow_patterns=[f"{name}/*", f"{name}/**"],
            token=os.environ.get("HF_TOKEN"),
        )
    return root


def read_json(path: str) -> list[dict]:
    """A JSON list, a JSON object (holding a list, or {id: record}) or JSON lines."""
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        d = [json.loads(line) for line in txt.splitlines() if line.strip()]
    if isinstance(d, dict):
        lists = [v for v in d.values() if isinstance(v, list)]
        if lists:
            d = lists[0]
        else:
            d = [dict(id=k, **v) if isinstance(v, dict) else dict(id=k, document=v) for k, v in d.items()]
    return d


def _pick(rec: dict, keys) -> str | None:
    for k in keys:
        if k in rec and rec[k] is not None:
            return k
    low = {k.lower(): k for k in rec}
    for k in keys:
        if k in low:
            return low[k]
    return None


def clean_text(t) -> str:
    """Whitespace normalised; page-number lines and runs of separators removed."""
    if isinstance(t, list):
        t = " ".join(str(x) for x in t)
    t = str(t or "")
    t = re.sub(r"(?m)^\s*(page\s*)?\d{1,4}\s*$", " ", t, flags=re.I)
    t = re.sub(r"[-_=*]{4,}", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalise(records: list[dict], need_summary: bool) -> list[dict]:
    if not records:
        return []
    ik, dk, sk = _pick(records[0], ID_KEYS), _pick(records[0], DOC_KEYS), _pick(records[0], SUM_KEYS)
    if dk is None:
        raise KeyError(f"no document field in {list(records[0])}")
    if need_summary and sk is None:
        raise KeyError(f"no summary field in {list(records[0])}")
    out = []
    for i, r in enumerate(records):
        d = dict(id=str(r[ik]) if ik else str(i), document=clean_text(r.get(dk)))
        if sk is not None:
            d["summary"] = clean_text(r.get(sk))
        out.append(d)
    return out


def find_file(root: str, kind: str) -> str:
    hits = []
    for base, _, files in os.walk(root):
        hits += [os.path.join(base, f) for f in files if f.lower().startswith(kind) and ".json" in f.lower()]
    if not hits:
        raise FileNotFoundError(f"no {kind}*.json under {root}")
    return sorted(hits, key=len)[0]


def load_data(
    root: str = DEFAULT_ROOT, fetch: bool = True, heldout_frac: float = 0.05, seed: int = 42, log=None
) -> dict[str, list[dict]]:
    """train / held (our validation part of the training data) / test."""
    if fetch:
        download(root)
    train = normalise(read_json(find_file(root, "train")), True)
    test = normalise(read_json(find_file(root, "test")), False)
    test_docs = {d["document"] for d in test}
    n0 = len(train)
    train = [d for d in train if d["document"] and d["summary"] and d["document"] not in test_docs]
    rng = random.Random(seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    k = max(min(400, int(round(len(train) * heldout_frac))), min(20, len(train) // 5))
    held = [train[i] for i in sorted(order[:k])]
    train = [train[i] for i in sorted(order[k:])]
    if log:
        log(
            f"[data] train {n0} -> {len(train)} + {len(held)} held out (removed {n0 - len(train) - len(held)} "
            f"empty / test documents); test {len(test)}"
        )
    return dict(train=train, held=held, test=test)


# --- sentences
_SPLIT = re.compile(r"(?<=[.?!;:])\s+(?=[\"'(\[A-Z0-9])")


def sentences(text: str) -> list[str]:
    """Sentence split for legal text: abbreviations (No., vs., Ltd., s., Art., single letters ...) and very short
    pieces are joined to the next piece."""
    parts = [p for p in _SPLIT.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out:
            last = out[-1]
            tail = re.findall(r"([A-Za-z.]+)[.:;]$", last)
            if (tail and tail[0].lower().rstrip(".") in ABBREV) or len(last.split()) < 3:
                out[-1] = last + " " + p
                continue
        out.append(p)
    return out


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content(ws) -> list[str]:
    return [w for w in ws if w not in STOP and len(w) > 1]


# --- scores
def _ngrams(ws, n):
    return Counter(zip(*[ws[i:] for i in range(n)]))


def rouge_n(ref: list[str], hyp: list[str], n: int) -> tuple[float, float, float]:
    r, h = _ngrams(ref, n), _ngrams(hyp, n)
    ov = sum((r & h).values())
    p = ov / max(1, sum(h.values()))
    rc = ov / max(1, sum(r.values()))
    return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)


def lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    idx: dict[str, list[int]] = {}
    for j, w in enumerate(b):
        idx.setdefault(w, []).append(j)
    prev = [0] * (len(b) + 1)
    for w in a:
        cur = prev[:]
        pos = idx.get(w)
        if pos:
            for j in pos:
                v = prev[j] + 1
                if v > cur[j + 1]:
                    cur[j + 1] = v
        for j in range(1, len(b) + 1):
            if cur[j] < cur[j - 1]:
                cur[j] = cur[j - 1]
        prev = cur
    return prev[-1]


def rouge_l(ref: list[str], hyp: list[str]) -> tuple[float, float, float]:
    m = lcs(ref, hyp)
    p, r = m / max(1, len(hyp)), m / max(1, len(ref))
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def repetition(ws: list[str], n: int = 3) -> float:
    """Share of n-grams that already occurred earlier in the same text (0 = no repetition)."""
    g = list(zip(*[ws[i:] for i in range(n)]))
    return 1 - len(set(g)) / len(g) if g else 0.0


def score_all(refs: list[str], hyps: list[str], docs: list[str] | None = None) -> dict:
    """Mean ROUGE-1/2/L F1 (word level, lower case, no stemming), lengths, repetition and the share of new word pairs
    (not in the document: how abstractive)."""
    r1, r2, rl, rep, novel, lh, lr = [], [], [], [], [], [], []
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        a, b = words(ref), words(hyp)
        r1.append(rouge_n(a, b, 1)[2])
        r2.append(rouge_n(a, b, 2)[2])
        rl.append(rouge_l(a, b)[2])
        rep.append(repetition(b))
        lh.append(len(b))
        lr.append(len(a))
        if docs is not None:
            dbi = set(zip(*[words(docs[i])[k:] for k in range(2)]))
            hb = list(zip(b, b[1:]))
            novel.append(sum(x not in dbi for x in hb) / max(1, len(hb)))
    out = dict(
        rouge1=float(np.mean(r1)),
        rouge2=float(np.mean(r2)),
        rougeL=float(np.mean(rl)),
        repetition3=float(np.mean(rep)),
        words_hyp=float(np.mean(lh)),
        words_ref=float(np.mean(lr)),
        n=len(refs),
    )
    out["rouge_mean"] = (out["rouge1"] + out["rougeL"]) / 2
    if novel:
        out["novel_bigrams"] = float(np.mean(novel))
    return out


def fmt(s: dict) -> str:
    return (
        f"ROUGE-1 {s['rouge1']:.4f}  ROUGE-2 {s['rouge2']:.4f}  ROUGE-L {s['rougeL']:.4f}  | words "
        f"{s['words_hyp']:.0f} (reference {s['words_ref']:.0f}), repeated trigrams {s['repetition3']:.1%}"
        + (f", new word pairs {s['novel_bigrams']:.1%}" if "novel_bigrams" in s else "")
    )


# --- tokens and windows
def tokenize_doc(doc: str, encode) -> tuple[list[int], list[int], list[str]]:
    """(token ids of the document, start token of every sentence, the sentences); sentences are encoded one by one
    (with a leading space after the first) so sentence boundaries are known in tokens."""
    sents = sentences(doc) or [doc]
    ids, starts = [], []
    for i, s in enumerate(sents):
        starts.append(len(ids))
        ids += encode(s if i == 0 else " " + s)
    return ids, starts, sents


def windows(n_tokens: int, window: int) -> list[tuple[int, int]]:
    """[start, end) of windows of exactly `window` tokens; the last one ends at the document's end (it may overlap the
    previous one); a document shorter than `window` is one window."""
    if n_tokens <= window:
        return [(0, n_tokens)]
    out = [(s, s + window) for s in range(0, n_tokens - window + 1, window)]
    if out[-1][1] < n_tokens:
        out.append((n_tokens - window, n_tokens))
    return out


def window_of(token_pos: int, wins: list[tuple[int, int]]) -> int:
    """The first window that holds token_pos (in the overlap of the last two windows: the earlier one)."""
    for i, (s, e) in enumerate(wins):
        if s <= token_pos < e:
            return i
    return len(wins) - 1


def align(doc_sents: list[str], sum_sents: list[str]) -> list[int]:
    """For every summary sentence the index of the most similar document sentence (F1 of their content-word sets; ties:
    the earlier sentence)."""
    dsets = [set(content(words(s))) for s in doc_sents]
    inv: dict[str, list[int]] = {}
    for i, ws in enumerate(dsets):
        for w in ws:
            inv.setdefault(w, []).append(i)
    out = []
    for s in sum_sents:
        ws = set(content(words(s)))
        if not ws:
            out.append(-1)
            continue
        ov: Counter = Counter()
        for w in ws:
            for i in inv.get(w, ()):
                ov[i] += 1
        best, bi = 0.0, -1
        for i, o in ov.items():
            f = 2 * o / (len(ws) + len(dsets[i]))
            if f > best or (f == best and i < bi):
                best, bi = f, i
        out.append(bi)
    return out


def make_pairs(
    docs: list[dict], encode, window: int, tgt_tokens: int, empty_frac: float = 0.2, seed: int = 42, eos_ok: bool = True
) -> tuple[list[dict], dict]:
    """Training pairs {src, tgt, doc, w, n_w}: every window of every document with its summary part (summary sentences
    whose best-matching document sentence lies in it, in summary order, cut to tgt_tokens at a sentence end when
    possible)."""
    rng = random.Random(seed)
    pairs, n_empty, n_sum_tok, cut = [], 0, 0, 0
    for d in docs:
        ids, starts, sents = tokenize_doc(d["document"], encode)
        wins = windows(len(ids), window)
        ss = sentences(d.get("summary", "")) or ([d["summary"]] if d.get("summary") else [])
        best = align(sents, ss)
        parts: list[list[str]] = [[] for _ in wins]
        for s, b in zip(ss, best):
            if b < 0:
                continue
            parts[window_of(starts[b], wins)].append(s)
        for wi, (a, e) in enumerate(wins):
            tgt: list[int] = []
            for k, s in enumerate(parts[wi]):
                t = encode(s if k == 0 else " " + s)
                if len(tgt) + len(t) > tgt_tokens:
                    if not tgt:
                        tgt = t[:tgt_tokens]
                    cut += 1
                    break
                tgt += t
            if not tgt:
                if rng.random() >= empty_frac:
                    continue
                n_empty += 1
            n_sum_tok += len(tgt)
            pairs.append(dict(src=ids[a:e], tgt=tgt, doc=d["id"], w=wi, n_w=len(wins)))
    info = dict(
        pairs=len(pairs),
        empty_targets=n_empty,
        docs=len(docs),
        targets_cut=cut,
        target_tokens_mean=n_sum_tok / max(1, len(pairs)),
    )
    return pairs, info


# --- length model
def fit_length(docs: list[dict]) -> dict:
    """log(summary words) = a + b log(document words), least squares on the training documents; predictions are kept
    between the 5th and 95th percentile of the training summaries."""
    x = np.log([max(1, len(d["document"].split())) for d in docs])
    y = np.log([max(1, len(d["summary"].split())) for d in docs])
    b, a = np.polyfit(x, y, 1) if len(docs) > 2 else (0.0, float(np.mean(y)) if len(y) else 0.0)
    lo, hi = np.percentile(np.exp(y), [5, 95]) if len(y) else (10, 1000)
    return dict(
        a=float(a), b=float(b), lo=float(lo), hi=float(hi), median=float(np.median(np.exp(y))) if len(y) else 100
    )


def predict_words(model: dict, doc: str, scale: float = 1.0) -> int:
    w = math.exp(model["a"] + model["b"] * math.log(max(1, len(doc.split()))))
    return int(round(min(model["hi"], max(model["lo"], w)) * scale))


def allocate(win_lengths: list[int], total: int, minimum: int = 16) -> list[int]:
    """Total target tokens shared among the windows in proportion to their length (at least `minimum` each)."""
    s = max(1, sum(win_lengths))
    return [max(minimum, int(round(total * n / s))) for n in win_lengths]


# --- post-processing
def postprocess(parts: list[str], max_words: int | None = None, dedup: float = 0.8) -> str:
    """Joined window summaries: sentences that repeat an earlier sentence (content-word overlap >= dedup) are dropped,
    an unfinished last sentence is cut (if more than one sentence remains), and the text is cut to max_words at a
    sentence end."""
    kept, sets = [], []
    for p in parts:
        for s in sentences(p.strip()):
            ws = set(content(words(s)))
            if not ws:
                continue
            if any(len(ws & o) / len(ws) >= dedup for o in sets):
                continue
            kept.append(s.strip())
            sets.append(ws)
    if len(kept) > 1 and not re.search(r"[.?!\"')\]]$", kept[-1]):
        kept = kept[:-1]
    if max_words:
        out, n = [], 0
        for s in kept:
            k = len(s.split())
            if out and n + k > max_words:
                break
            out.append(s)
            n += k
        kept = out
    return " ".join(kept)


# --- files
def write_submission(test_docs: list[dict], summaries: dict, out_dir: str, fallback: dict | None = None) -> str:
    """prediction.json = [{"id", "summary"}] (one per test document, in test order) inside submission.zip."""
    os.makedirs(out_dir, exist_ok=True)
    rows, n_fb = [], 0
    for d in test_docs:
        s = (summaries.get(d["id"]) or "").strip()
        if not s:
            n_fb += 1
            s = ((fallback or {}).get(d["id"]) or "").strip() or (sentences(d["document"]) or [d["document"]])[0]
        rows.append({"id": d["id"], "summary": s})
    if len(rows) != len(test_docs) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("submission must have exactly one row per test document")
    path = os.path.join(out_dir, "prediction.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    z = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, "prediction.json")
    if n_fb:
        print(f"[submission] {n_fb} empty summaries replaced")
    return z


def plan_steps(
    step: int, planned: int, sec_per_step: float, seconds_left: float, predict_seconds: float, min_steps: int = 20
) -> int:
    """Training steps that fit before a deadline (as in Phase 5's lsi_data.plan_steps)."""
    spare = seconds_left - predict_seconds
    more = int(spare / sec_per_step) if sec_per_step > 0 and spare > 0 else 0
    floor = min(planned, max(step, min_steps))
    return max(floor, min(planned, step + more))


# --- explore
def explore(data: dict, encode=None, sample: int = 400, seed: int = 0) -> str:
    rng = random.Random(seed)
    tr = rng.sample(data["train"], min(sample, len(data["train"])))

    def q(x):
        x = np.asarray(x, float)
        return f"median {np.median(x):.0f}, mean {x.mean():.0f}, p90 {np.percentile(x, 90):.0f}, max {x.max():.0f}"

    lines = [
        f"train {len(data['train'])}, held out {len(data['held'])}, test {len(data['test'])}",
        "document words: " + q([len(d["document"].split()) for d in tr]),
        "summary words:  " + q([len(d["summary"].split()) for d in tr]),
        "test document words: " + q([len(d["document"].split()) for d in data["test"]]),
    ]
    if encode is not None:
        sub = tr[:150]
        lines.append("document tokens: " + q([len(encode(d["document"])) for d in sub]))
        lines.append("summary tokens:  " + q([len(encode(d["summary"])) for d in sub]))
    uni, pos = [], []
    for d in tr[:200]:
        dw, sw = set(words(d["document"])), words(d["summary"])
        uni.append(np.mean([w in dw for w in sw]) if sw else 0)
        ds = sentences(d["document"])
        for b in align(ds, sentences(d["summary"])):
            if b >= 0:
                pos.append(b / max(1, len(ds) - 1))
    lines.append(f"summary words found in the document: {np.mean(uni):.1%}")
    lines.append(
        "where summary sentences match the document (5 position bins): "
        + " ".join(f"{x:.2f}" for x in np.histogram(pos, bins=5, range=(0, 1))[0] / max(1, len(pos)))
    )
    return "\n".join(lines)
