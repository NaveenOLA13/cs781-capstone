"""Phase 7 (prior case retrieval): data, our own BM25 over word n-grams, year and citation-count signals, micro-F1@7,
learning to rank (fusion of lexical and model scores), the prediction file."""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import zipfile
from collections import Counter

import numpy as np

HF_REPO = "Exploration-Lab/CS-781-Capstone"
DEFAULT_ROOT = "/kaggle/working/data/Phase_7_PCR"
K = 7
SPLITS = ("train", "dev", "test")
YEAR = re.compile(r"\b(18[5-9]\d|19\d\d|20[0-2]\d)\b")


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
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        d = [json.loads(line) for line in txt.splitlines() if line.strip()]
    if isinstance(d, dict):
        lists = [v for v in d.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
        if lists:
            d = lists[0]
        else:
            d = [dict(v, id=k) if isinstance(v, dict) else dict(id=k, text=v) for k, v in d.items()]
    return d


def _text(v) -> str:
    if isinstance(v, list):
        v = " ".join(_text(x) for x in v)
    elif isinstance(v, dict):
        v = " ".join(_text(x) for x in v.values())
    return re.sub(r"\s+", " ", str(v or "")).strip()


def normalise(records: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(records):
        rid = r.get("id", r.get("doc_id", r.get("case_id", r.get("query_id", r.get("candidate_id", i)))))
        text = r.get("text", r.get("document", r.get("sentences", r.get("content", ""))))
        rel = r.get("relevant_candidates", r.get("relevant", r.get("labels")))
        d = dict(id=str(rid), text=_text(text))
        if rel is not None:
            d["rel"] = list(dict.fromkeys(str(x) for x in rel))
        out.append(d)
    return out


def find_file(root: str, name: str) -> str:
    hits = []
    for base, _, files in os.walk(root):
        hits += [os.path.join(base, f) for f in files if f.lower() == name]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return sorted(hits, key=len)[0]


def load_data(root: str = DEFAULT_ROOT, fetch: bool = True, log=None) -> dict:
    """{split: (queries, pool)}; a query is {"id", "text", "rel" (train / dev: relevant candidate ids)}, a pool
    document {"id", "text"}."""
    if fetch:
        download(root)
    data = {}
    for s in SPLITS:
        q = normalise(read_json(find_file(root, f"{s}.json")))
        p = normalise(read_json(find_file(root, f"{s}_candidates.json")))
        seen, pool = set(), []
        for d in p:
            if d["id"] not in seen:
                seen.add(d["id"])
                pool.append(d)
        data[s] = (q, pool)
        if log:
            rel = [x for d in q for x in d.get("rel", [])]
            inside = sum(x in seen for x in rel)
            log(
                f"[data] {s}: {len(q)} queries, pool {len(pool)} candidates"
                + (f" ({len(p) - len(pool)} duplicates dropped)" if len(p) != len(pool) else "")
                + (
                    f", {len(rel)} relevant ({len(rel) / max(1, len(q)):.2f} per query, {inside / max(1, len(rel)):.1%} "
                    f"in the pool)"
                    if rel
                    else ""
                )
            )
    return data


# --- metrics
def micro_at_k(preds: dict, gold: dict, k: int = K) -> dict:
    """Micro precision / recall / F1 over the first k predictions of every query (the competition's metric: hits summed
    over queries; precision over all predictions, recall over all gold candidates)."""
    hit = n_pred = n_gold = 0
    for q, g in gold.items():
        p = list(preds.get(q, []))[:k]
        gs = set(g)
        hit += sum(x in gs for x in p)
        n_pred += len(p)
        n_gold += len(gs)
    pr, rc = hit / max(1, n_pred), hit / max(1, n_gold)
    return dict(f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, precision=pr, recall=rc, hits=hit, queries=len(gold))


def fmt(m: dict) -> str:
    return f"micro-F1@7 {m['f1']:.4f} (P {m['precision']:.4f}, R {m['recall']:.4f})"


def write_prediction(preds: dict, queries: list[dict], pool: list[dict], out_dir: str, k: int = K) -> str:
    """prediction.json = {query id: [k unique candidate ids from the pool, best first]} inside submission.zip; checked
    against the brief's rules (every query, exactly k, unique, from the pool, not the query itself)."""
    pool_ids = [d["id"] for d in pool]
    pset = set(pool_ids)
    out = {}
    for q in queries:
        p = [c for c in dict.fromkeys(preds.get(q["id"], [])) if c in pset and c != q["id"]][:k]
        for c in pool_ids:
            if len(p) == k:
                break
            if c not in p and c != q["id"]:
                p.append(c)
        out[q["id"]] = p
    if len(out) != len(queries) or any(len(v) != k or len(set(v)) != k for v in out.values()):
        raise ValueError("prediction must have exactly k unique candidates for every query")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prediction.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    z = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(path, "prediction.json")
    with zipfile.ZipFile(z) as zf:
        back = json.loads(zf.read("prediction.json"))
    assert back == out
    return z


# --- word n-grams
_WORD = re.compile(r"[a-z0-9]+")


class Vocab:
    """Word -> integer id (grows; shared by queries and candidates of all splits)."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def encode(self, text: str) -> np.ndarray:
        get = self.ids.setdefault
        return np.fromiter((get(w, len(self.ids)) for w in _WORD.findall(text.lower())), dtype=np.int64)


def ngrams(ids: np.ndarray, n: int, buckets: int = 1 << 24) -> np.ndarray:
    """Integer codes of the word n-grams of a document (n >= 2 hashed into `buckets`; unigrams keep the word id)."""
    if n == 1:
        return ids
    if len(ids) < n:
        return np.zeros(0, dtype=np.int64)
    m = len(ids) - n + 1
    h = ids[:m] % buckets
    for k in range(1, n):
        h = (h * 1000003 + ids[k : m + k]) % buckets
    return h


class BM25:
    """Okapi BM25 over the word n-grams of a candidate pool, as a sparse matrix of term weights (one row per document):
    weight(d, t) = idf(t) * tf (k1 + 1) / (tf + k1 (1 - b + b |d| / avg |d|)), idf(t) = log(1 + (N - df + 0.5) / (df
    + 0.5))."""

    def __init__(self, docs: list[np.ndarray], n: int = 1, k1: float = 1.2, b: float = 0.75, buckets: int = 1 << 24):
        import scipy.sparse as sp

        self.n, self.k1, self.b, self.buckets = n, k1, b, buckets
        rows, cols, tfs, dls = [], [], [], []
        for i, ids in enumerate(docs):
            h = ngrams(ids, n, buckets)
            t, c = np.unique(h, return_counts=True)
            rows.append(np.full(len(t), i, dtype=np.int32))
            cols.append(t)
            tfs.append(c.astype(np.float32))
            dls.append(len(h))
        rows_a = np.concatenate(rows) if rows else np.zeros(0, np.int32)
        cols_a = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        tf = np.concatenate(tfs) if tfs else np.zeros(0, np.float32)
        dl = np.asarray(dls, dtype=np.float32)
        self.n_docs = len(docs)
        uniq, inv, df = np.unique(cols_a, return_inverse=True, return_counts=True)
        self.terms = uniq
        idf = np.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg = float(dl.mean()) if len(dl) else 1.0
        w = idf[inv] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows_a] / max(avg, 1e-9)))
        self.W = sp.csc_matrix((w.astype(np.float32), (rows_a, inv.reshape(-1))), shape=(self.n_docs, len(uniq)))

    def score(self, ids: np.ndarray, qtf: str = "log") -> np.ndarray:
        h = ngrams(ids, self.n, self.buckets)
        if not len(h) or not self.n_docs or not len(self.terms):
            return np.zeros(self.n_docs, dtype=np.float32)
        t, c = np.unique(h, return_counts=True)
        pos = np.searchsorted(self.terms, t)
        ok = (pos < len(self.terms)) & (self.terms[np.minimum(pos, len(self.terms) - 1)] == t)
        pos, c = pos[ok], c[ok].astype(np.float32)
        if not len(pos):
            return np.zeros(self.n_docs, dtype=np.float32)
        qw = 1 + np.log(c) if qtf == "log" else (np.ones_like(c) if qtf == "binary" else c)
        return np.asarray(self.W[:, pos] @ qw, dtype=np.float32).ravel()


# --- other signals
def max_year(text: str) -> int:
    """The latest year mentioned in a judgment (about its decision year; 0 if none)."""
    ys = [int(y) for y in YEAR.findall(text)]
    return max(ys) if ys else 0


def citation_counts(queries: list[dict]) -> Counter:
    return Counter(c for q in queries for c in q.get("rel", []))


# --- lexical score matrices
def lexical(data: dict, ns=(1, 2, 3, 4, 5), k1: float = 1.2, b: float = 0.75, qtf: str = "log", log=print) -> dict:
    """BM25 of every query against its own split's pool for each n-gram order, each row divided by its maximum (the
    query itself gets -1)."""
    vocab = Vocab()
    out: dict = {"ns": list(ns), "single": {}, "settings": dict(k1=k1, b=b, qtf=qtf)}
    for s in SPLITS:
        q, pool = data[s]
        pids = [d["id"] for d in pool]
        pos = {c: j for j, c in enumerate(pids)}
        pdocs = [vocab.encode(d["text"]) for d in pool]
        qdocs = [vocab.encode(d["text"]) for d in q]
        selfs = [(i, pos[d["id"]]) for i, d in enumerate(q) if d["id"] in pos]
        mats = {}
        for n in ns:
            bm = BM25(pdocs, n, k1, b)
            sc = np.zeros((len(q), len(pool)), dtype=np.float32)
            for i, ids in enumerate(qdocs):
                sc[i] = bm.score(ids, qtf)
            for i, j in selfs:
                sc[i, j] = 0.0
            sc /= np.maximum(sc.max(1, keepdims=True), 1e-9)
            for i, j in selfs:
                sc[i, j] = -1.0
            mats[n] = sc
            del bm
        total = np.sum([mats[n] for n in ns], 0) / len(ns)
        split = dict(
            qids=[d["id"] for d in q],
            pids=pids,
            gold={d["id"]: d["rel"] for d in q if "rel" in d},
            bm25=mats,
            sum=total.astype(np.float32),
            self=selfs,
            qyear=np.array([max_year(d["text"]) for d in q], dtype=np.int32),
            pyear=np.array([max_year(d["text"]) for d in pool], dtype=np.int32),
        )
        out[s] = split
        if split["gold"]:
            for n in list(ns) + ["sum"]:
                m = eval_matrix(split, mats[n] if n != "sum" else total)
                out["single"][f"{s}_bm25_{n}"] = m
                log(f"[bm25] {s} {n if n == 'sum' else str(n) + '-grams':>8s}: {fmt(m)}")
    return out


def save_lexical(L: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(L, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".tmp", path)


def load_lexical(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def rank_matrix(split: dict, M: np.ndarray, k: int = K) -> dict:
    """{query id: top-k candidate ids} from a [Q, P] score matrix (the query itself excluded)."""
    M = np.array(M, dtype=np.float32, copy=True)
    for i, j in split["self"]:
        M[i, j] = -np.inf
    top = np.argsort(-M, axis=1, kind="stable")[:, :k]
    return {q: [split["pids"][j] for j in top[i]] for i, q in enumerate(split["qids"])}


def eval_matrix(split: dict, M: np.ndarray, k: int = K) -> dict:
    return micro_at_k(rank_matrix(split, M, k), split["gold"], k)


# --- learning to rank
def _ranks(M: np.ndarray, selfs) -> np.ndarray:
    """Rank (0 = best) of every candidate in every row."""
    M = np.array(M, dtype=np.float32, copy=True)
    for i, j in selfs:
        M[i, j] = -np.inf
    order = np.argsort(-M, axis=1, kind="stable")
    r = np.empty_like(order)
    rows = np.arange(M.shape[0])[:, None]
    r[rows, order] = np.arange(M.shape[1])[None, :]
    return r


def prior_counts(L: dict, split: str) -> np.ndarray:
    """[Q, P] how often each candidate is cited by labelled queries of the other splits (train labels for dev, train +
    dev for test; for train itself: the other training queries, leaving the query's own labels out)."""
    sp = L[split]
    if split == "train":
        cnt = Counter(c for g in sp["gold"].values() for c in g)
    elif split == "dev":
        cnt = Counter(c for g in L["train"]["gold"].values() for c in g)
    else:
        cnt = Counter(c for s in ("train", "dev") for g in L[s]["gold"].values() for c in g)
    base = np.array([cnt.get(c, 0) for c in sp["pids"]], dtype=np.float32)
    P = np.repeat(base[None, :], len(sp["qids"]), 0)
    if split == "train":
        pos = {c: j for j, c in enumerate(sp["pids"])}
        for i, q in enumerate(sp["qids"]):
            for c in sp["gold"].get(q, []):
                if c in pos:
                    P[i, pos[c]] -= 1
    return np.log1p(np.maximum(P, 0))


def build(
    L: dict, split: str, dense: dict | None = None, top: int = 100, dense_top: int = 50, use_prior: bool = True
) -> dict:
    """Candidate list of every query (BM25 top `top` by the averaged normalised n-gram scores, plus each model score's
    top `dense_top`) and its feature rows."""
    sp = L[split]
    dense = dense or {}
    lex_rank = _ranks(sp["sum"], sp["self"])
    d_rank = {k: _ranks(v, sp["self"]) for k, v in dense.items()}
    prior = prior_counts(L, split) if use_prior else None
    names = [f"bm25_{n}" for n in L["ns"]] + ["bm25_avg", "bm25_logrank", "later", "year_gap", "year_known"]
    names += ["prior"] if use_prior else []
    for k in dense:
        names += [f"{k}", f"{k}_logrank"]
    cands, feats, ys = [], [], []
    for i, q in enumerate(sp["qids"]):
        sel = lex_rank[i] < top
        for k in dense:
            sel |= d_rank[k][i] < dense_top
        for _, j in [x for x in sp["self"] if x[0] == i]:
            sel[j] = False
        idx = np.nonzero(sel)[0]
        qy, py = sp["qyear"][i], sp["pyear"][idx]
        known = ((py > 0) & (qy > 0)).astype(np.float32)
        cols = [sp["bm25"][n][i, idx] for n in L["ns"]] + [
            sp["sum"][i, idx],
            np.log1p(lex_rank[i, idx]),
            (known * (py > qy)).astype(np.float32),
            known * np.clip((qy - py) / 50.0, -1, 2),
            known,
        ]
        if use_prior:
            cols.append(prior[i, idx])
        for k in dense:
            cols += [dense[k][i, idx], np.log1p(d_rank[k][i, idx])]
        cands.append(idx)
        feats.append(np.stack(cols, 1).astype(np.float64))
        if sp["gold"]:
            g = set(sp["gold"].get(q, []))
            ys.append(np.array([sp["pids"][j] in g for j in idx], dtype=np.float64))
    recall = None
    if sp["gold"]:
        n_g = sum(len(v) for v in sp["gold"].values())
        recall = sum(y.sum() for y in ys) / max(1, n_g)
    return dict(
        split=split,
        names=names,
        cands=cands,
        X=standardise(feats),
        ys=ys or None,
        recall=recall,
        qids=sp["qids"],
        pids=sp["pids"],
        gold=sp["gold"],
    )


def standardise(feats: list[np.ndarray]) -> list[np.ndarray]:
    """Per query: every feature as (x - mean) / std over that query's candidate list."""
    out = []
    for f in feats:
        mu, sd = f.mean(0, keepdims=True), f.std(0, keepdims=True)
        out.append((f - mu) / np.where(sd > 1e-9, sd, 1.0))
    return out


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 100) -> np.ndarray:
    """L2-regularised logistic regression by Newton steps; positives weighted up to balance the classes."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    pw = max(1.0, (len(y) - y.sum()) / max(1.0, y.sum()))
    sw = np.where(y > 0, pw, 1.0)
    reg = l2 * np.r_[np.ones(Xb.shape[1] - 1), 0.0]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (sw * (p - y)) + reg * w
        H = Xb.T @ (Xb * (sw * p * (1 - p))[:, None]) + np.diag(reg + 1e-6)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w[:-1]


def _hits(Xs, ys, w, k=K) -> int:
    h = 0
    for X, y in zip(Xs, ys):
        s = X @ w
        top = np.argpartition(-s, min(k, len(s) - 1))[:k] if len(s) > k else np.arange(len(s))
        h += int(y[top].sum())
    return h


def fit_coord(Xs, ys, w0: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Coordinate ascent on the number of correct candidates in the top 7 (= micro-F1@7 for a fixed query set),
    starting from w0 (Metzler & Croft 2007 style)."""
    w = w0.astype(np.float64).copy()
    scale = max(1e-3, float(np.abs(w).max()))
    best = _hits(Xs, ys, w)
    for _ in range(rounds):
        improved = False
        for f in range(len(w)):
            for d in (-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2):
                v = w.copy()
                v[f] += d * scale
                h = _hits(Xs, ys, v)
                if h > best:
                    best, w, improved = h, v, True
        if not improved:
            break
    return w


def fit(Xs, ys, method: str = "logreg", l2: float = 1.0) -> np.ndarray:
    w = fit_logreg(np.vstack(Xs), np.concatenate(ys), l2)
    return fit_coord(Xs, ys, w) if method == "coord" else w


def predict(B: dict, w: np.ndarray, k: int = K) -> dict:
    out = {}
    for q, X, idx in zip(B["qids"], B["X"], B["cands"]):
        s = X @ w
        order = np.argsort(-s, kind="stable")[:k]
        out[q] = [B["pids"][idx[j]] for j in order]
    return out


def crossfit(B: dict, method: str = "logreg", folds: int = 5, seed: int = 0, l2: float = 1.0) -> dict:
    """Honest estimate on a labelled split: every query is ranked by a ranker fitted on the other folds."""
    n = len(B["X"])
    order = list(range(n))
    random.Random(seed).shuffle(order)
    parts = [order[i::folds] for i in range(folds)]
    preds = {}
    for p in parts:
        tr = [i for i in range(n) if i not in set(p)]
        w = fit([B["X"][i] for i in tr], [B["ys"][i] for i in tr], method, l2)
        sub = dict(B, qids=[B["qids"][i] for i in p], X=[B["X"][i] for i in p], cands=[B["cands"][i] for i in p])
        preds.update(predict(sub, w))
    return micro_at_k(preds, B["gold"])


def describe_weights(names, w) -> str:
    return ", ".join(f"{n} {v:+.2f}" for n, v in sorted(zip(names, w), key=lambda t: -abs(t[1])))


# --- submissions
def _queries_pool(data_or_L, split):
    sp = data_or_L[split]
    return [dict(id=q) for q in sp["qids"]], [dict(id=c) for c in sp["pids"]]


def lexical_submissions(L: dict, out_dir: str, top: int = 100, log=print) -> dict:
    """Two lexical-only systems (no neural model): (a) BM25 averaged over the n-gram orders; (b) a ranker on the BM25
    scores + year + citation counts, fitted on the training queries."""
    res = {"single": L["single"]}
    m = eval_matrix(L["dev"], L["dev"]["sum"])
    q, p = _queries_pool(L, "test")
    res["bm25_avg"] = dict(
        dev=m,
        submission=write_prediction(rank_matrix(L["test"], L["test"]["sum"]), q, p, os.path.join(out_dir, "bm25_avg")),
    )
    log(f"[lexical] BM25 averaged over {L['ns']}-grams: dev {fmt(m)}")
    Btr, Bdv, Bte = (build(L, s, top=top) for s in SPLITS)
    log(
        f"[lexical] candidates: top {top} by BM25 hold {Btr['recall']:.1%} (train) / {Bdv['recall']:.1%} (dev) of "
        f"the cited cases"
    )
    best = None
    for method in ("logreg", "coord"):
        w = fit(Btr["X"], Btr["ys"], method)
        md = micro_at_k(predict(Bdv, w), Bdv["gold"])
        cf = crossfit(Bdv, method)
        log(f"[lexical] ranker ({method}) fitted on train: dev {fmt(md)} | fitted on dev folds: {fmt(cf)}")
        log(f"          weights: {describe_weights(Btr['names'], w)}")
        res[f"ltr_{method}"] = dict(dev=md, dev_crossfit=cf, weights=dict(zip(Btr["names"], map(float, w))))
        if best is None or md["f1"] > res[f"ltr_{best}"]["dev"]["f1"]:
            best = method
    w = fit(Btr["X"] + Bdv["X"], Btr["ys"] + Bdv["ys"], best)
    res["lexical_ltr"] = dict(
        method=best,
        dev=res[f"ltr_{best}"]["dev"],
        submission=write_prediction(predict(Bte, w), q, p, os.path.join(out_dir, "lexical_ltr")),
    )
    log(f"[lexical] test predictions: ranker ({best}) refitted on train + dev -> {res['lexical_ltr']['submission']}")
    res["recall_top"] = dict(train=Btr["recall"], dev=Bdv["recall"], top=top)
    with open(os.path.join(out_dir, "lexical_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


def fused_submission(
    L: dict, dense_dev: dict, dense_test: dict, out_dir: str, top: int = 100, dense_top: int = 50, log=print
) -> dict:
    """Our model's scores ({name: [Q, P]} for dev and test, same query / pool order as L) alone and fused with the
    lexical features by a ranker."""
    res = {"dense_alone": {}}
    for k, M in dense_dev.items():
        res["dense_alone"][k] = eval_matrix(L["dev"], M)
        log(f"[fusion] model score '{k}' alone: dev {fmt(res['dense_alone'][k])}")
    Bl = build(L, "dev", top=top)
    Bd = build(L, "dev", dense_dev, top=top, dense_top=dense_top)
    Bt = build(L, "test", dense_test, top=top, dense_top=dense_top)
    res["recall_candidates"] = Bd["recall"]
    best = None
    for method in ("logreg", "coord"):
        lex = crossfit(Bl, method)
        fus = crossfit(Bd, method)
        res[f"lexical_{method}"], res[f"fused_{method}"] = lex, fus
        log(
            f"[fusion] dev, 5-fold cross-fitted ranker ({method}): lexical only {fmt(lex)} | lexical + model {fmt(fus)}"
        )
        if best is None or fus["f1"] > res[f"fused_{best}"]["f1"]:
            best = method
    w = fit(Bd["X"], Bd["ys"], best)
    res.update(method=best, dev=res[f"fused_{best}"], weights=dict(zip(Bd["names"], map(float, w))))
    log(f"[fusion] ranker ({best}) on all of dev: {describe_weights(Bd['names'], w)}")
    q, p = _queries_pool(L, "test")
    res["submission"] = write_prediction(predict(Bt, w), q, p, os.path.join(out_dir, "submission"))
    return res


# --- explore
def explore(data: dict, sample: int = 200, seed: int = 0) -> str:
    rng = random.Random(seed)
    lines = []

    def q(x):
        x = np.asarray(x, float)
        return (
            f"median {np.median(x):.0f}, mean {x.mean():.0f}, p90 {np.percentile(x, 90):.0f}, max {x.max():.0f}"
            if len(x)
            else "-"
        )

    pools = {s: {d["id"] for d in data[s][1]} for s in SPLITS}
    queries = {s: {d["id"] for d in data[s][0]} for s in SPLITS}
    for s in SPLITS:
        qs, pool = data[s]
        lines.append(
            f"{s}: {len(qs)} queries, pool {len(pool)}; query words "
            + q([len(d["text"].split()) for d in rng.sample(qs, min(sample, len(qs)))])
            + "; candidate words "
            + q([len(d["text"].split()) for d in rng.sample(pool, min(sample, len(pool)))])
        )
        if qs and "rel" in qs[0]:
            lines.append(
                f"  relevant per query: {q([len(d['rel']) for d in qs])}; queries that are also in their own "
                f"pool: {sum(d['id'] in pools[s] for d in qs)}"
            )
    lines.append(
        f"pool overlap: train-dev {len(pools['train'] & pools['dev'])}, train-test "
        f"{len(pools['train'] & pools['test'])}, dev-test {len(pools['dev'] & pools['test'])}; "
        f"test queries in the train pool {len(queries['test'] & pools['train'])}, in the dev pool "
        f"{len(queries['test'] & pools['dev'])}"
    )
    tr = citation_counts(data["train"][0])
    dv = data["dev"][0]
    seen = sum(c in tr for d in dv for c in d.get("rel", []))
    lines.append(
        f"dev relevant candidates also cited by some training query: {seen} of "
        f"{sum(len(d.get('rel', [])) for d in dv)}; most cited in train: {tr.most_common(3)}"
    )
    later = tot = no_year = 0
    pool = {x["id"]: x for x in data["dev"][1]}
    for d in dv:
        qy = max_year(d["text"])
        no_year += qy == 0
        for c in d.get("rel", []):
            if c in pool and qy:
                tot += 1
                later += max_year(pool[c]["text"]) > qy
    lines.append(
        f"dev: queries without any year {no_year}; cited cases whose latest year is after the query's latest "
        f"year: {later} of {tot}"
    )
    ex = data["dev"][0][0]["text"][:600] if data["dev"][0] else ""
    lines.append(f"start of a dev query: {ex}")
    return "\n".join(lines)
