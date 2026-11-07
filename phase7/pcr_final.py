"""Phase 7 final ranking (CPU only, no training): our runs' saved similarity scores + better lexical signals.

What it adds over the ranking each run did on its own:
  1. exact judgment dates: the header's "DATE OF JUDGMENT: dd/mm/yyyy" (else the year of its CITATION line, else the
     latest year in the text outside the CITATOR INFO list). A candidate that is certainly decided after the query
     cannot be cited by it: removed (or, in the other variant, given as a feature).
  2. "similar cases cite similar cases": the query's most similar labelled queries (BM25 3-5-grams between query
     texts; train queries for dev, train + dev queries for test) vote for the cases they cite.
  3. bigger candidate lists: the top cases of the 3-, 4-, 5-gram BM25 lists, the averaged list, the votes and each of
     our model scores (after the date filter), instead of only the averaged BM25 top 100.
  4. several of our models together: per model the average of its best runs' standardised scores (mean, maxsim,
     maxsim_sym), and the average over the four models.
  5. citation strings in the headnotes (variants with and without them): the query's own citation (e.g. "1966 AIR
     1053") listed in a candidate's CITATOR INFO (the later cases that cited the candidate), a candidate's citation
     written in the query, and later cases that cited both. This reads the publisher's citator metadata, not
     retrieval from the judgment text, so it is reported separately ("citator" vs "nocitator" variants).
  6. citation contexts: the 40 words before and after every <CITATION> mask of the query are short queries about
     one cited case each; BM25 2-/3-gram scores of the best-matching context and of all contexts together.
  7. three rankers (logistic regression, pairwise logistic regression, coordinate ascent on top-7 hits), every
     variant measured by 5-fold cross-fitting on dev; the best ones are refitted on all of dev and written as
     named submissions <variant>__dev<score>.zip.

    import pcr_final as PF
    PF.main(results_roots=["/kaggle/input"], out_dir="/kaggle/working/final")
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import shutil
import time
import zipfile
from collections import Counter

import numpy as np

import p7_paths  # noqa: F401
import pcr_data as PD  # noqa: E402

AGGS3 = ("mean", "maxsim", "maxsim_sym")
MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"), 1)}
_DOJ = r"date\s+of\s+(?:the\s+)?judg(?:e)?ment\s*[:\-]?\s*"
_DOJ_NUM = re.compile(_DOJ + r"(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4})", re.I)
_DOJ_DMY = re.compile(_DOJ + r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\.?,?\s+(\d{4})", re.I)
_DOJ_MDY = re.compile(_DOJ + r"([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", re.I)
_CIT = re.compile(r"(?<!<)\bcitation\s*:\s*(\d{4})\b", re.I)            # header "CITATION: 1966 ...", not <CITATION>
_CITATOR = re.compile(r"citator\s+info\s*:?(.{0,4000}?)(?=\b(?:act|headnote|judgment|bench)\s*:)", re.I | re.S)
_Y = r"(1[89]\d\d|20[0-2]\d)"
_SEG_CIT = re.compile(r"(?<!<)\bCITATION\s*:\s*(.{0,300}?)(?=CITATOR\s+INFO|\bACT\s*:|HEADNOTE|$)", re.I | re.S)
_AIR1 = re.compile(_Y + r"\s*AIR\s*(\d{1,4})\b", re.I)                         # 1966 AIR 1053
_AIR2 = re.compile(r"\bAIR\s*" + _Y + r"\s*S\.?\s*C\.?\s*(\d{1,4})\b", re.I)    # AIR 1966 SC 1053
_SC = re.compile(r"\b" + _Y + r"\s*S\.?\s*C\.?\s*(\d{1,4})\b")                 # 1966 SC 1053 / 1966 SC1053


# ------------------------------------------------------------------ citation strings
def _keys(s: str) -> set:
    return {(int(a), int(b)) for rx in (_AIR1, _AIR2, _SC) for a, b in rx.findall(s or "")}


def cite_keys(text: str) -> tuple[set, set, set]:
    """(own AIR citations from the CITATION line, AIR citations listed in CITATOR INFO = later cases that cited this
    one, AIR citations written anywhere else in the text); each as {(year, page)}."""
    m = _SEG_CIT.search(text[:6000])
    own = _keys(m.group(1)) if m else set()
    t = _CITATOR.search(text[:20000])
    tor = _keys(t.group(1)) if t else set()
    body = _CITATOR.sub(" ", _SEG_CIT.sub(" ", text, count=1), count=1)
    return own, tor, _keys(body)


def citation_matrices(q_texts: list[str], p_texts: list[str]):
    """back [Q, P]: the query's own citation is in the candidate's CITATOR INFO (the candidate lists the query among
    the later cases that cited it); fwd [Q, P]: the candidate's own citation is written in the query; co [Q, P]: how
    many later cases cited both (log1p)."""
    qk = [cite_keys(t) for t in q_texts]
    pk = [cite_keys(t) for t in p_texts]
    idx_tor, idx_own = {}, {}
    for j, (own, tor, _) in enumerate(pk):
        for k in tor:
            idx_tor.setdefault(k, []).append(j)
        for k in own:
            idx_own.setdefault(k, []).append(j)
    Q, P = len(q_texts), len(p_texts)
    back = np.zeros((Q, P), dtype=np.float32)
    fwd = np.zeros((Q, P), dtype=np.float32)
    co = np.zeros((Q, P), dtype=np.float32)
    for i, (own, tor, men) in enumerate(qk):
        for k in own:
            for j in idx_tor.get(k, ()):
                back[i, j] = 1.0
        for k in men - own:
            for j in idx_own.get(k, ()):
                fwd[i, j] = 1.0
        for k in tor:
            for j in idx_tor.get(k, ()):
                co[i, j] += 1.0
    return back, fwd, np.log1p(co)


# ------------------------------------------------------------------------------ dates
def doc_date(text: str) -> dict:
    """{"lo", "hi", "ord", "src"}: the judgment year lies in [lo, hi] (None = unknown); ord = yyyymmdd if the exact
    date is known."""
    head = text[:6000]
    for rx, order in ((_DOJ_NUM, "dmy_num"), (_DOJ_DMY, "dmy"), (_DOJ_MDY, "mdy")):
        m = rx.search(head)
        if not m:
            continue
        a, b, y = m.groups()
        try:
            y = int(y)
            if order == "dmy_num":
                d, mo = int(a), int(b)
            elif order == "dmy":
                d, mo = int(a), MONTHS.get(b[:3].lower(), 0)
            else:
                d, mo = int(b), MONTHS.get(a[:3].lower(), 0)
        except ValueError:
            continue
        if 1850 <= y <= 2030 and 1 <= mo <= 12 and 1 <= d <= 31:
            return dict(lo=y, hi=y, ord=y * 10000 + mo * 100 + d, src="date of judgment")
        if 1850 <= y <= 2030:
            return dict(lo=y, hi=y, ord=None, src="date of judgment (year)")
    m = _CIT.search(head)
    if m and 1850 <= int(m.group(1)) <= 2030:
        y = int(m.group(1))
        return dict(lo=y - 1, hi=y, ord=None, src="citation year")
    years = [int(y) for y in PD.YEAR.findall(_CITATOR.sub(" ", text))]
    if years:
        return dict(lo=max(years), hi=None, ord=None, src="latest year in the text")
    return dict(lo=None, hi=None, ord=None, src="none")


def date_matrices(q_dates: list[dict], p_dates: list[dict]):
    """later [Q, P] (the candidate is certainly decided after the query), gap [Q, P] (query year - candidate year,
    / 50), known [Q, P]."""
    Q, P = len(q_dates), len(p_dates)
    later = np.zeros((Q, P), dtype=bool)
    gap = np.zeros((Q, P), dtype=np.float32)
    known = np.zeros((Q, P), dtype=np.float32)
    p_ord = np.array([d["ord"] or 0 for d in p_dates], dtype=np.int64)
    p_lo = np.array([d["lo"] if d["lo"] is not None else -1 for d in p_dates], dtype=np.int64)
    p_y = np.array([d["lo"] if d["lo"] is not None else (d["hi"] or -1) for d in p_dates], dtype=np.int64)
    for i, q in enumerate(q_dates):
        if q["ord"]:
            later[i] |= (p_ord > 0) & (p_ord > q["ord"])
        if q["hi"] is not None:
            later[i] |= (p_lo >= 0) & (p_lo > q["hi"])
        qy = q["hi"] if q["hi"] is not None else q["lo"]
        if qy is not None:
            ok = p_y >= 0
            gap[i, ok] = np.clip((qy - p_y[ok]) / 50.0, -1, 2)
            known[i, ok] = 1.0
    return later, gap, known


# --------------------------------------------------------------- similar cases' citations
def knn_votes(data: dict, split: str, k: int = 20, ns=(3, 4, 5), log=print) -> np.ndarray:
    """[Q, P]: for every query of the split, the cases cited by its k most similar labelled queries (BM25 between
    query texts), each vote weighted by that query's similarity / the best similarity. Labelled queries: train for
    dev; train + dev for test (a query never votes for itself)."""
    q, pool = data[split]
    lab = list(data["train"][0]) + (list(data["dev"][0]) if split == "test" else [])
    pos = {c: j for j, c in enumerate(d["id"] for d in pool)}
    vocab = PD.Vocab()
    ldocs = [vocab.encode(d["text"]) for d in lab]
    qdocs = [vocab.encode(d["text"]) for d in q]
    S = np.zeros((len(q), len(lab)), dtype=np.float32)
    for n in ns:
        bm = PD.BM25(ldocs, n)
        M = np.stack([bm.score(x) for x in qdocs]) if q else np.zeros((0, len(lab)), np.float32)
        S += M / np.maximum(M.max(1, keepdims=True), 1e-9)
    lab_ids = [d["id"] for d in lab]
    V = np.zeros((len(q), len(pool)), dtype=np.float32)
    for i, d in enumerate(q):
        row = S[i].copy()
        row[[j for j, x in enumerate(lab_ids) if x == d["id"]]] = -1
        top = np.argsort(-row)[:k]
        best = max(float(row[top[0]]), 1e-9) if len(top) else 1.0
        for j in top:
            if row[j] <= 0:
                continue
            for c in lab[j].get("rel", []):
                if c in pos and c != d["id"]:
                    V[i, pos[c]] += row[j] / best
    if q and "rel" in q[0]:
        preds = PD.rank_matrix(dict(qids=[d["id"] for d in q], pids=list(pos), self=[]), V)
        m = PD.micro_at_k(preds, {d["id"]: d["rel"] for d in q})
        log(f"[votes] {split}: the cases cited by the {k} most similar labelled queries, alone: {PD.fmt(m)}")
    return V


# ------------------------------------------------------------------ citation contexts
MASK = "<CITATION>"


def contexts(text: str, width: int = 40) -> list[str]:
    """The words around every masked citation of a query (width words before and after, other masks removed): each is
    a short query about one cited case."""
    toks = text.split()
    out = []
    for i, t in enumerate(toks):
        if MASK in t:
            win = toks[max(0, i - width):i] + toks[i + 1:i + 1 + width]
            win = [w for w in win if MASK not in w]
            if len(win) >= 5:
                out.append(" ".join(win))
    return out


def context_matrices(data: dict, split: str, ns=(2, 3), width: int = 40, log=print) -> dict:
    """{"ctx_max_n": [Q, P] best BM25 n-gram score of the candidate over the query's citation contexts (each context's
    scores divided by its maximum), "ctx_all_n": BM25 of all contexts together (divided by the row maximum)}."""
    q, pool = data[split]
    vocab = PD.Vocab()
    pdocs = [vocab.encode(d["text"]) for d in pool]
    ctx = [contexts(d["text"], width) for d in q]
    out = {}
    for n in ns:
        bm = PD.BM25(pdocs, n)
        mx = np.zeros((len(q), len(pool)), dtype=np.float32)
        al = np.zeros((len(q), len(pool)), dtype=np.float32)
        for i, cs in enumerate(ctx):
            if not cs:
                continue
            for c in cs:
                sc = bm.score(vocab.encode(c))
                m = sc.max()
                if m > 0:
                    np.maximum(mx[i], sc / m, out=mx[i])
            sc = bm.score(vocab.encode(" ".join(cs)))
            al[i] = sc / max(float(sc.max()), 1e-9)
        out[f"ctx_max_{n}"], out[f"ctx_all_{n}"] = mx, al
    n_ctx = [len(c) for c in ctx]
    msg = (f"[contexts] {split}: {np.mean([c > 0 for c in n_ctx]):.0%} of queries have masked citations, "
           f"{np.mean(n_ctx):.1f} contexts per query")
    if q and "rel" in q[0]:
        sp = dict(qids=[d["id"] for d in q], pids=[d["id"] for d in pool],
                  self=[(i, j) for i, d in enumerate(q) for j, c in enumerate(pool) if c["id"] == d["id"]])
        gold = {d["id"]: d["rel"] for d in q}
        msg += "; alone: " + ", ".join(f"{k} {PD.micro_at_k(PD.rank_matrix(sp, M), gold)['f1']:.4f}"
                                       for k, M in out.items())
    log(msg)
    return out


# ------------------------------------------------------------------------- our runs
def find_runs(roots, L: dict, log=print, unzip_dir: str = "_unzipped") -> dict:
    """{run name: {"model", "dev": {agg: [Q, P]}, "test": {...}, "dev_f1": {agg: f1}}} for every finished run under
    roots (folders of pcr_runs/<run>/scores_dev.npz; pcr_results*.zip files are unpacked into unzip_dir first)."""
    roots = [roots] if isinstance(roots, str) else list(roots)
    unz = unzip_dir
    shutil.rmtree(unz, ignore_errors=True)
    for r in roots:
        for z in glob.glob(os.path.join(r, "**", "*.zip"), recursive=True):
            if "pcr_results" in os.path.basename(z):
                dst = os.path.join(unz, os.path.splitext(os.path.basename(z))[0])
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dst)
    runs = {}
    for r in roots + ([unz] if os.path.isdir(unz) else []):
        for f in sorted(glob.glob(os.path.join(r, "**", "pcr_runs", "*", "scores_dev.npz"), recursive=True)):
            d = os.path.dirname(f)
            name = os.path.basename(d)
            if name in runs or not os.path.exists(os.path.join(d, "scores_test.npz")):
                continue
            try:
                with open(os.path.join(d, "ids.json")) as fh:
                    ids = json.load(fh)
                if any(ids[s]["qids"] != L[s]["qids"] or ids[s]["pids"] != L[s]["pids"] for s in ("dev", "test")):
                    log(f"[runs] {name}: scored on other data, skipped")
                    continue
                sd, st = np.load(f), np.load(os.path.join(d, "scores_test.npz"))
                model = name.split("_")[0]
                if os.path.exists(os.path.join(d, "metrics.json")):
                    with open(os.path.join(d, "metrics.json")) as fh:
                        model = json.load(fh).get("model", model)
                dev = {a: sd[a].astype(np.float32) for a in sd.files}
                runs[name] = dict(model=model, dir=d, dev=dev, test={a: st[a].astype(np.float32) for a in st.files},
                                  dev_f1={a: PD.eval_matrix(L["dev"], M)["f1"] for a, M in dev.items()})
            except Exception as e:  # noqa: BLE001
                log(f"[runs] {name}: unreadable ({type(e).__name__}: {e})")
    log(f"[runs] {len(runs)} runs: " + ", ".join(f"{n} ({max(r['dev_f1'].values()):.3f})" for n, r in runs.items()))
    return runs


def _z(M):
    mu, sd = M.mean(1, keepdims=True), M.std(1, keepdims=True)
    return (M - mu) / np.where(sd > 1e-9, sd, 1.0)


def dense_sets(runs: dict, per_model: int = 4, log=print) -> dict:
    """Feature sets of our model scores: {"all": {...}, "models": {...}} for dev and test. Per model: the average of
    the standardised scores of its best per_model runs (by dev micro-F1@7 alone; the zero-shot run left out)."""
    by_model: dict = {}
    for n, r in runs.items():
        if "zeroshot" in n or "frozen" in n:
            continue
        by_model.setdefault(r["model"], []).append((max(r["dev_f1"].values()), n))
    per = {}
    for m, lst in sorted(by_model.items()):
        best = [n for _, n in sorted(lst, reverse=True)[:per_model]]
        log(f"[dense] {m}: {', '.join(best)}")
        per[m] = {s: {a: np.mean([_z(runs[n][s][a]) for n in best if a in runs[n][s]], 0) for a in AGGS3
                      if any(a in runs[n][s] for n in best)} for s in ("dev", "test")}
    out = {"all": {"dev": {}, "test": {}}, "models": {"dev": {}, "test": {}}}
    for s in ("dev", "test"):
        for a in AGGS3:
            have = [per[m][s][a] for m in per if a in per[m][s]]
            if have:
                out["all"][s][f"all_{a}"] = np.mean(have, 0)
        for m in per:
            for a in ("mean", "maxsim_sym"):
                if a in per[m][s]:
                    out["models"][s][f"{m}_{a}"] = per[m][s][a]
    out["all+models"] = {s: {**out["all"][s], **out["models"][s]} for s in ("dev", "test")}
    return out


# ---------------------------------------------------------------------- features
def _ranks(M):
    order = np.argsort(-M, axis=1, kind="stable")
    r = np.empty_like(order)
    r[np.arange(M.shape[0])[:, None], order] = np.arange(M.shape[1])[None, :]
    return r


def build(L: dict, split: str, extra: dict, dense: dict, filt: bool, k_lex: int = 100, k_vote: int = 40,
          k_dense: int = 60, use_cit: bool = False) -> dict:
    """Candidate lists and standardised feature rows for every query of a split."""
    sp, ex = L[split], extra[split]
    Q, P = len(sp["qids"]), len(sp["pids"])
    bad = np.zeros((Q, P), dtype=bool)
    for i, j in sp["self"]:
        bad[i, j] = True
    if filt:
        bad |= ex["later"]

    def masked(M):
        return np.where(bad, -np.inf, M)
    R = {f"bm25_{n}": _ranks(masked(sp["bm25"][n])) for n in L["ns"]}
    R["bm25_avg"] = _ranks(masked(sp["sum"]))
    R["votes"] = _ranks(masked(ex["votes"]))
    ctx = ex.get("ctx", {})
    Rc = {k: _ranks(masked(v)) for k, v in ctx.items()}
    Rd = {k: _ranks(masked(v)) for k, v in dense.items()}
    sel = np.zeros((Q, P), dtype=bool)
    for k in ("bm25_3", "bm25_4", "bm25_5", "bm25_avg"):
        if k in R:
            sel |= R[k] < k_lex
    sel |= (R["votes"] < k_vote) & (ex["votes"] > 0)
    for k in Rc:
        sel |= (Rc[k] < k_dense) & (ctx[k] > 0)
    for k in Rd:
        sel |= Rd[k] < k_dense
    if use_cit:
        sel |= (ex["back"] > 0) | (ex["fwd"] > 0) | (ex["co"] > 0)
    sel &= ~bad
    names = [f"bm25_{n}" for n in L["ns"]] + ["bm25_avg", "bm25_avg_lr", "bm25_4_lr", "bm25_5_lr", "later",
                                               "year_gap", "year_known", "prior", "votes", "votes_lr"]
    for k in ctx:
        names += [k, f"{k}_lr"]
    if use_cit:
        names += ["cite_back", "cite_fwd", "cocited"]
    for k in dense:
        names += [k, f"{k}_lr"]
    cands, X, ys = [], [], []
    gold = sp["gold"]
    for i, q in enumerate(sp["qids"]):
        idx = np.nonzero(sel[i])[0]
        cols = [sp["bm25"][n][i, idx] for n in L["ns"]]
        cols += [sp["sum"][i, idx], np.log1p(R["bm25_avg"][i, idx]),
                 np.log1p(R["bm25_4"][i, idx]) if "bm25_4" in R else np.zeros(len(idx)),
                 np.log1p(R["bm25_5"][i, idx]) if "bm25_5" in R else np.zeros(len(idx)),
                 ex["later"][i, idx].astype(np.float32), ex["gap"][i, idx], ex["known"][i, idx],
                 ex["prior"][i, idx], np.log1p(ex["votes"][i, idx]), np.log1p(R["votes"][i, idx])]
        for k in ctx:
            cols += [ctx[k][i, idx], np.log1p(Rc[k][i, idx])]
        if use_cit:
            cols += [ex["back"][i, idx], ex["fwd"][i, idx], ex["co"][i, idx]]
        for k in dense:
            cols += [dense[k][i, idx], np.log1p(Rd[k][i, idx])]
        cands.append(idx)
        X.append(np.stack(cols, 1).astype(np.float64))
        if gold:
            g = set(gold.get(q, []))
            ys.append(np.array([sp["pids"][j] in g for j in idx], dtype=np.float64))
    rec = None
    if gold:
        rec = sum(y.sum() for y in ys) / max(1, sum(len(v) for v in gold.values()))
    return dict(split=split, names=names, cands=cands, X=PD.standardise(X), ys=ys or None, recall=rec,
                qids=sp["qids"], pids=sp["pids"], gold=gold)


# ---------------------------------------------------------------------- rankers
def fit_pairwise(Xs, ys, l2: float = 1.0, negs: int = 40, seed: int = 0) -> np.ndarray:
    """Pairwise logistic regression (RankNet with a linear scorer): P(cited case above a non-cited one) =
    sigmoid(w . (x_pos - x_neg)); negatives: the hardest by the current lexical average plus random ones."""
    rng = np.random.default_rng(seed)
    D = []
    for X, y in zip(Xs, ys):
        p, n = np.nonzero(y > 0)[0], np.nonzero(y == 0)[0]
        if not len(p) or not len(n):
            continue
        if len(n) > negs:
            n = rng.choice(n, negs, replace=False)
        D.append((X[p][:, None, :] - X[n][None, :, :]).reshape(-1, X.shape[1]))
    D = np.vstack(D)
    w = PD.fit_logreg(np.vstack([D, -D]), np.r_[np.ones(len(D)), np.zeros(len(D))], l2)
    return w


def fit(Xs, ys, method: str) -> np.ndarray:
    if method == "pairwise":
        return fit_pairwise(Xs, ys)
    w = PD.fit_logreg(np.vstack(Xs), np.concatenate(ys), 1.0)
    return PD.fit_coord(Xs, ys, w) if method == "coord" else w


def crossfit(B: dict, method: str, folds: int = 5, seed: int = 0) -> dict:
    n = len(B["X"])
    order = list(range(n))
    random.Random(seed).shuffle(order)
    preds = {}
    for k in range(folds):
        test = set(order[k::folds])
        tr = [i for i in range(n) if i not in test]
        w = fit([B["X"][i] for i in tr], [B["ys"][i] for i in tr], method)
        sub = dict(B, qids=[B["qids"][i] for i in sorted(test)], X=[B["X"][i] for i in sorted(test)],
                   cands=[B["cands"][i] for i in sorted(test)])
        preds.update(PD.predict(sub, w))
    return PD.micro_at_k(preds, B["gold"])


# -------------------------------------------------------------------------- main
def prepare_extra(data: dict, L: dict, log=print) -> dict:
    extra = {}
    for s in ("dev", "test"):
        q, p = data[s]
        qd, pd_ = [doc_date(d["text"]) for d in q], [doc_date(d["text"]) for d in p]
        later, gap, known = date_matrices(qd, pd_)
        src = Counter(d["src"] for d in qd + pd_)
        msg = f"[dates] {s}: " + ", ".join(f"{k} {v / len(qd + pd_):.0%}" for k, v in src.most_common())
        msg += f"; candidates certainly later than their query: {later.mean():.1%} of all pairs"
        if L[s]["gold"]:
            pos = {c: j for j, c in enumerate(L[s]["pids"])}
            bad = tot = 0
            for i, qq in enumerate(L[s]["qids"]):
                for c in L[s]["gold"][qq]:
                    if c in pos:
                        tot += 1
                        bad += bool(later[i, pos[c]])
            msg += f"; cited cases wrongly marked later: {bad} of {tot}"
        log(msg)
        back, fwd, co = citation_matrices([d["text"] for d in q], [d["text"] for d in p])
        for i, j in L[s]["self"]:
            back[i, j] = fwd[i, j] = co[i, j] = 0.0
        if L[s]["gold"]:
            G = np.zeros_like(back, dtype=bool)
            pos = {c: j for j, c in enumerate(L[s]["pids"])}
            for i, qq in enumerate(L[s]["qids"]):
                for c in L[s]["gold"][qq]:
                    if c in pos:
                        G[i, pos[c]] = True
            for name, M in (("query in the candidate's CITATOR INFO", back), ("candidate's citation in the query", fwd),
                            ("cited by the same later case", co)):
                h = M > 0
                log(f"[citations] {s}, {name}: finds {(h & G).sum()} of {G.sum()} cited cases "
                    f"({(h & G).sum() / max(1, G.sum()):.1%}), precision {(h & G).sum() / max(1, h.sum()):.1%}")
        extra[s] = dict(later=later, gap=gap, known=known, prior=PD.prior_counts(L, s),
                        votes=knn_votes(data, s, log=log), back=back, fwd=fwd, co=co,
                        ctx=context_matrices(data, s, log=log))
    return extra


def write_named(B: dict, w: np.ndarray, L: dict, out_dir: str, name: str) -> str:
    q = [dict(id=x) for x in L["test"]["qids"]]
    p = [dict(id=x) for x in L["test"]["pids"]]
    tmp = os.path.join(out_dir, "_tmp", name)
    z = PD.write_prediction(PD.predict(B, w), q, p, tmp)
    dst = os.path.join(out_dir, f"{name}.zip")
    shutil.copy(z, dst)
    return dst


def main(results_roots=("/kaggle/input",), out_dir: str = "/kaggle/working/final", data_root: str = PD.DEFAULT_ROOT,
         lexical_path: str | None = None, fetch: bool = True, top_write: int = 4, log=print) -> list:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    data = PD.load_data(data_root, fetch=fetch, log=log)
    if lexical_path and os.path.exists(lexical_path):
        L = PD.load_lexical(lexical_path)
    else:
        L = PD.lexical(data, log=log)
        if lexical_path:
            PD.save_lexical(L, lexical_path)
    extra = prepare_extra(data, L, log)
    runs = find_runs(results_roots, L, log, unzip_dir=os.path.join(out_dir, "_unzipped"))
    sets = {"lexical": {"dev": {}, "test": {}}}
    if runs:
        sets.update(dense_sets(runs, log=log))
    rows = []
    for use_cit in (True, False):
        for sname, ds in sets.items():
            for filt in (True, False):
                Bd = build(L, "dev", extra, ds["dev"], filt, use_cit=use_cit)
                for method in ("logreg", "pairwise", "coord"):
                    m = crossfit(Bd, method)
                    rows.append(dict(name=f"{sname}_{'citator' if use_cit else 'nocitator'}_"
                                          f"{'datefilter' if filt else 'datefeature'}_{method}",
                                     set=sname, filt=filt, use_cit=use_cit, method=method, f1=m["f1"],
                                     p=m["precision"], r=m["recall"], recall_cands=Bd["recall"],
                                     features=len(Bd["names"])))
                    log(f"[final] {rows[-1]['name']:52s} dev (5-fold cross-fitted) {PD.fmt(m)}; candidates hold "
                        f"{Bd['recall']:.1%} of the cited cases; {len(Bd['names'])} features")
    rows.sort(key=lambda r: -r["f1"])
    chosen = rows[:top_write]
    best_plain = next((r for r in rows if not r["use_cit"]), None)
    if best_plain is not None and best_plain not in chosen:
        chosen.append(best_plain)                     # the best version without the citator lists, for the report
    written = []
    for r in chosen:
        ds = sets[r["set"]]
        Bd = build(L, "dev", extra, ds["dev"], r["filt"], use_cit=r["use_cit"])
        Bt = build(L, "test", extra, ds["test"], r["filt"], use_cit=r["use_cit"])
        w = fit(Bd["X"], Bd["ys"], r["method"])
        name = f"final_{r['name']}__dev{r['f1']:.4f}"
        path = write_named(Bt, w, L, out_dir, name)
        r["submission"] = path
        r["weights"] = dict(zip(Bd["names"], map(float, w)))
        written.append(path)
        log(f"[final] wrote {path}\n        weights: {PD.describe_weights(Bd['names'], w)}")
    shutil.rmtree(os.path.join(out_dir, "_tmp"), ignore_errors=True)
    shutil.rmtree(os.path.join(out_dir, "_unzipped"), ignore_errors=True)
    with open(os.path.join(out_dir, "final_results.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    lines = [f"{r['f1']:.4f}  P {r['p']:.4f}  R {r['r']:.4f}  cands {r['recall_cands']:.1%}  {r['name']}"
             + (f"  -> {os.path.basename(r['submission'])}" if r.get("submission") else "") for r in rows]
    text = "final ranking variants, dev micro-F1@7 (5-fold cross-fitted), best first:\n" + "\n".join(lines)
    with open(os.path.join(out_dir, "final_results.txt"), "w") as f:
        f.write(text + "\n")
    log(text)
    log(f"[final] done in {(time.time() - t0) / 60:.1f} min; submit the files in {out_dir} from the top")
    return rows
