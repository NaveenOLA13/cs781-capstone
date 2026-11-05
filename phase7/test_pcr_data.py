"""Self-test of pcr_data (no torch, a few seconds): python test_pcr_data.py -> every line PASS.

Fake data: every split has its own pool of cases (random legal-ish words + a year); every query copies a few
5-word phrases from 3-8 older pool cases (its relevant candidates) and has a later year. Checks: loading (text as a
list of sentences), BM25 against a slow direct formula, micro-F1@7, the prediction file rules, lexical retrieval
finds the planted citations, citation counts leave a training query's own labels out, rankers, fusion with an
informative model score beats lexical alone.
"""
import json
import math
import os
import random
import sys
import tempfile
import zipfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcr_data as PD  # noqa: E402

WORDS = [f"w{i}" for i in range(400)] + ["appeal", "court", "section", "act", "held", "order", "petition", "the", "of"]


def _sentences(words, rng):
    out, i = [], 0
    while i < len(words):
        k = rng.randint(6, 14)
        out.append(" ".join(words[i:i + k]) + ".")
        i += k
    return out


def make_fake_pcr(root, n_q=(60, 20, 20), n_pool=(150, 80, 80), seed=0, as_dict=False):
    rng = random.Random(seed)
    base = os.path.join(root, "Phase_7_PCR")
    gold_all = {}
    for split, nq, npool in zip(("train", "dev", "test"), n_q, n_pool):
        pool = []
        for j in range(npool):
            words = [rng.choice(WORDS) for _ in range(rng.randint(150, 300))]
            year = rng.randint(1950, 2000)
            pool.append(dict(id=f"{split[:2]}c{j}", words=words, year=year))
        queries = []
        for i in range(nq):
            rel = rng.sample(pool, rng.randint(3, 8))
            words = [rng.choice(WORDS) for _ in range(200)]
            for c in rel:                                     # copied phrases from most cited cases
                for _ in range(rng.choice([0, 1, 3, 3])):
                    s = rng.randint(0, len(c["words"]) - 6)
                    pos = rng.randint(0, len(words))
                    words[pos:pos] = c["words"][s:s + 5]
            year = max(c["year"] for c in rel) + rng.randint(1, 10)
            q = dict(id=f"{split[:2]}q{i}", text=_sentences(words + ["decided", "in", str(year)], rng))
            if split != "test":
                q["relevant_candidates"] = [c["id"] for c in rel]
            gold_all[q["id"]] = [c["id"] for c in rel]
            queries.append(q)
        cands = [dict(id=c["id"], text=_sentences(c["words"] + ["dated", "1", "1", str(c["year"])], rng)) for c in pool]
        d = os.path.join(base, split.capitalize())
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{split}.json"), "w") as f:
            json.dump(queries, f)
        with open(os.path.join(d, f"{split}_candidates.json"), "w") as f:
            json.dump({c["id"]: {"text": c["text"]} for c in cands} if as_dict else cands, f)
    return base, gold_all


def test_load():
    with tempfile.TemporaryDirectory() as tmp:
        root, _ = make_fake_pcr(tmp, as_dict=True)
        data = PD.load_data(root, fetch=False)
        q, p = data["train"]
        assert len(q) == 60 and len(p) == 150 and isinstance(q[0]["text"], str) and q[0]["rel"]
        assert "rel" not in data["test"][0][0] and p[0]["id"].startswith("trc")


def test_bm25_formula():
    docs = [np.array(x) for x in ([1, 2, 3, 2], [2, 4, 5], [1, 1, 1, 6, 7, 8], [9])]
    q = np.array([2, 1, 1, 7])
    k1, b = 1.2, 0.75
    bm = PD.BM25(docs, 1, k1, b)
    avg = np.mean([len(d) for d in docs])
    exp = []
    for d in docs:
        c = Counter(d.tolist())
        s = 0.0
        for t, qt in Counter(q.tolist()).items():
            df = sum(t in x.tolist() for x in docs)
            if not df:
                continue
            idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
            tf = c.get(t, 0)
            s += (1 + math.log(qt)) * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
        exp.append(s)
    assert np.allclose(bm.score(q), exp, atol=1e-5), (bm.score(q), exp)
    bi = PD.BM25(docs, 2)
    assert bi.score(np.array([2, 3]))[0] > 0 and bi.score(np.array([2, 3]))[1] == 0
    assert len(PD.ngrams(np.array([1, 2]), 3)) == 0


def test_metric_and_file():
    gold = {"a": ["1", "2", "3"], "b": ["4"]}
    m = PD.micro_at_k({"a": ["1", "9", "2"], "b": ["5"]}, gold, k=7)
    assert m["hits"] == 2 and abs(m["precision"] - 2 / 4) < 1e-9 and abs(m["recall"] - 2 / 4) < 1e-9
    with tempfile.TemporaryDirectory() as tmp:
        qs = [dict(id="a"), dict(id="b")]
        pool = [dict(id=str(i)) for i in range(10)] + [dict(id="a")]
        z = PD.write_prediction({"a": ["a", "1", "1", "2"], "b": []}, qs, pool, tmp)
        out = json.loads(zipfile.ZipFile(z).read("prediction.json"))
        assert set(out) == {"a", "b"} and all(len(v) == 7 and len(set(v)) == 7 for v in out.values())
        assert "a" not in out["a"] and out["a"][:2] == ["1", "2"]


def test_lexical_ltr_fusion():
    with tempfile.TemporaryDirectory() as tmp:
        root, gold_all = make_fake_pcr(tmp)
        data = PD.load_data(root, fetch=False)
        L = PD.lexical(data, ns=(1, 2, 3, 5), log=lambda *a: None)
        m5 = L["single"]["dev_bm25_5"]
        assert m5["f1"] > 0.35, m5                   # copied 5-word phrases are found
        assert L["single"]["dev_bm25_1"]["f1"] < m5["f1"]
        P = PD.prior_counts(L, "train")
        sp = L["train"]
        pos = {c: j for j, c in enumerate(sp["pids"])}
        g0 = sp["gold"][sp["qids"][0]]
        cnt = Counter(c for g in sp["gold"].values() for c in g)
        assert abs(P[0, pos[g0[0]]] - math.log1p(cnt[g0[0]] - 1)) < 1e-6      # own label left out
        path = os.path.join(tmp, "lex.pkl")
        PD.save_lexical(L, path)
        L = PD.load_lexical(path)
        out = os.path.join(tmp, "runs")
        res = PD.lexical_submissions(L, out, top=30, log=lambda *a: None)
        assert res["lexical_ltr"]["dev"]["f1"] > 0.4, res["lexical_ltr"]
        sub = json.loads(zipfile.ZipFile(res["lexical_ltr"]["submission"]).read("prediction.json"))
        test_pool = {d["id"] for d in data["test"][1]}
        assert len(sub) == 20 and all(len(v) == 7 and set(v) <= test_pool for v in sub.values())
        # an informative model score (gold + noise) fused with the lexical features beats lexical alone
        rng = np.random.default_rng(0)

        def fake_dense(split):
            sp = L[split]
            M = rng.normal(0, 1, (len(sp["qids"]), len(sp["pids"]))).astype(np.float32)
            for i, q in enumerate(sp["qids"]):
                for c in gold_all[q]:
                    M[i, sp["pids"].index(c)] += 2.5
            return {"mean": M}
        fr = PD.fused_submission(L, fake_dense("dev"), fake_dense("test"), os.path.join(out, "fake"), top=30,
                                 dense_top=10, log=lambda *a: None)
        assert fr["dev"]["f1"] > fr["lexical_logreg"]["f1"], fr
        assert fr["weights"]["mean"] > 0 and os.path.exists(fr["submission"])
        Bd = PD.build(L, "dev", top=30)
        assert Bd["recall"] > 0.8 and Bd["X"][0].shape[1] == len(Bd["names"])
        txt = PD.explore(data)
        assert "pool overlap" in txt and "after the query" in txt


def test_ranker_alone():
    rng = np.random.default_rng(1)
    Xs, ys = [], []
    for _ in range(40):
        X = rng.normal(0, 1, (30, 3))
        y = (X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.5, 30) > 1.5).astype(float)
        Xs.append(X)
        ys.append(y)
    w = PD.fit_logreg(np.vstack(Xs), np.concatenate(ys))
    assert w[0] > 0 and w[1] < 0 and abs(w[2]) < abs(w[1])
    wc = PD.fit_coord(Xs, ys, w)
    assert PD._hits(Xs, ys, wc) >= PD._hits(Xs, ys, w)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
