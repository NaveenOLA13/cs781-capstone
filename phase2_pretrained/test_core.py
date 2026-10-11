"""Tests for lner_core (no GPU needed). Run: python test_core.py"""
import json
import os
import random
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lner_core as C

TYPES = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "COURT"]
NAMES = {"APP": ["M/s. Shree Ram Mills Ltd.", "Rajesh Kumar", "Union of India", "Smt. Asha Rani"],
         "RESP": ["State of U.P.", "Smt. Kamla Devi", "The Commissioner, Kanpur Nagar"],
         "A.COUNSEL": ["Mr. A. K. Sharma", "Ms. Neha Gupta"],
         "R.COUNSEL": ["Mr. S. P. Singh, A.G.A.", "Sri Ravi Shukla"],
         "JUDGE": ["Hon'ble Justice R. Mehta"],
         "COURT": ["High Court of Judicature at Allahabad"]}
FILLER = ("the petition was filed under Section 482 Cr.P.C. ; heard counsel for the parties ,"
          " and perused the record ( dated 12.03.2019 ) .").split()


def legal_docs(n=150, seed=0, rare=None):
    rng = random.Random(seed)
    w = {t: 1.0 for t in TYPES}
    w.update(rare or {})
    docs = []
    for i in range(n):
        text, spans = "", []
        for _ in range(rng.randint(40, 700)):
            text += rng.choice([" ", " ", "  ", "\n"]) if text else ""
            if rng.random() < 0.06:
                t = rng.choices(TYPES, [w[x] for x in TYPES])[0]
                nm = rng.choice(NAMES[t])
                spans.append(C.Span(len(text), len(text) + len(nm), t))
                text += nm
                if rng.random() < 0.3:
                    text += rng.choice([",", ".", ")"])
            else:
                text += rng.choice(FILLER)
        docs.append(C.Doc(f"case{i}", text, spans, rng.choice(["fold1", "fold2"])))
    return docs


def deberta_like_offsets(text, piece=4):
    """Mimics DeBERTa-v3 fast tokenizer offsets: leading space inside the first piece,
    long words split, punctuation separate, plus occasional standalone '▁' tokens."""
    offs = []
    for m in re.finditer(r"\s*(\w+|[^\w\s])", text):
        ws, a, b = m.start(), m.start(1), m.end(1)
        if not text[a].isalnum() and a > ws:
            offs.append((ws, a))  # standalone '▁' before punctuation
            ws = a
        for i, p in enumerate(range(a, b, piece)):
            offs.append((ws if i == 0 else p, min(p + piece, b)))
    return C.normalize_offsets(text, offs)


def noisy_probs(lab, scheme, rng, under=0.6):
    """A 'model' that is right on most tokens but under-confident on entities, so the
    per-label bias tuning has something to fix."""
    n, Cn = len(lab), len(scheme)
    P = rng.dirichlet(np.ones(Cn) * 0.3, size=n).astype(np.float32) * 0.15
    true = np.where(lab == C.IGNORE, scheme.O, lab)
    for k, t in enumerate(true):
        if t == scheme.O:
            P[k, scheme.O] += 0.85
        else:
            P[k, t] += 0.85 * (1 - under)
            P[k, scheme.O] += 0.85 * under
    return P / P.sum(1, keepdims=True)


def test_windows_and_merge():
    for n in [0, 1, 50, 510, 511, 1000, 4321]:
        w = C.make_windows(n, 510, 384)
        cov = np.zeros(n, int)
        for s, e in w:
            assert e - s <= 510
            cov[s:e] += 1
        assert (cov > 0).all()
    truth = np.eye(5)[np.random.default_rng(0).integers(0, 5, 3000)].astype(np.float32)
    w = C.make_windows(3000, 510, 384)
    m = C.merge_window_probs(3000, w, [truth[s:e] for s, e in w])
    assert (m.argmax(1) == truth.argmax(1)).all()


def test_viterbi_fixes_orphan_I():
    S = C.LabelScheme(["PER"])
    logp = np.log(np.array([[0.15, 0.3, 0.55], [0.3, 0.1, 0.6], [0.9, 0.05, 0.05]]))
    assert [S.labels[p] for p in logp.argmax(1)] == ["I-PER", "I-PER", "O"]  # invalid by argmax
    path = C.viterbi(logp, S)
    assert [S.labels[p] for p in path] == ["B-PER", "I-PER", "O"]


def test_ceiling_is_one():
    docs = legal_docs()
    S = C.LabelScheme(C.label_set(docs))
    rules = C.learn_trim_rules(docs)
    offs = {d.id: deberta_like_offsets(d.text) for d in docs}
    res, _ = C.ceiling_check(docs, offs, S, rules)
    assert res["macro_f1"] == 1.0, res
    # unit-level (ensemble path) ceiling too
    units = {d.id: C.word_units(d.text) for d in docs}
    res_u, _ = C.ceiling_check(docs, units, S, rules)
    assert res_u["macro_f1"] == 1.0, res_u


def test_trimming_learned_from_gold():
    docs = legal_docs(60)
    rules = C.learn_trim_rules(docs)
    rules["_kr"], rules["_kl"] = set(rules["keep_right"]), set(rules["keep_left"])
    t = "by State of U.P., through M/s. Shree Ram Mills Ltd.). Rajesh Kumar,"
    expect = {"State of U.P.,": "State of U.P.", "M/s. Shree Ram Mills Ltd.).": "M/s. Shree Ram Mills Ltd.",
              "Rajesh Kumar,": "Rajesh Kumar"}
    for surf, want in expect.items():
        s = t.index(surf)
        out = C.trim_span(t, C.Span(s, s + len(surf), "X"), rules)
        assert t[out.start:out.end] == want, t[out.start:out.end]


def test_tuning_recovers_underconfident_model():
    docs = legal_docs(80, seed=5, rare={"JUDGE": 0.3, "COURT": 0.3})
    S = C.LabelScheme(C.label_set(docs))
    rules = C.learn_trim_rules(docs)
    rng = np.random.default_rng(1)
    prep, gold = {}, {}
    for d in docs:
        offs = deberta_like_offsets(d.text)
        lab = C.spans_to_bio(offs, d.spans, S)
        prep[d.id] = C.Prepared(d.text, offs, noisy_probs(lab, S, rng), S)
        gold[d.id] = d.spans
    res = C.tune(prep, gold, S, rules, log=lambda *a: None)
    assert res["f1_final"] > res["f1_start"] + 0.2, res
    assert all(b > 0 for b in res["biases"].values()), res["biases"]
    return res


def test_consistency_and_propagation():
    S = C.LabelScheme(["APP", "RESP"])
    text = "Rajesh Kumar filed. Later Rajesh Kumar argued. Rajesh Kumar won."
    a, b, c = [m.start() for m in re.finditer("Rajesh Kumar", text)]
    preds = [(C.Span(a, a + 12, "APP"), 0.9), (C.Span(b, b + 12, "RESP"), 0.4)]
    cons = C.make_consistent(text, preds)
    assert {s.label for s, _ in cons} == {"APP"}
    prop = C.propagate(text, cons)
    assert sorted(s.start for s, _ in prop) == [a, b, c]


def test_folds_balanced():
    docs = legal_docs(300, seed=2, rare={"JUDGE": 0.2})
    f = C.assign_folds(docs, 5, seed=3)
    for t in TYPES:
        per = [sum(1 for d in docs for s in d.spans if s.label == t and f[d.id] == k) for k in range(5)]
        assert max(per) - min(per) <= max(3, 0.12 * np.mean(per)), (t, per)
    nd = np.bincount(list(f.values()))
    assert nd.max() - nd.min() <= 0.15 * nd.mean() + 2, nd


def test_unit_projection_ensemble_two_tokenizers():
    docs = legal_docs(20, seed=9)
    S = C.LabelScheme(C.label_set(docs))
    rules = C.learn_trim_rules(docs)
    prep, gold = {}, {}
    for d in docs:
        Us = []
        for piece in (3, 5):
            offs = deberta_like_offsets(d.text, piece)
            lab = C.spans_to_bio(offs, d.spans, S)
            oh = np.eye(len(S), dtype=np.float32)[np.where(lab == C.IGNORE, S.O, lab)]
            units, U = C.project_to_units(d.text, offs, oh, S)
            Us.append(U)
        prep[d.id] = C.Prepared(d.text, units, np.mean(Us, 0), S)
        gold[d.id] = d.spans
    assert C.macro_f1(gold, C.predict_spans(prep, S, rules), S.types)["macro_f1"] == 1.0


def test_loader_formats():
    text = "Appeal by Rajesh Kumar against State of U.P."
    variants = [
        [{"id": "a", "text": text, "entities": [{"start": 10, "end": 22, "label": "APP", "text": "Rajesh Kumar"}]}],
        [{"doc_id": "a", "text": text, "spans": [[10, 22, "APP"]]}],
        [{"id": "a", "data": {"text": text}, "annotations": [{"value": {"start": 10, "end": 22, "labels": ["APP"]}}]}],
        [{"id": "a", "text": text, "entities": json.dumps([{"start_offset": 10, "end_offset": 22, "type": "APP"}])}],
        [{"id": "a", "text": text, "labels": {"APP": [[10, 22]]}}],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for k, rows in enumerate(variants):
            for kind in ("json", "jsonl"):
                p = os.path.join(tmp, f"fold1_{k}.{kind}")
                with open(p, "w") as f:
                    f.write(json.dumps(rows) if kind == "json" else "\n".join(json.dumps(r) for r in rows))
                d = C.load_file(p)[0]
                assert d.spans == [C.Span(10, 22, "APP")], (k, kind, d.spans)
        os.makedirs(os.path.join(tmp, "Phase_2_LNER"))
        for name in ("fold1.json", "fold2.json", "test.json"):
            with open(os.path.join(tmp, "Phase_2_LNER", name), "w") as f:
                json.dump(variants[0], f)
        found = C.find_files(os.path.join(tmp, "Phase_2_LNER"))
        assert set(found) == {"fold1", "fold2", "test"}, found
        tr, te, _ = C.load_data(os.path.join(tmp, "Phase_2_LNER"))
        assert len(tr) == 2 and len({d.id for d in tr}) == 2 and len(te) == 1


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                out = fn()
                print(f"PASS {name}" + (f"  {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})}" if out else ""))
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e!r}"[:600])
    sys.exit(1 if fails else 0)