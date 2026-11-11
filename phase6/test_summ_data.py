"""Self-test of the Phase 6 data helpers (no torch). python test_summ_data.py -> every line PASS"""

import json
import os
import random
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summ_data as D  # noqa: E402

FILL = [
    "the record shows the hearing was held",
    "counsel for both parties were heard at length",
    "the documents were placed before the bench",
    "several earlier orders were referred to",
    "the matter was listed again after notice",
    "written submissions were filed by the state",
]
KEYS = [
    "the appellant was convicted under section 302",
    "the high court reversed the acquittal",
    "the land acquisition notice was quashed",
    "the appeal is allowed with costs",
    "the tenant was evicted for default in rent",
    "the bail application was rejected",
]


def make_fake_summ(tmp, n_train=60, n_test=8, n_sent=(8, 16), seed=0, dup_test=1):
    """train.json / test.json like the Phase 6 data: a document = filler sentences + 2 key sentences; the summary = the
    key sentences (reworded slightly)."""
    rng = random.Random(seed)

    def doc():
        s = [rng.choice(FILL) + f" on day {rng.randint(1, 30)}." for _ in range(rng.randint(*n_sent))]
        keys = rng.sample(KEYS, 2)
        for k in keys:
            s.insert(rng.randint(0, len(s)), k.capitalize() + ".")
        summ = " ".join(("In short " + k + ".") for k in keys)
        return " ".join(x[0].upper() + x[1:] for x in s), summ

    root = os.path.join(tmp, "Phase_6_Summ")
    os.makedirs(os.path.join(root, "Train"))
    os.makedirs(os.path.join(root, "Test"))
    train = [dict(id=f"tr{i}", document=d, summary=s) for i, (d, s) in enumerate(doc() for _ in range(n_train))]
    test = [dict(id=f"te{i}", document=doc()[0]) for i in range(n_test)]
    for i in range(dup_test):
        train.append(dict(id=f"dup{i}", document=test[i]["document"], summary="x y z."))
    json.dump(train, open(os.path.join(root, "Train", "train.json"), "w"))
    json.dump(test, open(os.path.join(root, "Test", "test.json"), "w"))
    return root


def test_sentences_words():
    t = "The appellant filed Civil Appeal No. 5 of 1990 vs. the State. The court held so. Appeal is dismissed."
    s = D.sentences(t)
    assert len(s) == 3 and s[0].endswith("State.") and s[2] == "Appeal is dismissed.", s
    assert D.words("Sec. 302, I.P.C.!") == ["sec", "302", "i", "p", "c"]
    assert D.content(["the", "appeal", "is", "allowed", "a"]) == ["appeal", "allowed"]


def test_windows_align_pairs():
    assert D.windows(10, 4) == [(0, 4), (4, 8), (6, 10)] and D.windows(3, 4) == [(0, 3)]
    assert D.windows(8, 4) == [(0, 4), (4, 8)]
    wins = D.windows(10, 4)
    assert [D.window_of(p, wins) for p in (0, 3, 4, 7, 8, 9)] == [0, 0, 1, 1, 2, 2]
    doc = ["Counsel were heard.", "The appellant was convicted under section 302.", "The appeal is allowed."]
    assert D.align(doc, ["Appeal allowed with no order.", "He was convicted under 302.", "the of"]) == [2, 1, -1]
    enc = str.split
    d = dict(
        id="a",
        document="Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu.",
        summary="Iota kappa summary. Alpha beta summary.",
    )
    pairs, info = D.make_pairs([d], enc, window=4, tgt_tokens=50, empty_frac=0.0)
    assert [p["w"] for p in pairs] == [0, 2] and pairs[0]["tgt"] == ["Alpha", "beta", "summary."], pairs
    assert pairs[1]["src"] == ["Iota", "kappa", "lambda", "mu."] and info["empty_targets"] == 0
    pairs, info = D.make_pairs([d], enc, window=4, tgt_tokens=50, empty_frac=1.0)
    assert len(pairs) == 3 and pairs[1]["tgt"] == [] and info["empty_targets"] == 1
    pairs, info = D.make_pairs([d], enc, window=4, tgt_tokens=2, empty_frac=0.0)
    assert all(len(p["tgt"]) <= 2 for p in pairs) and info["targets_cut"] == 2
    ids, starts, sents = D.tokenize_doc(d["document"], enc)
    assert starts == [0, 4, 8] and len(ids) == 12 and len(sents) == 3


def test_rouge():
    a = "the appeal is allowed".split()
    assert D.rouge_n(a, a, 1)[2] == 1.0 and D.rouge_n(a, ["x"], 1)[2] == 0.0
    assert D.lcs("a b c d".split(), "a c d b".split()) == 3 and D.lcs([], ["a"]) == 0
    assert abs(D.rouge_l("a b c d".split(), "a c d b".split())[2] - 0.75) < 1e-9
    assert D.repetition("a b c a b c".split()) == 0.25
    s = D.score_all(["the appeal is allowed"], ["the appeal is allowed"], docs=["the appeal is allowed today"])
    assert s["rouge1"] == 1.0 and s["rougeL"] == 1.0 and s["novel_bigrams"] == 0.0 and "ROUGE-1" in D.fmt(s)


def test_length_post_submission():
    docs = [dict(document=" ".join(["w"] * n), summary=" ".join(["s"] * (n // 5))) for n in range(100, 2000, 50)]
    m = D.fit_length(docs)
    assert 0.8 < m["b"] < 1.2 and D.predict_words(m, " ".join(["w"] * 1000)) > D.predict_words(m, "w w w")
    assert D.allocate([100, 100, 50], 250) == [100, 100, 50] and min(D.allocate([5, 500], 100)) == 16
    out = D.postprocess(
        [
            "The appeal is allowed with costs. The appeal is thus allowed with costs.",
            "The parties shall bear costs. And then the",
            "",
        ],
        max_words=None,
    )
    assert out == "The appeal is allowed with costs. The parties shall bear costs.", out
    assert D.postprocess(["One two three. Four five six. Seven eight nine."], max_words=5) == "One two three."
    with tempfile.TemporaryDirectory() as tmp:
        test = [dict(id="a", document="First sentence here. Second."), dict(id="b", document="Only one.")]
        z = D.write_submission(test, {"a": "Sum A."}, tmp, fallback={})
        rows = json.loads(zipfile.ZipFile(z).read("prediction.json"))
        assert rows == [{"id": "a", "summary": "Sum A."}, {"id": "b", "summary": "Only one."}]
    assert D.plan_steps(20, 1000, 2.0, seconds_left=1000, predict_seconds=600) == 220


def test_load_and_formats():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_summ(tmp, n_train=60, n_test=8)
        data = D.load_data(root, fetch=False)
        assert len(data["test"]) == 8 and len(data["train"]) + len(data["held"]) == 60
        assert not any(d["id"].startswith("dup") for d in data["train"] + data["held"])
        assert {"id", "document", "summary"} <= set(data["train"][0]) and "summary" not in data["test"][0]
        assert "summary words" in D.explore(data, encode=str.split)
        p = os.path.join(tmp, "x.jsonl")
        open(p, "w").write(
            '{"doc_id": 1, "text": "A b.", "abstract": "S."}\n{"doc_id": 2, "text": "C d.", "abstract": "T."}\n'
        )
        recs = D.normalise(D.read_json(p), True)
        assert recs[1] == {"id": "2", "document": "C d.", "summary": "T."}
        json.dump({"data": [{"id": "q", "document": "Z.", "summary": "Y."}]}, open(p, "w"))
        assert D.normalise(D.read_json(p), True)[0]["id"] == "q"


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
