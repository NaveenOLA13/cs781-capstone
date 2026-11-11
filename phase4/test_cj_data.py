"""Self-test of cj_data.py (no torch needed). python test_cj_data.py -> every line PASS"""

import json
import os
import random
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cj_data as D  # noqa: E402

FILLER = [
    "court",
    "section",
    "appeal",
    "order",
    "petitioner",
    "respondent",
    "act",
    "the",
    "of",
    "high",
    "tribunal",
    "evidence",
    "facts",
    "learned",
    "counsel",
    "submitted",
    "held",
    "that",
    "and",
    "is",
]


def fake_text(rng, label, n_words):
    body = [rng.choice(FILLER) for _ in range(n_words)]
    end = ["accordingly", "the", "appeal", "is"] + (["allowed"] * 3 if label == 1 else ["dismissed"] * 3)
    return " ".join(body + end)


def make_fake_data(root, n_train=40, n_dev=12, n_test=10, fmt="list", seed=0, n_words=(20, 120)):
    """root/Phase_4_CJPE/{Train/train.json, Train/dev.json, Test/test.json}; returns the Phase_4_CJPE folder."""
    rng = random.Random(seed)
    base = os.path.join(root, "Phase_4_CJPE")
    os.makedirs(os.path.join(base, "Train"), exist_ok=True)
    os.makedirs(os.path.join(base, "Test"), exist_ok=True)

    def cases(prefix, n, labelled):
        out = []
        for i in range(n):
            lab = i % 2 if i < n - 2 else 1
            c = {"id": f"{prefix}_{i:03d}", "text": fake_text(rng, lab, rng.randint(*n_words))}
            if labelled:
                c["label"] = lab
            out.append(c)
        return out

    for name, sub, n, labelled in (
        ("train", "Train", n_train, True),
        ("dev", "Train", n_dev, True),
        ("test", "Test", n_test, False),
    ):
        cs = cases(name, n, labelled)
        obj = (
            cs
            if fmt == "list"
            else {c["id"]: {k: (str(v) if k == "label" else v) for k, v in c.items() if k != "id"} for c in cs}
        )
        with open(os.path.join(base, sub, f"{name}.json"), "w") as f:
            json.dump(obj, f)
    return base


def word_encode(text):
    return [3 + sum(map(ord, w)) % 597 for w in text.split()]


# --- tests
def test_load_formats():
    with tempfile.TemporaryDirectory() as tmp:
        for fmt in ("list", "dict"):
            root = make_fake_data(os.path.join(tmp, fmt), fmt=fmt)
            data = D.load_data(root, fetch=False)
            assert [len(data[s]) for s in ("train", "dev", "test")] == [40, 12, 10]
            assert all(d["label"] in (0, 1) for s in ("train", "dev") for d in data[s])
            assert all(d["label"] is None for d in data["test"])
            assert data["train"][0]["id"] == "train_000" and data["train"][0]["text"].endswith("allowed") is False
            assert "train" in D.summary(data)
        bad = os.path.join(tmp, "bad.json")
        for content, need in (
            ([{"id": "a", "text": "x"}], True),
            ([{"id": "a", "text": "x", "label": 2}], True),
            ([{"text": "x", "label": 1}], True),
        ):
            json.dump(content, open(bad, "w"))
            try:
                D.load_split(bad, need_label=need)
                raise AssertionError(f"{content} should fail")
            except ValueError:
                pass
        json.dump([{"id": "a", "text": ["first", "sentence"], "label": "1"}], open(bad, "w"))
        assert D.load_split(bad, True) == [dict(id="a", text="first sentence", label=1)]
        assert D.load_split(bad, False, read_label=False) == [dict(id="a", text="first sentence", label=None)]
        root = make_fake_data(os.path.join(tmp, "testlab"))
        tp = os.path.join(root, "Test", "test.json")
        cases = json.load(open(tp))
        json.dump([dict(c, label=1, expert_1=None) for c in cases], open(tp, "w"))
        assert all(d["label"] is None for d in D.load_data(root, fetch=False)["test"])
    assert [D.to_label(v) for v in (0, 1, True, "0", "1", 1.0, "1.0", [1])] == [0, 1, 1, 0, 1, 1, 1, 1]


def test_clean_splits():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_data(tmp)
        paths = {
            s: os.path.join(root, sub, f"{s}.json")
            for s, sub in (("train", "Train"), ("dev", "Train"), ("test", "Test"))
        }
        raw = {s: json.load(open(p)) for s, p in paths.items()}
        dev2 = raw["dev"] + raw["dev"][:3] + [dict(c, label=1) for c in raw["test"]] + [dict(raw["train"][0], id="x")]
        train2 = raw["train"] + raw["train"][:5] + [dict(raw["test"][0], id="t-copy", label=0)]
        json.dump(dev2, open(paths["dev"], "w"))
        json.dump(train2, open(paths["train"], "w"))
        logs = []
        data = D.load_data(root, fetch=False, log=logs.append)
        assert [len(data[s]) for s in ("train", "dev", "test")] == [40, 12, 10]
        test_ids = {d["id"] for d in data["test"]}
        test_texts = {d["text"] for d in data["test"]}
        for s in ("train", "dev"):
            assert not any(d["id"] in test_ids or d["text"] in test_texts for d in data[s]), s
        assert len({d["text"] for d in data["train"]}) == 40 and len({d["text"] for d in data["dev"]}) == 12
        text = "\n".join(logs)
        assert "dev: 26 -> 12 cases (removed 10 test cases, 3 repeats, 1 also in train)" in text, text
        assert "train: 46 -> 40 cases (removed 1 test cases, 5 repeats)" in text, text
        unclean = D.load_data(root, fetch=False, clean=False)
        assert [len(unclean[s]) for s in ("train", "dev")] == [46, 26]


def test_windows():
    rng = random.Random(1)
    text = " ".join(rng.choice(FILLER) + str(rng.randint(0, 99)) for _ in range(500))
    full = word_encode(text)
    for n in (1, 7, 64, 200):
        assert D.encode_window(text, word_encode, n, "tail") == full[-n:]
        assert D.encode_window(text, word_encode, n, "head") == full[:n]
        ht = D.encode_window(text, word_encode, n, "head_tail")
        assert ht == full[: n // 4] + full[len(full) - (n - n // 4) :], n
    short = "only five words in it"
    for w in D.WINDOWS:
        assert D.encode_window(short, word_encode, 64, w) == word_encode(short)
    long_text = " ".join(["word"] * 3000) + " end"
    assert D.encode_window(long_text, word_encode, 40, "head_tail") == (
        word_encode("word") * 10 + word_encode("word") * 29 + word_encode("end")
    )
    assert D.tail_text("abc def ghi", 5) == "ghi" and D.head_text("abc def ghi", 6) == "abc"


def test_encode_docs_cache_and_workers():
    calls = {"n": 0}

    def counting(text):
        calls["n"] += 1
        return word_encode(text)

    rng = random.Random(2)
    docs = [dict(id=str(i), text=fake_text(rng, i % 2, rng.randint(5, 300)), label=i % 2) for i in range(40)]
    want = [word_encode(d["text"])[-32:] for d in docs]
    with tempfile.TemporaryDirectory() as tmp:
        a = D.encode_docs(docs, counting, 32, "tail", workers=1, cache_dir=tmp, log=lambda *x: None)
        assert a == want and calls["n"] > 0
        files = os.listdir(tmp)
        assert len(files) == 1 and files[0].endswith("_tail_32.pkl"), files
        calls["n"] = 0
        b = D.encode_docs(docs, counting, 32, "tail", workers=1, cache_dir=tmp, log=lambda *x: None)
        assert b == want and calls["n"] == 1
        c = D.encode_docs(docs[::-1], word_encode, 32, "tail", workers=3, log=lambda *x: None)
        assert c == want[::-1]
    lengths = [len(x) for x in want]
    assert "fill the whole 32-token window" in D.length_report(lengths, 32)


def test_chunks():
    ids = list(range(1000))
    ch = D.chunk_ids(ids, 512, 448, 3)
    assert [(c[0], c[-1], len(c)) for c in ch] == [(40, 551, 512), (488, 999, 512)]
    ch = D.chunk_ids(list(range(2000)), 512, 448, 3)
    assert [(c[0], c[-1]) for c in ch] == [(592, 1103), (1040, 1551), (1488, 1999)]
    assert D.chunk_ids(list(range(300)), 512, 448, 8) == [list(range(300))]
    assert D.chunk_ids([], 512, 448, 3) == [[]]
    assert D.chunk_span(512, 448, 8) == 3648
    long = list(range(5000))
    tail = long[-D.chunk_span(512, 448, 4) :]
    assert D.chunk_ids(tail, 512, 448, 4) == D.chunk_ids(long, 512, 448, 4)


def test_batches_cover_everything_once():
    lengths = [5, 5, 9, 9, 9, 3, 12, 5, 9, 1, 12]
    for exact in (False, True):
        for shuffle in (False, True):
            bs = D.batches(lengths, 2, shuffle=shuffle, seed=3, exact=exact)
            assert sorted(i for b in bs for i in b) == list(range(len(lengths)))
            assert all(len(b) <= 2 for b in bs)
            if exact:
                assert all(len({lengths[i] for i in b}) == 1 for b in bs)


def test_report_and_threshold():
    y = [1, 1, 1, 0, 0, 0, 0, 1]
    p = [0.9, 0.8, 0.4, 0.3, 0.6, 0.1, 0.2, 0.7]
    r = D.report(y, p, 0.5)
    assert (r["tp"], r["tn"], r["fp"], r["fn"]) == (3, 3, 1, 1)
    assert abs(r["class1"]["precision"] - 0.75) < 1e-12 and abs(r["class1"]["recall"] - 0.75) < 1e-12
    assert abs(r["macro_f1"] - 0.75) < 1e-12 and abs(r["accuracy"] - 0.75) < 1e-12
    t, f = D.tune_threshold([1, 1, 0, 0], [0.9, 0.35, 0.3, 0.1])
    assert f == 1.0 and 0.3 < t <= 0.35, (t, f)
    t, f = D.tune_threshold([1, 0], [0.9, 0.1])
    assert f == 1.0 and t == 0.5
    allone = D.report([1, 0, 0], [0.9, 0.9, 0.9])
    assert allone["class0"]["f1"] == 0.0 and abs(allone["macro_f1"] - 0.25) < 1e-12
    assert "class 1" in D.format_report(r)


def test_class_weights_and_folds():
    assert D.class_weights([0, 0, 0, 0, 1]) == [1.0, 2.0]
    assert D.class_weights([0, 1, 0, 1]) == [1.0, 1.0]
    docs = [dict(id=str(i), text="x", label=int(i % 3 == 0)) for i in range(30)]
    folds = D.make_folds(docs, k=5)
    assert sorted(i for f in folds for i in f) == list(range(30))
    assert all(sum(docs[i]["label"] for i in f) == 2 for f in folds)


def test_submission():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_data(tmp)
        test = D.load_data(root, fetch=False)["test"]
        preds = {d["id"]: i % 2 for i, d in enumerate(test)}
        z = D.write_submission(test, preds, os.path.join(tmp, "sub"))
        with zipfile.ZipFile(z) as zf:
            assert zf.namelist() == ["prediction.json"]
            rows = json.loads(zf.read("prediction.json"))
        assert rows == [{"id": d["id"], "pred": i % 2} for i, d in enumerate(test)]
        for bad in (
            {**preds, "extra": 1},
            {k: v for k, v in list(preds.items())[1:]},
            {**preds, test[0]["id"]: 2},
            {**preds, test[0]["id"]: 0.5},
            {**preds, test[0]["id"]: True},
        ):
            try:
                D.write_submission(test, bad, os.path.join(tmp, "bad"))
                raise AssertionError("should fail")
            except ValueError:
                pass


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
