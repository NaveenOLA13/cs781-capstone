"""Self-test for rr_data.py (no torch, runs in a second).   python test_rr_data.py"""
import json
import os
import random
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_data as D  # noqa: E402

LABELS = ["Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute", "Dissent",
          "PrecedentReliedUpon", "PrecedentNotReliedUpon", "PrecedentOverruled", "RulingByLowerCourt",
          "RatioOfTheDecision", "RulingByPresentCourt", "None"]
WORDS = {l: [f"{l.lower()}{k}" for k in range(5)] for l in LABELS}


def fake_doc(rng, i, n):
    """A judgment whose roles come in blocks; each sentence uses words of its role."""
    blocks = ["None", "Fact", "Issue", "ArgumentPetitioner", "ArgumentRespondent", "Statute",
              "PrecedentReliedUpon", "RatioOfTheDecision", "RulingByPresentCourt"]
    labels = [blocks[min(len(blocks) - 1, k * len(blocks) // n)] for k in range(n)]
    sents = [" ".join(rng.choice(WORDS[l]) for _ in range(rng.randint(1, 12))) + " ." for l in labels]
    if n > 3:
        sents[2] = "Pvt."                                   # a fragment, like the real data
    return sents, labels


def make_fake_data(root, n_train=6, n_dev=2, n_test=2, vocab_format="list", seed=0):
    rng = random.Random(seed)
    os.makedirs(os.path.join(root, "Train"), exist_ok=True)
    os.makedirs(os.path.join(root, "Test"), exist_ok=True)
    vocab = {"list": LABELS, "label2id": {l: i for i, l in enumerate(LABELS)},
             "id2label": {str(i): l for i, l in enumerate(LABELS)}}[vocab_format]
    json.dump(vocab, open(os.path.join(root, "Train", "label_vocab.json"), "w"))
    for dom in ("CL", "IT"):
        for split, k in (("train", n_train), ("dev", n_dev)):
            recs = []
            for i in range(k):
                s, l = fake_doc(rng, i, rng.randint(12, 40))
                recs.append({"id": f"{dom}_{split}_{i}", "text": s, "labels": l})
            json.dump(recs, open(os.path.join(root, "Train", f"{dom}_{split}.json"), "w"))
        test = {f"{dom}_test_{i}": fake_doc(rng, i, rng.randint(12, 40))[0] for i in range(n_test)}
        json.dump(test, open(os.path.join(root, "Test", f"{dom}_test.json"), "w"))
    return root


def word_encode(s):
    return [3 + (sum(map(ord, w)) % 500) for w in s.split()]


def test_load_and_vocab_formats():
    for fmt in ("list", "label2id", "id2label"):
        with tempfile.TemporaryDirectory() as tmp:
            make_fake_data(tmp, vocab_format=fmt)
            assert D.load_labels(tmp) == LABELS, fmt
            docs = D.load_docs(tmp)
            assert {(d["split"], d["domain"]) for d in docs} == {(s, c) for s in ("train", "dev", "test")
                                                                 for c in ("CL", "IT")}
            assert all(d["labels"] is None for d in docs if d["split"] == "test")
            assert all(len(d["labels"]) == len(d["sents"]) for d in docs if d["split"] != "test")
            D.check_labels(docs, LABELS)


def test_length_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_data(tmp)
        p = os.path.join(tmp, "Train", "CL_dev.json")
        recs = json.load(open(p))
        recs[0]["labels"] = recs[0]["labels"][:-1]
        json.dump(recs, open(p, "w"))
        try:
            D.load_docs(tmp)
            assert False
        except ValueError as e:
            assert "labels" in str(e)


def test_pos_bucket():
    assert [D.pos_bucket(i, 18) for i in (0, 17)] == [0, 9]
    assert [D.pos_bucket(i, 420) for i in (17, 400, 419)] == [0, 9, 9]
    assert D.pos_bucket(0, 1) == 0 and D.pos_bucket(5, 3) == 9
    assert sorted({D.pos_bucket(i, 100) for i in range(100)}) == list(range(10))


def test_truncate_and_encode():
    ids = list(range(200))
    t = D.truncate(ids, 128, 32)
    assert len(t) == 128 and t[:96] == list(range(96)) and t[-32:] == list(range(168, 200))
    assert D.truncate([1, 2, 3], 128, 32) == [1, 2, 3]
    docs = [{"sents": ["a b c", "", " ".join(["w"] * 300)]}]
    raw = D.encode_docs(docs, word_encode, max_len=16, bos_id=1, eos_id=2, fallback_id=0)
    assert raw == [3, 0, 300]
    a, b, c = docs[0]["ids"]
    assert a[0] == 1 and a[-1] == 2 and len(a) == 5
    assert b == [1, 2]                                    # empty sentence: just bos + eos
    assert len(c) == 16 and c[0] == 1 and c[-1] == 2
    docs2 = [{"sents": [""]}]
    D.encode_docs(docs2, word_encode, max_len=16)
    assert docs2[0]["ids"] == [[0]]                        # no specials: fallback token
    print("     ", D.length_report(raw, 16))


def test_batches_cover_everything_once():
    rng = random.Random(1)
    lengths = [rng.randint(1, 100) for _ in range(1001)]
    for shuffle in (False, True):
        b = D.length_batches(lengths, 32, shuffle, seed=3)
        flat = [i for x in b for i in x]
        assert sorted(flat) == list(range(1001)) and max(map(len, b)) <= 32
    b = D.length_batches(lengths, 32, False)
    assert all(lengths[x[i]] <= lengths[x[i + 1]] for x in b for i in range(len(x) - 1))
    pad = sum(max(lengths[i] for i in x) * len(x) - sum(lengths[i] for i in x) for x in D.length_batches(lengths, 32, True))
    assert pad < 0.1 * sum(lengths), pad                  # length bucketing keeps padding small
    for shuffle in (False, True):                         # exact: one length per batch, no padding
        b = D.length_batches(lengths, 32, shuffle, seed=3, exact=True)
        assert sorted(i for x in b for i in x) == list(range(1001)) and max(map(len, b)) <= 32
        assert all(len({lengths[i] for i in x}) == 1 for x in b)
    assert len(D.length_batches(lengths, 32, True, seed=1, exact=True)) == \
        len(D.length_batches(lengths, 32, True, seed=2, exact=True))   # step count does not depend on seed


def test_class_weights():
    y = [0] * 100 + [1] * 25 + [2] * 1
    w, cnt = D.class_weights(y, 4)
    assert cnt == [100, 25, 1, 0]
    assert abs(w[0] - 1) < 1e-9 and abs(w[1] - 2) < 1e-9 and abs(w[2] - 10) < 1e-9 and w[3] == 10


def test_scores_by_hand():
    labels = ["A", "B", "C"]
    gold = [0, 0, 0, 1, 1, 2]
    pred = [0, 0, 1, 1, 0, 1]
    s = D.scores(gold, pred, labels)
    # A: tp2 fp1 fn1 -> p=r=2/3 ; B: tp1 fp2 fn1 -> p=1/3 r=1/2 ; C: 0
    fa, fb = 2 / 3, 2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2)
    assert abs(s["per_label"]["A"]["f1"] - fa) < 1e-9 and abs(s["per_label"]["B"]["f1"] - fb) < 1e-9
    assert abs(s["macro_f1"] - (fa + fb) / 3) < 1e-9 and s["confusion"][2][1] == 1
    s2 = D.scores([0, 1], [0, 1], labels)                 # C absent from gold and pred
    assert s2["macro_f1"] == 1.0 and abs(s2["macro_f1_all13"] - 2 / 3) < 1e-9
    try:
        from sklearn.metrics import f1_score
        assert abs(f1_score(gold, pred, average="macro") - s["macro_f1"]) < 1e-9
        print("      matches sklearn f1_score(average='macro')")
    except ImportError:
        pass


def test_evaluate_report_and_submission():
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_data(os.path.join(tmp, "d"))
        docs = D.load_docs(os.path.join(tmp, "d"))
        dev = [d for d in docs if d["split"] == "dev"]
        test = [d for d in docs if d["split"] == "test"]
        perfect = {d["id"]: list(d["labels"]) for d in dev}
        allfact = {d["id"]: ["Fact"] * len(d["sents"]) for d in dev}
        ra, rb = D.evaluate(dev, allfact, LABELS), D.evaluate(dev, perfect, LABELS)
        assert rb["all"]["macro_f1"] == 1.0 and set(rb) == {"all", "CL", "IT"}
        assert ra["all"]["macro_f1"] < 0.3
        txt = D.format_report({"Stage A": ra, "Stage B": rb}, LABELS)
        assert "macro-F1 CL" in txt and "Fact" in txt
        D.confusion_csv(ra["all"]["confusion"], LABELS, os.path.join(tmp, "cm.csv"))
        preds = {d["id"]: ["Fact"] * len(d["sents"]) for d in test}
        z = D.write_submission(test, preds, LABELS, os.path.join(tmp, "sub"))
        with zipfile.ZipFile(z) as f:
            assert f.namelist() == ["prediction.json"]
            back = json.loads(f.read("prediction.json"))
        assert list(back) == [d["id"] for d in test]
        for broken in ({**preds, test[0]["id"]: preds[test[0]["id"]][:-1]},
                       {k: v for k, v in list(preds.items())[1:]},
                       {**preds, test[0]["id"]: ["Facts"] * len(test[0]["sents"])}):
            try:
                D.write_submission(test, broken, LABELS, os.path.join(tmp, "bad"))
                assert False, "should have failed"
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
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)