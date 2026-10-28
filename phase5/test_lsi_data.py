"""Self-test of lsi_data.py and the ensemble in lsi_grid.py (no torch needed).   python test_lsi_data.py -> PASS"""
import json
import os
import random
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsi_data as L  # noqa: E402

KEYWORDS = {3: "murder", 7: "theft", 11: "fraud", 20: "dowry", 31: "bail"}   # token ids distinct from the filler


def make_fake_lsi(tmp, n_train=60, n_dev=24, n_test=12, n_words=(30, 120), seed=0, extra_test_in_dev=2,
                  at_end=False):
    """Phase_5_LSI-like folders; a document's statutes are the ids whose keyword appears in its text (at random
    places, or at the end of the document with at_end=True: easier for the tiny test models)."""
    rng = random.Random(seed)
    filler = [f"w{i}" for i in range(50)]

    def doc(i, split):
        labs = rng.sample(sorted(KEYWORDS), rng.choice([1, 1, 2]))
        if 20 in labs and rng.random() < 0.6:          # statute 20 is rare
            labs = [x for x in labs if x != 20] or [3]
        words = [rng.choice(filler) for _ in range(rng.randint(*n_words))]
        for x in labs:
            words.insert(len(words) if at_end else rng.randrange(len(words) + 1), KEYWORDS[x])
        return {"id": f"{split}_{i:03d}", "text": " ".join(words), "labels": sorted(labs)}

    root = os.path.join(tmp, "Phase_5_LSI")
    os.makedirs(os.path.join(root, "Train"))
    os.makedirs(os.path.join(root, "Test"))
    train = [doc(i, "train") for i in range(n_train)]
    dev = [doc(i, "dev") for i in range(n_dev)]
    test = [doc(i, "test") for i in range(n_test)]
    dev += [dict(t) for t in test[:extra_test_in_dev]]      # leak: test documents inside dev (must be removed)
    json.dump(train, open(os.path.join(root, "Train", "train.json"), "w"))
    json.dump(dev, open(os.path.join(root, "Train", "dev.json"), "w"))
    json.dump([{"id": t["id"], "text": t["text"]} for t in test], open(os.path.join(root, "Test", "test.json"), "w"))
    return root


def test_load_and_clean():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_lsi(tmp)
        logs = []
        data = L.load_data(root, fetch=False, log=logs.append)
        assert len(data["train"]) == 60 and len(data["dev"]) == 24 and len(data["test"]) == 12
        assert any("removed 2 test documents" in m for m in logs), logs
        assert all(d["labels"] is None for d in data["test"])
        assert all(d["labels"] and d["labels"] == sorted(set(d["labels"])) for d in data["dev"])
        labels = L.label_space(data)
        assert labels == sorted(KEYWORDS)
        Y = L.multi_hot(data["dev"], labels)
        assert Y.shape == (24, 5) and Y.sum() == sum(len(d["labels"]) for d in data["dev"])
        assert "statute ids (train + dev): 5" in L.summary(data)


def test_to_labels():
    assert L.to_labels([3, "7", 7.0, " 11 "]) == [3, 7, 11]
    assert L.to_labels(5) == [5] and L.to_labels(None) == []
    for bad in ([1.5], [True]):
        try:
            L.to_labels(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_scores():
    Y = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], bool)
    P = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], bool)
    s = L.scores(Y, P)
    # statute 0: tp 1 fn 1 -> F1 2/3; statute 1: tp 2 -> 1; statute 2 has no gold -> not in macro
    assert abs(s["macro_f1"] - (2 / 3 + 1) / 2) < 1e-9 and s["statutes_scored"] == 2
    assert abs(s["micro_f1"] - 2 * 3 / (2 * 3 + 1 + 1)) < 1e-9
    assert abs(s["subset_acc"] - 1 / 3) < 1e-9


def test_thresholds():
    rng = np.random.RandomState(0)
    n, k = 400, 4
    Y = rng.rand(n, k) < [0.5, 0.3, 0.2, 0.1]
    Y[~Y.any(1), 0] = True                     # every document has at least one statute (as in the data)
    # statute j is separable at a different cut-off: positives score above c_j, negatives below
    cuts = np.array([0.2, 0.4, 0.6, 0.8])
    P = np.where(Y, cuts + rng.rand(n, k) * (1 - cuts) * 0.9 + 0.01, rng.rand(n, k) * cuts * 0.95)
    tg, mg = L.tune_global(Y, P)
    tl = L.tune_per_label(Y, P, tg, min_pos=3)
    assert L.scores(Y, L.apply(P, tl))["macro_f1"] > 0.99 > mg
    ch = L.choose_thresholds(Y, P)
    assert ch["kind"] == "per_label" and ch["cv_per_label"] > ch["cv_global"], ch
    # min_one: a document with nothing above threshold still gets its top statute
    pred = L.apply(np.array([[0.1, 0.3], [0.9, 0.2]]), 0.5)
    assert pred.tolist() == [[False, True], [True, False]]
    # rare statutes keep the global threshold
    Yr = np.zeros((50, 2), bool)
    Yr[:2, 1] = True
    Yr[:, 0] = rng.rand(50) < 0.5
    assert L.tune_per_label(Yr, rng.rand(50, 2), 0.37, min_pos=3)[1] == 0.37


def test_plan_steps():
    # plenty of time: every planned step
    assert L.plan_steps(20, 1000, 1.0, seconds_left=5000, predict_seconds=600) == 1000
    # 400 s left after the predictions at 2 s per step: 200 more steps
    assert L.plan_steps(20, 1000, 2.0, seconds_left=1000, predict_seconds=600) == 220
    # deadline already passed: stop at the steps done (at least min_steps)
    assert L.plan_steps(50, 1000, 1.0, seconds_left=-10, predict_seconds=600) == 50
    assert L.plan_steps(5, 1000, 1.0, seconds_left=-10, predict_seconds=600, min_steps=20) == 20
    assert L.plan_steps(5, 12, 1.0, seconds_left=-10, predict_seconds=600, min_steps=20) == 12


def test_submission():
    with tempfile.TemporaryDirectory() as tmp:
        docs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        P = np.array([[0.9, 0.8, 0.1], [0.2, 0.1, 0.3], [0.6, 0.1, 0.95]])
        z = L.write_submission(docs, [5, 9, 12], P, 0.5, tmp)
        rows = json.loads(zipfile.ZipFile(z).read("prediction.json"))
        assert rows == [{"id": "a", "pred": [5, 9]}, {"id": "b", "pred": [12]}, {"id": "c", "pred": [5, 12]}]
        try:
            L.write_submission(docs[:2], [5, 9, 12], P, 0.5, tmp)
            raise AssertionError("size mismatch accepted")
        except ValueError:
            pass


def test_probs_and_ensemble():
    import lsi_grid as G
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_lsi(tmp, n_dev=80)
        data = L.load_data(root, fetch=False)
        labels = L.label_space(data)
        Y = L.multi_hot(data["dev"], labels)
        rng = np.random.RandomState(1)
        ids_dev, ids_test = [d["id"] for d in data["dev"]], [d["id"] for d in data["test"]]
        base = os.path.join(tmp, "runs")
        # three runs, each wrong (scores flipped) on a different statute: the average is right everywhere
        for r, name in enumerate(("m_full", "m_lorma", "m_adapter")):
            d = os.path.join(base, name)
            os.makedirs(d)
            P = np.where(Y, 0.75, 0.25) + rng.uniform(-0.05, 0.05, Y.shape)
            P[:, r] = 1 - P[:, r]
            L.save_probs(os.path.join(d, "dev_probs.json"), ids_dev, P)
            L.save_probs(os.path.join(d, "test_probs.json"), ids_test, rng.rand(len(ids_test), len(labels)))
            json.dump(labels, open(os.path.join(d, "labels.json"), "w"))
        back = L.load_probs(os.path.join(base, "m_full", "dev_probs.json"), ids_dev)
        assert back.shape == Y.shape
        other = os.path.join(base, "other")
        os.makedirs(other)
        json.dump([1, 2], open(os.path.join(other, "labels.json"), "w"))
        L.save_probs(os.path.join(other, "dev_probs.json"), ids_dev, np.zeros((len(ids_dev), 2)))
        L.save_probs(os.path.join(other, "test_probs.json"), ids_test, np.zeros((len(ids_test), 2)))
        logs = []
        res = G.ensemble([base], out_dir=os.path.join(tmp, "ens"), data_root=root, fetch=False, log=logs.append)
        assert "skipped other" in "\n".join(logs)
        assert len(res["members"]) == 3 and res["dev_chosen"]["macro_f1"] > res["single_best_dev_f1"] + 0.1, res
        rows = json.loads(zipfile.ZipFile(res["submission"]).read("prediction.json"))
        assert [r["id"] for r in rows] == ids_test and all(r["pred"] for r in rows)


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
