"""Self-test of cj_ensemble.py (no torch needed).   python test_cj_ensemble.py  -> every line PASS"""
import json
import os
import random
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cj_data as D  # noqa: E402
import cj_ensemble as E  # noqa: E402
from test_cj_data import make_fake_data  # noqa: E402


def _run(folder, dev_ids, test_ids, y, wrong, rng):
    """Probabilities that are right except on the cases in `wrong`."""
    os.makedirs(folder, exist_ok=True)
    dev = {i: (0.8 if y[i] else 0.2) if i not in wrong else (0.3 if y[i] else 0.7) for i in dev_ids}
    dev = {i: min(1, max(0, p + rng.uniform(-0.05, 0.05))) for i, p in dev.items()}
    test = {i: rng.random() for i in test_ids}
    json.dump(dev, open(os.path.join(folder, "dev_probs.json"), "w"))
    json.dump(test, open(os.path.join(folder, "test_probs.json"), "w"))


def test_ensemble_picks_complementary_runs():
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_data(tmp, n_dev=60)
        data = D.load_data(root, fetch=False)
        dev_ids = [d["id"] for d in data["dev"]]
        test_ids = [d["id"] for d in data["test"]]
        y = {d["id"]: d["label"] for d in data["dev"]}
        base = os.path.join(tmp, "runs")
        # three runs, each wrong on a different fifth of dev: the average of the three is right everywhere
        for k, name in enumerate(("a_full", "b_lorma", "c_adapter")):
            _run(os.path.join(base, name), dev_ids, test_ids, y, set(dev_ids[k * 12:(k + 1) * 12]), rng)
        _run(os.path.join(base, "d_bad"), dev_ids, test_ids, y, set(dev_ids[:40]), rng)
        _run(os.path.join(base, "a_full", "hier_k8", "gru_attn"), dev_ids, test_ids, y, set(dev_ids[48:60]), rng)
        quick = os.path.join(base, "quick")
        os.makedirs(quick)
        json.dump({dev_ids[0]: 0.5}, open(os.path.join(quick, "dev_probs.json"), "w"))
        json.dump({test_ids[0]: 0.5}, open(os.path.join(quick, "test_probs.json"), "w"))
        runs = E.find_runs(base)
        assert set(runs) == {"a_full", "b_lorma", "c_adapter", "d_bad", "a_full/hier_k8/gru_attn", "quick"}, runs
        logs = []
        res = E.ensemble(runs, os.path.join(tmp, "ens"), data_root=root, fetch=False, log=logs.append)
        assert "skipped quick" in "\n".join(logs)
        assert res["dev_tuned"]["macro_f1"] == 1.0 and res["single_best_dev_f1"] < 0.97, res
        assert "d_bad" not in res["members"] and 2 <= len(res["members"]) <= 5, res["members"]
        assert [s["dev_f1"] for s in res["steps"]] == sorted(s["dev_f1"] for s in res["steps"])
        with zipfile.ZipFile(res["submission"]) as z:
            rows = json.loads(z.read("prediction.json"))
        assert [r["id"] for r in rows] == test_ids
        assert set(json.load(open(os.path.join(tmp, "ens", "test_probs.json")))) == set(test_ids)


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
