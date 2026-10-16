"""Self-test of the final-submission code (rr_final.py, the train_on_dev and b_variants options of
rr_kd.teacher_oof, and neighbour context) on a tiny stand-in model and fake data. Needs torch and peft.
About two minutes.     python test_rr_final.py   -> every line PASS

Checks: every train and dev judgment is predicted exactly once by a copy that did not see it, for every
Stage B version; plain and smoothed submissions are complete for every version; smoothing weights (shared
and per domain) come from the given list; the probabilities export and the summary are written; a
finished ensemble is reused.
"""
import gzip
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_data as D  # noqa: E402
from rr_final import run_final  # noqa: E402
from test_rr_data import make_fake_data  # noqa: E402
from test_rr_kd import _check_zip  # noqa: E402
from test_rr_peft import ToyLlama  # noqa: E402
from test_rr_pipeline import _bb  # noqa: E402

torch.manual_seed(0)
FAST = dict(epochs=2, b_epochs=2, batch_size=16, max_len=32, lr_head=3e-3, log_every=0)
VERSIONS = ("gru", "gru_attn", "attn")


def test_run_final_context_and_versions():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=4)
        out = os.path.join(tmp, "runs")
        bb = _bb(ToyLlama())
        kw = dict(prefix="f", folds=2, data_root=data, out_dir=out, weights=(0, 0.5, 1.0), context=True,
                  context_tokens=8, b_variants=VERSIONS, **FAST)
        res = run_final(bb, **kw)
        t = torch.load(os.path.join(out, "f_ensemble", "teacher.pt"), map_location="cpu", weights_only=False)
        labelled = [d for d in D.load_docs(data) if d["split"] in ("train", "dev")]
        ids = sorted(d["id"] for d in labelled)
        assert sorted(t["train"]) == ids and sorted(sum(t["folds"], [])) == ids    # train + dev, each once
        assert t["dev"] == {} and t["train_on_dev"] is True and len(t["test"]) == 4
        assert t["main_version"] == "gru" and set(t["versions"]) == set(VERSIONS)
        for v in VERSIONS:
            assert sorted(t["versions"][v]["train"]) == ids and len(t["versions"][v]["test"]) == 4
            for d in labelled:
                assert t["versions"][v]["train"][d["id"]].shape == (len(d["labels"]), 13)
            for kind in ("plain", "smoothed"):
                _check_zip(os.path.join(out, "f_subs", v, kind, "submission.zip"), data)
        sm = json.load(open(os.path.join(out, "f_subs", "smoothing.json")))
        assert sm["best"] in VERSIONS and res["best"] == sm["best"]
        for v in VERSIONS:
            r = sm["versions"][v]
            assert r["w_shared"] in (0, 0.5, 1.0) and set(r["w_domain"]) == {"CL", "IT"}
            assert all(w in (0, 0.5, 1.0) for w in r["w_domain"].values())
            assert r["per_domain"]["all"] >= r["one_weight"]["all"] - 1e-9        # per domain includes shared
        m = res["metrics"]
        assert m["train_on_dev"] and m["config"]["context"] and m["main_version"] == "gru"
        assert set(m["stage_b_versions"]) == set(VERSIONS) and m["folds"][0]["fit_judgments"] == 6
        assert set(m["folds"][0]["stage_b_best_epochs"]) == set(VERSIONS)
        saved = torch.load(os.path.join(out, "f_ensemble", "ensemble_fold1.pt"), map_location="cpu",
                           weights_only=False)
        assert set(saved["stageB_versions"]) == set(VERSIONS)
        with gzip.open(os.path.join(out, "f_probs.json.gz"), "rt", encoding="utf-8") as f:
            ex = json.load(f)
        assert ex["versions"] == list(VERSIONS) and len(ex["pool"]) == len(labelled) and len(ex["test"]) == 4
        for row in ex["pool"] + ex["test"]:
            for v in VERSIONS:
                assert len(row["probs"][v]) == len(row["sents"])
        text = open(os.path.join(out, "f_summary.txt")).read()
        for s in ("FINAL ENSEMBLE", "previous and next sentence", "Stage B versions", "gru_attn", "best"):
            assert s in text, s
        again = run_final(bb, **kw)                                                # reused, not retrained
        assert again["best"] == res["best"]
        print(f"      best version {res['best']}; per-domain weights "
              f"{sm['versions'][res['best']]['w_domain']} (fake data)")


def test_run_final_plain_single_version():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=4)
        out = os.path.join(tmp, "runs")
        res = run_final(_bb(ToyLlama()), prefix="g", folds=2, data_root=data, out_dir=out,
                        weights=(0, 1.0), **FAST)
        assert res["best"] == "gru" and not res["metrics"]["config"]["context"]
        _check_zip(os.path.join(out, "g_subs", "gru", "smoothed", "submission.zip"), data)
        assert "one sentence at a time" in open(os.path.join(out, "g_summary.txt")).read()


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
