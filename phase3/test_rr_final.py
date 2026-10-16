"""Self-test of the final-submission code (rr_final.py and the train_on_dev option of rr_kd.teacher_oof)
on a tiny stand-in model and fake data. Needs torch and peft. About a minute.
    python test_rr_final.py   -> every line PASS

Checks: every train and dev judgment is predicted exactly once by a copy that did not see it, both
submissions (plain average and smoothed) are complete, the smoothing weight comes from the given list,
the summary is written, and a finished ensemble is reused.
"""
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


def test_run_final():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=4)
        out = os.path.join(tmp, "runs")
        bb = _bb(ToyLlama())
        res = run_final(bb, prefix="f", folds=2, data_root=data, out_dir=out, weights=(0, 0.5, 1.0), **FAST)
        t = torch.load(os.path.join(out, "f_ensemble", "teacher.pt"), map_location="cpu", weights_only=False)
        labelled = [d for d in D.load_docs(data) if d["split"] in ("train", "dev")]
        ids = sorted(d["id"] for d in labelled)
        assert sorted(t["train"]) == ids and sorted(sum(t["folds"], [])) == ids    # train + dev, each once
        assert t["dev"] == {} and t["train_on_dev"] is True and len(t["test"]) == 4
        for d in labelled:
            assert t["train"][d["id"]].shape == (len(d["labels"]), 13)
        _check_zip(os.path.join(out, "f_ensemble", "stageB", "submission.zip"), data)
        _check_zip(os.path.join(out, "f_smoothed", "submission.zip"), data)
        sm = json.load(open(os.path.join(out, "f_smoothed", "smoothing.json")))
        assert sm["chosen"] in (0, 0.5, 1.0) and set(sm["weights"]) == {"0", "0.5", "1.0"}
        m = res["metrics"]
        assert m["train_on_dev"] and m["scored_on"].startswith("out-of-fold") and m["copies"] == 2
        assert m["folds"][0]["fit_judgments"] == 6                                   # 12 labelled, 2 groups
        assert set(m["stage_b"]["dev"]) == {"all", "CL", "IT"}
        assert os.path.exists(os.path.join(out, "f_ensemble", "ensemble_fold2.pt"))
        text = open(os.path.join(out, "f_summary.txt")).read()
        assert "FINAL ENSEMBLE" in text and "smoothed" in text
        again = run_final(bb, prefix="f", folds=2, data_root=data, out_dir=out, weights=(0, 0.5, 1.0), **FAST)
        assert again["smoothing"]["chosen"] == res["smoothing"]["chosen"]
        print(f"      chosen smoothing weight {sm['chosen']} (fake data)")


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