"""Self-test of the whole-document stage (lsi_hier.py), knowledge distillation from a Stage B teacher (lsi_grid.plan_kd
+ lsi_train) and the report, on tiny stand-in models. Needs torch. About 2 minutes.
python test_lsi_hier.py  -> every line PASS

Checks: every Stage B version gives the same scores for a document alone and inside a padded batch; the "_res"
version starts exactly as the run (max over chunk scores); run_stage_b trains all versions from chunks.pt and writes
probabilities (also for training documents), metrics, submissions and hier.txt; a Stage B version and an ensemble can
be KD teachers (the student starts from the run's checkpoint with the run's settings); report() lists everything.
"""
import json
import os
import sys
import tempfile
import zipfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsi_grid as G  # noqa: E402
import lsi_hier as H  # noqa: E402
import lsi_train as S  # noqa: E402
from test_cj_train import _bb  # noqa: E402
from test_lsi_data import make_fake_lsi  # noqa: E402
from test_lsi_train import FAST  # noqa: E402


def test_stage_b_padding_and_start():
    torch.manual_seed(0)
    d, n_lab = 12, 5
    a_f, a_z = torch.randn(1, 2, d), torch.randn(1, 2, n_lab)
    b_f, b_z = torch.randn(1, 4, d), torch.randn(1, 4, n_lab)
    F_ = torch.zeros(2, 4, d)
    Z = torch.zeros(2, 4, n_lab)
    F_[0, :2], F_[1] = a_f[0], b_f[0]
    Z[0, :2], Z[1] = a_z[0], b_z[0]
    M = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    for v in H.VERSIONS:
        m = H.StageB(d, n_lab, v, hidden=16, dropout=0.0, prior_logit=torch.zeros(n_lab)).eval()
        with torch.no_grad():
            alone = m(a_f, a_z, torch.ones(1, 2, dtype=torch.bool))
            batched = m(F_, Z, M)[:1]
        assert torch.allclose(alone, batched, atol=1e-5), (v, alone, batched)
        if v.endswith("_res"):                                     # starts as the run: max over chunk scores
            assert torch.allclose(m(F_, Z, M), torch.stack([a_z[0].max(0).values, b_z[0].max(0).values]),
                                  atol=1e-6)
    # chunk dropout keeps at least one chunk and never mixes documents
    import random
    feats = [torch.randn(3, d).half(), torch.randn(1, d).half()]
    logits = [torch.randn(3, n_lab).half(), torch.randn(1, n_lab).half()]
    F2, Z2, M2 = H.make_batch(feats, logits, [0, 1], "cpu", p_drop=0.9, rng=random.Random(0))
    assert M2.sum(1).min() >= 1 and M2[1].sum() == 1


def test_stage_b_kd_report():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_lsi(tmp, n_train=200, n_dev=60, n_test=20, n_words=(20, 70), at_end=True)
        out = os.path.join(tmp, "runs")
        common = dict(FAST, data_root=root, out_dir=out, cache_dir=os.path.join(tmp, "cache"), epochs=3)
        run = S.run(_bb("llama"), name="llama", technique="full", lr_body=1e-3, save_chunks=True, ckpt_path="toy",
                    **common)["out_dir"]
        res = H.run_stage_b(run, epochs=4, hidden=32, batch=16, patience=4, log=lambda *a: None)
        assert set(res) == set(H.VERSIONS)
        for v in H.VERSIONS:
            folder = os.path.join(run, "stageB", v)
            m = json.load(open(os.path.join(folder, "metrics.json")))
            assert {"cv_global", "cv_per_label", "dev_chosen", "best_epoch", "params", "history"} <= set(m)
            assert json.load(open(os.path.join(folder, "labels.json"))) == [3, 7, 11, 20, 31]
            assert len(json.load(open(os.path.join(folder, "train_probs.json")))) == 200
            rows = json.loads(zipfile.ZipFile(os.path.join(folder, "submission", "submission.zip"))
                              .read("prediction.json"))
            assert len(rows) == 20 and all(r["pred"] for r in rows)
        res_m = json.load(open(os.path.join(run, "stageB", "gru_attn_res", "metrics.json")))
        run_m = json.load(open(os.path.join(run, "metrics.json")))
        assert abs(res_m["history"][0]["dev_macro_f1"] - run_m["dev_global"]["macro_f1"]) < 0.05   # epoch 0 = run
        assert "best on dev" in open(os.path.join(run, "stageB", "hier.txt")).read()
        # knowledge distillation with a Stage B teacher: the student starts from the run's checkpoint and settings
        teacher = os.path.join(run, "stageB", "gru_attn")
        p = G.plan_kd(teacher, out_dir=out)
        assert p["sft_dir"] == run and p["ckpt"] == "toy" and p["model"] == "llama"
        assert [j["run_name"] for j in p["jobs"]] == ["llama_student_kd", "llama_student"]
        s_kd = p["jobs"][0]["settings"]
        assert s_kd["kd_teacher"] == os.path.join(teacher, "train_probs.json") and s_kd["student_layers"] == 0.5
        assert s_kd["eval_chunks"] == 6 and s_kd["save_chunks"] is False
        assert "kd_teacher" not in p["jobs"][1]["settings"]
        st = S.run(_bb("llama"), name="llama", technique="full", run_name="llama_student_kd",
                   **{k: v for k, v in s_kd.items() if k not in ("data_root", "epochs")}, data_root=root, epochs=1)
        assert st["metrics"]["kd"]["statutes_shared"] == 5 and st["metrics"]["student"]["layers_student"] == 2
        # an ensemble as the teacher (needs sft_run for the student's checkpoint and settings)
        ens = G.ensemble([out], out_dir=os.path.join(tmp, "ens"), data_root=root, fetch=False, log=lambda *a: None)
        if ens["train_probs"]:
            pe = G.plan_kd(os.path.join(tmp, "ens"), out_dir=out, sft_run=run)
            assert pe["jobs"][0]["settings"]["kd_teacher"] == os.path.join(tmp, "ens", "train_probs.json")
        text = G.report([out], out_path=os.path.join(tmp, "all.txt"))
        assert "Stage B" in text and "fine-tuned" in text and "KD student" in text and "best:" in text


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
