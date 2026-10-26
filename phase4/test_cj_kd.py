"""Self-test of knowledge distillation (cj_kd.py + the KD parts of cj_train.py) on tiny stand-in models.
Needs torch. About 1-2 minutes.     python test_cj_kd.py  -> every line PASS

Checks: the teacher term is zero when the student already matches the teacher and pulls the student towards it;
cutting blocks makes a smaller model that still runs and leaves the original untouched; the whole chain teacher SFT
-> whole-case teacher (train_probs.json) -> students with and without the teacher; the student run can be rebuilt
(for whole-case reading) and the comparison table is written.
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cj_hier as H  # noqa: E402
import cj_kd as K  # noqa: E402
import cj_train as R  # noqa: E402
from test_cj_data import make_fake_data  # noqa: E402
from test_cj_train import DEV, FAST, _bb  # noqa: E402

torch.manual_seed(0)


def test_kd_loss():
    p = torch.tensor([0.9, 0.2, 0.5])
    z = torch.log(p) - torch.log1p(-p)
    match = torch.stack([torch.zeros(3), z], -1)                 # student score difference = teacher's
    assert abs(R.kd_loss(match, p, 2.0).item()) < 1e-5
    assert abs(R.kd_loss(match, p, 1.0).item()) < 1e-5
    logits = torch.zeros(3, 2, requires_grad=True)
    loss = R.kd_loss(logits, p, 2.0)
    assert loss.item() > 0
    loss.backward()
    g = logits.grad[:, 1] - logits.grad[:, 0]                    # gradient on the score difference
    assert g[0] < 0 and g[1] > 0 and abs(g[2]) < 1e-6, g         # pushed up for p = 0.9, down for p = 0.2


def test_cut_blocks():
    for kind in ("llama", "gpt"):
        bb = _bb(kind)
        n_before = sum(p.numel() for p in bb.model.parameters())
        model, info = R.make_model(bb, dict(student_layers=0.5), log=lambda *a: None)
        assert (info["layers_full"], info["layers_student"]) == (3, 2), info
        assert info["params_student"] < info["params_full"] == n_before
        assert sum(p.numel() for p in bb.model.parameters()) == n_before    # the original is untouched
        clf = bb.classifier(2, model=model).to(DEV).float().eval()
        ids, mask = R.collate([[5, 6, 7, 8], [9, 10]], bb.pad_id, DEV)
        with torch.no_grad():
            assert clf(ids, mask).shape == (2, 2)
        one, info1 = R.make_model(bb, dict(student_layers=1), log=lambda *a: None)
        assert info1["layers_student"] == 1
        same, none = R.make_model(bb, dict(student_layers=0), log=lambda *a: None)
        assert none is None and sum(p.numel() for p in same.parameters()) == n_before
        try:
            R.make_model(bb, dict(student_layers=3), log=lambda *a: None)
            raise AssertionError("keeping every block is not a student")
        except ValueError:
            pass


def test_teacher_to_students():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_data(tmp, n_train=40, n_dev=12, n_test=10, n_words=(40, 160))
        common = dict(FAST, data_root=root, out_dir=os.path.join(tmp, "runs"), cache_dir=os.path.join(tmp, "cache"),
                      max_len=24, chunk_stride=16, train_chunks=3)
        teacher_run = R.run(_bb("llama"), name="llama", technique="full", **common)["out_dir"]
        H.run_hier(teacher_run, versions=("mean",), bb=_bb("llama"), k=3, epochs=2, hidden=32, batch=8,
                   log=lambda *a: None)
        hier_dir = os.path.join(teacher_run, "hier_k3")
        p = K.plan(hier_dir, out_dir=os.path.join(tmp, "kd"), ckpt="toy", tok_workers=1, log_every=0,
                   eval_batch_size=16, fit_sample=20, lr_head=3e-3, batch_size=8)
        f1 = {v: json.load(open(os.path.join(hier_dir, v, "metrics.json")))["dev_tuned"]["macro_f1"]
              for v in ("last", "vote", "mean")}
        chosen = os.path.basename(os.path.dirname(p["teacher_file"]))
        assert f1[chosen] == max(f1.values()), (chosen, f1)       # the best on dev (ties: any of them)
        assert p["sft_dir"] == teacher_run and p["model"] == "llama"
        assert [j["run_name"] for j in p["jobs"]] == ["llama_student_kd", "llama_student"]
        s_kd = p["jobs"][0]["settings"]
        assert s_kd["train_chunks"] == 3 and s_kd["max_len"] == 24 and s_kd["student_layers"] == 0.5
        assert s_kd["kd_teacher"] == p["teacher_file"] and "kd_teacher" not in p["jobs"][1]["settings"]
        teacher_m = json.load(open(os.path.join(teacher_run, "metrics.json")))
        for job in p["jobs"]:
            res = R.run(_bb("llama"), name="llama", technique="full", **job["settings"])
            m = res["metrics"]
            assert m["student"]["layers_student"] == 2 and m["total_params"] < teacher_m["total_params"]
            log = open(os.path.join(res["out_dir"], "log.txt")).read()
            if job["run_name"].endswith("_kd"):
                assert m["kd"]["cases"] == 40 and m["kd"]["cases_missing"] == 0, m["kd"]
                assert "[kd] teacher" in log and all(h["teacher_term"] is not None for h in m["history"])
            else:
                assert m["kd"] is None and "[kd]" not in log
            assert os.path.exists(os.path.join(res["out_dir"], "submission_tuned", "submission.zip"))
        # a student run can be rebuilt for whole-case reading: 'last' reproduces its own dev predictions
        st = os.path.join(tmp, "kd", "llama_student_kd")
        H.run_hier(st, versions=(), bb=_bb("llama"), k=3, log=lambda *a: None)
        run_dev = json.load(open(os.path.join(st, "dev_probs.json")))
        last = json.load(open(os.path.join(st, "hier_k3", "last", "dev_probs.json")))
        assert max(abs(run_dev[i] - last[i]) for i in run_dev) < 2e-3
        text = K.summary(p)
        assert "llama_student_kd" in text and "llama_student " in text and "best student on dev" in text
        assert os.path.exists(os.path.join(tmp, "kd", "kd.txt"))


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
