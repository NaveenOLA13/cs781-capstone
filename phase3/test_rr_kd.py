"""Self-test of knowledge distillation (rr_kd.py and the kd_* settings of rr_train.py) on tiny stand-in
models and fake data. Needs torch and peft. About two minutes.   python test_rr_kd.py -> every line PASS

Checks: the teacher term is 0 when the student already matches the teacher; the student keeps only the
first blocks (also for a model that loops over a stored layer count) and the teacher's model is not
changed; teacher probabilities reach the right sentences (a student taught only by a gold-like teacher
learns the roles); bad teacher files are refused; the whole experiment (teacher out-of-fold, two
students, comparison, reuse of finished runs).
"""
import json
import os
import sys
import tempfile
import zipfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_data as D  # noqa: E402
import rr_kd as KD  # noqa: E402
from rr_train import kd_loss, run, teacher_targets  # noqa: E402
from test_rr_data import make_fake_data  # noqa: E402
from test_rr_peft import ToyLlama  # noqa: E402
from test_rr_pipeline import ToyGPT, WordTok, _bb  # noqa: E402

torch.manual_seed(0)
FAST = dict(epochs=2, b_epochs=2, batch_size=16, max_len=32, lr_head=3e-3, log_every=0)


class ToyLlamaCount(ToyLlama):
    """Loops over a stored layer count (`for i in range(self.n_layers)`), as some implementations do."""

    def __init__(self):
        super().__init__()
        self.n_layers = len(self.layers)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for i in range(self.n_layers):
            x = self.layers[i](x)
        return self.output(self.norm(x))


def _check_zip(path, data):
    test = {}
    for name in ("CL_test.json", "IT_test.json"):
        test.update(json.load(open(os.path.join(data, "Test", name))))
    with zipfile.ZipFile(path) as z:
        pred = json.loads(z.read("prediction.json"))
    assert set(pred) == set(test) and all(len(pred[k]) == len(test[k]) for k in test), path


# ------------------------------------------------------------------------------- tests
def test_kd_loss():
    lp = torch.log_softmax(torch.randn(5, 13), -1)
    assert kd_loss(lp, lp, 2.0).abs().item() < 1e-5                    # student = teacher -> nothing to learn
    z = torch.randn(5, 13, requires_grad=True)
    loss = kd_loss(z, lp, 2.0)
    loss.backward()
    assert loss.item() > 0 and z.grad.abs().sum() > 0
    q = torch.log_softmax(z.detach(), -1)                               # T = 1: plain KL(teacher || student)
    want = (lp.exp() * (lp - q)).sum(-1).mean().item()
    assert abs(kd_loss(z.detach(), lp, 1.0).item() - want) < 1e-5


def test_make_student():
    for model_cls, blocks, fields in ((ToyLlama, "layers", []), (ToyGPT, "h", []),
                                      (ToyLlamaCount, "layers", ["model.n_layers"])):
        bb = _bb(model_cls())
        sbb, info = KD.make_student(bb, WordTok(), keep=0.5, verbose=False)
        assert (info["blocks"], info["layers_teacher"], info["layers_student"]) == (blocks, 2, 1), info
        assert info["layer_count_fields"] == fields, info
        assert len(getattr(bb.model, blocks)) == 2 and len(getattr(sbb.model, blocks)) == 1
        assert info["params_student"] < info["params_teacher"]
        clf = sbb.classifier(13, pos_buckets=10)
        ids = torch.tensor([[5, 6, 7, 0], [8, 9, 10, 11]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        assert clf(ids, mask, pos=torch.tensor([0, 9])).shape == (2, 13)
    try:
        KD.make_student(_bb(ToyLlama()), WordTok(), keep=2, verbose=False)
        raise AssertionError("keeping every block should fail")
    except ValueError:
        pass


def test_teacher_reaches_the_right_sentences():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=6)
        labels = D.load_labels(data)
        train = [d for d in D.load_docs(data) if d["split"] == "train"]
        li = {l: i for i, l in enumerate(labels)}

        def gold_like(d):                                               # 97% on the gold role
            p = torch.full((len(d["labels"]), len(labels)), 0.0025)
            p[torch.arange(len(d["labels"])), torch.tensor([li[l] for l in d["labels"]])] = 0.97
            return p.log()

        good = {"labels": labels, "train": {d["id"]: gold_like(d) for d in train}}
        path = os.path.join(tmp, "teacher.pt")
        torch.save(good, path)
        assert [t.shape[0] for t in teacher_targets(path, train, labels)] == [len(d["labels"]) for d in train]
        for name, broken in (("shape", {**good, "train": {**good["train"], train[0]["id"]: torch.zeros(1, 13)}}),
                             ("missing", {**good, "train": {d["id"]: gold_like(d) for d in train[1:]}}),
                             ("labels", {**good, "labels": labels[::-1]})):
            bad = os.path.join(tmp, f"bad_{name}.pt")
            torch.save(broken, bad)
            try:
                teacher_targets(bad, train, labels)
                raise AssertionError(f"{name}: should fail")
            except ValueError:
                pass
        bb = _bb(ToyLlama())
        common = dict(data_root=data, out_dir=os.path.join(tmp, "runs"), method="full", epochs=3, b_epochs=2,
                      batch_size=16, max_len=32, lr_backbone=3e-3, lr_head=3e-3, log_every=0, save_model=False)
        plain = run(bb, run_name="plain", **common)["metrics"]
        taught = run(bb, run_name="taught", kd_teacher=path, kd_alpha=0.0, **common)["metrics"]   # teacher only
        a_plain = plain["stage_a"]["dev"]["all"]["macro_f1_all13"]
        a_taught = taught["stage_a"]["dev"]["all"]["macro_f1_all13"]
        print(f"      Stage A dev 13-role avg: gold labels {a_plain:.3f}, gold-like teacher only {a_taught:.3f}")
        assert a_taught > max(0.15, 0.6 * a_plain), (a_plain, a_taught)
        assert taught["stage_a"]["history"][0]["kd_loss"] >= 0 and "kd_loss" not in plain["stage_a"]["history"][0]
        assert taught["config"]["kd_teacher"] == path and plain["config"]["kd_teacher"] is None
        try:
            run(bb, run_name="x", kd_teacher=os.path.join(tmp, "nope.pt"), **common)
            raise AssertionError("a missing teacher file should fail")
        except ValueError as e:
            assert "not found" in str(e)


def test_run_kd_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=4)
        out = os.path.join(tmp, "runs")
        bb = _bb(ToyLlama())
        before = {k: v.clone() for k, v in bb.model.state_dict().items()}
        res = KD.run_kd(bb, WordTok(), prefix="k", folds=2, data_root=data, out_dir=out, **FAST)
        assert set(res) == {"teacher", "student", "student_kd"} and all(res.values()), res.keys()
        t = torch.load(os.path.join(out, "k_teacher", "teacher.pt"), map_location="cpu", weights_only=False)
        train = [d for d in D.load_docs(data) if d["split"] == "train"]
        assert set(t["train"]) == {d["id"] for d in train}
        for d in train:
            lp = t["train"][d["id"]]
            assert lp.shape == (len(d["labels"]), 13)
            assert torch.allclose(lp.exp().sum(-1), torch.ones(len(d["labels"])), atol=1e-4)
        assert sorted(sum(t["folds"], [])) == sorted(d["id"] for d in train)      # each judgment held out once
        assert len(t["dev"]) == 4 and len(t["test"]) == 4
        assert os.path.exists(os.path.join(out, "k_teacher", "teacher_fold2.pt"))
        _check_zip(os.path.join(out, "k_teacher", "stageB", "submission.zip"), data)
        for key in ("student", "student_kd"):
            m = res[key]
            _check_zip(os.path.join(out, f"k_{key}", m["best"], "submission.zip"), data)
            assert m["stage_a"]["total_params"] < res["teacher"]["stage_a"]["total_params"]
        assert res["student_kd"]["config"]["kd_teacher"].endswith("teacher.pt")
        assert res["student"]["config"]["kd_teacher"] is None
        text = open(os.path.join(out, "k_comparison.txt")).read()
        for s in ("KNOWLEDGE DISTILLATION SUMMARY", "student_kd", "effect of the teacher term", "Stage B, CL",
                  "average probability of the gold role"):
            assert s in text, s
        rows = json.load(open(os.path.join(out, "k_comparison.json")))
        assert rows["student_kd"]["kd"]["student"]["layers_student"] == 1
        for k, v in bb.model.state_dict().items():                               # the Phase 1 model is unchanged
            assert torch.equal(v.cpu(), before[k]), k
        again = KD.run_kd(bb, WordTok(), prefix="k", folds=2, data_root=data, out_dir=out, **FAST)  # reused
        assert (again["student_kd"]["stage_b"]["dev"]["all"]["macro_f1_all13"]
                == res["student_kd"]["stage_b"]["dev"]["all"]["macro_f1_all13"])


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