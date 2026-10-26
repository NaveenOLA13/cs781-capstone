"""Self-test of Phase 5 training (lsi_train.py) on tiny stand-in models and fake documents whose statutes are given
by keywords. Needs torch. About 2-4 minutes.     python test_lsi_train.py  -> every line PASS

Checks: chunking helpers; full fine-tuning learns the fake task with max-pooling over chunks; LoRMA and a model whose
output changes with padding (same-length chunk groups) run end to end; submissions are valid; the teacher's
train_probs.json drives a half-size KD student; the ensemble combines the runs.
"""
import json
import os
import sys
import tempfile
import zipfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsi_grid as G  # noqa: E402
import lsi_train as S  # noqa: E402
from test_cj_train import _bb  # noqa: E402
from test_lsi_data import make_fake_lsi  # noqa: E402

FAST = dict(max_len=16, doc_tokens=96, train_chunks=3, eval_chunks=6, epochs=4, batch_size=8, eval_batch_size=24,
            log_every=0, fit_sample=20, tok_workers=1, fetch_data=False, lr_head=3e-3)


def test_helpers():
    assert S.doc_chunks(list(range(70)), 32) == [list(range(32)), list(range(32, 64)), list(range(38, 70))]
    assert S.doc_chunks(list(range(84)), 32) == [list(range(32)), list(range(32, 64)), list(range(64, 84))]
    assert S.doc_chunks(list(range(10)), 32) == [list(range(10))] and S.doc_chunks([], 32) == [[]]
    assert S.spread(3, 8) == [0, 1, 2] and S.spread(10, 4) == [0, 3, 6, 9]
    assert S.doc_batches([3, 3, 3, 1], 6) == [[0, 1], [2, 3]] and S.doc_batches([9, 1], 4) == [[0], [1]]
    z = torch.tensor([[0., 1.], [2., -1.], [5., 5.]])
    assert S.aggregate(z, [2, 1], "max").tolist() == [[2., 1.], [5., 5.]]
    assert S.aggregate(z, [2, 1], "mean").tolist() == [[1., 0.], [5., 5.]]


def test_runs_kd_ensemble():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_lsi(tmp, n_train=240, n_dev=60, n_test=20, n_words=(20, 70), at_end=True)
        out = os.path.join(tmp, "runs")
        common = dict(FAST, data_root=root, out_dir=out, cache_dir=os.path.join(tmp, "cache"))
        res = S.run(_bb("llama"), name="llama", technique="full", lr_body=1e-3, save_train_probs=True, **common)
        m = res["metrics"]
        log = open(os.path.join(res["out_dir"], "log.txt")).read()
        print(f"    llama full: dev macro-F1 {m['dev_global']['macro_f1']:.3f} (global), cross-fitted "
              f"{max(m['cv_global'], m['cv_per_label']):.3f}")
        assert m["dev_global"]["macro_f1"] > 0.55, m["dev_global"]
        assert m["statutes"] == 5 and "[train] using epoch" in log and "chunks of" in log
        for sub in ("submission", "submission_global"):
            rows = json.loads(zipfile.ZipFile(os.path.join(res["out_dir"], sub, "submission.zip"))
                              .read("prediction.json"))
            assert len(rows) == 20 and all(r["pred"] and set(r["pred"]) <= {3, 7, 11, 20, 31} for r in rows)
        tp = json.load(open(os.path.join(res["out_dir"], "train_probs.json")))
        assert len(tp) == 240 and all(len(v) == 5 for v in tp.values())
        pl = json.load(open(os.path.join(res["out_dir"], "per_label.json")))
        assert pl["labels"] == [3, 7, 11, 20, 31] and len(pl["thresholds"]) == 5
        # LoRMA (parametrised weights) and a padding-sensitive encoder-decoder (same-length chunk groups)
        r2 = S.run(_bb("llama"), name="llama", technique="lorma", epochs=1, **{k: v for k, v in common.items()
                                                                            if k != "epochs"})
        assert r2["metrics"]["trainable_share"] < 0.5
        r3 = S.run(_bb("encdec"), name="encdec", technique="bitfit", epochs=1, **{k: v for k, v in common.items()
                                                                               if k != "epochs"})
        assert "same-length chunk groups" in open(os.path.join(r3["out_dir"], "log.txt")).read()
        # knowledge distillation: half-size student with the teacher's training probabilities
        r4 = S.run(_bb("llama"), name="llama", technique="full", run_name="llama_student_kd", student_layers=0.5,
                   kd_teacher=os.path.join(res["out_dir"], "train_probs.json"), lr_body=1e-3, **common)
        m4 = r4["metrics"]
        assert m4["student"]["layers_student"] == 2 and m4["total_params"] < m["total_params"]
        assert m4["kd"]["statutes_shared"] == 5 and all(h["teacher_term"] is not None for h in m4["history"])
        # comparison table and ensemble over the llama runs
        table = S.comparison([m, r2["metrics"], m4])
        assert "llama_student_kd" in table and "llama_full" in table
        ens = G.ensemble([out], out_dir=os.path.join(tmp, "ens"), data_root=root, fetch=False, log=lambda *a: None)
        assert ens["members"] and os.path.exists(ens["submission"])
        assert G.compare_all(out).count("\n") >= 5


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
