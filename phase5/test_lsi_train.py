"""Self-test of Phase 5 training (lsi_train.py) on tiny stand-in models and fake documents whose statutes are given
by keywords. Needs torch. About 2-4 minutes.     python test_lsi_train.py  -> every line PASS

Checks: chunking helpers; the asymmetric loss; LoRA and QLoRA (4-bit NF4 round trip, runs on a transformer and on
the GRU encoder-decoder); the finishing-time control (finish_by); full fine-tuning learns the fake task with max-pooling over chunks
and saves every chunk's vector and scores (chunks.pt) for Stage B; LoRMA with the asymmetric loss and a model whose
output changes with padding (same-length chunk groups) run end to end; submissions are valid; the teacher's
train_probs.json drives a half-size KD student; the ensemble combines the runs (and averages their train_probs).
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


def test_asl_loss():
    y = torch.tensor([[1., 0., 0.]])
    z = torch.tensor([[4., -4., 0.]], requires_grad=True)          # confident right, confident right, unsure
    loss = S.asl_loss(z, y, gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    loss.backward()
    g = z.grad[0]
    assert g[0] < 0 and abs(g[1]) < 1e-9 and g[2] > 0, g           # easy negative (p < clip): no gradient at all
    assert S.asl_loss(torch.tensor([[9., -9.]]), torch.tensor([[1., 0.]])) < 1e-3
    zb = torch.tensor([[0.5, 0.5]])
    assert S.asl_loss(zb, torch.tensor([[1., 0.]]), 0.0, 0.0, 0.0) > S.asl_loss(zb, torch.tensor([[1., 0.]]))


def test_runs_kd_ensemble():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_lsi(tmp, n_train=240, n_dev=60, n_test=20, n_words=(20, 70), at_end=True)
        out = os.path.join(tmp, "runs")
        common = dict(FAST, data_root=root, out_dir=out, cache_dir=os.path.join(tmp, "cache"))
        res = S.run(_bb("llama"), name="llama", technique="full", lr_body=1e-3, save_chunks=True, ckpt_path="toy",
                    **common)
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
        ch = torch.load(os.path.join(res["out_dir"], "chunks.pt"), weights_only=False)
        assert ch["labels"] == [3, 7, 11, 20, 31] and set(ch["splits"]) == {"train", "dev", "test"}
        for s_, n_ in (("train", 240), ("dev", 60), ("test", 20)):
            sp = ch["splits"][s_]
            assert len(sp["ids"]) == len(sp["feats"]) == len(sp["logits"]) == n_
            assert all(1 <= len(f) <= 6 and len(f) == len(z) and z.shape[1] == 5 for f, z in zip(sp["feats"],
                                                                                                 sp["logits"]))
        # the saved dev probabilities = maximum over the saved chunk scores
        dp = json.load(open(os.path.join(res["out_dir"], "dev_probs.json")))
        sp = ch["splits"]["dev"]
        diff = max(abs(torch.sigmoid(z.float().max(0).values) - torch.tensor(dp[i])).max().item()
                   for i, z in zip(sp["ids"], sp["logits"]))
        assert diff < 5e-3, diff
        pl = json.load(open(os.path.join(res["out_dir"], "per_label.json")))
        assert pl["labels"] == [3, 7, 11, 20, 31] and len(pl["thresholds"]) == 5
        # LoRMA (parametrised weights) and a padding-sensitive encoder-decoder (same-length chunk groups)
        r2 = S.run(_bb("llama"), name="llama", technique="lorma", epochs=1, loss="asl",
                   **{k: v for k, v in common.items() if k != "epochs"})
        assert r2["metrics"]["trainable_share"] < 0.5 and r2["metrics"]["loss"] == "asl"
        assert "asymmetric" in open(os.path.join(r2["out_dir"], "log.txt")).read()
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
        assert ens["train_probs"] == (set(ens["members"]) <= {"llama_full"})   # only llama_full saved train probs
        assert G.compare_all(out).count("\n") >= 5


def test_lora_qlora():
    import cj_tech as T
    torch.manual_seed(0)
    W = torch.randn(70, 33)                         # 2,310 values: not a multiple of the block (64) -> padding
    packed, absmax = T.nf4_quantize(W, 64)
    lv = torch.tensor(T.NF4)
    back = T.nf4_dequantize(packed, absmax, lv, W.shape, 64)
    assert back.shape == W.shape and packed.dtype == torch.uint8 and packed.numel() == 37 * 64 // 2
    rel = ((back - W).norm() / W.norm()).item()
    assert rel < 0.2, rel
    exact = lv[torch.randint(0, 16, (4, 64))] * 3.0     # NF4 values x one scale (3 = the block maximum): no error
    exact[:, 0] = 3.0
    p2, a2 = T.nf4_quantize(exact, 64)
    assert torch.allclose(T.nf4_dequantize(p2, a2, lv, exact.shape, 64), exact, atol=1e-5)
    lo = T.LoRA(5, 4, 2, 4.0)
    w = torch.randn(5, 4)
    assert torch.equal(lo(w), w)                     # B starts at 0: no change at the start
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_lsi(tmp, n_train=120, n_dev=30, n_test=10, n_words=(20, 70), at_end=True)
        common = dict(FAST, data_root=root, out_dir=os.path.join(tmp, "runs"), cache_dir=os.path.join(tmp, "cache"),
                      epochs=1)
        for kind, tech in (("llama", "lora"), ("llama", "qlora"), ("encdec", "qlora")):
            r = S.run(_bb(kind), name=kind, technique=tech, **common)
            m = r["metrics"]
            assert m["trainable_params"] < m["total_params"], (kind, tech)
            assert os.path.exists(os.path.join(r["out_dir"], "submission", "submission.zip"))
            if tech == "qlora":
                info = m["technique_info"]
                assert info["quantized_params"] > 0 and info["rel_error_mean"] < 0.2, info
                assert info["frozen_mb_4bit"] < info["frozen_mb_full"] / 4, info
                assert "4-bit NF4" in open(os.path.join(r["out_dir"], "log.txt")).read()


def test_deadline():
    import time
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_lsi(tmp, n_train=400, n_dev=40, n_test=10, n_words=(20, 70), at_end=True)
        common = dict(FAST, data_root=root, out_dir=os.path.join(tmp, "runs"), cache_dir=os.path.join(tmp, "cache"),
                      batch_size=4)
        common.pop("epochs")
        # deadline already passed: only the 20 probe steps, then every result is still written
        r = S.run(_bb("llama"), name="llama", technique="full", run_name="cut", epochs=3, finish_by=time.time() + 1,
                  **common)
        b = r["metrics"]["budget"]
        per_epoch = -(-r["metrics"]["data"]["train"] // 4)
        assert b["planned_steps"] == 3 * per_epoch and b["steps"] == 20 and b["cut"], b
        assert len(r["metrics"]["history"]) == 1 and "[budget] trained 20 of" in open(
            os.path.join(r["out_dir"], "log.txt")).read()
        assert os.path.exists(os.path.join(r["out_dir"], "submission", "submission.zip"))
        # plenty of time: every planned step
        r2 = S.run(_bb("llama"), name="llama", technique="lorma", run_name="free", epochs=1,
                   finish_by=time.time() + 3600, **common)
        b2 = r2["metrics"]["budget"]
        assert b2["steps"] == b2["planned_steps"] == per_epoch and not b2["cut"], b2


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
