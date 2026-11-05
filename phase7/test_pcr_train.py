"""Self-test of Phase 7 training on tiny stand-in models (decoder-only with a tied head, Llama-style, encoder-decoder
with a bidirectional GRU encoder) and fake case data. Needs torch. About 3-6 minutes.
python test_pcr_train.py  -> every line PASS

Checks: chunking; the case similarities against slow direct formulas; the ranking and KD losses; (IA)^3 (no change at
the start, gradients reach every vector, a change once trained) on all three toys; LoRMA on GRU weights with cuDNN
kept equals cuDNN switched off (GPU only); half-size students (blocks / RNN layers); full, LoRMA, QLoRA, (IA)^3,
partial runs and loss / negative / pooling variants write scores, a valid fused submission and metrics; a KD
student learns from a finished teacher's case vectors (listwise and Margin-MSE); finish_by cuts training and still
writes everything; zero-shot (no training); ensemble, best_run, report; mid-training texts.
"""
import copy
import json
import math
import os
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "phase4"))
import p7_paths  # noqa: E402,F401
import cj_tech as T  # noqa: E402
import pcr_data as PD  # noqa: E402
import pcr_grid as G  # noqa: E402
import pcr_model as PM  # noqa: E402
import pcr_train as S  # noqa: E402
from cj_models import make_backbone  # noqa: E402
from test_cj_train import TOYS, WordTok  # noqa: E402
from test_pcr_data import make_fake_pcr  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAST = dict(max_len=64, chunk_len=64, dim=16, batch_queries=4, epochs=2, q_chunks=2, d_chunks=2, max_chunks=6,
            val_chunks=3, val_every=1, enc_batch=8, eval_batch=16, log_every=5, fetch_data=False, tok_workers=1,
            hard_negs=2, neg_top=10, top=30, dense_top=10, lr_head=3e-3)
QUIET = dict(log=lambda *a, **k: None)


class ToyEncDecE(TOYS["encdec"]):
    """The toy encoder-decoder with model.encode(src), as our Phase 1 encoder-decoder has."""

    def encode(self, src):
        return self.encoder(self.emb(src))[0]


def bb_of(kind):
    torch.manual_seed(1)
    model = ToyEncDecE() if kind == "encdec_e" else TOYS[kind]()
    bb = make_backbone(model, WordTok(), max_len=64, verbose=False)
    bb.model.eval()                 # as cj_models.load_model returns the real checkpoints
    return bb


def test_chunks():
    assert PM.chunk_starts(50, 64, 8) == [0]
    s = PM.chunk_starts(300, 64, 8)
    assert len(s) == 5 and s[0] == 0 and s[-1] == 300 - 64                  # covers the case, ends at its end
    assert all(b - a <= 64 for a, b in zip(s, s[1:]))
    s = PM.chunk_starts(10_000, 64, 8)
    assert len(s) == 8 and s[0] == 0 and s[-1] == 10_000 - 64                # spread over a long case
    cs = PM.doc_chunks(np.arange(300), 64, 8)
    assert all(len(c) == 64 for c in cs) and cs[-1][-1] == 299
    g = PM._groups([5, 3, 5, 5, 3], 2, exact=True)
    assert all(len({[5, 3, 5, 5, 3][i] for i in x}) == 1 for x in g) and sum(map(len, g)) == 5


def test_aggregate():
    torch.manual_seed(0)
    Q = F.normalize(torch.randn(3, 4, 8), dim=-1)
    P = F.normalize(torch.randn(5, 3, 8), dim=-1)
    Qn, Pn = torch.tensor([4, 2, 1]), torch.tensor([3, 1, 2, 3, 2])
    for i, n in enumerate(Qn.tolist()):
        Q[i, n:] = 0
    for j, n in enumerate(Pn.tolist()):
        P[j, n:] = 0
    out = PM.aggregate(Q.half(), Qn, P.half(), Pn, PM.AGGS, torch.device("cpu"), block=2)
    for i in range(3):
        for j in range(5):
            q, p = Q[i, :Qn[i]].half().float(), P[j, :Pn[j]].half().float()
            Sm = q @ p.T
            exp = dict(mean=float(F.normalize(q.sum(0), dim=0) @ F.normalize(p.sum(0), dim=0)),
                       maxsim=float(Sm.max(1).values.mean()),
                       maxsim_sym=float((Sm.max(1).values.mean() + Sm.max(0).values.mean()) / 2),
                       max=float(Sm.max()), first=float(q[0] @ p[0]))
            for a, v in exp.items():
                assert abs(out[a][i, j] - v) < 1e-4, (a, i, j, out[a][i, j], v)
    mv = PM.mean_vectors(Q.half(), Qn)
    assert torch.allclose(mv @ PM.mean_vectors(P.half(), Pn).T, torch.tensor(out["mean"]), atol=1e-4)


def test_losses():
    Sm = torch.tensor([[0.9, 0.1, 0.5, 0.2], [0.3, 0.8, 0.2, 0.7]])
    tgt = torch.tensor([0, 1])
    rel = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)
    t = 0.1
    r0 = -math.log(math.exp(9) / (math.exp(9) + math.exp(1) + math.exp(2)))
    r1 = -math.log(math.exp(8) / sum(math.exp(x / t) for x in (0.3, 0.8, 0.2, 0.7)))
    assert abs(PM.ranking_loss(Sm, tgt, rel, "infonce", t).item() - (r0 + r1) / 2) < 1e-4
    l0 = -(math.log(math.exp(9) + math.exp(5)) - math.log(sum(math.exp(x / t) for x in (0.9, 0.1, 0.5, 0.2))))
    assert abs(PM.ranking_loss(Sm, tgt, rel, "listwise", t).item() - (l0 + r1) / 2) < 1e-4
    assert abs(PM.ranking_loss(Sm, tgt, rel, "pairwise", t, margin=0.2).item() - 0.1 / 5) < 1e-5
    assert PM.kd_loss(Sm, Sm.clone(), "listwise", 0.05, 0.05, 1.0, tgt).abs().item() < 1e-6
    assert PM.kd_loss(Sm, Sm.clone(), "margin_mse", 0.05, 0.05, 1.0, tgt, ~rel).item() < 1e-9
    assert PM.kd_loss(Sm, Sm.flip(1), "listwise", 0.05, 0.05, 1.0, tgt).item() > 0.1


def test_ia3():
    ids = torch.randint(3, 600, (2, 20), device=DEV)
    mask = torch.ones_like(ids)
    expect = {"gpt": 12, "llama": 21, "encdec": 1, "encdec_e": 1}
    for kind, n_vec in expect.items():
        bb = bb_of(kind)
        clf = bb.classifier(16, model=copy.deepcopy(bb.model)).to(DEV)
        clf.eval()
        with torch.no_grad():
            before = clf(ids, mask)
        used = PM.used_params(clf, ids, DEV)
        hooks = T.apply(clf, "ia3", used, dict(T.TECH_DEFAULTS), 1, **QUIET)
        assert len(clf.ia3) == n_vec, (kind, len(clf.ia3))
        clf.eval()
        with torch.no_grad():
            assert torch.allclose(clf(ids, mask), before, atol=1e-5), kind           # starts as no change
        clf.train()
        clf(ids, mask).pow(2).sum().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in hooks.body_params), kind
        assert not any(p.requires_grad for p in clf.encoder.model.parameters()), kind
        with torch.no_grad():
            for v in clf.ia3:
                v.l.mul_(1.5)
        clf.eval()
        with torch.no_grad():
            assert not torch.allclose(clf(ids, mask), before, atol=1e-4), kind


def test_rnn_cudnn_kept():
    if not torch.cuda.is_available():
        print("    (no GPU: cuDNN test skipped)")
        return
    ids = torch.randint(3, 600, (3, 40), device="cuda")
    res = []
    for keep in (False, True):
        bb = bb_of("encdec_e")
        clf = bb.classifier(16, model=copy.deepcopy(bb.model)).cuda()
        used = PM.used_params(clf, ids, "cuda")
        T.apply(clf, "lorma", used, dict(T.TECH_DEFAULTS, rnn_cudnn=keep), 1, **QUIET)
        torch.manual_seed(5)
        with torch.no_grad():
            for n, p in sorted(clf.named_parameters()):
                if p.requires_grad and n.endswith(".B"):
                    p.copy_(torch.randn_like(p) * 0.1)
        clf.train()
        clf.drop.p = 0.0
        out = clf(ids, torch.ones_like(ids))
        out.pow(2).sum().backward()
        grads = {n: p.grad.detach().clone() for n, p in clf.named_parameters() if p.grad is not None}
        res.append((out.detach(), grads))
    (o1, g1), (o2, g2) = res
    assert torch.allclose(o1, o2, atol=1e-4), (o1 - o2).abs().max()
    assert set(g1) == set(g2) and any(".parametrizations." in n for n in g1)
    for n in g1:
        assert torch.allclose(g1[n], g2[n], atol=1e-3, rtol=1e-3), (n, (g1[n] - g2[n]).abs().max())


def test_students():
    ids = torch.randint(3, 600, (2, 20))
    for kind, kept, n_full in (("llama", [0, 2], 3), ("encdec", [0], 2), ("encdec_e", [0], 2)):
        bb = bb_of(kind)
        model = copy.deepcopy(bb.model)
        clf = bb.classifier(16, model=model)
        used = PM.used_params(clf, ids, torch.device("cpu"))
        info = PM.make_student(model, 0.5, used, **QUIET)
        assert info["kept"] == kept and info["layers_full"] == n_full, info
        assert info["params_student"] < info["params_full"]
        clf2 = bb.classifier(16, model=model)
        clf2.eval()
        assert clf2(ids, torch.ones_like(ids)).shape == (2, 16)


def _check(res, tech, n_test=20):
    m, out = res["metrics"], res["out_dir"]
    assert {"dev_dense", "best_agg", "dev_fused", "val_history", "submission", "dense_submission"} <= set(m)
    assert set(m["dev_dense"]) == set(PM.AGGS) and m["dev_fused"] is not None, m["fusion"]
    for s, nq, npool in (("dev", 20, 80), ("test", n_test, 80)):
        z = np.load(os.path.join(out, f"scores_{s}.npz"))
        assert set(z.files) == set(PM.AGGS) and z["mean"].shape == (nq, npool)
    for sub in (m["submission"], m["dense_submission"]):
        pred = json.loads(zipfile.ZipFile(sub).read("prediction.json"))
        assert len(pred) == n_test and all(len(v) == 7 and len(set(v)) == 7 for v in pred.values())
    if tech not in ("full",):
        assert m["trainable_share"] < 0.6, (tech, m["trainable_share"])
    h = [x["loss"] for x in m["history"]]
    assert all(math.isfinite(x) for x in h), h


def test_runs_kd_deadline_report():
    with tempfile.TemporaryDirectory() as tmp:
        root, _ = make_fake_pcr(tmp)
        out = os.path.join(tmp, "runs")
        common = dict(FAST, data_root=root, out_dir=out, cache_dir=os.path.join(tmp, "cache"), ckpt_path="toy")
        r = S.run(bb_of("llama"), name="llama", technique="full", save_train=True, lr_body=3e-3,
                  **dict(common, epochs=4))
        _check(r, "full")
        m = r["metrics"]
        assert os.path.exists(os.path.join(r["out_dir"], "teacher_vectors.pt")) and m["train_dense"] is not None
        h = m["history"]
        assert min(x["loss"] for x in h[2:]) < h[0]["loss"], h
        print(f"    toy llama full: dev alone {m['dev_dense']['mean']['f1']:.3f} (start "
              f"{m['val_history'][0]['f1']:.3f}), fused {m['dev_fused']['f1']:.3f}, lexical {m['dev_lexical']['f1']:.3f}")
        for kind, tech, extra in (("gpt", "lorma", {}), ("llama", "qlora", {}), ("gpt", "ia3", {}),
                                  ("encdec", "ia3", {}), ("encdec_e", "lorma", {}), ("llama", "partial", {}),
                                  ("gpt", "lorma", dict(run_name="gpt_lorma_listwise", loss="listwise")),
                                  ("gpt", "lorma", dict(run_name="gpt_lorma_pairwise", loss="pairwise",
                                                        neg_source="random", pooling="last"))):
            rr = S.run(bb_of(kind), name=kind, technique=tech, **dict(common, **extra))
            _check(rr, tech)
            if kind == "encdec_e":
                text = open(os.path.join(rr["out_dir"], "log.txt")).read()
                assert "cuDNN kept" in text and "RNN)" in text, text[-500:]
        # knowledge distillation: half-size student with the llama_full teacher (listwise, then Margin-MSE)
        st = S.run(bb_of("llama"), name="llama", technique="full", run_name="llama_student_kd", student_layers=0.5,
                   kd_teacher=r["out_dir"], **common)
        ms = st["metrics"]
        assert ms["student"]["kept"] == [0, 2] and ms["total_params"] < m["total_params"], ms["student"]
        assert ms["kd"]["teacher_technique"] == "full" and all(x["kd"] is not None for x in ms["history"])
        _check(st, "full")
        mm = S.run(bb_of("llama"), name="llama", technique="full", run_name="llama_student_mm", student_layers=0.5,
                   kd_teacher=r["out_dir"], kd_loss="margin_mse", **common)
        assert mm["metrics"]["kd"]["loss"] == "margin_mse"
        # finish_by: no time left -> only the first steps, fewer chunks per case, everything still written
        fb = S.run(bb_of("gpt"), name="gpt", technique="lorma", run_name="gpt_cut", finish_by=time.time() + 1,
                   **dict(common, epochs=8))
        b = fb["metrics"]["budget"]
        assert b["cut"] and fb["metrics"]["steps"] <= 20 and b["max_chunks"] == 4, b
        _check(fb, "lorma")
        # zero-shot: no training at all (Phase 1 representations + an untrained projection)
        zs = S.run(bb_of("llama"), name="llama", technique="frozen", run_name="llama_zeroshot", **dict(common, epochs=0))
        assert zs["metrics"]["steps"] == 0 and len(zs["metrics"]["val_history"]) == 1
        # grid helpers
        assert G.best_run(out, ["llama_full", "gpt_lorma"]) == os.path.join(out, "llama_full")
        ens = G.ensemble([os.path.join(out, "llama_full"), os.path.join(out, "gpt_lorma")],
                         os.path.join(out, "ensemble_toy"), lexical_path=os.path.join(tmp, "cache", "lexical.pkl"),
                         **QUIET)
        assert ens is not None and os.path.exists(ens["submission"])
        L = PD.load_lexical(os.path.join(tmp, "cache", "lexical.pkl"))
        PD.lexical_submissions(L, os.path.join(out, "lexical"), top=30, **QUIET)
        text = G.report([out], out_path=os.path.join(tmp, "all.txt"))
        assert "llama_student_kd" in text and "submission files" in text and "lexical baselines" in text
        assert "ensemble" in text


def test_mid_texts():
    import pcr_mid
    with tempfile.TemporaryDirectory() as tmp:
        root, _ = make_fake_pcr(tmp)
        data = PD.load_data(root, fetch=False)
        pool, held = pcr_mid.split_texts(data, dict(pcr_mid.MID_DEFAULTS, heldout_docs=10))
        test_texts = {d["text"] for d in data["test"][0] + data["test"][1]}
        assert not (set(pool) & test_texts) and not (set(pool) & set(held)) and len(held) == 10
        assert len(pool) + len(held) == 60 + 150 + 20 + 80


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                t0 = time.time()
                fn()
                print(f"PASS {name} ({time.time() - t0:.0f} s)", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
