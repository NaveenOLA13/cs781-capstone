"""Self-test of whole-case reading (cj_hier.py) and chunk training (train_chunks) on tiny stand-in models.
Needs torch. About 3 minutes.     python test_cj_hier.py  -> every line PASS

Checks: training on several chunks per case; the SFT run is rebuilt exactly from trained.pt (its own dev predictions
are reproduced by the 'last' reference) for several techniques; Stage B versions give the same scores for a case
alone and in a padded batch; run_hier writes probabilities, submissions and the comparison, and reuses the cache.
"""
import json
import os
import sys
import tempfile
import zipfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cj_data as D  # noqa: E402
import cj_hier as H  # noqa: E402
import cj_train as R  # noqa: E402
from test_cj_data import make_fake_data  # noqa: E402
from test_cj_train import FAST, _bb  # noqa: E402

torch.manual_seed(0)


def test_stage_b_padding():
    torch.manual_seed(0)
    for v in H.VERSIONS:
        m = H.HierHead(16, v, hidden=32, k=4).eval()
        a = torch.randn(1, 2, 16)
        b = torch.randn(1, 4, 16)
        x = torch.zeros(2, 4, 16)
        x[0, :2], x[1] = a[0], b[0]
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
        alone = m(a, torch.ones(1, 2, dtype=torch.bool))
        batched = m(x, mask)[:1]
        assert torch.allclose(alone, batched, atol=1e-5), (v, alone, batched)


def test_chunk_training_and_hier():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_data(tmp, n_train=40, n_dev=12, n_test=10, n_words=(40, 160))
        common = dict(FAST, data_root=root, out_dir=os.path.join(tmp, "runs"), cache_dir=os.path.join(tmp, "cache"),
                      max_len=24, chunk_stride=16, train_chunks=3)
        for tech in ("lorma", "adapter", "bitfit", "full"):
            bb = _bb("llama")
            res = R.run(bb, name="llama", technique=tech, **common)
            log = open(os.path.join(res["out_dir"], "log.txt")).read()
            assert "chunk examples from 40 cases" in log, log[:500]
            out = H.run_hier(res["out_dir"], versions=("gru_attn", "transformer", "mean") if tech == "lorma" else (),
                             bb=_bb("llama"), k=3, epochs=2, hidden=32, batch=8, log=lambda *a: None)
            # the rebuilt Stage A reproduces the SFT run's own dev predictions
            run_dev = json.load(open(os.path.join(res["out_dir"], "dev_probs.json")))
            last = json.load(open(os.path.join(res["out_dir"], "hier_k3", "last", "dev_probs.json")))
            diff = max(abs(run_dev[i] - last[i]) for i in run_dev)
            assert diff < 2e-3, (tech, diff)
            ch = torch.load(os.path.join(res["out_dir"], "chunks_k3.pt"), weights_only=False)
            assert len(ch["train"]["feats"]) == 40 and max(len(f) for f in ch["train"]["feats"]) <= 3
            assert any(len(f) > 1 for f in ch["dev"]["feats"])
            want = {"last", "vote"} | ({"gru_attn", "transformer", "mean"} if tech == "lorma" else set())
            assert set(out) == want, out.keys()
            test_ids = [d["id"] for d in D.load_data(root, fetch=False)["test"]]
            for v in want:
                with zipfile.ZipFile(os.path.join(res["out_dir"], "hier_k3", v, "submission_tuned",
                                                  "submission.zip")) as z:
                    assert [r["id"] for r in json.loads(z.read("prediction.json"))] == test_ids
            assert "best on dev" in open(os.path.join(res["out_dir"], "hier_k3", "hier.txt")).read()
            tp = json.load(open(os.path.join(res["out_dir"], "hier_k3", "vote", "train_probs.json")))
            assert len(tp) == 40 and all(0 <= v <= 1 for v in tp.values())
        again = H.extract(res["out_dir"], 3, bb=_bb("llama"), log=lambda *a: None)   # cached
        assert again["k"] == 3


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
