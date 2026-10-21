"""Self-test of mid-training (cj_mid.py) on tiny stand-in models. Needs torch. About a minute.
    python test_cj_mid.py  -> every line PASS
"""
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cj_mid as M  # noqa: E402
from test_cj_train import V, ToyEncDec, ToyGPT, ToyLlama, WordTok  # noqa: E402

torch.manual_seed(0)
PHRASES = ["the appeal is allowed and the order of the high court is set aside",
           "the petitioner submitted that the tribunal erred in law",
           "learned counsel for the respondent relied on the judgment of this court",
           "section 302 of the indian penal code was invoked against the accused"]


def fake_texts(n, seed):
    rng = random.Random(seed)
    return [" ".join(rng.choice(PHRASES) for _ in range(rng.randint(3, 12))) for _ in range(n)]


class ToyNanoGPT(ToyGPT):
    """nanoGPT style: scores for the last position only unless targets are given, then (scores, loss)."""

    def forward(self, idx, targets=None):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.h:
            x = b(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits[:, -1:, :], None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def test_windows():
    arrays = [np.arange(1, 6, dtype=np.int32), np.arange(10, 13, dtype=np.int32)]
    w = M.windows(arrays, eos_id=2, seq_len=3)
    assert w.tolist() == [[1, 2, 3, 4], [5, 2, 10, 11]]           # stream 1..5, eos, 10..12, eos -> 2 windows
    assert M.windows(arrays, eos_id=None, seq_len=1).tolist() == [[1, 2], [3, 4], [5, 10], [11, 12]]


def test_lm_loss_routes():
    x = torch.randint(3, V, (2, 9))
    y = torch.randint(3, V, (2, 9))
    m = ToyLlama()
    want = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    assert torch.allclose(M.lm_loss(m, x, y), want, atol=1e-5)
    n = ToyNanoGPT()
    want = F.cross_entropy(n(x, y)[0].reshape(-1, V), y.reshape(-1))
    assert torch.allclose(M.lm_loss(n, x, y), want, atol=1e-5)
    assert M.is_encoder_decoder(ToyEncDec()) and not M.is_encoder_decoder(ToyLlama())


def test_mid_training_lowers_heldout_loss():
    enc = WordTok().encode
    for cls in (ToyLlama, ToyGPT):
        torch.manual_seed(1)
        model = cls()
        before_w = {k: v.clone() for k, v in model.state_dict().items()}
        cfg = dict(tokens=12_000, seq_len=32, batch_size=8, lr=3e-3, warmup_steps=5, eval_every=0, log_every=0,
                   tok_workers=2, heldout_windows=32, data_parallel=False)
        res = M.mid_train_model(model, enc, 2, fake_texts(300, 1), fake_texts(30, 2), cfg, log=lambda *a: None)
        print(f"      {cls.__name__}: held-out loss {res['heldout_before']:.3f} -> {res['heldout_after']:.3f}")
        assert res["heldout_after"] < res["heldout_before"] - 0.5, res
        assert res["tokens"] <= 12_000 and res["steps"] >= 40 and res["history"][-1]["heldout"] == res["heldout_after"]
        assert any(not torch.equal(v.cpu(), before_w[k]) for k, v in model.state_dict().items())
    try:
        M.mid_train_model(ToyEncDec(), enc, 2, fake_texts(10, 3), fake_texts(5, 4), {}, log=lambda *a: None)
        raise AssertionError("an encoder-decoder should be refused")
    except NotImplementedError:
        pass


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
