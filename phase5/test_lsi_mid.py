"""Self-test of Phase 5 mid-training text selection (lsi_mid.py) and of one short mid-training pass on a tiny
stand-in model with LSI-like documents. Needs torch. About 30 seconds.     python test_lsi_mid.py -> PASS

Checks: the held-out documents are never trained on, dev text is used only if use_dev_text, test text never; the
Phase 4 next-token trainer lowers the held-out loss on these documents.
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lsi_data as L  # noqa: E402
import lsi_mid as LM  # noqa: E402
from test_lsi_data import make_fake_lsi  # noqa: E402


def test_split_texts():
    with tempfile.TemporaryDirectory() as tmp:
        data = L.load_data(make_fake_lsi(tmp, n_train=100, n_dev=20), fetch=False)
        cfg = dict(LM.MID_DEFAULTS, heldout_docs=300)
        pool, held = LM.split_texts(data, cfg)
        test_texts = {d["text"] for d in data["test"]}
        assert len(held) == 10 and not set(held) & set(pool) and not test_texts & set(pool)
        assert {d["text"] for d in data["dev"]} <= set(pool)
        pool2, _ = LM.split_texts(data, dict(cfg, use_dev_text=False))
        assert len(pool2) == 90
        assert LM.MID_DEFAULTS["out_dir"] == "/kaggle/working/mid5" and LM.MID_DEFAULTS["data_root"] == L.DEFAULT_ROOT


def test_mid_training_lowers_heldout_loss():
    import cj_mid as M
    from test_cj_train import TOYS, WordTok
    with tempfile.TemporaryDirectory() as tmp:
        data = L.load_data(make_fake_lsi(tmp, n_train=200, n_dev=20, n_words=(40, 90)), fetch=False)
        pool, held = LM.split_texts(data, dict(LM.MID_DEFAULTS, heldout_docs=20))
        torch.manual_seed(0)
        model = TOYS["llama"]()
        tok = WordTok()
        res = M.mid_train_model(model, tok.encode, 2, pool, held,
                                dict(tokens=12_000, seq_len=32, batch_size=8, lr=3e-3, warmup_steps=5, eval_every=0,
                                     log_every=0, heldout_windows=32, tok_workers=1, data_parallel=False),
                                log=lambda *a: None)
        assert res["heldout_after"] < res["heldout_before"], res


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
