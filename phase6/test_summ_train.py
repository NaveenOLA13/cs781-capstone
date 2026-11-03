"""Self-test of Phase 6 training and generation on tiny stand-in models (decoder-only with a tied head, Llama-style,
encoder-decoder with a bidirectional GRU) and fake documents. Needs torch. About 2-4 minutes.
python test_summ_train.py  -> every line PASS

Checks: beam search (lengths, minimum length, no repeated trigrams, several problems at once); batches; LoRA,
LoRMA, QLoRA, adapter and full runs write held-out scores, test summaries, a valid submission and trained.pt; the
loss goes down; the encoder-decoder path (exact batches, encoder cache, low-rank only on linear layers); a
half-size KD student learns from a finished teacher run; finish_by cuts training and still writes everything;
mid-training on documents + summaries (with finish_by); report and best_run.
"""
import json
import os
import random
import sys
import tempfile
import time
import zipfile

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "phase4"))
import summ_data as SD  # noqa: E402
import summ_grid as G  # noqa: E402
import summ_model as SM  # noqa: E402
import summ_train as S  # noqa: E402
from cj_models import make_backbone  # noqa: E402
from test_cj_train import TOYS, WordTok  # noqa: E402
from test_summ_data import make_fake_summ  # noqa: E402

FAST = dict(max_len=128, tgt_tokens=32, enc_window=48, epochs=2, batch_size=8, grad_accum=1, eval_docs=5, beams=2,
            gen_batch=4, log_every=2, fetch_data=False, lr_body=3e-3, empty_frac=0.3)


class ToyEncDecE(TOYS["encdec"]):
    """The toy encoder-decoder with model.encode(src) (as our Phase 1 encoder-decoder has): the pointer-generator
    can be added."""

    def encode(self, src):
        return self.encoder(self.emb(src))[0]


def bb_of(kind):
    torch.manual_seed(1)
    model = ToyEncDecE() if kind == "encdec_e" else TOYS[kind]()
    bb = make_backbone(model, WordTok(), max_len=128, verbose=False)
    bb.model.eval()                 # as cj_models.load_model returns the real checkpoints (cuDNN RNNs: eval mode!)
    return bb


def word_decoder(root):
    """The fake data's words back from WordTok ids (WordTok hashes words; the fake vocabulary is small)."""
    tok = WordTok()
    inv = {}
    for split in ("Train/train.json", "Test/test.json"):
        for d in json.load(open(os.path.join(root, split))):
            for w in (d["document"] + " " + d.get("summary", "")).split():
                inv.setdefault(tok.encode(w)[0], w)
    return lambda ids: " ".join(inv.get(int(t), "?") for t in ids)


def test_beam_search():
    V, eos = 12, 2

    def step(rows):
        lp = torch.full((rows.size(0), V), -6.0)
        lp[:, 3 + rows.size(1) % 3] = 0.0                        # prefers 3 4 5 3 4 5 ...
        lp[:, eos] = -1.0 if rows.size(1) >= 7 else -40.0
        return F.log_softmax(lp, -1)
    out = SM.beam_search(step, 2, eos, beams=3, max_new=[5, 12], min_new=[0, 9], no_repeat=3)
    assert len(out) == 2 and len(out[0]) <= 5 and 9 <= len(out[1]) <= 12, out
    for o in out:
        g = list(zip(o, o[1:], o[2:]))
        assert len(g) == len(set(g)), o                           # no repeated trigram
        assert eos not in o
    one = SM.beam_search(step, 1, eos, beams=1, max_new=4, no_repeat=0)
    assert one == [[3, 4, 5, 3]], one                             # greedy without the ban


def test_top_p_sampling():
    V, eos = 12, 2

    def step(rows):
        lp = torch.full((rows.size(0), V), -8.0)
        lp[:, 3 + rows.size(1) % 3] = 0.0                        # one clear favourite at every step
        lp[:, 7] = -0.5                                          # and a second likely token
        lp[:, eos] = -1.0 if rows.size(1) >= 6 else -40.0
        return F.log_softmax(lp, -1)
    a = SM.sample_search(step, 3, eos, max_new=[4, 10, 10], min_new=[0, 8, 0], no_repeat=3, top_p=0.9, seed=1)
    b = SM.sample_search(step, 3, eos, max_new=[4, 10, 10], min_new=[0, 8, 0], no_repeat=3, top_p=0.9, seed=1)
    assert a == b                                                 # seeded: reproducible
    assert len(a[0]) <= 4 and 8 <= len(a[1]) <= 10 and len(a[2]) <= 10, a
    for o in a:
        g = list(zip(o, o[1:], o[2:]))
        assert len(g) == len(set(g)) and eos not in o, o
    tiny = SM.sample_search(step, 1, eos, max_new=3, no_repeat=0, top_p=0.01, temperature=1.0)
    assert tiny == [[3, 4, 5]], tiny                              # top-p near 0 = greedy


def test_batches():
    pairs = [dict(src=[5, 6, 7], tgt=[8, 9]), dict(src=[5, 6], tgt=[8]), dict(src=[1, 2, 3], tgt=[])]
    x, y = SM.dec_batch(pairs[:2], [4], eos_id=2, pad_id=0, device="cpu")
    assert x[0].tolist() == [2, 5, 6, 7, 4, 8, 9, 2] and y[0].tolist() == [-100] * 5 + [8, 9, 2]
    assert x[1].tolist() == [2, 5, 6, 4, 8, 2, 0, 0] and y[1].tolist() == [-100] * 4 + [8, 2, -100, -100]
    src, seq, yy = SM.ed_batch([pairs[0], pairs[2]], bos_id=1, eos_id=2, pad_id=0, device="cpu")
    assert seq.tolist() == [[1, 8, 9, 2], [1, 2, 0, 0]] and yy.tolist() == [[8, 9, 2], [2, -100, -100]]
    b = SM.batches(pairs, 2, exact=True, rng=random.Random(0))
    assert all(len({len(pairs[i]["src"]) for i in g}) == 1 for g in b) and sum(map(len, b)) == 3


def _check(res, n_test, tech):
    m, out = res["metrics"], res["out_dir"]
    assert {"held", "held_raw", "trainable_params", "steps", "submission", "decoding", "decodings"} <= set(m), m.keys()
    assert {"beam", "top_p"} <= set(m["decodings"]) and m["decoding"] in m["decodings"]
    assert m["held"]["rouge_mean"] == max(v["held"]["rouge_mean"] for v in m["decodings"].values())
    assert len(m["val_history"]) >= 2 and 0 <= m["best_epoch"] <= len(m["val_history"]), m["val_history"]
    rows = json.loads(zipfile.ZipFile(os.path.join(out, "submission", "submission.zip")).read("prediction.json"))
    assert len(rows) == n_test and all(isinstance(r["summary"], str) and r["summary"] for r in rows), rows[:2]
    assert os.path.exists(os.path.join(out, "trained.pt")) and os.path.exists(os.path.join(out, "held_summaries.json"))
    if tech != "full":
        assert m["trainable_share"] < 0.6, (tech, m["trainable_share"])


def test_runs_kd_deadline_report():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_summ(tmp, n_train=120, n_test=6)
        dec = word_decoder(root)
        out = os.path.join(tmp, "runs")
        common = dict(FAST, data_root=root, out_dir=out, cache_dir=os.path.join(tmp, "cache"), ckpt_path="toy")
        r = S.run(bb_of("llama"), name="llama", technique="full", decode=dec, **dict(common, epochs=4))
        _check(r, 6, "full")
        h = r["metrics"]["history"]
        assert len(h) >= 2 and h[-1]["loss"] < h[0]["loss"], h
        print(f"    toy llama full: held-out {SD.fmt(r['metrics']['held'])}")
        for kind, tech in (("gpt", "lora"), ("llama", "lorma"), ("llama", "qlora"), ("gpt", "adapter"),
                           ("encdec", "adapter"), ("encdec_e", "lora"), ("encdec_e", "qlora")):
            rr = S.run(bb_of(kind), name=kind, technique=tech, decode=dec, **common)
            _check(rr, 6, tech)
            text = open(os.path.join(rr["out_dir"], "log.txt")).read()
            if kind == "encdec":                                  # no encode(): no pointer-generator
                assert not rr["metrics"]["copy"] and "no pointer-generator" in text
            if kind == "encdec_e":                                # pointer-generator trained with the technique
                assert rr["metrics"]["copy"] and "[copy] pointer-generator" in text and "linear layers only" in text
                st_ = torch.load(os.path.join(rr["out_dir"], "trained.pt"), weights_only=False)["state"]
                assert any(k.startswith("copy.") for k in st_), list(st_)[:5]
        # knowledge distillation: half-size student (evenly spaced blocks) with the llama_full teacher
        st = S.run(bb_of("llama"), name="llama", technique="full", run_name="llama_student_kd", decode=dec,
                   student_layers=0.5, kd_teacher=r["out_dir"], decodings=("beam", "top_p", "beam_long"), **common)
        assert set(st["metrics"]["decodings"]) == {"beam", "top_p", "beam_long"}
        ms = st["metrics"]
        assert ms["student"]["kept"] == [0, 2] and ms["total_params"] < r["metrics"]["total_params"], ms["student"]
        assert ms["kd"]["teacher_technique"] == "full" and all(x["kd"] is not None for x in ms["history"])
        # finish_by: no time left -> only the first steps, everything still written
        fb = S.run(bb_of("gpt"), name="gpt", technique="lora", run_name="gpt_cut", decode=dec,
                   finish_by=time.time() + 1, **dict(common, epochs=5))
        assert fb["metrics"]["budget"]["cut"] and fb["metrics"]["steps"] <= 20, fb["metrics"]["budget"]
        _check(fb, 6, "lora")
        text = G.report([out], out_path=os.path.join(tmp, "all.txt"))
        assert "submission files" in text and "llama_student_kd" in text
        assert G.best_run(out, ["llama_full", "llama_lorma"]) is not None


def test_mid_training():
    import cj_mid as M
    import summ_mid as SMID
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_summ(tmp, n_train=100, n_test=5)
        data = SD.load_data(root, fetch=False)
        pool, held = SMID.split_texts(data, dict(SMID.MID_DEFAULTS, heldout_docs=10))
        test_docs = {d["document"] for d in data["test"]}
        held_docs = {d["document"] for d in data["held"]}        # (fake summaries repeat across documents)
        assert not (set(pool) & test_docs) and not (set(pool) & held_docs) and len(pool) == 2 * len(data["train"])
        assert set(held) <= held_docs
        torch.manual_seed(0)
        model = TOYS["llama"]()
        cfg = dict(tokens=8_000, seq_len=32, batch_size=8, lr=3e-3, warmup_steps=5, eval_every=0, log_every=0,
                   heldout_windows=16, tok_workers=1, data_parallel=False)
        res = M.mid_train_model(model, WordTok().encode, 2, pool, held, cfg, log=lambda *a: None)
        assert res["heldout_after"] < res["heldout_before"], res
        res2 = M.mid_train_model(TOYS["llama"](), WordTok().encode, 2, pool, held,
                                 dict(cfg, tokens=200_000, finish_by=time.time() + 1), log=lambda *a: None)
        assert res2["steps"] == 20, res2["steps"]                 # deadline passed: stopped at the first check


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
