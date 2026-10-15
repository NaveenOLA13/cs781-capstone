"""End-to-end self-test of the Phase 3 code on tiny stand-in models and fake data.
Runs in about a minute (GPU or CPU).      python test_rr_pipeline.py   -> every line PASS

Checks: tokenizer adapters, automatic hidden-state detection for decoder / encoder-decoder /
nanoGPT / hidden_states-method / one-way-RNN models, clear failures, and a full run
(Stage A + Stage B + both submissions) that leaves the original model untouched.
"""
import json
import math
import os
import sys
import tempfile
import zipfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_backbone import make_backbone, make_encode, special_ids  # noqa: E402
from rr_head import count_params  # noqa: E402
from rr_train import run  # noqa: E402
from rr_cv import VARIANTS, run_cv  # noqa: E402
from test_rr_data import make_fake_data  # noqa: E402

torch.manual_seed(0)
V, D_ = 600, 32


# ------------------------------------------------------------------- toy tokenizers
def word_ids(text):
    return [3 + sum(map(ord, w)) % (V - 3) for w in text.split()]


class WordTok:
    """encode(text) -> list[int]; <pad>=0, <s>=1, </s>=2 via token_to_id."""
    special = {"<pad>": 0, "<s>": 1, "</s>": 2}

    def encode(self, text):
        return word_ids(text)

    def token_to_id(self, t):
        return self.special.get(t)


class Encoding:
    def __init__(self, ids):
        self.ids = ids


class HFLikeTok(WordTok):
    def encode(self, text):
        return Encoding(super().encode(text))


class SPLikeTok:
    """sentencepiece-like: encode_as_ids, piece_to_id returns unk (1) for unknown, pad_id() = -1."""

    def encode_as_ids(self, text):
        return word_ids(text)

    def piece_to_id(self, p):
        return {"</s>": 2}.get(p, 1)

    def pad_id(self):
        return -1


# ----------------------------------------------------------------------- toy models
class Block(nn.Module):
    def __init__(self, d, causal):
        super().__init__()
        self.causal = causal
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        T = x.size(1)
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        att = q @ k.transpose(1, 2) / math.sqrt(x.size(-1))
        if self.causal:
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float("-inf"))
        x = x + self.proj(att.softmax(-1) @ v)
        return x + self.ff(self.ln2(x))


class ToyGPT(nn.Module):
    """Causal decoder, tied LM head, positions up to max_pos."""

    def __init__(self, max_pos=256):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, D_), nn.Embedding(max_pos, D_)
        self.h = nn.ModuleList([Block(D_, True) for _ in range(2)])
        self.ln_f = nn.LayerNorm(D_)
        self.lm_head = nn.Linear(D_, V, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.h:
            x = b(x)
        return self.lm_head(self.ln_f(x))


class ToyNanoGPT(ToyGPT):
    """nanoGPT: logits for the last position only unless targets are given."""

    def forward(self, idx, targets=None):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.h:
            x = b(x)
        x = self.ln_f(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        return self.lm_head(x), None


class ToyEncDec(nn.Module):
    """Encoder-decoder: embedding outside a BiGRU encoder, forward(src, tgt)."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_)
        self.encoder = nn.GRU(D_, D_ // 2, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(D_, D_, batch_first=True)
        self.lm_head = nn.Linear(D_, V)

    def forward(self, src, tgt):
        enc, _ = self.encoder(self.emb(src))
        dec, _ = self.decoder(self.emb(tgt) + enc.mean(1, keepdim=True))
        return self.lm_head(dec)


class TransformerEncoderStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D_)
        self.pos = nn.Embedding(256, D_)
        self.layers = nn.ModuleList([Block(D_, False) for _ in range(2)])

    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device))
        for b in self.layers:
            x = b(x)
        return x


class ToyEncDecEmbInside(nn.Module):
    """Encoder module includes its embeddings; forward needs (src, tgt, teacher_forcing)."""

    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoderStack()
        self.dec_emb = nn.Embedding(V, D_)
        self.out = nn.Linear(D_, V)

    def forward(self, src, tgt, teacher_forcing):
        return self.out(self.dec_emb(tgt) + self.encoder(src).mean(1, keepdim=True))


class ToyWithHiddenStates(ToyNanoGPT):
    def hidden_states(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.h:
            x = b(x)
        return self.ln_f(x)


class ToyOneWayRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_)
        self.rnn = nn.GRU(D_, D_, batch_first=True)
        self.lm_head = nn.Linear(D_, V)

    def forward(self, idx):
        return self.lm_head(self.rnn(self.emb(idx))[0])


class Weird(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 4)

    def forward(self, a, b, c):
        return self.w(a)


# ---------------------------------------------------------------------------- tests
def test_tokenizer_adapters():
    s = "The appellant filed an appeal"
    ref = WordTok().encode(s)
    for tok in (WordTok(), HFLikeTok(), SPLikeTok(), WordTok().encode):
        assert make_encode(tok)(s) == ref, type(tok)
    assert special_ids(WordTok()) == {"pad": 0, "bos": 1, "eos": 2}
    sp = special_ids(SPLikeTok())
    assert sp["pad"] is None and sp["eos"] == 2 and sp["bos"] is None, sp   # unk (1) is not taken as bos


def _bb(model, **kw):
    return make_backbone(model, WordTok(), max_len=64, verbose=False, **kw)


def test_routes_detected():
    cases = [
        (ToyGPT(), "LM head (lm_head) while running model(ids)", "decoder", "last"),
        (ToyNanoGPT(), "targets=ids", "decoder", "last"),
        (ToyWithHiddenStates(), "hidden_states", "decoder", "last"),
        (ToyEncDec(), "model.encoder while running model(ids, ids)", "encoder", "mean"),
        (ToyEncDecEmbInside(), "model.encoder(ids)", "encoder", "mean"),
        (ToyOneWayRNN(), "LM head", "decoder", "last"),
    ]
    for model, route, kind, pooling in cases:
        bb = _bb(model)
        assert route in bb.route and bb.kind == kind and bb.pooling == pooling, (type(model).__name__, bb.route,
                                                                                bb.kind)
        clf = bb.classifier(13, pos_buckets=10)
        ids = torch.tensor([[5, 6, 7, 0], [8, 9, 10, 11]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        assert clf(ids, mask, pos=torch.tensor([0, 9])).shape == (2, 13)
        print(f"      {type(model).__name__:20s} -> {bb.route} | {bb.kind}, pooling {bb.pooling}")


def test_explicit_route_and_failures():
    m = ToyEncDecEmbInside()
    bb = _bb(m, call=lambda mm, ids, mask: mm.encoder(ids))
    assert bb.kind == "encoder" and bb.mode == "output"
    try:
        _bb(Weird())
        assert False, "should fail"
    except RuntimeError as e:
        assert "screenshot" in str(e)
    try:
        make_backbone(ToyGPT(max_pos=32), WordTok(), max_len=64, verbose=False)
        assert False, "should fail"
    except RuntimeError as e:
        assert "cannot take 64 tokens" in str(e)
    try:
        make_backbone(ToyGPT(), WordTok(), pad_id=V + 5, verbose=False)
        assert False, "should fail"
    except ValueError as e:
        assert "vocabulary" in str(e)


def _check_run(res, data_dir, model, before):
    out = res["out_dir"]
    for stage in ("stageA", "stageB"):
        z = os.path.join(out, stage, "submission.zip")
        with zipfile.ZipFile(z) as f:
            assert f.namelist() == ["prediction.json"]
            pred = json.loads(f.read("prediction.json"))
        test = {}
        for name in ("CL_test.json", "IT_test.json"):
            test.update(json.load(open(os.path.join(data_dir, "Test", name))))
        assert set(pred) == set(test) and all(len(pred[k]) == len(test[k]) for k in test)
    m = json.load(open(os.path.join(out, "metrics.json")))
    assert set(m["stage_a"]["dev"]) == {"all", "CL", "IT"}
    h = m["stage_a"]["history"]
    assert h[-1]["loss"] < h[0]["loss"], [x["loss"] for x in h]
    for k, v in model.state_dict().items():                       # the original model is untouched
        assert torch.equal(v.cpu(), before[k]), f"original weights changed: {k}"
    for f in ("log.txt", "config.json", "features.pt", "stageA_model.pt", "stageB_model.pt",
              "stageA/dev_confusion.csv", "stageB/dev_predictions.json"):
        assert os.path.exists(os.path.join(out, f)), f
    return m


def test_full_run_decoder():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=8)
        model = ToyGPT()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        bb = _bb(model)
        res = run(bb, data_root=data, out_dir=os.path.join(tmp, "runs"), run_name="gpt", epochs=4,
                  b_epochs=4, batch_size=16, max_len=32, lr_backbone=3e-3, lr_head=3e-3, log_every=0)
        m = _check_run(res, data, model, before)
        assert m["config"]["exact_batches"] is False            # causal decoder: padding has no effect
        print(f"      dev macro-F1: stage A {m['stage_a']['best_dev_score']:.3f}, "
              f"stage B {m['stage_b']['best_dev_score']:.3f} (fake data)")


def test_full_and_frozen_run_encdec():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=4)
        model = ToyEncDec()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        bb = _bb(model)
        res = run(bb, data_root=data, out_dir=os.path.join(tmp, "runs"), run_name="ed", epochs=2, b_epochs=2,
                  batch_size=16, max_len=32, lr_backbone=3e-3, log_every=0)
        m = _check_run(res, data, model, before)
        assert m["config"]["exact_batches"] is True             # BiGRU: padding changes outputs
        used = count_params(model.emb) + count_params(model.encoder)
        head = (D_ + 32) * 13 + 13 + 10 * 32
        assert m["stage_a"]["trainable_params"] == used + head, (m["stage_a"]["trainable_params"], used + head)
        trained = res["stage_a"].encoder.model
        assert torch.equal(trained.decoder.weight_hh_l0.cpu(), before["decoder.weight_hh_l0"])  # unused: unchanged
        assert not torch.equal(trained.encoder.weight_hh_l0.cpu(), before["encoder.weight_hh_l0"])
        res = run(bb, data_root=data, out_dir=os.path.join(tmp, "runs"), run_name="ed_frozen", method="frozen",
                  epochs=1, b_epochs=1, batch_size=16, max_len=32, log_every=0)
        m = _check_run_frozen(res, head)
        assert torch.equal(res["stage_a"].encoder.model.encoder.weight_hh_l0.cpu(), before["encoder.weight_hh_l0"])


def _check_run_frozen(res, head):
    m = res["metrics"]
    assert m["stage_a"]["trainable_params"] == head, (m["stage_a"]["trainable_params"], head)
    return m


def test_limit_and_bad_setting():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=6)
        bb = _bb(ToyGPT())
        res = run(bb, data_root=data, out_dir=os.path.join(tmp, "runs"), run_name="q", limit_train_docs=2,
                  epochs=1, b_epochs=1, max_len=32, log_every=0, save_model=False)
        assert res["metrics"]["data"]["train"]["docs"] == 4
        try:
            run(bb, data_root=data, epoch=3)
            assert False
        except ValueError as e:
            assert "unknown settings" in str(e)


def test_out_of_fold_run():
    for model_cls, name in ((ToyGPT, "gpt"), (ToyEncDec, "encdec")):
        with tempfile.TemporaryDirectory() as tmp:
            data = make_fake_data(os.path.join(tmp, "data"), n_train=6)
            model = model_cls()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            bb = _bb(model)
            res = run_cv(bb, data_root=data, out_dir=os.path.join(tmp, "runs"), run_name=name, folds=3, epochs=2,
                         b_epochs=4, batch_size=16, max_len=32, lr_backbone=3e-3, lr_head=3e-3, log_every=0)
            out, m = res["out_dir"], res["metrics"]
            assert res["chosen"] in VARIANTS and m["chosen"] == res["chosen"]
            test = {}
            for f in ("CL_test.json", "IT_test.json"):
                test.update(json.load(open(os.path.join(data, "Test", f))))
            for v in VARIANTS:
                with zipfile.ZipFile(os.path.join(out, v, "submission.zip")) as z:
                    pred = json.loads(z.read("prediction.json"))
                assert set(pred) == set(test) and all(len(pred[k]) == len(test[k]) for k in test), v
                assert set(m["dev"][v]) == {"all", "CL", "IT"} and set(m["oof_train"][v]) == {"all", "CL", "IT"}
            assert len(m["folds"]) == 3 and sorted(sum(m["folds"], [])) == sorted(m["folds"][0] + m["folds"][1] + m["folds"][2])
            assert len({d for f in m["folds"] for d in f}) == 12                       # every judgment held out once
            assert m["bias"]["oof_13avg"]["per_domain"] >= m["bias"]["oof_13avg"]["none"] - 1e-9
            for f in ("oof.pt", "metrics.json", "log.txt", "stageA_fold1.pt", "stageA_fold3.pt", "stageB_fold2.pt"):
                assert os.path.exists(os.path.join(out, f)), f
            saved = torch.load(os.path.join(out, "stageA_fold1.pt"), map_location="cpu", weights_only=False)
            assert not any("decoder" in k or "lm_head" in k for k in saved["trained_weights"]) or name == "gpt"
            for k, v in model.state_dict().items():
                assert torch.equal(v.cpu(), before[k]), f"original weights changed: {k}"
            print(f"      {name}: chosen {res['chosen']}, dev 13-role avg "
                  + ", ".join(f"{v} {m['dev'][v]['all']['macro_f1_all13']:.3f}" for v in VARIANTS))


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