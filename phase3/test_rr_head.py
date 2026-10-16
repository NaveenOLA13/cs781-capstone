"""Self-test for rr_head.py on tiny stand-in models (runs in seconds on CPU).
Run:  python test_rr_head.py      -> every line should say PASS
"""
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rr_head import (ContextClassifier, HiddenStates, Pooler, SentenceAttention, SentenceClassifier,  # noqa: E402
                     count_params, distance_buckets, find_lm_head, freeze_backbone, load_phase1, padding_check)

torch.manual_seed(0)
V, D, NL = 50, 16, 13


class ToyGPT(nn.Module):
    """Causal decoder with a weight-tied LM head, like a small GPT."""

    def __init__(self, last_only=False):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(64, D)
        layer = nn.TransformerEncoderLayer(D, 2, 4 * D, dropout=0.0, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.tok.weight
        self.last_only = last_only

    def forward(self, idx):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        causal = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        x = self.ln_f(self.blocks(x, mask=causal))
        self.last_x = x
        if self.last_only:
            x = x[:, [-1], :]
        return self.lm_head(x)


class ToyEncDec(nn.Module):
    """Encoder-decoder with a BiGRU encoder."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.encoder = nn.GRU(D, D // 2, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(D, D, batch_first=True)
        self.lm_head = nn.Linear(D, V)

    def forward(self, idx):
        e = self.emb(idx)
        enc, _ = self.encoder(e)
        dec, _ = self.decoder(e + enc)
        return self.lm_head(dec)


def batch(seqs, pad=0):
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad)
    mask = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


def test_finds_tied_lm_head_and_skips_it():
    m = ToyGPT().eval()
    assert find_lm_head(m) == "lm_head"
    calls = []
    m.lm_head.register_forward_hook(lambda mod, a, out: calls.append(1))
    hs = HiddenStates(m)
    ids, mask = batch([[3, 4, 5, 6]])
    h = hs(ids, mask)
    assert h.shape == (1, 4, D)
    assert torch.allclose(h, m.last_x)          # exactly the final normed hidden states
    assert not calls                            # LM head never executed


def test_untied_head_not_confused_with_feed_forward():
    m = ToyGPT()
    m.lm_head = nn.Linear(D, V, bias=False)      # untied; position table (64) == feed-forward width (64)
    assert find_lm_head(m) == "lm_head", find_lm_head(m)


def test_classifier_shapes_and_width_inference():
    for pooling in Pooler.KINDS:
        clf = SentenceClassifier(ToyGPT(), NL, pooling=pooling).eval()
        assert clf.d_model == D
        ids, mask = batch([[1, 2, 3], [4, 5, 6, 7, 8]])
        assert clf(ids, mask).shape == (2, NL), pooling


def test_padding_does_not_leak_causal():
    clf = SentenceClassifier(ToyGPT(), NL, pooling="last")
    for pooling in ("last", "mean", "max", "attn", "first"):
        clf.pool = Pooler(D, pooling)
        diff = padding_check(clf, [3, 4, 5], [6, 7, 8, 9, 10, 11, 12])
        assert diff < 1e-5, (pooling, diff)


def test_left_padding_last_token():
    p = Pooler(D, "last")
    h = torch.randn(2, 5, D)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
    out = p(h, mask)
    assert torch.equal(out[0], h[0, 2]) and torch.equal(out[1], h[1, 4])
    first = Pooler(D, "first")(h, mask)
    assert torch.equal(first[0], h[0, 0]) and torch.equal(first[1], h[1, 2])


def test_gradients_full_and_frozen():
    clf = SentenceClassifier(ToyGPT(), NL, pooling="last")
    ids, mask = batch([[1, 2, 3], [4, 5, 6, 7]])
    loss = nn.functional.cross_entropy(clf(ids, mask), torch.tensor([0, 5]))
    loss.backward()
    assert clf.encoder.model.tok.weight.grad is not None      # full fine-tuning reaches backbone
    assert clf.head.weight.grad is not None
    n_all = count_params(clf)
    freeze_backbone(clf)
    n_head = count_params(clf)
    assert n_head == D * NL + NL and n_head < n_all, (n_head, n_all)


def test_nanogpt_style_last_only_fails_loudly():
    try:
        SentenceClassifier(ToyGPT(last_only=True), NL)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "hidden_states" in str(e)


def test_hidden_states_method_is_preferred():
    m = ToyGPT(last_only=True)
    m.hidden_states = lambda idx: (m(idx), m.last_x)[1]   # what you'd add to your model class
    clf = SentenceClassifier(m, NL, pooling="mean").eval()
    ids, mask = batch([[1, 2, 3, 4]])
    assert clf(ids, mask).shape == (1, NL)


def test_encdec_encoder_output():
    clf = SentenceClassifier(ToyEncDec(), NL, pooling="mean", module_name="encoder", mode="module_out").eval()
    assert clf.d_model == D                     # BiGRU: 2 x D/2
    ids, mask = batch([[1, 2, 3], [4, 5, 6, 7, 8]])
    assert clf(ids, mask).shape == (2, NL)
    diff = padding_check(clf, [3, 4, 5], [6, 7, 8, 9, 10, 11, 12])
    print(f"     (BiGRU encoder padding difference = {diff:.4f}; non-zero is expected without packing)")


def test_load_phase1_roundtrip():
    src = ToyGPT()
    sd = {"module." + k: v for k, v in src.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ck.pt")
        torch.save({"model": sd, "config": {"arch": "toy"}}, path)
        dst = ToyGPT()
        ck = load_phase1(dst, path)
        assert ck["config"]["arch"] == "toy"
        for k, v in src.state_dict().items():
            assert torch.equal(v, dst.state_dict()[k]), k
        try:
            load_phase1(ToyEncDec(), path)
            assert False, "mismatched architecture should raise"
        except RuntimeError:
            pass


def test_distance_buckets():
    def ref(i, j):
        d = j - i
        a = abs(d)
        b = a if a <= 2 else 3 if a <= 4 else 4 if a <= 8 else 5 if a <= 16 else 6 if a <= 32 else 7
        return 0 if d == 0 else (b if d > 0 else b + 7)
    S = 70
    got = distance_buckets(S)
    assert got.shape == (S, S) and int(got.max()) == 14 and int(got.min()) == 0
    for i in range(S):
        for j in range(S):
            assert int(got[i, j]) == ref(i, j), (i, j, int(got[i, j]), ref(i, j))


def test_attention_starts_switched_off_and_knows_order():
    F_ = 24
    x = torch.randn(9, F_)
    torch.manual_seed(1)
    gru = ContextClassifier(F_, NL, d_ctx=16, arch="gru").eval()
    torch.manual_seed(1)
    gru_attn = ContextClassifier(F_, NL, d_ctx=16, arch="gru_attn").eval()
    assert torch.allclose(gru(x), gru_attn(x), atol=1e-6)          # same start: attention output is 0
    attn = ContextClassifier(F_, NL, d_ctx=16, arch="attn").eval()
    assert attn(x).shape == (9, NL) and attn(x.unsqueeze(0)).shape == (1, 9, NL)
    att = SentenceAttention(16, heads=4).eval()
    assert att(torch.randn(1, 5, 16)).abs().max() == 0                # switched off at the start
    nn.init.normal_(att.o.weight)
    nn.init.normal_(att.dist.weight)
    h = torch.randn(1, 6, 16)
    perm = torch.tensor([5, 3, 0, 1, 4, 2])
    out, out_p = att(h), att(h[:, perm])
    assert not torch.allclose(out[:, perm], out_p, atol=1e-5)       # order matters through the distances
    assert att.last_weights.shape == (1, 4, 6, 6)
    m = ContextClassifier(F_, NL, d_ctx=16, arch="gru_attn")
    m(x).sum().backward()
    assert m.attn.o.weight.grad is not None and m.attn.o.weight.grad.abs().sum() > 0    # it can learn to turn on
    try:
        ContextClassifier(F_, NL, arch="lstm")
        raise AssertionError("unknown version should fail")
    except ValueError:
        pass


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
