"""Self-test of the PEFT code (rr_peft.py and its use in rr_train.py / rr_cv.py) on tiny stand-in models and fake data."""

import json
import math
import os
import sys
import tempfile
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_peft as P  # noqa: E402
from rr_cv import run_cv  # noqa: E402
from rr_train import run  # noqa: E402
from test_rr_data import make_fake_data  # noqa: E402
from test_rr_pipeline import D_, V, ToyEncDec, ToyGPT, _bb  # noqa: E402

torch.manual_seed(0)
HEAD = (D_ + 32) * 13 + 13 + 10 * 32
FAST = dict(epochs=3, b_epochs=2, batch_size=16, max_len=32, lr_head=3e-3, log_every=0)


# --- a Llama-style toy model
class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x):
        T = x.size(1)
        att = self.wq(x) @ self.wk(x).transpose(1, 2) / math.sqrt(x.size(-1))
        att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float("-inf"))
        return self.wo(att.softmax(-1) @ self.wv(x))


class FeedForward(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention_norm, self.ffn_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention, self.feed_forward = Attention(d), FeedForward(d, 48)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class ToyLlama(nn.Module):
    """Meta-Llama layout: tok_embeddings, layers[i].attention.{wq,wk,wv,wo}, feed_forward.{w1,w2,w3}, output."""

    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(V, D_)
        self.layers = nn.ModuleList([LlamaBlock(D_) for _ in range(2)])
        self.norm = nn.LayerNorm(D_)
        self.output = nn.Linear(D_, V, bias=False)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))


# --- tests
def test_compact_names():
    assert P.compact(["layers.0.attention.wq", "layers.1.attention.wq", "h.0.ff.2"]) == [
        "layers.*.attention.wq (2)",
        "h.*.ff.2 (1)",
    ]


def test_choose_targets():
    llama = ToyLlama()
    t, ff, _ = P.choose_targets(llama, "lora", D_, V)
    assert len(t) == 14 and "output" not in t and all(n.startswith("layers.") for n in t), t
    t, ff, note = P.choose_targets(llama, "ia3", D_, V)
    assert t == [f"layers.{i}.{n}" for i in (0, 1) for n in ("attention.wk", "attention.wv", "feed_forward.w2")], t
    assert ff == ["layers.0.feed_forward.w2", "layers.1.feed_forward.w2"] and "key and value" in note, (ff, note)
    gpt = ToyGPT()
    t, _, _ = P.choose_targets(gpt, "lora", D_, V)
    assert len(t) == 8 and "lm_head" not in t, t
    t, ff, note = P.choose_targets(gpt, "ia3", D_, V)
    assert t == ["h.0.qkv", "h.0.ff.2", "h.1.qkv", "h.1.ff.2"] and ff == ["h.0.ff.2", "h.1.ff.2"], (t, ff)
    assert "fused" in note
    t, _, _ = P.choose_targets(llama, "lora", D_, V, targets=["wq", "wv"])
    assert t == [f"layers.{i}.attention.{n}" for i in (0, 1) for n in ("wq", "wv")], t
    for bad, kw, msg in ((llama, dict(targets=["nope"]), "not found"), (ToyEncDec(), {}, "nn.GRU")):
        try:
            P.choose_targets(bad, "lora", D_, V, **kw)
            assert False, "should fail"
        except ValueError as e:
            assert msg in str(e), e


def _expected_trainable(model, method, cfg_r=8, init_r=12):
    targets, ff, _ = P.choose_targets(model, method, D_, V)
    mods = dict(model.named_modules())
    n = 0
    for t in targets:
        i, o = mods[t].in_features, mods[t].out_features
        if method == "ia3":
            n += i if t in ff else o
        elif method == "adalora":
            n += init_r * (i + o + 1)
        else:
            n += cfg_r * (i + o)
    return n + HEAD, targets


def _check_submissions(out, data):
    test = {}
    for name in ("CL_test.json", "IT_test.json"):
        test.update(json.load(open(os.path.join(data, "Test", name))))
    for stage in ("stageA", "stageB"):
        with zipfile.ZipFile(os.path.join(out, stage, "submission.zip")) as f:
            pred = json.loads(f.read("prediction.json"))
        assert set(pred) == set(test) and all(len(pred[k]) == len(test[k]) for k in test)


def test_each_method_runs():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=6)
        model = ToyLlama()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        bb = _bb(model)
        for method in P.PEFT_METHODS:
            res = run(bb, data_root=data, out_dir=os.path.join(tmp, "runs"), run_name=method, method=method, **FAST)
            out, m = res["out_dir"], res["metrics"]
            _check_submissions(out, data)
            want, targets = _expected_trainable(ToyLlama(), method)
            assert m["stage_a"]["trainable_params"] == want, (method, m["stage_a"]["trainable_params"], want)
            assert m["config"]["lr_peft"] == P.DEFAULT_LR[method] and m["stage_a"]["method"] == method
            assert set(m["stage_a"]["dev"]) == {"all", "CL", "IT"}
            for k, v in model.state_dict().items():
                assert torch.equal(v.cpu(), before[k]), f"{method}: original weights changed: {k}"
            trained = res["stage_a"].encoder.model
            mods = dict(trained.named_modules())
            for t in targets:
                assert torch.equal(mods[t].base_layer.weight.detach().cpu(), before[t + ".weight"]), (method, t)
            assert torch.equal(trained.tok_embeddings.weight.detach().cpu(), before["tok_embeddings.weight"])
            if method in ("lora", "rslora"):
                assert any(mods[t].lora_B["default"].weight.abs().sum() > 0 for t in targets), method
                scale = 16 / 8 if method == "lora" else 16 / math.sqrt(8)
                assert abs(mods[targets[0]].scaling["default"] - scale) < 1e-6, (method, mods[targets[0]].scaling)
            elif method == "ia3":
                assert any((mods[t].ia3_l["default"] - 1).abs().sum() > 0 for t in targets)
                assert [t for t in targets if mods[t].is_feedforward] == [t for t in targets if "w2" in t]
            else:
                hooks = res["stage_a"].peft_hooks
                total = hooks.total_steps
                assert 0 < hooks.step <= total and hooks.step >= total - int(0.2 * total), (hooks.step, total)
                info = m["stage_a"]["peft"]
                assert 0 < info["final_rank_total"] <= 8 * len(targets), info
                assert len(info["final_rank_by_layer"]) == len(targets), info["final_rank_by_layer"]
                assert any(mods[t].lora_E["default"].abs().sum() > 0 for t in targets)
                reg = hooks.extra_loss()
                assert torch.is_tensor(reg) and reg.item() >= 0
            assert not os.path.exists(os.path.join(out, "stageA_model.pt"))
            saved = torch.load(os.path.join(out, "stageA_adapter.pt"), map_location="cpu", weights_only=False)
            keys = list(saved["trained_weights"])
            assert keys and all(
                ("lora_" in k or "ia3_" in k or ".ranknum." in k or k.split(".")[0] in ("pool", "head", "pos_emb"))
                for k in keys
            ), keys
            assert saved["peft_config"] is not None
            assert m["stage_a"]["saved_mb"] < 1.0
            a13, b13 = (m[s]["dev"]["all"]["macro_f1_all13"] for s in ("stage_a", "stage_b"))
            print(
                f"      {method:8s} trainable {m['stage_a']['trainable_params']:6d} (incl. head), "
                f"dev 13-role avg A {a13:.3f} B {b13:.3f} (fake data)"
            )


def test_lora_on_tied_gpt():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=3)
        model = ToyGPT()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        res = run(
            _bb(model),
            data_root=data,
            out_dir=os.path.join(tmp, "runs"),
            run_name="g",
            method="lora",
            **{**FAST, "epochs": 1},
        )
        trained = res["stage_a"].encoder.model
        assert isinstance(trained.lm_head, nn.Linear) and not hasattr(trained.lm_head, "base_layer")
        assert trained.lm_head.weight is trained.wte.weight
        assert torch.equal(trained.wte.weight.detach().cpu(), before["wte.weight"])
        assert res["metrics"]["stage_a"]["trainable_params"] == _expected_trainable(ToyGPT(), "lora")[0]


def test_run_methods_compare_reuse_and_failure():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=3)
        out = os.path.join(tmp, "runs")
        common = dict(data_root=data, out_dir=out, **{**FAST, "epochs": 1, "b_epochs": 1})
        bb = _bb(ToyLlama())
        res = P.run_methods(bb, methods=("frozen", "lora"), prefix="t", **common)
        assert set(res) == {"frozen", "lora"} and all(res.values())
        text = open(os.path.join(out, "t_comparison.txt")).read()
        for s in ("COST", "lora", "frozen", "Stage B, CL", "Stage A, IT", "Pet/Resp"):
            assert s in text, s
        rows = json.load(open(os.path.join(out, "t_comparison.json")))
        assert rows["frozen"]["trainable"] == HEAD < rows["lora"]["trainable"]
        assert rows["lora"]["peft"]["layers"] == 14 and os.path.exists(rows["lora"]["submission"])
        again = P.run_methods(bb, methods=("frozen", "lora"), prefix="t", **common)
        assert again["lora"]["stage_a"]["trainable_params"] == res["lora"]["stage_a"]["trainable_params"]
        bad = P.run_methods(_bb(ToyEncDec()), methods=("frozen", "lora"), prefix="ed", **common)
        assert bad["lora"] is None and bad["frozen"], "the RNN model has no Linear layers for LoRA"
        assert "failed runs: lora" in open(os.path.join(out, "ed_comparison.txt")).read()
        for kw in (dict(methods=("lorra",)), dict(method="lora")):
            try:
                P.run_methods(bb, prefix="x", **kw, **common)
                assert False, "should fail"
            except ValueError:
                pass


def test_cross_validation_with_adalora():
    with tempfile.TemporaryDirectory() as tmp:
        data = make_fake_data(os.path.join(tmp, "data"), n_train=4)
        res = run_cv(
            _bb(ToyLlama()),
            data_root=data,
            out_dir=os.path.join(tmp, "runs"),
            run_name="cv",
            method="adalora",
            folds=2,
            **{**FAST, "epochs": 2},
        )
        saved = torch.load(os.path.join(res["out_dir"], "stageA_fold1.pt"), map_location="cpu", weights_only=False)
        keys = list(saved["trained_weights"])
        assert keys and all("lora_" in k or k.split(".")[0] in ("pool", "head", "pos_emb") for k in keys), keys


if __name__ == "__main__":
    try:
        import peft

        print(f"peft {peft.__version__}, torch {torch.__version__}, {'GPU' if torch.cuda.is_available() else 'CPU'}")
    except ImportError:
        print("peft is not installed: pip install peft")
        sys.exit(1)
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
