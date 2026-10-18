"""Self-test of the Phase 4 techniques and training on tiny stand-in models (GPT-style with biases and a tied LM
head, Llama-style without biases, encoder-decoder with a bidirectional GRU encoder) and fake cases whose last words
give the outcome. Needs torch (no peft). About 3-5 minutes.     python test_cj_train.py  -> every line PASS

Checks: which weights each technique trains; LoRMA, AdaLoRA, adapters and added BitFit biases leave the model's
output unchanged at the start and do change it once trained (also through the GRU); the unfreezing schedule;
AdaLoRA ends with the target rank budget; full runs write complete submissions, probabilities and metrics, and a
full fine-tune learns the fake task; the grid helpers.
"""
import json
import math
import os
import sys
import tempfile
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cj_data as D  # noqa: E402
import cj_grid as G  # noqa: E402
import cj_tech as T  # noqa: E402
import cj_train as R  # noqa: E402
from cj_models import make_backbone  # noqa: E402
from test_cj_data import make_fake_data  # noqa: E402

torch.manual_seed(0)
V, D_ = 600, 32
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------ toy tokenizer
class WordTok:
    special = {"<pad>": 0, "<s>": 1, "</s>": 2}

    def encode(self, text):
        return [3 + sum(map(ord, w)) % (V - 3) for w in text.split()]

    def token_to_id(self, t):
        return self.special.get(t)


# --------------------------------------------------------------------------- toy models
class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        T_ = x.size(1)
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        att = q @ k.transpose(1, 2) / math.sqrt(x.size(-1))
        att = att.masked_fill(torch.triu(torch.ones(T_, T_, device=x.device), 1).bool(), float("-inf"))
        x = x + self.proj(att.softmax(-1) @ v)
        return x + self.ff(self.ln2(x))


class ToyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, D_), nn.Embedding(256, D_)
        self.h = nn.ModuleList([Block(D_) for _ in range(3)])
        self.ln_f = nn.LayerNorm(D_)
        self.lm_head = nn.Linear(D_, V, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.h:
            x = b(x)
        return self.lm_head(self.ln_f(x))


class LlamaBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention_norm, self.ffn_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.w1, self.w3 = nn.Linear(d, 48, bias=False), nn.Linear(d, 48, bias=False)
        self.w2 = nn.Linear(48, d, bias=False)

    def forward(self, x):
        h = self.attention_norm(x)
        T_ = x.size(1)
        att = self.wq(h) @ self.wk(h).transpose(1, 2) / math.sqrt(x.size(-1))
        att = att.masked_fill(torch.triu(torch.ones(T_, T_, device=x.device), 1).bool(), float("-inf"))
        x = x + self.wo(att.softmax(-1) @ self.wv(h))
        h = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class ToyLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(V, D_)
        self.layers = nn.ModuleList([LlamaBlock(D_) for _ in range(3)])
        self.norm = nn.LayerNorm(D_)
        self.output = nn.Linear(D_, V, bias=False)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))


class ToyEncDec(nn.Module):
    """Embedding shared by encoder and decoder, 2-layer bidirectional GRU encoder, forward(src, tgt)."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D_)
        self.encoder = nn.GRU(D_, D_ // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(D_, D_, batch_first=True)
        self.lm_head = nn.Linear(D_, V)

    def forward(self, src, tgt):
        enc, _ = self.encoder(self.emb(src))
        dec, _ = self.decoder(self.emb(tgt) + enc.mean(1, keepdim=True))
        return self.lm_head(dec)


TOYS = {"gpt": ToyGPT, "llama": ToyLlama, "encdec": ToyEncDec}
FAST = dict(epochs=2, batch_size=8, eval_batch_size=16, max_len=32, log_every=0, fit_sample=20, tok_workers=1,
            lr_head=3e-3, fetch_data=False)


def _bb(kind):
    torch.manual_seed(1)
    return make_backbone(TOYS[kind](), WordTok(), max_len=32, verbose=False)


def _setup(kind, tmp):
    """(bb, clf on DEV, used, train seqs, cfg) for a toy model on fake data."""
    root = make_fake_data(tmp, n_train=24, n_dev=8, n_test=6)
    bb = _bb(kind)
    cfg = R.resolve_cfg(dict(FAST, data_root=root, cache_dir=os.path.join(tmp, "cache"), out_dir=tmp))
    cfg["exact_batches"] = bool(bb.pad_effect > 1e-4)
    data, seqs = R.prepare_tokens(bb.encode, cfg, log=lambda *a: None)
    clf = bb.classifier(2, model=__import__("copy").deepcopy(bb.model)).to(DEV).float()
    used = R.find_used(clf, seqs["train"], cfg, DEV, bb.pad_id)
    return bb, clf, used, seqs, cfg, data


def _logits(clf, seqs, pad_id):
    clf.eval()
    with torch.no_grad():
        ids, mask = R.collate(seqs[:4], pad_id, DEV)
        return clf(ids, mask).float()


def _trained_names(clf):
    return {n for n, p in clf.encoder.model.named_parameters() if p.requires_grad}


# --------------------------------------------------------------------------------- tests
def test_structure():
    with tempfile.TemporaryDirectory() as tmp:
        for kind, (n_layers, layer_kind) in {"gpt": (3, "blocks"), "llama": (3, "blocks"),
                                              "encdec": (2, "RNN layers")}.items():
            bb, clf, used, *_ = _setup(kind, os.path.join(tmp, kind))
            model = clf.encoder.model
            bottom, layers, top, k = T.layer_groups(model, used)
            assert (len(layers), k) == (n_layers, layer_kind), (kind, len(layers), k)
            if kind == "encdec":
                assert bb.kind == "encoder" and bb.pad_effect > 1e-4
                assert not any(n.startswith(("decoder.", "lm_head.")) for n in used)       # decoder never used
                assert bottom == ["emb.weight"] and top == []
                assert all(n.startswith("encoder.") and n.endswith(("_l0", "_l0_reverse")) for n in layers[0])
                assert len(T.targets(model, used)) == 8                                     # ih, hh x 2 layers x 2 dirs
            else:
                assert bb.kind == "decoder"
                assert all(b[0].startswith(("h.0.", "layers.0.")) for b in layers[:1])
                assert top and all(n.startswith(("ln_f", "norm")) for n in top)
            if kind == "gpt":
                assert "lm_head.weight" not in [f"{m}.{p}" for m, _, p in T.targets(model, used)]   # tied: skipped


def test_trainable_sets():
    cfg = dict(T.TECH_DEFAULTS)
    with tempfile.TemporaryDirectory() as tmp:
        for kind in TOYS:
            for tech in T.TECHNIQUES:
                bb, clf, used, seqs, _, _ = _setup(kind, os.path.join(tmp, f"{kind}_{tech}"))
                before = _logits(clf, seqs["train"], bb.pad_id)
                h = T.apply(clf, tech, used, cfg, total_steps=30, log=lambda *a: None)
                after = _logits(clf, seqs["train"], bb.pad_id)
                assert torch.allclose(before, after, atol=1e-4), (kind, tech, (before - after).abs().max())
                model = clf.encoder.model
                names = _trained_names(clf)
                bottom, layers, top, _ = T.layer_groups(model, used) if tech not in ("lorma", "adalora") else (
                    None, None, None, None)
                if tech in ("frozen", "unfreeze"):
                    assert names == set(), (kind, tech, names)                  # unfreeze starts with the head only
                elif tech == "full":
                    assert names == used
                elif tech == "partial":
                    assert names == set(layers[-1]) | set(top), (kind, names)  # ceil(L / 3) = 1 layer for L = 2, 3
                elif tech == "bitfit":
                    assert names and all(n.split(".")[-1].startswith("bias") for n in names), names
                    if kind == "llama":
                        assert h.info["added_biases"] == 21 and h.info["existing_biases"] == 7, h.info
                elif tech in ("lorma", "adalora"):
                    want = ("A", "B") if tech == "lorma" else ("P", "lam", "Q")
                    assert names and all(n.split(".")[-1] in want for n in names), names
                    assert not any("original" in n for n in names)
                elif tech == "adapter":
                    assert names == set() and len(h.body_params) > 0
                    assert all(p.requires_grad for p in h.body_params)
                assert all(p.requires_grad for p in h.body_params)


def test_updates_change_the_model():
    """After one optimiser step, every technique's model output changes (also through the parametrized GRU)."""
    with tempfile.TemporaryDirectory() as tmp:
        for kind in TOYS:
            for tech in ("lorma", "adalora", "adapter", "bitfit", "partial"):
                bb, clf, used, seqs, _, _ = _setup(kind, os.path.join(tmp, f"{kind}_{tech}"))
                h = T.apply(clf, tech, used, dict(T.TECH_DEFAULTS), total_steps=30, log=lambda *a: None)
                before = _logits(clf, seqs["train"], bb.pad_id)
                opt = torch.optim.Adam(h.body_params, lr=1e-2)
                clf.train()
                for _ in range(2):          # AdaLoRA / LoRMA: the first step moves lambda / B, the second the rest
                    ids, mask = R.collate(seqs["train"][:4], bb.pad_id, DEV)
                    loss = F.cross_entropy(clf(ids, mask).float(), torch.tensor([0, 1, 0, 1], device=DEV))
                    extra = h.extra_loss()
                    if extra is not None:
                        loss = loss + extra
                    opt.zero_grad()
                    loss.backward()
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in h.body_params), (kind, tech)
                    opt.step()
                after = _logits(clf, seqs["train"], bb.pad_id)
                assert (before - after).abs().max() > 1e-6, (kind, tech)


def test_unfreeze_schedule():
    with tempfile.TemporaryDirectory() as tmp:
        bb, clf, used, *_ = _setup("llama", tmp)
        h = T.apply(clf, "unfreeze", used, dict(T.TECH_DEFAULTS, unfreeze_stages=4), total_steps=8,
                    log=lambda *a: None)
        model = clf.encoder.model
        bottom, layers, top, _ = T.layer_groups(model, used)
        seen = []
        for step in range(8):
            h.on_step_start(step)
            seen.append(_trained_names(clf))
        assert seen[0] == set() and seen[1] == set()                              # stage 0: steps 0-1
        assert seen[2] == set(layers[2]) | set(top)                                # stage 1: top layer
        assert seen[4] == set(layers[1]) | set(layers[2]) | set(top)               # stage 2: two layers
        assert seen[6] == used and seen[7] == used                                 # stage 3: all + embeddings
        assert [s["stage"] for s in h.info["schedule"]] == [0, 1, 2, 3]


def test_adalora_budget():
    with tempfile.TemporaryDirectory() as tmp:
        bb, clf, used, seqs, _, _ = _setup("gpt", tmp)
        cfg = dict(T.TECH_DEFAULTS, adalora_r0=6, adalora_r=2, adalora_deltaT=2)
        total = 40
        h = T.apply(clf, "adalora", used, cfg, total_steps=total, log=lambda *a: None)
        n = len(h.mods)
        opt = torch.optim.Adam(h.body_params, lr=1e-2)
        budgets = []
        clf.train()
        for step in range(1, total + 1):
            ids, mask = R.collate(seqs["train"][(step % 6) * 4:(step % 6) * 4 + 4], bb.pad_id, DEV)
            y = torch.tensor([0, 1, 0, 1], device=DEV)
            loss = F.cross_entropy(clf(ids, mask).float(), y) + h.extra_loss()
            opt.zero_grad()
            loss.backward()
            opt.step()
            h.after_step(step, stepped=True)
            budgets.append(h.budget)
        kept = sum(int(m.mask.sum()) for _, m in h.mods)
        assert budgets[0] == 6 * n and budgets[-1] == 2 * n and kept == 2 * n, (budgets[0], budgets[-1], kept)
        assert all(a >= b for a, b in zip(budgets, budgets[1:]))                 # the budget only shrinks
        s = h.summary()
        assert s["ranks_kept"] == 2 * n and s["final_rank_mean_by_layer_type"]


def _check_run(res, root, tech):
    m = res["metrics"]
    out = res["out_dir"]
    data = D.load_data(root, fetch=False)
    for key in ("tuned", "050"):
        with zipfile.ZipFile(os.path.join(out, f"submission_{key}", "submission.zip")) as z:
            rows = json.loads(z.read("prediction.json"))
        assert [r["id"] for r in rows] == [d["id"] for d in data["test"]]
        assert all(r["pred"] in (0, 1) and type(r["pred"]) is int for r in rows)
    probs = json.load(open(os.path.join(out, "dev_probs.json")))
    assert set(probs) == {d["id"] for d in data["dev"]} and all(0 <= v <= 1 for v in probs.values())
    assert set(json.load(open(os.path.join(out, "test_probs.json")))) == {d["id"] for d in data["test"]}
    saved = torch.load(os.path.join(out, "trained.pt"), map_location="cpu", weights_only=False)["state"]
    assert any(k.startswith("head.") for k in saved)
    if tech in ("lorma", "adalora", "adapter", "bitfit", "frozen"):
        assert not any("original" in k for k in saved) and m["trainable_share"] < 0.6, (tech, m["trainable_share"])
    for k in ("dev_050", "dev_tuned", "threshold", "train_fit", "trainable_params", "train_seconds",
              "dev_infer_seconds", "history", "technique_info"):
        assert k in m, k
    assert 0.0 <= m["dev_tuned"]["macro_f1"] <= 1.0 and m["dev_tuned"]["macro_f1"] >= m["dev_050"]["macro_f1"] - 1e-9


def test_runs_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_fake_data(tmp, n_train=40, n_dev=12, n_test=10)
        common = dict(FAST, data_root=root, out_dir=os.path.join(tmp, "runs"), cache_dir=os.path.join(tmp, "cache"))
        for kind in TOYS:
            bb = _bb(kind)
            techs = T.TECHNIQUES if kind == "llama" else ("full", "lorma", "adapter", "unfreeze")
            for tech in techs:
                res = R.run(bb, name=kind, technique=tech, **common)
                _check_run(res, root, tech)
        # a full fine-tune learns the fake task (the outcome is in the last words)
        res = R.run(_bb("llama"), name="learn", technique="full", **dict(common, epochs=8, lr_body=3e-3))
        f = res["metrics"]["dev_tuned"]["macro_f1"]
        print(f"      toy Llama, full fine-tuning: dev macro-F1 {f:.3f}")
        assert f >= 0.8, f
        rows = [R.load_metrics(common["out_dir"], f"llama_{t}") for t in T.TECHNIQUES]
        table = R.comparison(rows)
        assert all(f"llama_{t}" in table for t in T.TECHNIQUES)
        text = G.compare_all(common["out_dir"])
        assert "learn_full" in text and os.path.exists(os.path.join(common["out_dir"], "all_runs.txt"))
        try:
            R.run(bb, name="x", technique="nope", **common)
            raise AssertionError("an unknown technique should fail")
        except ValueError:
            pass


def test_split_jobs():
    jobs = G.split_jobs(T.TECHNIQUES, 2)
    assert sorted(t for j in jobs for t in j) == sorted(T.TECHNIQUES) and len(jobs) == 2
    loads = [sum(G.COST[t] for t in j) for j in jobs]
    assert abs(loads[0] - loads[1]) <= 1.0, loads
    assert G.split_jobs(["full"], 2) == [["full"]]


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
