"""Whole-case reading: the hierarchical (chunk-level) architecture of the best published CJPE systems, on our models.

The published ILDC models do not read only the end of a judgment. They split it into chunks, encode every chunk with
a fine-tuned language model, and combine the chunk vectors with a second, small model:
  ILDC paper (Malik et al. 2021): XLNet fine-tuned on 512-token chunks + BiGRU over the chunk vectors (77.1 macro-F1);
  HierBERT (Paul et al. 2022): chunk vectors of the last chunks + LSTM with attention (InLegalBERT 83.1);
  two-level encoder (Prasad et al. 2023): chunk vectors + transformer encoder over the chunks (83.7 accuracy);
  ToBERT / RoBERT (Pappagari et al. 2019): transformer / LSTM over chunk vectors.
Here Stage A is one of our SFT runs (best with train_chunks > 1, i.e. fine-tuned on chunks) and Stage B is one of:
  gru_attn     LayerNorm -> Linear -> BiGRU over the chunks -> attention pooling + last chunk -> 2 scores
  transformer  LayerNorm -> Linear + chunk position (counted from the end) -> 2 self-attention blocks (written by us,
               padding-aware) -> attention pooling + last chunk -> 2 scores
  mean         LayerNorm -> Linear -> average over the chunks -> 2 scores
and two untrained references: last (the SFT run's own prediction on the last chunk) and vote (average of the chunk
predictions). Stage B is trained on the training cases' chunk vectors, with chunk dropout (sometimes only the last
few chunks, sometimes without the last one) so it cannot rely on the last chunk alone; the epoch is chosen on dev.

    from cj_hier import run_hier
    run_hier("/kaggle/working/cj_runs_mid/llama_full", k=8)
Outputs in <run>/hier_k<k>/<version>/ (dev_probs.json, test_probs.json, metrics.json, submission_tuned/,
submission_050/) and the comparison <run>/hier_k<k>/hier.txt. Chunk vectors are cached in <run>/chunks_k<k>.pt.
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import cj_data as D
import cj_tech as T
import cj_train as R

VERSIONS = ("gru_attn", "transformer", "mean")
HIER_DEFAULTS = dict(k=8, hidden=256, dropout=0.2, lr=1e-3, weight_decay=0.01, epochs=15, patience=4, batch=64,
                     p_tail=0.3, p_drop_last=0.15, extract_batch=32, seed=42, device=None)


# --------------------------------------------------------------------------- Stage A
def rebuild(run_dir: str, bb=None, ckpt: str | None = None, device=None, log=print):
    """The SFT run's classifier, rebuilt: Phase 1 / mid-trained model + the same technique + its trained tensors."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        name = json.load(f)["model"]
    if bb is None:
        from cj_models import load_backbone
        bb, _ = load_backbone(name, max_len=cfg["max_len"], path=ckpt or cfg.get("ckpt_path"), log=log)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = R.make_model(bb, cfg, log=lambda *a: None)    # a KD student keeps only its first blocks
    clf = bb.classifier(2, model=model, head=cfg["head"], dropout=cfg["dropout"],
                        pooling=cfg["pooling"]).to(device).float()
    g = torch.Generator().manual_seed(0)
    top = max(8, min(bb.vocab or 1000, 1000))
    probe = [torch.randint(5, top, (32,), generator=g).tolist() for _ in range(4)]
    used = R.find_used(clf, probe, cfg, device, bb.pad_id)
    T.apply(clf, cfg["technique"], used, cfg, total_steps=1, log=lambda *a: None)
    state = torch.load(os.path.join(run_dir, "trained.pt"), map_location="cpu", weights_only=False)["state"]
    res = clf.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"trained.pt does not fit the rebuilt model: unexpected {res.unexpected_keys[:5]}")
    clf.eval()
    log(f"[hier] Stage A = {run_dir} ({name} + {cfg['technique']}), {len(state)} trained tensors loaded")
    return bb, clf, cfg, name


@torch.no_grad()
def chunk_vectors(clf, cases: list[list[list[int]]], cfg, device, pad_id, batch=32, log=print):
    """For every case: [n_chunks, d] chunk vectors (float16, CPU) and [n_chunks] P(label 1) of each chunk."""
    flat = [(i, j) for i, chunks in enumerate(cases) for j in range(len(chunks))]
    lens = [len(cases[i][j]) for i, j in flat]
    feats: list = [None] * len(flat)
    probs: list = [None] * len(flat)
    order = D.batches(lens, batch, shuffle=False, exact=cfg["exact_batches"])
    t0 = time.time()
    for n, b in enumerate(order, 1):
        ids, mask = R.collate([cases[flat[k][0]][flat[k][1]] for k in b], pad_id, device)
        with R._autocast(device, cfg["amp"]):
            f = clf.features(ids, mask)
            p = torch.softmax(clf.head(f).float(), -1)[:, 1]
        f, p = f.half().cpu(), p.cpu()
        for r, k in enumerate(b):
            feats[k], probs[k] = f[r], p[r].item()
        if n % 500 == 0:
            el = time.time() - t0
            left = el / n * (len(order) - n) / 60
            log(f"  chunk vectors: {n}/{len(order)} batches, {el / 60:.1f} min, about {left:.1f} min left")
    out_f, out_p, k = [], [], 0
    for chunks in cases:
        m = len(chunks)
        out_f.append(torch.stack(feats[k:k + m]))
        out_p.append(torch.tensor(probs[k:k + m]))
        k += m
    return out_f, out_p


def extract(run_dir: str, k: int = 8, bb=None, ckpt=None, device=None, batch=32, log=print) -> dict:
    """Chunk vectors of the last k chunks of every train / dev / test case (cached in <run>/chunks_k<k>.pt)."""
    path = os.path.join(run_dir, f"chunks_k{k}.pt")
    if os.path.exists(path):
        log(f"[hier] reusing {path}")
        return torch.load(path, map_location="cpu", weights_only=False)
    bb, clf, cfg, name = rebuild(run_dir, bb, ckpt, device, log)
    dev_ = next(clf.parameters()).device
    data = D.load_data(cfg["data_root"], fetch=cfg["fetch_data"], log=log)
    data["train"] = R.limit(data["train"], cfg["limit_train"], cfg["seed"])
    data["dev"] = R.limit(data["dev"], cfg["limit_dev"], cfg["seed"])
    n = cfg["max_len"] - (bb.bos_id is not None) - (bb.eos_id is not None)
    stride = cfg.get("chunk_stride", 448)
    out = dict(k=k, stride=stride, max_len=n, run=run_dir, model=name, technique=cfg["technique"])
    t0 = time.time()
    for s, ds in data.items():
        ids = D.encode_docs(ds, bb.encode, D.chunk_span(n, stride, k), "tail", workers=cfg["tok_workers"],
                            cache_dir=cfg["cache_dir"], log=log)
        wrap = lambda x: ([bb.bos_id] if bb.bos_id is not None else []) + (x or [0]) + (  # noqa: E731
            [bb.eos_id] if bb.eos_id is not None else [])
        cases = [[wrap(c) for c in D.chunk_ids(x, n, stride, k)] for x in ids]
        log(f"[hier] {s}: {len(cases)} cases, {sum(len(c) for c in cases)} chunks "
            f"(average {sum(len(c) for c in cases) / max(1, len(cases)):.1f} per case)")
        f, p = chunk_vectors(clf, cases, cfg, dev_, bb.pad_id, batch, log)
        out[s] = dict(ids=[d["id"] for d in ds], labels=[d["label"] for d in ds], feats=f, probs=p)
    out["minutes"] = (time.time() - t0) / 60
    torch.save(out, path)
    log(f"[hier] chunk vectors saved to {path} ({os.path.getsize(path) / 2 ** 20:.0f} MB, {out['minutes']:.1f} min)")
    clf.cpu()
    return out


# --------------------------------------------------------------------------- Stage B
class MaskedAttention(nn.Module):
    """Self-attention over the chunks of a case, written out by hand; padded chunks are never attended to."""

    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):                                  # x [B, K, d], mask [B, K] (True = real chunk)
        B, K, d = x.shape
        q, k, v = (f(x).view(B, K, self.h, self.dk).transpose(1, 2) for f in (self.q, self.k, self.v))
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        att = att.masked_fill(~mask[:, None, None, :], float("-inf"))
        w = torch.nan_to_num(att.softmax(-1))
        return self.o((self.drop(w) @ v).transpose(1, 2).reshape(B, K, d))


class Block(nn.Module):
    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MaskedAttention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        return x + self.ff(self.n2(x))


class HierHead(nn.Module):
    """Chunk vectors of a case [B, K, d] (right-padded, mask [B, K]) -> 2 scores."""

    def __init__(self, d: int, version: str, hidden: int = 256, k: int = 8, dropout: float = 0.2):
        super().__init__()
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}")
        self.version = version
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, hidden)
        self.drop = nn.Dropout(dropout)
        if version == "gru_attn":
            self.gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        elif version == "transformer":
            self.pos = nn.Embedding(k + 1, hidden)               # 0 = last chunk, 1 = the one before, ...
            self.blocks = nn.ModuleList([Block(hidden, 4, dropout) for _ in range(2)])
        self.score = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden * (1 if version == "mean" else 2), 2)

    def forward(self, x, mask):
        lengths = mask.sum(1)
        h = self.drop(F.gelu(self.inp(self.norm(x.float()))))
        if self.version == "mean":
            m = mask.unsqueeze(-1).float()
            return self.out(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1)))
        if self.version == "gru_attn":
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=mask.size(1))
        else:
            K = mask.size(1)
            from_end = lengths[:, None] - 1 - torch.arange(K, device=x.device)[None, :]
            from_end = from_end.clamp(0, self.pos.num_embeddings - 1)
            h = h + self.pos(from_end)
            for blk in self.blocks:
                h = blk(h, mask)
        a = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.nan_to_num(a.softmax(-1))
        pooled = (w.unsqueeze(-1) * h).sum(1)
        last = h[torch.arange(h.size(0), device=h.device), lengths - 1]
        return self.out(self.drop(torch.cat([pooled, last], -1)))


def _pad(feats, idx, device, keep=None):
    """Right-padded batch [B, K, d] + mask for cases idx; keep[i] = (start, end) chunk range to use."""
    sel = [feats[i] if keep is None else feats[i][keep[j][0]:keep[j][1]] for j, i in enumerate(idx)]
    K = max(len(s) for s in sel)
    d = sel[0].size(-1)
    x = torch.zeros(len(sel), K, d, dtype=torch.float16)
    mask = torch.zeros(len(sel), K, dtype=torch.bool)
    for j, s in enumerate(sel):
        x[j, :len(s)] = s
        mask[j, :len(s)] = True
    return x.to(device), mask.to(device)


@torch.no_grad()
def predict_b(model, feats, device, batch=256):
    model.eval()
    out = []
    for i in range(0, len(feats), batch):
        idx = list(range(i, min(i + batch, len(feats))))
        x, m = _pad(feats, idx, device)
        out += torch.softmax(model(x, m).float(), -1)[:, 1].cpu().tolist()
    return out


def train_b(version, ch: dict, cfg: dict, device, log=print):
    rng = random.Random(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    tr, dv = ch["train"], ch["dev"]
    d = tr["feats"][0].size(-1)
    model = HierHead(d, version, cfg["hidden"], ch["k"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    w = torch.tensor(D.class_weights(tr["labels"]), dtype=torch.float32, device=device)
    y = torch.tensor(tr["labels"], dtype=torch.long)
    best, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        order = list(range(len(tr["feats"])))
        rng.shuffle(order)
        tot, n = 0.0, 0
        for i in range(0, len(order), cfg["batch"]):
            idx = order[i:i + cfg["batch"]]
            keep = []
            for c in idx:                                          # chunk dropout
                m = len(tr["feats"][c])
                s, e = 0, m
                if m > 1 and rng.random() < cfg["p_tail"]:
                    s = m - rng.randint(1, m - 1)                  # only the last few chunks
                if e - s > 1 and rng.random() < cfg["p_drop_last"]:
                    e -= 1                                         # without the last chunk
                keep.append((s, e))
            x, mask = _pad(tr["feats"], idx, device, keep)
            loss = F.cross_entropy(model(x, mask).float(), y[idx].to(device), weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        p = predict_b(model, dv["feats"], device)
        t, f = D.tune_threshold(dv["labels"], p)
        hist.append(dict(epoch=ep, loss=tot / max(1, n), dev_f1_tuned=f, threshold=t,
                         dev_f1_050=D.report(dv["labels"], p, 0.5)["macro_f1"]))
        mark = ""
        if f > best:
            best, best_ep, bad, mark = f, ep, 0, "  <- best so far"
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"[hier {version}] epoch {ep}  loss {tot / max(1, n):.4f}  dev macro-F1 {f:.4f} at {t:.2f}{mark}")
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model, dict(history=hist, best_epoch=best_ep)


def _write(out_dir, ch, p_dev, p_test, extra, data_test, p_train=None):
    os.makedirs(out_dir, exist_ok=True)
    if p_train is not None:                                   # for knowledge distillation (teacher probabilities)
        with open(os.path.join(out_dir, "train_probs.json"), "w") as f:
            json.dump(dict(zip(ch["train"]["ids"], p_train)), f)
    y = ch["dev"]["labels"]
    t, _ = D.tune_threshold(y, p_dev)
    r50, rt = D.report(y, p_dev, 0.5), D.report(y, p_dev, t)
    with open(os.path.join(out_dir, "dev_probs.json"), "w") as f:
        json.dump(dict(zip(ch["dev"]["ids"], p_dev)), f)
    with open(os.path.join(out_dir, "test_probs.json"), "w") as f:
        json.dump(dict(zip(ch["test"]["ids"], p_test)), f)
    subs = {}
    for key, thr in (("tuned", t), ("050", 0.5)):
        preds = {i: int(p >= thr) for i, p in zip(ch["test"]["ids"], p_test)}
        subs[key] = D.write_submission(data_test, preds, os.path.join(out_dir, f"submission_{key}"))
    m = dict(dev_050=r50, dev_tuned=rt, threshold=t, submissions=subs, **extra)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1, default=str)
    return m


def run_hier(run_dir: str, versions=VERSIONS, bb=None, ckpt=None, log=print, **overrides) -> dict:
    """Chunk vectors from the SFT run in run_dir, then every Stage B version; returns {version: metrics}."""
    cfg = {**HIER_DEFAULTS, **overrides}
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    ch = extract(run_dir, cfg["k"], bb, ckpt, device, cfg["extract_batch"], log)
    with open(os.path.join(run_dir, "config.json")) as f:
        rcfg = json.load(f)
    test_docs = D.load_data(rcfg["data_root"], fetch=rcfg["fetch_data"])["test"]
    base = os.path.join(run_dir, f"hier_k{cfg['k']}")
    res = {}
    refs = {"last": lambda p: p[-1].item(), "vote": lambda p: p.mean().item()}
    for name, fn in refs.items():
        res[name] = _write(os.path.join(base, name), ch, [fn(p) for p in ch["dev"]["probs"]],
                           [fn(p) for p in ch["test"]["probs"]], dict(version=name, trained=False), test_docs,
                           [fn(p) for p in ch["train"]["probs"]])
    for v in versions:
        t0 = time.time()
        model, info = train_b(v, ch, cfg, device, log)
        res[v] = _write(os.path.join(base, v), ch, predict_b(model, ch["dev"]["feats"], device),
                        predict_b(model, ch["test"]["feats"], device),
                        dict(version=v, trained=True, minutes=(time.time() - t0) / 60,
                             params=sum(p.numel() for p in model.parameters()), **info), test_docs,
                        predict_b(model, ch["train"]["feats"], device))
    lines = [f"Whole-case reading on {run_dir} (last {ch['k']} chunks, stride {ch['stride']}):",
             f"{'version':14s} {'dev F1@0.5':>11s} {'dev F1@t':>9s} {'t':>5s} {'P0':>6s} {'R0':>6s} {'P1':>6s} "
             f"{'R1':>6s}"]
    for v, m in sorted(res.items(), key=lambda kv: -kv[1]["dev_tuned"]["macro_f1"]):
        t = m["dev_tuned"]
        lines.append(f"{v:14s} {m['dev_050']['macro_f1']:11.4f} {t['macro_f1']:9.4f} {t['threshold']:5.2f} "
                     f"{t['class0']['precision']:6.3f} {t['class0']['recall']:6.3f} {t['class1']['precision']:6.3f} "
                     f"{t['class1']['recall']:6.3f}")
    best = max(res, key=lambda v: res[v]["dev_tuned"]["macro_f1"])
    lines.append(f"'last' = the SFT run itself (last chunk only); 'vote' = average of the chunk predictions.\n"
                 f"best on dev: {best} -> submit {res[best]['submissions']['tuned']}")
    text = "\n".join(lines)
    with open(os.path.join(base, "hier.txt"), "w") as f:
        f.write(text + "\n")
    log(text)
    return res


def run_hier_many(run_dirs, gpus: int | None = None, poll: int = 60, **overrides) -> None:
    """run_hier for several SFT runs at once, one process per GPU (python cj_hier.py --run DIR ...)."""
    import subprocess
    import sys
    n_gpu = torch.cuda.device_count() if gpus is None else gpus
    here = os.path.dirname(os.path.abspath(__file__))
    queue = list(run_dirs)
    procs = []
    while queue or procs:
        while queue and len(procs) < max(1, n_gpu):
            busy = {g for _, g, _, _ in procs}
            gpu = next(g for g in range(max(1, n_gpu)) if g not in busy)
            run = queue.pop(0)
            log_path = os.path.join(run, "hier_log.txt")
            env = dict(os.environ)
            if n_gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            fh = open(log_path, "w")
            p = subprocess.Popen([sys.executable, "-u", os.path.join(here, "cj_hier.py"), "--run", run,
                                  "--settings", json.dumps(overrides)], stdout=fh, stderr=subprocess.STDOUT,
                                 env=env, cwd=here)
            procs.append((p, gpu, fh, run))
            print(f"[hier] {run} on GPU {gpu} (log: {log_path})", flush=True)
        time.sleep(poll)
        for p, gpu, fh, run in list(procs):
            try:
                with open(os.path.join(run, "hier_log.txt"), encoding="utf-8", errors="replace") as f:
                    lines = [ln.rstrip() for ln in f if ln.strip()]
            except OSError:
                lines = []
            state = "running" if p.poll() is None else f"finished (exit {p.returncode})"
            print(f"[hier] GPU {gpu} {os.path.basename(run)} {state}: {lines[-1][:140] if lines else ''}", flush=True)
            if p.poll() is not None:
                fh.close()
                procs.remove((p, gpu, fh, run))
                if p.returncode != 0:
                    print("\n".join(lines[-25:]))
    for run in run_dirs:
        k = overrides.get("k", HIER_DEFAULTS["k"])
        path = os.path.join(run, f"hier_k{k}", "hier.txt")
        if os.path.exists(path):
            print("\n" + open(path).read())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--settings", default="{}")
    a = ap.parse_args()
    s = json.loads(a.settings)
    versions = tuple(s.pop("versions", VERSIONS))
    run_hier(a.run, versions=versions, **s)
