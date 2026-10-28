"""Phase 5 post-training: whole-document reading (Stage B) on top of a fine-tuned run.

A fine-tuned run (lsi_train, save_chunks=True) saves, for every train / dev / test document, the vector of each chunk
(up to eval_chunks chunks spread over the first doc_tokens tokens) and that chunk's statute scores (chunks.pt).
Stage B is a small network trained on them; the Phase 1 model is not needed again:

    chunk vectors -> LayerNorm + Linear (d -> hidden) + GELU  [-> BiGRU over the chunks, in document order]
    -> label-wise attention: every statute has its own query vector and attends over the chunks (as in CAML / LAAT
       for long multi-label documents), giving one document vector per statute -> one score per statute
    [+ alpha x the run's own score (maximum over the chunks): "_res" versions start exactly as the run]
    [+ co-occurrence layer: scores += W sigmoid(scores); W (statutes x statutes) starts at 0: "_co" versions]

Versions: attn (no GRU), gru_attn, gru_attn_res, gru_attn_co. The run's scores of TRAINING documents are in-sample
(the run was trained on them), so the "_res" version may lean on them too much; the others learn from the chunk
vectors alone (as Phase 4's whole-case reading, +4.8 dev macro-F1 there). Loss as in the run (BCE with pos_weight,
or the asymmetric loss); chunk dropout on training documents; the epoch with the best dev macro-F1 (global
threshold) is kept. Thresholds as in lsi_train (global or per statute, chosen by cross-fitting on dev).

    from lsi_hier import run_stage_b
    run_stage_b("/kaggle/working/lsi_runs/llama_full")     # -> <run>/stageB/<version>/ + <run>/stageB/hier.txt

Each version folder has labels.json, dev / test / train probabilities (train = in-sample, used as a KD teacher),
metrics.json, per_label.json, submission/ (chosen thresholds) and submission_global/.
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import p5_paths  # noqa: F401  (Phase 4 building blocks)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import lsi_data as L  # noqa: E402

VERSIONS = ("attn", "gru_attn", "gru_attn_res", "gru_attn_co")
HIER_DEFAULTS = dict(hidden=256, dropout=0.2, lr=1e-3, weight_decay=0.01, epochs=15, patience=3, batch=64,
                     p_drop=0.3, seed=42, device=None, eval_batch=256)


class StageB(nn.Module):
    def __init__(self, d: int, n_lab: int, version: str = "gru_attn", hidden: int = 256, dropout: float = 0.2,
                 prior_logit=None):
        super().__init__()
        if version not in VERSIONS:
            raise ValueError(f"version must be one of {VERSIONS}")
        self.version = version
        self.use_gru = version.startswith("gru")
        self.use_co = version.endswith("_co")
        self.use_res = version.endswith("_res")
        self.inp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        if self.use_gru:
            self.rnn = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.att = nn.Linear(hidden, hidden)
        self.query = nn.Parameter(torch.randn(n_lab, hidden) * hidden ** -0.5)
        self.out_w = nn.Parameter(torch.zeros(n_lab, hidden))      # 0 at the start ("_res": Stage B = the run)
        b0 = torch.zeros(n_lab) if (self.use_res or prior_logit is None) else torch.as_tensor(prior_logit).float()
        self.out_b = nn.Parameter(b0.clone())                        # else: each statute's training frequency
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=self.use_res)
        if self.use_co:
            self.co = nn.Linear(n_lab, n_lab)
            nn.init.zeros_(self.co.weight)
            nn.init.zeros_(self.co.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """feats [B, K, d], logits [B, K, statutes] (the run's chunk scores), mask [B, K] (True = real chunk)."""
        g = self.inp(feats)
        if self.use_gru:
            lengths = mask.sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(g, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            g, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=feats.size(1))
        u = torch.tanh(self.att(g))
        s = torch.einsum("bkh,lh->blk", u, self.query).masked_fill(~mask[:, None, :], float("-inf"))
        a = torch.softmax(s, -1)
        doc = torch.einsum("blk,bkh->blh", a, self.drop(g))
        z = (doc * self.out_w).sum(-1) + self.out_b
        if self.use_res:
            zmax = logits.masked_fill(~mask[..., None], float("-inf")).max(1).values
            z = self.alpha * zmax + z
        if self.use_co:
            z = z + self.co(torch.sigmoid(z))
        return z


# ------------------------------------------------------------------------------ batches
def make_batch(feats: list, logits: list, idx: list[int], device, p_drop: float = 0.0, rng=None):
    """Padded tensors for documents idx; with p_drop > 0 every chunk is dropped with that probability (at least one
    chunk per document is kept)."""
    fs, zs = [], []
    for i in idx:
        f, z = feats[i], logits[i]
        if p_drop > 0 and len(f) > 1:
            keep = [j for j in range(len(f)) if rng.random() >= p_drop] or [rng.randrange(len(f))]
            f, z = f[keep], z[keep]
        fs.append(f)
        zs.append(z)
    K = max(len(f) for f in fs)
    B, d, n_lab = len(fs), fs[0].shape[1], zs[0].shape[1]
    F_ = torch.zeros(B, K, d)
    Z = torch.zeros(B, K, n_lab)
    M = torch.zeros(B, K, dtype=torch.bool)
    for r, (f, z) in enumerate(zip(fs, zs)):
        F_[r, :len(f)] = f.float()
        Z[r, :len(z)] = z.float()
        M[r, :len(f)] = True
    return F_.to(device), Z.to(device), M.to(device)


@torch.no_grad()
def predict_b(model: StageB, feats: list, logits: list, device, batch: int = 256) -> np.ndarray:
    was = model.training
    model.eval()
    out = []
    for i in range(0, len(feats), batch):
        F_, Z, M = make_batch(feats, logits, list(range(i, min(i + batch, len(feats)))), device)
        out.append(torch.sigmoid(model(F_, Z, M)).cpu().numpy())
    model.train(was)
    return np.concatenate(out)


def max_probs(logits: list) -> np.ndarray:
    """The run's own document probabilities: maximum chunk score per statute."""
    return np.stack([torch.sigmoid(z.float().max(0).values).numpy() for z in logits])


# ------------------------------------------------------------------------------ training
def train_b(version: str, ch: dict, Y_tr: np.ndarray, Y_dev: np.ndarray, run_cfg: dict, cfg: dict, device,
            log=print):
    """Trains one Stage B version; returns (model, info)."""
    import lsi_train as S
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    tr, dv = ch["train"], ch["dev"]
    d, n_lab = tr["feats"][0].shape[1], tr["logits"][0].shape[1]
    prior = (Y_tr.sum(0) + 0.5) / (len(Y_tr) + 1.0)
    model = StageB(d, n_lab, version, cfg["hidden"], cfg["dropout"], np.log(prior / (1 - prior))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pos = Y_tr.sum(0)
    pw = np.minimum(((len(Y_tr) - pos) / np.maximum(pos, 1)) ** run_cfg.get("pos_weight_power", 0.5),
                    run_cfg.get("pos_weight_max", 30.0))
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    use_asl = run_cfg.get("loss", "bce") == "asl"
    P0 = predict_b(model, dv["feats"], dv["logits"], device, cfg["eval_batch"])
    t0_, best = L.tune_global(Y_dev, P0)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, history, bad = 0, [dict(epoch=0, dev_macro_f1=best, threshold=t0_)], 0
    log(f"[stageB {version}] epoch 0 ({'= the run itself' if model.use_res else 'statute frequencies only'}): "
        f"dev macro-F1 {best:.4f} at threshold {t0_:.2f}")
    order = list(range(len(Y_tr)))
    t_start = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        rng.shuffle(order)
        tot, n = 0.0, 0
        for i in range(0, len(order), cfg["batch"]):
            idx = order[i:i + cfg["batch"]]
            F_, Z, M = make_batch(tr["feats"], tr["logits"], idx, device, cfg["p_drop"], rng)
            y = torch.tensor(Y_tr[idx], dtype=torch.float32, device=device)
            z = model(F_, Z, M)
            if use_asl:
                loss = S.asl_loss(z, y, run_cfg.get("asl_gamma_neg", 4.0), run_cfg.get("asl_gamma_pos", 0.0),
                                  run_cfg.get("asl_clip", 0.05))
            else:
                loss = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        P = predict_b(model, dv["feats"], dv["logits"], device, cfg["eval_batch"])
        t, m = L.tune_global(Y_dev, P)
        history.append(dict(epoch=ep, loss=tot / max(1, n), dev_macro_f1=m, threshold=t))
        mark = ""
        if m > best:
            best, best_epoch, bad, mark = m, ep, 0, "  <- best so far"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"[stageB {version}] epoch {ep}  loss {tot / max(1, n):.4f}  dev macro-F1 {m:.4f} at {t:.2f}{mark}")
        if bad >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    info = dict(version=version, best_epoch=best_epoch, history=history, minutes=(time.time() - t_start) / 60,
                params=sum(p.numel() for p in model.parameters()), alpha=float(model.alpha.item()))
    return model, info


def load_chunks(run_dir: str) -> dict:
    path = os.path.join(run_dir, "chunks.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: train the run with save_chunks=True")
    return torch.load(path, map_location="cpu", weights_only=False)


def _labels_of(run_cfg: dict, ch: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    data = L.load_data(run_cfg["data_root"], fetch=run_cfg.get("fetch_data", True))
    if run_cfg.get("limit_train"):
        data["train"] = data["train"][:run_cfg["limit_train"]]
    for s in ("train", "dev", "test"):
        if [d["id"] for d in data[s]] != ch["splits"][s]["ids"]:
            raise ValueError(f"chunks.pt {s} documents differ from the data (other data or settings?)")
    return data, L.multi_hot(data["train"], ch["labels"]), L.multi_hot(data["dev"], ch["labels"])


def _write(out_dir, data, labels, P_dev, P_test, P_train, Y_dev, extra, min_pos, seed):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    ids = {s: [d["id"] for d in data[s]] for s in data}
    L.save_probs(os.path.join(out_dir, "dev_probs.json"), ids["dev"], P_dev)
    L.save_probs(os.path.join(out_dir, "test_probs.json"), ids["test"], P_test)
    if P_train is not None:
        L.save_probs(os.path.join(out_dir, "train_probs.json"), ids["train"], P_train)
    th = L.choose_thresholds(Y_dev, P_dev, min_pos, seed)
    subs = dict(chosen=L.write_submission(data["test"], labels, P_test, th["thresholds"],
                                          os.path.join(out_dir, "submission")),
                global_=L.write_submission(data["test"], labels, P_test, th["t_global"],
                                           os.path.join(out_dir, "submission_global")))
    with open(os.path.join(out_dir, "per_label.json"), "w") as f:
        json.dump(dict(labels=labels, dev_f1=th["dev_chosen"]["per_label_f1"], dev_support=th["dev_chosen"]["support"],
                       thresholds=[float(x) for x in th["thresholds"]]), f)
    m = dict(threshold_kind=th["kind"], t_global=th["t_global"], cv_global=th["cv_global"],
             cv_per_label=th["cv_per_label"], dev_global=L.short(th["dev_global"]),
             dev_chosen=L.short(th["dev_chosen"]), submissions=subs, **extra)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1, default=str)
    return m


def run_stage_b(run_dir: str, versions=VERSIONS, log=print, **overrides) -> dict:
    """Every Stage B version on one fine-tuned run; returns {version: metrics}; writes <run>/stageB/hier.txt."""
    unknown = set(overrides) - set(HIER_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = {**HIER_DEFAULTS, **overrides}
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(os.path.join(run_dir, "config.json")) as f:
        run_cfg = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        run_m = json.load(f)
    ch = load_chunks(run_dir)
    labels = ch["labels"]
    data, Y_tr, Y_dev = _labels_of(run_cfg, ch)
    sp = ch["splits"]
    n_chunks = {s: sum(len(x) for x in sp[s]["feats"]) for s in sp}
    log(f"[stageB] {run_dir}: {len(labels)} statutes; chunks per document (mean): "
        + ", ".join(f"{s} {n_chunks[s] / max(1, len(sp[s]['feats'])):.2f}" for s in ("train", "dev", "test")))
    base = os.path.join(run_dir, "stageB")
    res = {}
    for v in versions:
        model, info = train_b(v, sp, Y_tr, Y_dev, run_cfg, cfg, device, log)
        P = {s: predict_b(model, sp[s]["feats"], sp[s]["logits"], device, cfg["eval_batch"]) for s in sp}
        res[v] = _write(os.path.join(base, v), data, labels, P["dev"], P["test"], P["train"], Y_dev,
                        dict(run=run_dir, model=run_m.get("model"), technique=run_m.get("technique"), **info),
                        run_cfg.get("min_pos", 3), run_cfg.get("seed", 42))
        log(L.fmt(res[v]["dev_chosen"], f"[stageB {v}] dev ({res[v]['threshold_kind']} thresholds): ")
            + f"; cross-fitted {max(res[v]['cv_global'], res[v]['cv_per_label']):.4f}")
        del model
    text = table(run_m, res, run_dir)
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "hier.txt"), "w") as f:
        f.write(text + "\n")
    log(text)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def table(run_m: dict, res: dict, run_dir: str) -> str:
    hdr = (f"{'version':18s} {'params':>9s} {'macro@t':>8s} {'macro':>7s} {'cv':>7s} {'thr':>9s} {'micro':>7s} "
           f"{'subset':>7s} {'ep':>3s} {'min':>6s}")
    lines = [f"Whole-document reading (Stage B) on {run_dir}:", hdr, "-" * len(hdr)]

    def row(name, m, params, ep, minutes):
        thr = f"g {m['t_global']:.2f}" if m["threshold_kind"] == "global" else "per-stat"
        return (f"{name:18s} {params:>9s} {m['dev_global']['macro_f1']:8.4f} {m['dev_chosen']['macro_f1']:7.4f} "
                f"{max(m['cv_global'], m['cv_per_label']):7.4f} {thr:>9s} {m['dev_chosen']['micro_f1']:7.4f} "
                f"{m['dev_chosen']['subset_acc']:7.4f} {ep:>3} {minutes:6.1f}")
    lines.append(row("run (max)", run_m, "-", run_m.get("best_epoch", "-"), run_m.get("train_seconds", 0) / 60))
    for v, m in sorted(res.items(), key=lambda kv: -max(kv[1]["cv_global"], kv[1]["cv_per_label"])):
        lines.append(row(v, m, f"{m['params'] / 1e6:.2f}M", m["best_epoch"], m["minutes"]))
    best = max(res, key=lambda v: max(res[v]["cv_global"], res[v]["cv_per_label"])) if res else None
    lines.append("run (max) = the fine-tuned run itself (document score = maximum over chunks); cv = cross-fitted "
                 "macro-F1 (honest estimate, used for ranking); _res = adds the run's own scores; _co = statute "
                 "co-occurrence layer; ep = chosen epoch.")
    if best:
        lines.append(f"best on dev: {best} -> submit {res[best]['submissions']['chosen']}")
    return "\n".join(lines)


def run_stage_b_many(run_dirs, **overrides) -> dict:
    out = {}
    for r in run_dirs:
        if not os.path.exists(os.path.join(r, "chunks.pt")):
            print(f"[stageB] skipped {r}: no chunks.pt (run it with save_chunks=True)")
            continue
        out[r] = run_stage_b(r, **overrides)
    return out
