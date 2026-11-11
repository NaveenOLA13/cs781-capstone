"""Phase 7 training: one Phase 1 model + one fine-tuning technique -> a prior-case retriever (bi-encoder)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import p7_paths  # noqa: F401
import cj_tech as T  # noqa: E402
import cj_train as R  # noqa: E402
import pcr_data as PD  # noqa: E402
import pcr_model as PM  # noqa: E402

DEFAULTS = dict(
    data_root=PD.DEFAULT_ROOT,
    out_dir="/kaggle/working/pcr_runs",
    run_name=None,
    device=None,
    seed=42,
    technique="lorma",
    max_len=512,
    chunk_len=512,
    dim=256,
    pooling="mean",
    dropout=0.1,
    loss="infonce",
    tau=0.05,
    margin=0.2,
    hard_negs=2,
    neg_top=50,
    neg_source="bm25",
    q_chunks=3,
    d_chunks=3,
    train_max_chunks=64,
    batch_queries=8,
    epochs=12,
    lr_body=None,
    lr_head=1e-3,
    weight_decay=0.01,
    warmup=0.05,
    grad_clip=1.0,
    amp=True,
    enc_batch=16,
    eval_batch=64,
    max_chunks=24,
    val_chunks=6,
    val_every=2,
    select_by="dense",
    aggs=PM.AGGS,
    fuse_aggs=("mean", "maxsim"),
    top=100,
    dense_top=50,
    fuse=True,
    finish_by=0.0,
    student_layers=0,
    kd_teacher=None,
    kd_alpha=0.5,
    kd_T=1.0,
    kd_loss="listwise",
    save_train=False,
    save_weights=True,
    ckpt_path=None,
    cache_dir="/kaggle/working/pcr_cache",
    lexical_path=None,
    log_every=50,
    fetch_data=True,
    limit_train=0,
    tok_workers=4,
    exact=None,
    checkpointing=True,
    **{**T.TECH_DEFAULTS, "rnn_cudnn": True},
)


def resolve_cfg(overrides: dict) -> dict:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **overrides}
    if cfg["technique"] not in T.TECHNIQUES:
        raise ValueError(f"technique must be one of {T.TECHNIQUES}")
    if cfg["loss"] not in PM.LOSSES or cfg["kd_loss"] not in PM.KD_LOSSES:
        raise ValueError(f"loss must be one of {PM.LOSSES}, kd_loss one of {PM.KD_LOSSES}")
    cfg["aggs"] = list(cfg["aggs"])
    cfg["fuse_aggs"] = [a for a in cfg["fuse_aggs"] if a in cfg["aggs"]] or ["mean"]
    if "mean" not in cfg["aggs"] or set(cfg["aggs"]) - set(PM.AGGS):
        raise ValueError(f"aggs: a list of {PM.AGGS} including 'mean'")
    if cfg["neg_source"] not in ("bm25", "random") or cfg["select_by"] not in ("dense", "fused"):
        raise ValueError("neg_source: 'bm25' | 'random'; select_by: 'dense' | 'fused'")
    if cfg["lr_body"] is None:
        cfg["lr_body"] = T.BODY_LR[cfg["technique"]] or 0.0
    if cfg["kd_teacher"] and not os.path.exists(os.path.join(cfg["kd_teacher"], "teacher_vectors.pt")):
        raise FileNotFoundError(f"{cfg['kd_teacher']} has no teacher_vectors.pt (run the teacher with save_train=True)")
    cfg["chunk_len"] = min(cfg["chunk_len"], cfg["max_len"])
    return cfg


# --- data
def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def tokens_for(texts: dict, encode, cache_dir: str, workers: int = 4, log=print) -> dict:
    """{key: token ids (numpy int16/int32)} for {key: text}; tokenised once and cached in cache_dir."""
    import cj_data as D
    from cj_mid import tokenize

    path = os.path.join(cache_dir, f"pcr_tokens_{D.fingerprint(encode)}.npz") if cache_dir else None
    have: dict = {}
    if path and os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            off, flat = z["offsets"], z["flat"]
            have = {k: flat[off[i] : off[i + 1]] for i, k in enumerate(z["keys"].tolist())}
        except Exception as e:  # noqa: BLE001
            log(f"[data] token cache unreadable ({type(e).__name__}): tokenising again")
            have = {}
    todo = [k for k in texts if k not in have]
    if todo:
        t0 = time.time()
        log(f"[data] tokenising {len(todo)} cases with {workers} process(es)...")
        arrs = tokenize([texts[k] for k in todo], encode, workers, dtype=np.int32)
        for k, a in zip(todo, arrs):
            have[k] = a
        n_tok = sum(len(a) for a in arrs)
        log(f"[data] {n_tok / 1e6:.1f}M tokens in {(time.time() - t0) / 60:.1f} min")
        if path:
            keys = list(have)
            lens = np.array([len(have[k]) for k in keys], dtype=np.int64)
            flat = np.concatenate([have[k] for k in keys]) if keys else np.zeros(0, np.int32)
            dtype = np.int16 if (not len(flat) or flat.max() < 32767) else np.int32
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp.npz"
            np.savez(tmp, flat=flat.astype(dtype), offsets=np.r_[0, np.cumsum(lens)], keys=np.array(keys))
            os.replace(tmp, path)
    return {k: have[k] for k in texts}


def prepare(encode, cfg: dict, log=print) -> dict:
    """Data, token ids of every case, lexical matrices (cached), training items with hard negatives."""
    data = PD.load_data(cfg["data_root"], fetch=cfg["fetch_data"], log=log)
    keys, texts = {}, {}
    for s in PD.SPLITS:
        q, p = data[s]
        keys[s] = ([_key(d["text"]) for d in q], [_key(d["text"]) for d in p])
        for d in q + p:
            texts.setdefault(_key(d["text"]), d["text"])
    tok = tokens_for(texts, encode, cfg["cache_dir"], cfg["tok_workers"], log)
    lex_path = cfg["lexical_path"] or os.path.join(cfg["cache_dir"], "lexical.pkl")
    if os.path.exists(lex_path):
        L = PD.load_lexical(lex_path)
    else:
        log("[data] computing the BM25 matrices (cached for later runs)...")
        L = PD.lexical(data, log=log)
        PD.save_lexical(L, lex_path)
    for s in PD.SPLITS:
        if L[s]["qids"] != [d["id"] for d in data[s][0]] or L[s]["pids"] != [d["id"] for d in data[s][1]]:
            raise RuntimeError(f"{lex_path} was made from other data ({s}): delete it")
    lens = np.array([len(tok[k]) for k in set(keys["test"][1]) | set(keys["dev"][1])])
    info = dict(cases=len(texts), tokens_median=float(np.median(lens)), tokens_p90=float(np.percentile(lens, 90)))
    return dict(data=data, keys=keys, tok=tok, L=L, info=info)


def training_items(prep: dict, cfg: dict) -> list[tuple[int, list[int], list[int]]]:
    """(query index, its cited cases in the training pool, its hard negatives: the BM25 top neg_top cases that are not
    cited)."""
    L = prep["L"]["train"]
    q, pool = prep["data"]["train"]
    pos_of = {c: j for j, c in enumerate(L["pids"])}
    items = []
    for i, d in enumerate(q):
        pos = [pos_of[c] for c in d.get("rel", []) if c in pos_of and c != d["id"]]
        if not pos:
            continue
        gold = set(pos)
        order = np.argsort(-L["sum"][i], kind="stable")[: cfg["neg_top"] + len(gold) + 2]
        negs = [int(j) for j in order if j not in gold and pool[j]["id"] != d["id"]][: cfg["neg_top"]]
        items.append((i, pos, negs))
    if cfg["limit_train"]:
        items = items[: cfg["limit_train"]]
    return items


def case_chunks(prep, split, side, L, max_chunks, fill_id=0):
    keys = prep["keys"][split][0 if side == "q" else 1]
    out = []
    for k in keys:
        t = prep["tok"][k]
        out.append(PM.doc_chunks(t if len(t) else np.array([fill_id]), L, max_chunks))
    return out


def plan_steps(step, planned, sec_per_step, seconds_left, reserve, min_steps=20) -> int:
    spare = seconds_left - reserve
    more = int(spare / sec_per_step) if sec_per_step > 0 and spare > 0 else 0
    return max(min(planned, max(step, min_steps)), min(planned, step + more))


def _zrow(M):
    mu, sd = M.mean(1, keepdims=True), M.std(1, keepdims=True)
    return (M - mu) / np.where(sd > 1e-9, sd, 1.0)


# --- run
def run(bb, name: str = "model", **overrides) -> dict:
    cfg = resolve_cfg(overrides)
    tech = cfg["technique"]
    cfg["run_name"] = cfg["run_name"] or f"{name}_{tech}"
    R.set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    log = R.Logger(os.path.join(out, "log.txt"))
    kind = (
        " (KD student" + (" with a teacher)" if cfg["kd_teacher"] else ", no teacher)")
        if cfg["student_layers"]
        else (" (with a teacher)" if cfg["kd_teacher"] else "")
    )
    log(f"===== '{cfg['run_name']}': {name} + {tech}{kind} on {device} -> {out}")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    prep = prepare(bb.encode, cfg, log)
    data, L = prep["data"], prep["L"]
    CL = cfg["chunk_len"]
    exact = bb.pad_effect > 1e-4 if cfg["exact"] is None else bool(cfg["exact"])
    pad, fill = bb.pad_id, (bb.eos_id if bb.eos_id is not None else bb.pad_id)
    items = training_items(prep, cfg)
    tr_q = case_chunks(prep, "train", "q", CL, cfg["train_max_chunks"], fill)
    tr_p = case_chunks(prep, "train", "p", CL, cfg["train_max_chunks"], fill)
    gold_sets = {i: set(pos) for i, pos, _ in items}
    tr_pool = data["train"][1]
    tr_queries = data["train"][0]
    log(
        f"[data] {len(items)} training queries with cited cases in their pool ({sum(len(p) for _, p, _ in items)} "
        f"pairs); chunks of {CL} tokens; dev / test cases: median {prep['info']['tokens_median']:.0f} tokens, "
        f"p90 {prep['info']['tokens_p90']:.0f}; padding {'avoided (exact-length groups)' if exact else 'harmless'}"
    )

    # --- model + technique
    model = copy.deepcopy(bb.model).to(device).float()
    first = tr_q[items[0][0]][0][:64].astype(np.int64).tolist() if items else [1, 2, 3, 4]
    probe = torch.tensor([first] * 2, dtype=torch.long)
    student = None
    if cfg["student_layers"]:
        tmp = bb.classifier(cfg["dim"], model=model, pooling=cfg["pooling"], dropout=cfg["dropout"]).to(device)
        used0 = PM.used_params(tmp, probe, device)
        del tmp
        student = PM.make_student(model, cfg["student_layers"], used0, log)
    clf = bb.classifier(cfg["dim"], model=model, pooling=cfg["pooling"], dropout=cfg["dropout"]).to(device)
    clf.train()
    used = PM.used_params(clf, probe, device)
    per_epoch = max(1, math.ceil(len(items) / cfg["batch_queries"]))
    total = per_epoch * cfg["epochs"]
    orig_ids = {id(p) for p in model.parameters()}
    n_used = sum(p.numel() for n, p in model.named_parameters() if n in used)
    n_all = PM.count(model)
    hooks = T.apply(clf, tech, used, cfg, max(1, total), log)
    body = list(hooks.body_params)
    head = list(clf.head_parameters())
    n_train = sum(p.numel() for p in {id(p): p for p in body + head}.values())
    n_added = sum(p.numel() for p in {id(p): p for p in body}.values() if id(p) not in orig_ids)
    n_total = n_used + n_added + sum(p.numel() for p in head)
    log(
        f"[setup] {tech}: {n_train / 1e6:.3f}M trained weights of {n_total / 1e6:.1f}M ({n_train / n_total:.2%}); "
        f"case vectors of {cfg['dim']} values ({cfg['pooling']} pooling over tokens); loss {cfg['loss']} (tau "
        f"{cfg['tau']}); {cfg['batch_queries']} queries per step, each with 1 cited case + {cfg['hard_negs']} "
        f"{cfg['neg_source']} negatives (+ in-batch); {cfg['q_chunks']} / {cfg['d_chunks']} chunks per query / "
        f"candidate case; {per_epoch} steps per epoch x {cfg['epochs']}; lr body {cfg['lr_body']}, head {cfg['lr_head']}"
    )

    teacher, kd = None, None
    if cfg["kd_teacher"]:
        tv = torch.load(os.path.join(cfg["kd_teacher"], "teacher_vectors.pt"), map_location="cpu", weights_only=False)
        qrow = {q: i for i, q in enumerate(tv["qids"])}
        prow = {c: i for i, c in enumerate(tv["pids"])}
        tq = torch.stack([tv["q"][qrow[d["id"]]] for d in tr_queries]).float().to(device)
        tp = torch.stack([tv["p"][prow[d["id"]]] for d in tr_pool]).float().to(device)
        teacher = dict(q=tq, p=tp, tau=float(tv.get("tau", cfg["tau"])))
        with open(os.path.join(cfg["kd_teacher"], "metrics.json")) as f:
            tm = json.load(f)
        kd = dict(
            teacher=cfg["kd_teacher"],
            teacher_run=tm.get("run_name"),
            teacher_technique=tm.get("technique"),
            teacher_params=tm.get("total_params"),
            teacher_dev_f1=tm.get("dev_dense", {}).get("mean", {}).get("f1"),
            alpha=cfg["kd_alpha"],
            T=cfg["kd_T"],
            loss=cfg["kd_loss"],
        )
        log(
            f"[kd] teacher {kd['teacher_run']} ({kd['teacher_technique']}, {(kd['teacher_params'] or 0) / 1e6:.1f}M "
            f"weights, dev micro-F1@7 alone {kd['teacher_dev_f1']}); its case vectors of the training split are the "
            f"teacher signal; loss = {1 - cfg['kd_alpha']:.2f} x {cfg['loss']} + {cfg['kd_alpha']:.2f} x "
            f"{cfg['kd_loss']} (T = {cfg['kd_T']})"
        )

    groups = {}
    for p in body:
        groups.setdefault(("body", cfg["weight_decay"] if p.dim() >= 2 else 0.0), []).append(p)
    for p in head:
        groups.setdefault(("head", cfg["weight_decay"] if p.dim() >= 2 else 0.0), []).append(p)
    opt = torch.optim.AdamW(
        [
            dict(params=v, lr=cfg["lr_body"] if k[0] == "body" else cfg["lr_head"], weight_decay=k[1])
            for k, v in groups.items()
        ]
    )
    plan = dict(total=total, warm=max(1, int(cfg["warmup"] * total)))

    def lr_factor(s):
        if s < plan["warm"]:
            return (s + 1) / max(1, plan["warm"])
        return max(0.0, (plan["total"] - s) / max(1, plan["total"] - plan["warm"]))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = R._scaler(device, cfg["amp"])
    params = body + head

    # --- time budget
    max_chunks = cfg["max_chunks"]
    dev_val = (
        case_chunks(prep, "dev", "q", CL, cfg["val_chunks"], fill),
        case_chunks(prep, "dev", "p", CL, cfg["val_chunks"], fill),
    )

    def n_final(mc):
        n = 0
        for s in ("dev", "test") + (("train",) if cfg["save_train"] else ()):
            for side in ("q", "p"):
                n += sum(
                    len(PM.chunk_starts(len(prep["tok"][k]), CL, mc)) for k in prep["keys"][s][0 if side == "q" else 1]
                )
        return n

    sample = [c for cs in dev_val[1][:64] for c in cs][:96]
    PM.embed_all(clf, sample[:8], pad, device, cfg["eval_batch"], exact, cfg["amp"])
    t0 = time.time()
    PM.embed_all(clf, sample, pad, device, cfg["eval_batch"], exact, cfg["amp"])
    speed = len(sample) / max(1e-3, time.time() - t0)
    val_sec = sum(len(c) for side in dev_val for c in side) / speed * 1.1 + 5
    fin_sec = n_final(max_chunks) / speed * 1.15 + 180
    budget = None
    if cfg["finish_by"]:
        left = cfg["finish_by"] - time.time()
        while fin_sec > 0.45 * left and max_chunks > 4:
            max_chunks = max(4, max_chunks // 2)
            fin_sec = n_final(max_chunks) / speed * 1.15 + 180
        budget = dict(
            finish_by=cfg["finish_by"],
            planned_steps=total,
            final_seconds=fin_sec,
            val_seconds=val_sec,
            max_chunks=max_chunks,
            cut=False,
            probe=min(20, total),
            margin=120,
        )
        log(
            f"[budget] finish by {time.strftime('%H:%M', time.gmtime(cfg['finish_by']))} UTC ({left / 60:.0f} min); "
            f"encoding speed {speed:.0f} chunks/s; final encoding of dev + test{' + train' if cfg['save_train'] else ''}"
            f" (up to {max_chunks} chunks per case) about {fin_sec / 60:.0f} min; one validation {val_sec / 60:.1f} min"
        )
    else:
        log(f"[speed] {speed:.0f} chunks/s; final encoding about {fin_sec / 60:.0f} min")

    # --- validation
    def validate():
        Qv, Qn = PM.encode_cases(clf, dev_val[0], pad, device, cfg["eval_batch"], exact, cfg["amp"])
        Pv, Pn = PM.encode_cases(clf, dev_val[1], pad, device, cfg["eval_batch"], exact, cfg["amp"])
        S = PM.aggregate(Qv, Qn, Pv, Pn, ["mean"], device)["mean"]
        m = PD.eval_matrix(L["dev"], S)
        fz = PD.eval_matrix(L["dev"], _zrow(S) + _zrow(L["dev"]["sum"]))
        return m, fz

    keep = [n for n, p in clf.named_parameters() if p.requires_grad]

    def snapshot():
        sd = clf.state_dict()
        return {n: sd[n].detach().to("cpu", copy=True) for n in keep if n in sd}

    t_v = time.time()
    m0, f0 = validate()
    val_history = [dict(epoch=0, step=0, f1=m0["f1"], fused_quick=f0["f1"])]
    sel = (lambda m, f: m["f1"]) if cfg["select_by"] == "dense" else (lambda m, f: f["f1"])
    best_v, best_state, best_ep = sel(m0, f0), None, 0
    log(
        f"[valid] before training: dev {PD.fmt(m0)} alone ('mean', {cfg['val_chunks']} chunks per case); with BM25 "
        f"(z-scores added) {f0['f1']:.4f}  ({(time.time() - t_v) / 60:.1f} min)"
    )

    # --- training
    H = cfg["hard_negs"]
    step, t_start, tot, tot_kd, seen, val_spent = 0, time.time(), 0.0, 0.0, 0, 0.0
    history, no_grad_warned = [], False
    stop = total == 0
    clf.train()
    ep = 0
    for ep in range(cfg["epochs"]):
        if stop:
            break
        order = list(range(len(items)))
        rng.shuffle(order)
        for b0 in range(0, len(order), cfg["batch_queries"]):
            if step >= plan["total"]:
                stop = True
                break
            batch = [items[i] for i in order[b0 : b0 + cfg["batch_queries"]]]
            hooks.on_step_start(step)
            q_idx, cand, target, chunks, owner = [], [], [], [], []
            for qi, pos, negs in batch:
                q_idx.append(qi)
                target.append(len(cand))
                if cfg["neg_source"] == "bm25" and negs:
                    ns = rng.sample(negs, min(H, len(negs)))
                else:
                    ns = []
                    while len(ns) < H:
                        j = rng.randrange(len(tr_pool))
                        if j not in gold_sets[qi] and tr_pool[j]["id"] != tr_queries[qi]["id"]:
                            ns.append(j)
                cand += [rng.choice(pos)] + ns
            B, N = len(q_idx), len(cand)
            for b, qi in enumerate(q_idx):
                cs = tr_q[qi]
                for c in sorted(rng.sample(range(len(cs)), min(cfg["q_chunks"], len(cs)))):
                    chunks.append(cs[c])
                    owner.append(b)
            for k, j in enumerate(cand):
                cs = tr_p[j]
                for c in sorted(rng.sample(range(len(cs)), min(cfg["d_chunks"], len(cs)))):
                    chunks.append(cs[c])
                    owner.append(B + k)
            rel = torch.tensor(
                [[j in gold_sets[qi] or tr_pool[j]["id"] == tr_queries[qi]["id"] for j in cand] for qi in q_idx],
                device=device,
            )
            tgt = torch.tensor(target, device=device)
            V = PM.embed_with_grad(clf, chunks, pad, device, cfg["enc_batch"], exact, cfg["amp"], cfg["checkpointing"])
            own = torch.tensor(owner, device=device)
            docs = F.normalize(torch.zeros(B + N, V.size(1), device=device).index_add(0, own, V), dim=-1)
            S = docs[:B] @ docs[B:].T
            loss = PM.ranking_loss(S, tgt, rel, cfg["loss"], cfg["tau"], cfg["margin"])
            kd_t = None
            if teacher is not None:
                Tt = teacher["q"][q_idx] @ teacher["p"][cand].T
                ncols = torch.zeros_like(rel)
                for b, t_ in enumerate(target):
                    ncols[b, t_ + 1 : t_ + 1 + H] = True
                kd_t = PM.kd_loss(S, Tt, cfg["kd_loss"], cfg["tau"], teacher["tau"], cfg["kd_T"], tgt, ncols & ~rel)
                loss = (1 - cfg["kd_alpha"]) * loss + cfg["kd_alpha"] * kd_t
            extra = hooks.extra_loss()
            if extra is not None:
                loss = loss + extra
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if not no_grad_warned:
                none = sum(1 for p in body if p.grad is None)
                if none:
                    log(f"[warning] {none} of {len(body)} trained tensors of the technique got no gradient")
                no_grad_warned = True
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            sc = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            hooks.after_step(step, stepped=scaler.get_scale() >= sc)
            tot += loss.item()
            tot_kd += kd_t.item() if kd_t is not None else 0.0
            seen += 1
            if budget is not None and step >= budget["probe"] and (step == budget["probe"] or step % 50 == 0):
                sps = (time.time() - t_start - val_spent) / step
                n_val_left = max(1, (plan["total"] - step) // max(1, per_epoch * cfg["val_every"]) + 1)
                reserve = budget["final_seconds"] + n_val_left * val_sec + budget["margin"]
                new = plan_steps(step, total, sps, cfg["finish_by"] - time.time(), reserve, budget["probe"])
                if abs(new - plan["total"]) > max(10, 0.02 * plan["total"]) or (new < plan["total"] and new <= step):
                    plan["total"] = new
                    plan["warm"] = min(plan["warm"], max(1, int(cfg["warmup"] * new)))
                    budget["cut"] = new < total
                    log(
                        f"[budget] {sps:.2f} s per step, {(cfg['finish_by'] - time.time()) / 60:.0f} min left -> "
                        f"{new} of {total} planned steps ({new / per_epoch:.1f} epochs)"
                    )
            if cfg["log_every"] and step % cfg["log_every"] == 0:
                el = time.time() - t_start
                rec = dict(
                    step=step,
                    loss=tot / max(1, seen),
                    kd=tot_kd / max(1, seen) if teacher is not None else None,
                    minutes=el / 60,
                )
                history.append(rec)
                log(
                    f"  step {step}/{plan['total']}  loss {rec['loss']:.4f}"
                    + (f"  teacher term {rec['kd']:.4f}" if teacher is not None else "")
                    + f"  {el / 60:.1f} min, about {el / step * (plan['total'] - step) / 60:.1f} min left"
                )
                tot, tot_kd, seen = 0.0, 0.0, 0
        last = stop or ep == cfg["epochs"] - 1 or step >= plan["total"]
        if step > val_history[-1]["step"] and ((ep + 1) % cfg["val_every"] == 0 or last):
            t_v = time.time()
            m, fz = validate()
            val_spent += time.time() - t_v
            val_history.append(dict(epoch=ep + 1, step=step, f1=m["f1"], fused_quick=fz["f1"]))
            mark = ""
            if sel(m, fz) > best_v:
                best_v, best_state, best_ep, mark = sel(m, fz), snapshot(), ep + 1, "  <- best so far"
            log(f"[valid] epoch {ep + 1}: dev {PD.fmt(m)} alone; with BM25 {fz['f1']:.4f}{mark}")
            clf.train()
        if step >= plan["total"]:
            stop = True
    if best_state is not None and best_ep != val_history[-1]["epoch"]:
        clf.load_state_dict(best_state, strict=False)
        log(f"[valid] using the weights after epoch {best_ep} (best dev {best_v:.4f})")
    elif best_state is None and step > 0:
        log("[valid] no epoch beat the starting point on dev: keeping the final weights anyway")
    train_seconds = time.time() - t_start
    if budget is not None:
        budget.update(steps=step, epochs_done=step / per_epoch)
    log(
        f"[train] {step} steps ({step / per_epoch:.2f} epochs) in {train_seconds / 60:.1f} min"
        + (" (cut to finish in time)" if budget and budget["cut"] else "")
    )

    # --- final encoding + scores
    t0 = time.time()
    scores, vec_mb = {}, 0.0
    for s in ("dev", "test"):
        Qv, Qn = PM.encode_cases(
            clf, case_chunks(prep, s, "q", CL, max_chunks, fill), pad, device, cfg["eval_batch"], exact, cfg["amp"]
        )
        Pv, Pn = PM.encode_cases(
            clf, case_chunks(prep, s, "p", CL, max_chunks, fill), pad, device, cfg["eval_batch"], exact, cfg["amp"]
        )
        vec_mb += (int(Qn.sum()) + int(Pn.sum())) * cfg["dim"] * 2 / 2**20
        ta = time.time()
        scores[s] = PM.aggregate(Qv, Qn, Pv, Pn, cfg["aggs"], device)
        agg_s = time.time() - ta
        np.savez_compressed(os.path.join(out, f"scores_{s}.npz"), **scores[s])
        del Qv, Pv
    with open(os.path.join(out, "ids.json"), "w") as f:
        json.dump({s: dict(qids=L[s]["qids"], pids=L[s]["pids"]) for s in ("dev", "test")}, f)
    enc_seconds = time.time() - t0
    train_dense = None
    if cfg["save_train"]:
        Qv, Qn = PM.encode_cases(
            clf,
            case_chunks(prep, "train", "q", CL, max_chunks, fill),
            pad,
            device,
            cfg["eval_batch"],
            exact,
            cfg["amp"],
        )
        Pv, Pn = PM.encode_cases(
            clf,
            case_chunks(prep, "train", "p", CL, max_chunks, fill),
            pad,
            device,
            cfg["eval_batch"],
            exact,
            cfg["amp"],
        )
        qm, pm = PM.mean_vectors(Qv, Qn), PM.mean_vectors(Pv, Pn)
        torch.save(
            dict(
                qids=L["train"]["qids"],
                pids=L["train"]["pids"],
                q=qm.half(),
                p=pm.half(),
                tau=cfg["tau"],
                run=cfg["run_name"],
            ),
            os.path.join(out, "teacher_vectors.pt"),
        )
        train_dense = PD.eval_matrix(L["train"], (qm @ pm.T).numpy())
        log(
            f"[train split] our model alone ('mean') on the training queries it learned from: {PD.fmt(train_dense)} "
            f"(compare with dev: how much it memorised)"
        )
        del Qv, Pv
    log(
        f"[encode] dev + test{' + train' if cfg['save_train'] else ''}: {(time.time() - t0) / 60:.1f} min "
        f"(up to {max_chunks} chunks per case; case vectors {vec_mb:.0f} MB in float16; all similarities of the last "
        f"split {agg_s:.1f} s)"
    )
    dev_dense = {a: PD.eval_matrix(L["dev"], scores["dev"][a]) for a in cfg["aggs"]}
    for a, m in dev_dense.items():
        log(f"[dev] our model alone, '{a}': {PD.fmt(m)}")
    best_agg = max(dev_dense, key=lambda a: dev_dense[a]["f1"])
    qs, ps = [dict(id=q) for q in L["test"]["qids"]], [dict(id=c) for c in L["test"]["pids"]]
    dense_sub = PD.write_prediction(
        PD.rank_matrix(L["test"], scores["test"][best_agg]), qs, ps, os.path.join(out, "dense_only")
    )
    fusion, sub = None, None
    if cfg["fuse"]:
        try:
            fusion = PD.fused_submission(
                L,
                {a: scores["dev"][a] for a in cfg["fuse_aggs"]},
                {a: scores["test"][a] for a in cfg["fuse_aggs"]},
                out,
                cfg["top"],
                cfg["dense_top"],
                log,
            )
            sub = fusion["submission"]
        except Exception as e:  # noqa: BLE001
            log(f"[fusion] failed: {type(e).__name__}: {e}")
    saved_mb = None
    if cfg["save_weights"] and step > 0:
        keep_ = {n for n, p in clf.named_parameters() if p.requires_grad}
        sd = clf.state_dict()
        state = {n: sd[n].detach().cpu() for n in sd if n in keep_ or n.endswith("mask")}
        path = os.path.join(out, "trained.pt")
        torch.save({"state": state, "config": cfg, "technique": tech, "model": name}, path)
        saved_mb = os.path.getsize(path) / 2**20
    metrics = dict(
        run_name=cfg["run_name"],
        model=name,
        technique=tech,
        dev_dense=dev_dense,
        best_agg=best_agg,
        dev_fused=fusion["dev"] if fusion else None,
        dev_lexical=fusion.get("lexical_" + fusion["method"]) if fusion else None,
        fusion={k: v for k, v in (fusion or {}).items() if k != "submission"} or None,
        train_dense=train_dense,
        val_history=val_history,
        best_epoch=best_ep,
        trainable_params=n_train,
        total_params=n_total,
        trainable_share=n_train / max(1, n_total),
        model_params_all=n_all,
        steps=step,
        epochs_done=step / per_epoch,
        train_seconds=train_seconds,
        encode_seconds=enc_seconds,
        chunks_per_second=speed,
        max_chunks=max_chunks,
        vectors_mb=vec_mb,
        peak_gpu_gb=R._peak_gb(device),
        saved_mb=saved_mb,
        history=history,
        data=prep["info"],
        student=student,
        kd=kd,
        budget=budget,
        technique_info=hooks.summary(),
        submission=sub,
        dense_submission=dense_sub,
        config=cfg,
    )
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=str)
    log(
        f"[done] dev micro-F1@7: our model alone {dev_dense[best_agg]['f1']:.4f} ('{best_agg}')"
        + (
            f", fused with the lexical features {fusion['dev']['f1']:.4f} (lexical alone "
            f"{metrics['dev_lexical']['f1']:.4f}), cross-fitted on dev"
            if fusion
            else ""
        )
        + f" | submit {sub or dense_sub}"
    )
    clf.cpu()
    opt.state.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dict(out_dir=out, metrics=metrics)


def load_metrics(out_dir: str, run_name: str) -> dict | None:
    p = os.path.join(out_dir, run_name, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def comparison(rows: list[dict]) -> str:
    hdr = (
        f"{'run':30s} {'trained':>9s} {'share':>7s} {'total':>7s} {'mean':>7s} {'maxsim':>7s} {'sym':>7s} "
        f"{'max':>7s} {'first':>7s} {'lexical':>8s} {'fused':>7s} {'train':>7s} {'epochs':>6s} {'min':>6s} "
        f"{'GB':>5s} {'ch/s':>6s}"
    )
    lines = [hdr, "-" * len(hdr)]

    def key(m):
        return -((m.get("dev_fused") or {}).get("f1") or 0.0), -m["dev_dense"]["mean"]["f1"]

    for m in sorted(rows, key=key):
        d = m["dev_dense"]

        def g(a):
            return f"{d[a]['f1']:7.4f}" if a in d else f"{'-':>7s}"

        fu = m.get("dev_fused") or {}
        lx = m.get("dev_lexical") or {}
        tr = m.get("train_dense") or {}
        lines.append(
            f"{m['run_name']:30s} {m['trainable_params'] / 1e6:8.2f}M {m['trainable_share']:7.2%} "
            f"{m['total_params'] / 1e6:6.1f}M {g('mean')} {g('maxsim')} {g('maxsim_sym')} {g('max')} "
            f"{g('first')} {lx.get('f1', float('nan')):8.4f} {fu.get('f1', float('nan')):7.4f} "
            f"{tr.get('f1', float('nan')):7.4f} {m['epochs_done']:6.2f} {m['train_seconds'] / 60:6.1f} "
            f"{(m.get('peak_gpu_gb') or 0):5.1f} {m.get('chunks_per_second', 0):6.0f}"
        )
    lines.append(
        "dev micro-F1@7. mean / maxsim / sym / max / first: our model alone with each case similarity (whole "
        "pool ranked); lexical: the BM25 + year + citation-count ranker alone; fused: that ranker with our "
        "model's scores added (both 5-fold cross-fitted on dev); train: our model alone on the training "
        "queries (memorising); min: training minutes; GB: peak GPU memory; ch/s: chunks encoded per second."
    )
    return "\n".join(lines)
