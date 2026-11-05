"""Phase 6 training: one Phase 1 model + one fine-tuning technique -> summaries of long legal documents.

    from cj_models import load_backbone
    from summ_train import run
    bb, info = load_backbone("llama", max_len=1024)
    res = run(bb, name="llama", technique="lora", ckpt_path=".../best.pt")

Divide and conquer (summ_data): every document is cut into windows; training pairs = window -> the summary sentences
that match it; at test time every window is summarised (beam search, no repeated trigrams) with a length budget
shared out from the predicted summary length, and the parts are joined and cleaned (repeated sentences removed).
Loss: cross-entropy with label smoothing on the target tokens; KD students add T^2 x KL(teacher || student).
finish_by (UNIX time): the run measures its training and generation speed and trains only the steps that fit, so
held-out scoring, test summaries and the submission are always written in time.

Outputs in out_dir/run_name/: log.txt, config.json, metrics.json (held-out ROUGE etc.), held_summaries.json,
test_summaries.json, submission/submission.zip, trained.pt (only the trained tensors).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import p6_paths  # noqa: F401
import cj_tech as T  # noqa: E402
import cj_train as R  # noqa: E402
import summ_data as SD  # noqa: E402
import summ_model as SM  # noqa: E402
from cj_mid import is_encoder_decoder  # noqa: E402

DEFAULTS = dict(
    data_root=SD.DEFAULT_ROOT, out_dir="/kaggle/working/summ_runs", run_name=None, device=None, seed=42,
    technique="lora", max_len=1024, tgt_tokens=0, enc_window=512, empty_frac=0.2, heldout_frac=0.05,
    epochs=1, batch_size=8, grad_accum=2, lr_body=None, weight_decay=0.01, warmup=0.03, label_smoothing=0.1,
    grad_clip=1.0, amp=True, finish_by=0.0,
    student_layers=0, kd_teacher=None, kd_alpha=0.5, kd_T=2.0,
    beams=4, no_repeat=3, len_pen=1.0, min_frac=0.0, max_new_scale=1.3, length_scale=1.0, gen_batch=8,
    decodings=("beam", "top_p"), top_p=0.9, top_k=0, temperature=0.7,
    eval_docs=40, gen_test=True, ckpt_path=None, cache_dir="/kaggle/working/summ_cache", log_every=200,
    save_weights=True, fetch_data=True, lowrank_rnn=False, limit_train=0, copy=True, val_docs=60,
    **T.TECH_DEFAULTS,
)


# decoding name -> (search, minimum share of the length budget written, length budget scale)
DECODINGS = {"beam": ("beam", None, 1.0), "top_p": ("top_p", None, 1.0), "beam_long": ("beam", 0.5, 1.3)}
LABELS = {"beam": "beam search", "top_p": "top-p sampling", "beam_long": "beam search, longer"}


def resolve_cfg(overrides: dict) -> dict:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    cfg = {**DEFAULTS, **overrides}
    if cfg["technique"] not in T.TECHNIQUES:
        raise ValueError(f"technique must be one of {T.TECHNIQUES}")
    if cfg["lr_body"] is None:
        cfg["lr_body"] = T.BODY_LR[cfg["technique"]] or 1e-3
    cfg["decodings"] = [cfg["decodings"]] if isinstance(cfg["decodings"], str) else list(cfg["decodings"])
    if not cfg["decodings"] or set(cfg["decodings"]) - set(DECODINGS):
        raise ValueError(f"decodings: a list of {DECODINGS}")
    return cfg


# ------------------------------------------------------------------------------ data
def prepare(encode, cfg: dict, encdec: bool, log=print) -> dict:
    """Data, training pairs and generation items (cached per tokenizer + window settings)."""
    data = SD.load_data(cfg["data_root"], fetch=cfg["fetch_data"], heldout_frac=cfg["heldout_frac"],
                        seed=cfg["seed"], log=log)
    if cfg["limit_train"]:
        data["train"] = data["train"][:cfg["limit_train"]]
    prompt = [] if encdec else list(encode(SM.PROMPT))
    tgt = cfg["tgt_tokens"] or (256 if encdec else min(256, cfg["max_len"] // 3))
    window = SM.window_budget(encdec, cfg["max_len"], len(prompt), tgt, cfg["enc_window"])
    fp = hashlib.md5(json.dumps([list(encode("The appeal is allowed, Section 302 I.P.C.")), window, tgt,
                                 cfg["empty_frac"], cfg["seed"], cfg["heldout_frac"], cfg["limit_train"],
                                 len(data["train"]), len(data["test"]), cfg["val_docs"], "v2"]).encode()).hexdigest()[:12]
    path = os.path.join(cfg["cache_dir"], f"summ_{fp}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                prep = pickle.load(f)
            log(f"[data] cached windows and pairs: {path}")
            prep["data"] = data
            return prep
        except Exception:  # noqa: BLE001  (damaged cache: rebuild)
            pass
    t0 = time.time()
    pairs, info = SD.make_pairs(data["train"], encode, window, tgt, cfg["empty_frac"], cfg["seed"])
    lm = SD.fit_length(data["train"])
    samp = data["train"][:300]
    tpw = sum(len(encode(d["summary"])) for d in samp) / max(1, sum(len(d["summary"].split()) for d in samp))

    def items(docs):
        out = []
        for di, d in enumerate(docs):
            ids, _, _ = SD.tokenize_doc(d["document"], encode)
            wins = SD.windows(len(ids), window)
            words = SD.predict_words(lm, d["document"])
            alloc = SD.allocate([e - a for a, e in wins], int(words * tpw))
            for wi, ((a, e), al) in enumerate(zip(wins, alloc)):
                out.append(dict(doc=di, w=wi, src=ids[a:e], alloc=al))
        return out
    held_pairs, _ = SD.make_pairs(data["held"][:cfg["val_docs"]], encode, window, tgt, 0.0, cfg["seed"])
    prep = dict(pairs=pairs, info=info, window=window, tgt_tokens=tgt, prompt=prompt, length_model=lm,
                tokens_per_word=tpw, held_items=items(data["held"]), test_items=items(data["test"]),
                held_pairs=held_pairs)
    log(f"[data] windows of {window} tokens, targets up to {tgt} tokens: {info['pairs']} training pairs from "
        f"{info['docs']} documents ({info['empty_targets']} with an empty target, {info['targets_cut']} targets cut, "
        f"mean target {info['target_tokens_mean']:.0f} tokens); summary length model: words = exp({lm['a']:.2f} + "
        f"{lm['b']:.2f} log(document words)), {tpw:.2f} tokens per word; {len(prep['held_items'])} held-out and "
        f"{len(prep['test_items'])} test windows ({time.time() - t0:.0f} s)")
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(prep, f)
    os.replace(tmp, path)
    prep["data"] = data
    return prep


def gen_items(prep, split, n_docs, cfg, scale: float = 1.0):
    items = [dict(it) for it in prep[f"{split}_items"] if n_docs is None or it["doc"] < n_docs]
    for it in items:
        it["max_new"] = max(8, min(prep["tgt_tokens"],
                                   int(it["alloc"] * cfg["max_new_scale"] * cfg["length_scale"] * scale)))
    return items


def summaries(prep, items, outs, docs, decode, cfg, special, scale: float = 1.0) -> tuple[dict, dict]:
    """(cleaned summary per document id, raw joined parts per id)."""
    parts: dict[int, list] = {}
    for it, o in zip(items, outs):
        parts.setdefault(it["doc"], []).append((it["w"], decode([t for t in o if t not in special])))
    clean, raw = {}, {}
    for di, ps in parts.items():
        d = docs[di]
        texts = [t for _, t in sorted(ps)]
        mw = int(SD.predict_words(prep["length_model"], d["document"], cfg["length_scale"] * scale) * 1.15)
        clean[d["id"]] = SD.postprocess(texts, max_words=mw)
        raw[d["id"]] = " ".join(texts)
    return clean, raw


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


@torch.no_grad()
def valid_loss(wrap, pairs, ids, device, batch_size, encdec, amp) -> float:
    """Mean next-token cross-entropy (no label smoothing) on the summary parts of held-out windows."""
    was = wrap.training
    wrap.eval()
    tot, n = 0.0, 0
    for b in SM.batches(pairs, batch_size, encdec, random.Random(0)):
        with R._autocast(device, amp):
            lg, y = SM.logits_and_targets(wrap, [pairs[i] for i in b], ids, device)
        tot += F.cross_entropy(lg.float(), y, reduction="sum").item()
        n += y.numel()
    wrap.train(was)
    return tot / max(1, n)


def default_decode(log=print):
    """ids -> text for our bpe12k tokenizer: its own decode() if it has one and it round-trips, else the
    vocabulary of our tokenizer.json read directly by our own code (byte-level or 'Ġ' / '▁' word markers). No
    tokenizer library is used. Checked on a sample sentence."""
    from cj_models import load_tokenizer, tokenizer_candidates
    from rr_backbone import make_encode
    tok = load_tokenizer(log)
    encode = make_encode(tok)
    sample = "The appeal is allowed under Section 302 of the Indian Penal Code, 1860."

    def ok(fn):
        try:
            return SD.words(fn(encode(sample))) == SD.words(sample)
        except Exception:  # noqa: BLE001
            return False
    cands = []
    if hasattr(tok, "decode"):
        cands.append(("tokenizer.decode", lambda ids: tok.decode(list(ids))))
    files = [c for c in tokenizer_candidates() if c.endswith(".json") and os.path.isfile(c)]
    for path in files[:1]:
        try:
            with open(path, encoding="utf-8") as f:
                vocab = json.load(f)["model"]["vocab"]
            inv = {i: t for t, i in vocab.items()}
            b2u = _bytes_to_unicode()

            def by_bytes(ids, inv=inv):
                s = "".join(inv.get(int(i), "") for i in ids)
                try:
                    return bytes(b2u[c] for c in s).decode("utf-8", errors="replace")
                except KeyError:
                    return s.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
            cands.append(("vocabulary of tokenizer.json", by_bytes))
        except Exception:  # noqa: BLE001
            pass
    for name, fn in cands:
        if ok(fn):
            log(f"[decode] using {name}")
            return lambda ids, fn=fn: fn(ids).replace("Ġ", " ").strip()
    raise RuntimeError("no way to turn token ids back into text with the bpe12k tokenizer")


# ------------------------------------------------------------------------------- run
def run(bb, name: str = "model", decode=None, **overrides) -> dict:
    cfg = resolve_cfg(overrides)
    tech = cfg["technique"]
    cfg["run_name"] = cfg["run_name"] or f"{name}_{tech}"
    R.set_seed(cfg["seed"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    log = R.Logger(os.path.join(out, "log.txt"))
    encdec = is_encoder_decoder(bb.model)
    kind = (" (KD student" + (" with a teacher)" if cfg["kd_teacher"] else ", no teacher)") if cfg["student_layers"]
            else (" (with a teacher)" if cfg["kd_teacher"] else ""))
    log(f"===== '{cfg['run_name']}': {name} + {tech}{kind} on {device} -> {out} "
        f"({'encoder-decoder' if encdec else 'decoder-only'})")
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    decode = decode or default_decode(log)
    prep = prepare(bb.encode, cfg, encdec, log)
    data, pairs = prep["data"], prep["pairs"]
    eos = bb.eos_id if bb.eos_id is not None else 0
    ids = dict(bos=bb.bos_id if bb.bos_id is not None else eos, eos=eos, pad=bb.pad_id, prompt=prep["prompt"])
    special = {ids["bos"], ids["eos"], ids["pad"]}

    # ------------------------------------------------------------- model + technique
    model = copy.deepcopy(bb.model)
    student = SM.make_student(model, cfg["student_layers"], log) if cfg["student_layers"] else None
    wrap = SM.Wrap(model).to(device).float()
    wrap.train()                                  # Phase 1 checkpoints are loaded in eval mode
    long_ = [p for p in pairs if len(p["src"]) >= 48 and p["tgt"]]
    probe = ([dict(src=list(p["src"][:48]), tgt=list(p["tgt"][:16])) for p in long_[:2]] if long_ else
             [dict(src=list(pairs[0]["src"]), tgt=list(pairs[0]["tgt"][:16]) or [eos])])
    rng = random.Random(cfg["seed"])
    n_batches = len(SM.batches(pairs, cfg["batch_size"], encdec, random.Random(0)))
    per_epoch = max(1, n_batches // cfg["grad_accum"])
    total = per_epoch * cfg["epochs"]
    hooks = SM.apply_technique(wrap, tech, cfg, total, probe, ids, device, log)
    copy_params = []
    if encdec and cfg["copy"] and wrap.enable_copy(probe, ids, device, log):
        copy_params = list(wrap.copy.parameters())
    body = list(hooks.body_params) + copy_params
    n_train = sum(p.numel() for p in {id(p): p for p in body}.values())
    n_total = SM.count(wrap) + hooks.info.get("quantized_params", 0)
    teacher, kd = None, None
    if cfg["kd_teacher"]:
        with open(os.path.join(cfg["kd_teacher"], "config.json")) as f:
            tcfg = json.load(f)
        if tcfg.get("ckpt_path") != cfg["ckpt_path"]:
            log(f"[kd] warning: teacher started from {tcfg.get('ckpt_path')}, student from {cfg['ckpt_path']}")
        state = torch.load(os.path.join(cfg["kd_teacher"], "trained.pt"), map_location="cpu", weights_only=False)
        teacher = SM.load_trained(bb.model, tcfg, state["state"], probe, ids, device, log)
        kd = dict(teacher=cfg["kd_teacher"], teacher_technique=tcfg["technique"], alpha=cfg["kd_alpha"], T=cfg["kd_T"],
                  teacher_params=SM.count(teacher))
        log(f"[kd] teacher {cfg['kd_teacher']} ({tcfg['technique']}, {kd['teacher_params'] / 1e6:.1f}M weights); "
            f"loss = {1 - cfg['kd_alpha']:.2f} x gold + {cfg['kd_alpha']:.2f} x T^2 KL (T = {cfg['kd_T']})")
    log(f"[setup] {tech}: {n_train / 1e6:.3f}M trained weights of {n_total / 1e6:.1f}M ({n_train / n_total:.2%}); "
        f"{len(pairs)} pairs, batch {cfg['batch_size']} x {cfg['grad_accum']} accumulation, {per_epoch} steps per "
        f"epoch x {cfg['epochs']}; lr {cfg['lr_body']}, label smoothing {cfg['label_smoothing']}")

    groups: dict[tuple, list] = {}
    for p in body:
        groups.setdefault((cfg["lr_body"], cfg["weight_decay"] if p.dim() >= 2 else 0.0), []).append(p)
    opt = torch.optim.AdamW([dict(params=v, lr=k[0], weight_decay=k[1]) for k, v in groups.items()])
    plan = dict(total=total, warm=max(1, int(cfg["warmup"] * total)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, SM.lr_factor_fn(plan, cfg["warmup"]))
    scaler = R._scaler(device, cfg["amp"])
    gen = SM.Generator(wrap, ids, device, cfg["amp"], None if encdec else cfg["max_len"])
    held_items = gen_items(prep, "held", cfg["eval_docs"], cfg)
    test_items = gen_items(prep, "test", None, cfg) if cfg["gen_test"] else []

    # ------------------------------------------------------------------- time budget
    budget = None
    if cfg["finish_by"]:
        sample = [dict(it, max_new=8) for it in (test_items or held_items)[:cfg["gen_batch"]]]
        t0 = time.time()
        gen.run(sample, cfg["beams"], cfg["no_repeat"], cfg["len_pen"], cfg["gen_batch"])
        per_step = (time.time() - t0) / 8                          # seconds per new token for a full batch

        def gen_seconds(items):
            g: dict[int, list] = {}
            for it in items:
                g.setdefault(len(it["src"]), []).append(it["max_new"])
            steps = sum(max(v[k:k + cfg["gen_batch"]]) for v in g.values() for k in range(0, len(v), cfg["gen_batch"]))
            return steps * per_step * 1.25
        nd = len(cfg["decodings"])
        gsec = nd * gen_seconds(held_items) + gen_seconds(test_items)
        left = cfg["finish_by"] - time.time()
        if gsec > 0.6 * left and cfg["eval_docs"] > 15:
            held_items = gen_items(prep, "held", 15, cfg)
            gsec = nd * gen_seconds(held_items) + gen_seconds(test_items)
            log("[budget] little time: held-out scoring on 15 documents")
        budget = dict(finish_by=cfg["finish_by"], planned_steps=total, gen_seconds=gsec, margin=240, cut=False,
                      probe=min(20, total))
        log(f"[budget] finish by {time.strftime('%H:%M', time.gmtime(cfg['finish_by']))} UTC ({left / 60:.0f} min); "
            f"generating held-out + test summaries takes about {gsec / 60:.0f} min")

    # ------------------------------------------------------------------------ training
    keep = [n for n, p in wrap.named_parameters() if p.requires_grad]

    def snapshot():
        sd = wrap.state_dict()
        return {n: sd[n].detach().to("cpu", copy=True) for n in keep if n in sd}
    vpairs = prep["held_pairs"]
    t_v = time.time()
    v0 = valid_loss(wrap, vpairs, ids, device, cfg["batch_size"], encdec, cfg["amp"]) if vpairs else float("nan")
    t_v = time.time() - t_v
    if budget is not None:
        budget["margin"] += t_v * (cfg["epochs"] + 1)
    val_history = [dict(epoch=0, step=0, loss=v0)]
    best_v, best_state, best_ep = v0, None, 0
    log(f"[valid] held-out target loss before training: {v0:.4f} ({len(vpairs)} windows of "
        f"{min(cfg['val_docs'], len(data['held']))} held-out documents)")
    step, micro, t_start, tot, tot_kd, seen = 0, 0, time.time(), 0.0, 0.0, 0
    history = []
    wrap.train()
    stop = False
    for ep in range(cfg["epochs"]):
        for b in SM.batches(pairs, cfg["batch_size"], encdec, rng):
            if step >= plan["total"]:
                stop = True
                break
            if micro % cfg["grad_accum"] == 0:
                hooks.on_step_start(step)
            batch = [pairs[i] for i in b]
            with R._autocast(device, cfg["amp"]):
                loss, gold, kd_t = SM.seq_loss(wrap, batch, ids, device, cfg["label_smoothing"], teacher,
                                               cfg["kd_alpha"], cfg["kd_T"])
                extra = hooks.extra_loss()
                if extra is not None:
                    loss = loss + extra
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                micro = 0
                continue
            scaler.scale(loss / cfg["grad_accum"]).backward()
            micro += 1
            tot += gold.item()
            tot_kd += kd_t.item() if kd_t is not None else 0.0
            seen += 1
            if micro % cfg["grad_accum"]:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(body, cfg["grad_clip"])
            sc = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            hooks.after_step(step, stepped=scaler.get_scale() >= sc)
            if budget is not None and step >= budget["probe"] and (step == budget["probe"] or step % 100 == 0):
                sps = (time.time() - t_start) / step
                new = SD.plan_steps(step, total, sps, cfg["finish_by"] - time.time(),
                                    budget["gen_seconds"] + budget["margin"], budget["probe"])
                if abs(new - plan["total"]) > max(10, 0.02 * plan["total"]) or (new < plan["total"] and new <= step):
                    plan["total"] = new
                    plan["warm"] = min(plan["warm"], max(1, int(cfg["warmup"] * new)))
                    budget["cut"] = new < total
                    log(f"[budget] {sps:.2f} s per step, {(cfg['finish_by'] - time.time()) / 60:.0f} min left -> "
                        f"{new} of {total} planned steps ({new / per_epoch:.2f} epochs)")
            if cfg["log_every"] and step % cfg["log_every"] == 0:
                el = time.time() - t_start
                rec = dict(step=step, loss=tot / max(1, seen), kd=tot_kd / max(1, seen) if teacher is not None else None,
                           minutes=el / 60)
                history.append(rec)
                log(f"  step {step}/{plan['total']}  loss {rec['loss']:.4f}"
                    + (f"  teacher term {rec['kd']:.4f}" if teacher is not None else "")
                    + f"  {el / 60:.1f} min, about {el / step * (plan['total'] - step) / 60:.1f} min left")
                tot, tot_kd, seen = 0.0, 0.0, 0
        if vpairs and step > val_history[-1]["step"]:          # after every (possibly cut) epoch: keep the best
            v = valid_loss(wrap, vpairs, ids, device, cfg["batch_size"], encdec, cfg["amp"])
            val_history.append(dict(epoch=ep + 1, step=step, loss=v))
            mark = ""
            if not v > best_v:                                  # (also when the start value is nan)
                best_v, best_state, best_ep, mark = v, snapshot(), ep + 1, "  <- best so far"
            log(f"[valid] epoch {ep + 1}: held-out target loss {v:.4f}{mark}")
        if stop:
            break
    if best_state is not None and best_ep != val_history[-1]["epoch"]:
        wrap.load_state_dict(best_state, strict=False)
        log(f"[valid] using the weights after epoch {best_ep} (lowest held-out loss {best_v:.4f})")
    train_seconds = time.time() - t_start
    if budget is not None:
        budget.update(steps=step, epochs_done=step / per_epoch)
    log(f"[train] {step} steps ({step / per_epoch:.2f} epochs) in {train_seconds / 60:.1f} min"
        + (" (cut to finish in time)" if budget and budget["cut"] else ""))

    # ---------------------------------------------------------------- held-out + test
    def gen_kw(dname):
        search, mf, _ = DECODINGS[dname]
        return dict(beams=cfg["beams"], no_repeat=cfg["no_repeat"], len_pen=cfg["len_pen"], batch=cfg["gen_batch"],
                    min_frac=cfg["min_frac"] if mf is None else mf, decoding=search, top_p=cfg["top_p"],
                    top_k=cfg["top_k"], temperature=cfg["temperature"], seed=cfg["seed"])
    n_held = len({it["doc"] for it in held_items})
    hdocs = data["held"]
    decodings, best = {}, None
    held_s = 0.0
    for dname in cfg["decodings"]:                 # each decoding on the held-out documents; the best one is used
        t0 = time.time()
        sc_len = DECODINGS[dname][2]
        h_items = gen_items(prep, "held", n_held, cfg, sc_len)
        outs = gen.run(h_items, **gen_kw(dname))
        clean, raw = summaries(prep, h_items, outs, hdocs, decode, cfg, special, sc_len)
        used = [d for d in hdocs if d["id"] in clean]
        sc = SD.score_all([d["summary"] for d in used], [clean[d["id"]] for d in used], [d["document"] for d in used])
        sc_raw = SD.score_all([d["summary"] for d in used], [raw[d["id"]] for d in used])
        sec = time.time() - t0
        held_s += sec
        decodings[dname] = dict(held=sc, held_raw=sc_raw, seconds=sec)
        label = LABELS[dname]
        log(f"[held-out, {label}] " + SD.fmt(sc) + f"  ({len(used)} documents, {sec / 60:.1f} min)")
        log(f"[held-out, {label}] before clean-up: ROUGE-1 {sc_raw['rouge1']:.4f}, ROUGE-L {sc_raw['rougeL']:.4f}, "
            f"words {sc_raw['words_hyp']:.0f}, repeated trigrams {sc_raw['repetition3']:.1%}")
        if best is None or sc["rouge_mean"] > decodings[best]["held"]["rouge_mean"]:
            best = dname
            with open(os.path.join(out, "held_summaries.json"), "w", encoding="utf-8") as f:
                json.dump({d["id"]: dict(summary=clean[d["id"]], raw=raw[d["id"]], reference=d["summary"])
                           for d in used}, f, ensure_ascii=False)
    sc, sc_raw = decodings[best]["held"], decodings[best]["held_raw"]
    if len(decodings) > 1:
        log("[decoding] " + ", ".join(f"{k}: (ROUGE-1 + ROUGE-L) / 2 = {v['held']['rouge_mean']:.4f}"
                                      for k, v in decodings.items()) + f" -> test summaries with {best}")
    sub, test_s = None, 0.0
    if test_items:
        t0 = time.time()
        test_items = gen_items(prep, "test", None, cfg, DECODINGS[best][2])
        outs = gen.run(test_items, log=log, **gen_kw(best))
        tclean, traw = summaries(prep, test_items, outs, data["test"], decode, cfg, special, DECODINGS[best][2])
        sub = SD.write_submission(data["test"], tclean, os.path.join(out, "submission"), fallback=traw)
        with open(os.path.join(out, "test_summaries.json"), "w", encoding="utf-8") as f:
            json.dump(tclean, f, ensure_ascii=False)
        test_s = time.time() - t0
        log(f"[test] {len(tclean)} summaries, mean {np.mean([len(s.split()) for s in tclean.values()]):.0f} words "
            f"({test_s / 60:.1f} min) -> {sub}")
    saved_mb = None
    if cfg["save_weights"]:
        keep = {n for n, p in wrap.named_parameters() if p.requires_grad}
        sd = wrap.state_dict()
        state = {n: sd[n].detach().cpu() for n in sd if n in keep or n.endswith("mask")}
        path = os.path.join(out, "trained.pt")
        torch.save({"state": state, "config": cfg, "technique": tech, "model": name}, path)
        saved_mb = os.path.getsize(path) / 2 ** 20
    metrics = dict(run_name=cfg["run_name"], model=name, technique=tech, encdec=encdec, held=sc, held_raw=sc_raw,
                   decoding=best, decodings=decodings,
                   trainable_params=n_train, total_params=n_total, trainable_share=n_train / max(1, n_total),
                   steps=step, epochs_done=step / per_epoch, train_seconds=train_seconds, held_gen_seconds=held_s,
                   val_history=val_history, best_epoch=best_ep, copy=wrap.copy is not None,
                   test_gen_seconds=test_s, peak_gpu_gb=R._peak_gb(device), saved_mb=saved_mb, history=history,
                   data=prep["info"], window=prep["window"], tgt_tokens=prep["tgt_tokens"], student=student, kd=kd,
                   budget=budget, technique_info=hooks.summary(), submission=sub, config=cfg)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, default=str)
    log(f"[done] held-out ROUGE-1 {sc['rouge1']:.4f}, ROUGE-L {sc['rougeL']:.4f} ({best}) | submit {sub}")
    wrap.cpu()
    if teacher is not None:
        teacher.cpu()
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
    hdr = (f"{'run':28s} {'trained':>9s} {'share':>7s} {'decode':>9s} {'R-1':>7s} {'R-2':>7s} {'R-L':>7s} {'words':>6s} "
           f"{'ref':>6s} {'rep3':>6s} {'new2':>6s} {'beam':>7s} {'top-p':>7s} {'long':>7s} {'epochs':>6s} {'train':>7s}")
    lines = [hdr, "-" * len(hdr)]
    for m in sorted(rows, key=lambda m: -m["held"]["rouge_mean"]):
        h = m["held"]
        dec = m.get("decodings") or {}

        def rm(k):
            return f"{dec[k]['held']['rouge_mean']:7.4f}" if k in dec else f"{'-':>7s}"
        lines.append(f"{m['run_name']:28s} {m['trainable_params'] / 1e6:8.2f}M {m['trainable_share']:7.2%} "
                     f"{str(m.get('decoding', 'beam')):>9s} {h['rouge1']:7.4f} {h['rouge2']:7.4f} {h['rougeL']:7.4f} "
                     f"{h['words_hyp']:6.0f} {h['words_ref']:6.0f} {h['repetition3']:6.1%} "
                     f"{h.get('novel_bigrams', math.nan):6.1%} {rm('beam')} {rm('top_p')} {rm('beam_long')} {m['epochs_done']:6.2f} "
                     f"{m['train_seconds'] / 60:6.1f}m")
    lines.append("decode: the decoding chosen on the held-out documents and used for the test summaries; R-1 / R-2 / "
                 "R-L: ROUGE F1 on held-out training documents with it (our scorer, word level, no stemming); words: "
                 "mean summary length (ref = reference); rep3: repeated trigrams; new2: word pairs not in the document "
                 "(how abstractive); beam / top-p / long: (ROUGE-1 + ROUGE-L) / 2 with each decoding (long = beam search, "
                 "each window writes at least half of a 30% larger length budget).")
    return "\n".join(lines)
