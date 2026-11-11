"""Phase 3: knowledge distillation into a half-size student."""

from __future__ import annotations

import copy
import dataclasses
import gc
import json
import os
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

import rr_data as D
import rr_peft as P
import rr_train as T
from rr_backbone import make_backbone
from rr_head import count_params

KD_DEFAULTS = {**T.DEFAULTS, "run_name": "kd_teacher", "method": "lora", "folds": 5, "train_on_dev": False}
LAYER_COUNT_NAMES = ("n_layers", "n_layer", "num_layers", "num_hidden_layers", "n_blocks", "depth")


def _quiet(*args, **kwargs):
    pass


def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _short(rep):
    a = rep["all"]
    doms = ", ".join(f"{k} {v['macro_f1_all13']:.3f}" for k, v in rep.items() if k != "all")
    return f"13-role avg {a['macro_f1_all13']:.3f} ({doms}), accuracy {a['accuracy']:.3f}"


def _probs(outs):
    return [None if o is None else F.softmax(o.float(), -1) for o in outs]


def _avg(per_copy):
    """per_copy: [copy][doc] -> [S, L] probabilities."""
    return [None if runs[0] is None else torch.stack(runs).mean(0) for runs in zip(*per_copy)]


def _log(ps):
    return [None if p is None else torch.log(p.clamp_min(1e-12)) for p in ps]


# --- student
def find_blocks(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """The repeated blocks (layers) of the model: the nn.ModuleList of same-class items with the most weights."""
    best = None
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 2 and len({type(x) for x in m}) == 1:
            n = count_params(m, trainable_only=False)
            if best is None or n > best[2]:
                best = (name, m, n)
    if best is None:
        raise ValueError("no repeated blocks (an nn.ModuleList of layers) found to shorten")
    return best[0], best[1]


def _shrink_layer_count(model: nn.Module, n: int, k: int) -> list[str]:
    """Models that loop `for i in range(config.n_layers)` need the stored layer count changed too."""
    changed = []
    holders = [("model", model)] + [
        (a, getattr(model, a)) for a in ("config", "cfg", "args", "params") if getattr(model, a, None) is not None
    ]
    for hname, h in holders:
        for attr in LAYER_COUNT_NAMES:
            try:
                if isinstance(h, dict):
                    if h.get(attr) == n:
                        h[attr] = k
                        changed.append(f"{hname}[{attr!r}]")
                elif getattr(h, attr, None) == n and not isinstance(getattr(h, attr), bool):
                    if dataclasses.is_dataclass(h) and type(h).__dataclass_params__.frozen:  # type: ignore
                        object.__setattr__(h, attr, k)
                    else:
                        setattr(h, attr, k)
                    changed.append(f"{hname}.{attr}")
            except (AttributeError, TypeError):
                pass
    return changed


def make_student(bb, tokenizer, keep=0.5, verbose: bool = True):
    """A smaller copy of the Phase 1 model: only the first `keep` blocks (a share, or a number of blocks)."""
    student = copy.deepcopy(bb.model)
    name, blocks = find_blocks(student)
    n = len(blocks)
    k = keep if isinstance(keep, int) else max(1, int(round(n * float(keep))))
    if not 1 <= k < n:
        raise ValueError(f"keep must leave between 1 and {n - 1} of the {n} blocks (got {k})")
    parent_name, _, attr = name.rpartition(".")
    parent = student.get_submodule(parent_name) if parent_name else student
    setattr(parent, attr, nn.ModuleList(list(blocks)[:k]))
    changed = _shrink_layer_count(student, n, k)
    sbb = make_backbone(student, tokenizer, verbose=verbose)
    info = dict(
        blocks=name,
        layers_teacher=n,
        layers_student=k,
        layer_count_fields=changed,
        params_teacher=count_params(bb.model, trainable_only=False),
        params_student=count_params(student, trainable_only=False),
        route=sbb.route,
    )
    return sbb, info


# --- teacher
def teacher_oof(bb, **overrides) -> dict:
    """Trains the pipeline K times, each copy on all labelled judgments but one group, and saves teacher.pt:
    out-of-fold log-probabilities for every pooled judgment ("train"."""
    cfg = T.resolve_cfg(bb, overrides, KD_DEFAULTS)
    K, on_dev = cfg["folds"], bool(cfg["train_on_dev"])
    if K < 2:
        raise ValueError("folds must be at least 2")
    if cfg["kd_teacher"]:
        raise ValueError("the teacher is trained on gold labels only (no kd_teacher)")
    tag = "ensemble" if on_dev else "teacher"
    device, out, log = T.start_run(cfg, kind=f"{tag} ({K} copies, out-of-fold" + (", train + dev)" if on_dev else ")"))
    labels, train, dev, test = T.prepare(bb, cfg, log)
    pool = train + dev if on_dev else train
    li = {l: i for i, l in enumerate(labels)}
    domains = sorted({d["domain"] for d in pool})
    folds = D.make_folds(pool, K, cfg["seed"])
    if any(not f for f in folds):
        raise ValueError(f"need at least {K} labelled judgments for {K} folds")
    log(
        f"[{tag}] {len(pool)} judgments in {K} groups: "
        + " | ".join("+".join(f"{sum(pool[i]['domain'] == dm for i in f)} {dm}" for dm in domains) for f in folds)
        + ("; each copy picks its epochs on its held-out group" if on_dev else "; each copy picks its epochs on dev")
    )

    variants = list(cfg["b_variants"] or [cfg["b_arch"]])
    v0 = variants[0]
    if len(variants) > 1:
        log(
            f"[{tag}] Stage B versions trained on the same Stage A features in every copy: {', '.join(variants)} "
            f"(main: {v0})"
        )
    oof_v = {v: [None] * len(pool) for v in variants}
    oof_a = [None] * len(pool)
    dev_a, fold_info = [], []
    dev_bv = {v: [] for v in variants}
    test_bv = {v: [] for v in variants}
    peak, infer, saved, first = 0.0, 0.0, 0, None
    n_trainable = n_total = 0
    t_start = time.time()
    for k, held in enumerate(folds):
        t0 = time.time()
        held_set = set(held)
        fit = [d for i, d in enumerate(pool) if i not in held_set]
        held_docs = [pool[i] for i in held]
        select = held_docs if on_dev else dev
        T.set_seed(cfg["seed"] + k)
        clf, n_trainable, n_total = T.build_stage_a(bb, labels, cfg, device, fit, log if k == 0 else _quiet)
        T._reset_peak(device)
        log(f"[{tag} {k + 1}/{K}] Stage A on {len(fit)} judgments ({len(held_docs)} held out)")
        info_a = T.train_stage_a(clf, fit, select, labels, cfg, device, bb.pad_id, log)
        weight = info_a.pop("weight_tensor")
        held_pa = _probs(T.run_sentences(clf, held_docs, cfg, device, bb.pad_id))
        for i, p in zip(held, held_pa):
            oof_a[i] = p
        if not on_dev:
            ti = time.time()
            dev_a.append(_probs(T.run_sentences(clf, dev, cfg, device, bb.pad_id)))
            infer += time.time() - ti
        parts = [("fit", fit), ("held", held_docs), ("test", test)] + ([] if on_dev else [("dev", dev)])
        feats = {s: T.run_sentences(clf, ds, cfg, device, bb.pad_id, what="features") for s, ds in parts}
        peak = max(peak, T._gb(device) or 0.0)
        state_a = P.trained_state(clf)
        clf.cpu()
        del clf
        _free(device)
        ys = [torch.tensor([li[l] for l in d["labels"]], dtype=torch.long) for d in fit]
        sel_key = "held" if on_dev else "dev"
        rep_hv, best_ep, states_b = {}, {}, {}
        for v in variants:
            model_b, info_b = T.train_stage_b(
                feats["fit"], ys, feats[sel_key], select, labels, {**cfg, "b_arch": v}, device, weight, _quiet
            )
            held_p = _probs(T.predict_stage_b(model_b, feats["held"], device))
            for i, p in zip(held, held_p):
                oof_v[v][i] = p
            if not on_dev:
                dev_bv[v].append(_probs(T.predict_stage_b(model_b, feats["dev"], device)))
            test_bv[v].append(_probs(T.predict_stage_b(model_b, feats["test"], device)))
            rep_hv[v] = D.evaluate(held_docs, T.to_preds(held_docs, held_p, labels), labels)
            best_ep[v] = info_b["best_epoch"]
            states_b[v] = {n: t.detach().cpu() for n, t in model_b.state_dict().items()}
            del model_b
        rep_h = rep_hv[v0]
        rep_d = None if on_dev else D.evaluate(dev, T.to_preds(dev, dev_bv[v0][-1], labels), labels)
        if cfg["save_model"]:
            path = os.path.join(out, f"{tag}_fold{k + 1}.pt")
            torch.save(
                {
                    "stageA_trained": state_a,
                    "stageB": states_b[v0],
                    "stageB_versions": states_b,
                    "fold": k + 1,
                    "held_out": [d["id"] for d in held_docs],
                    "labels": labels,
                    "config": cfg,
                    "note": "Stage A: only the trained tensors (adapters + head); every other weight equals "
                    "the Phase 1 checkpoint. Stage B: whole models, one per version.",
                },
                path,
            )
            saved += os.path.getsize(path)
        minutes = (time.time() - t0) / 60
        first = first or info_a
        fold_info.append(
            dict(
                fold=k + 1,
                fit_judgments=len(fit),
                held_out=[d["id"] for d in held_docs],
                minutes=minutes,
                stage_a_best_epoch=info_a["best_epoch"],
                stage_b_best_epoch=best_ep[v0],
                stage_b_best_epochs=best_ep,
                held_out_13avg={d: v["macro_f1_all13"] for d, v in rep_h.items()},
                held_out_13avg_versions={v: {d: r["macro_f1_all13"] for d, r in rep_hv[v].items()} for v in variants},
                dev_13avg=None if rep_d is None else {d: v["macro_f1_all13"] for d, v in rep_d.items()},
            )
        )
        vs = (
            ""
            if len(variants) == 1
            else " | " + ", ".join(f"{v} {rep_hv[v]['all']['macro_f1_all13']:.3f}" for v in variants)
        )
        log(
            f"[{tag} {k + 1}/{K}] done in {minutes:.1f} min. Its {len(held_docs)} unseen judgments: {_short(rep_h)}"
            + vs
            + ("" if rep_d is None else f" | dev: {_short(rep_d)}")
        )
        del feats
        _free(device)

    # --- the K copies together
    oof, test_b, dev_b = oof_v[v0], test_bv[v0], dev_bv[v0]
    test_pb = _avg(test_b)
    rep_oof = D.evaluate(pool, T.to_preds(pool, oof, labels), labels)
    rep_oof_v = {v: D.evaluate(pool, T.to_preds(pool, oof_v[v], labels), labels) for v in variants}
    if len(variants) > 1:
        log(
            f"[{tag}] out-of-fold 13-role avg of each Stage B version: "
            + " | ".join(f"{v}: {_short(rep_oof_v[v])}" for v in variants)
        )
    rep_oof_a = D.evaluate(pool, T.to_preds(pool, oof_a, labels), labels)
    gold_p, top_p, n = 0.0, 0.0, 0
    for d, p in zip(pool, oof):
        if p is None:
            continue
        g = torch.tensor([li[l] for l in d["labels"]])
        gold_p += p[torch.arange(len(g)), g].sum().item()
        top_p += p.max(-1).values.sum().item()
        n += len(g)
    gold_p, top_p = gold_p / max(1, n), top_p / max(1, n)
    log("\n" + "=" * 78)
    if on_dev:
        rep_a, rep_b, dev_preds, ev_docs = rep_oof_a, rep_oof, T.to_preds(pool, oof, labels), pool
        log(f"OUT-OF-FOLD over all {len(pool)} labelled judgments (each predicted by the copy that did not see it)")
        log(D.format_report({"Stage A": rep_a, "Stage B": rep_b}, labels))
    else:
        dev_pa, dev_pb = _avg(dev_a), _avg(dev_b)
        rep_a = D.evaluate(dev, T.to_preds(dev, dev_pa, labels), labels)
        rep_b = D.evaluate(dev, T.to_preds(dev, dev_pb, labels), labels)
        dev_preds, ev_docs = T.to_preds(dev, dev_pb, labels), dev
        log(D.format_report({f"teacher A (avg of {K})": rep_a, f"teacher B (avg of {K})": rep_b}, labels))
        log(
            f"\n[teacher] answers for the training sentences (each from the copy that did not see it): "
            f"{_short(rep_oof)}"
        )
    log(f"[{tag}] out-of-fold: average probability given to the gold role {gold_p:.3f}, to the top role {top_p:.3f}")

    T._save_stage(out, "stageB", ev_docs, dev_preds, test, T.to_preds(test, test_pb, labels), rep_b, labels)
    L = len(labels)
    teacher = {
        "labels": labels,
        "train": {
            d["id"]: (torch.zeros(0, L) if p is None else torch.log(p.clamp_min(1e-12))) for d, p in zip(pool, oof)
        },
        "dev": {} if on_dev else {d["id"]: lp for d, lp in zip(dev, _log(_avg(dev_b))) if lp is not None},
        "test": {d["id"]: lp for d, lp in zip(test, _log(test_pb)) if lp is not None},
        "folds": [[pool[i]["id"] for i in f] for f in folds],
        "train_on_dev": on_dev,
        "config": cfg,
        "main_version": v0,
        "versions": {
            v: {
                "train": {
                    d["id"]: (torch.zeros(0, L) if p is None else torch.log(p.clamp_min(1e-12)))
                    for d, p in zip(pool, oof_v[v])
                },
                "test": {d["id"]: lp for d, lp in zip(test, _log(_avg(test_bv[v]))) if lp is not None},
            }
            for v in variants
        },
    }
    tpath = os.path.join(out, "teacher.pt")
    torch.save(teacher, tpath)
    total_min = (time.time() - t_start) / 60
    log(f"[{tag}] saved {tpath}; {K} copies in {total_min:.1f} min; submission {os.path.join(out, 'stageB')}")
    metrics = dict(
        run_name=cfg["run_name"],
        route=bb.route,
        kind=bb.kind,
        best="stageB",
        copies=K,
        train_on_dev=on_dev,
        scored_on="out-of-fold over train + dev" if on_dev else "dev",
        data={
            s: dict(docs=len(ds), sentences=sum(len(d["sents"]) for d in ds))
            for s, ds in (("train", train), ("dev", dev), ("test", test))
        },
        stage_a=dict(
            dev=rep_a,
            train_fit=None,
            trainable_params=n_trainable,
            total_params=n_total,
            peak_gpu_gb=peak,
            saved_mb=saved / 2**20,
            method=cfg["method"],
            dev_infer_seconds=None if on_dev else infer,
            history=first["history"],
            best_epoch=first["best_epoch"],
            train_seconds=time.time() - t_start,
        ),
        stage_b=dict(dev=rep_b),
        oof_train=dict(stage_b=rep_oof, stage_a=rep_oof_a, gold_prob=gold_p, top_prob=top_p),
        stage_b_versions={v: rep_oof_v[v] for v in variants},
        main_version=v0,
        folds=fold_info,
        config=cfg,
    )
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return dict(out_dir=out, teacher_path=tpath, metrics=metrics)


# --- experiment
def _delta(a, b, stage):
    return ", ".join(
        f"{d} {b[stage]['dev'][d]['macro_f1_all13'] - a[stage]['dev'][d]['macro_f1_all13']:+.4f}"
        for d in a[stage]["dev"]
    )


def kd_summary(results: dict, student_info: dict | None) -> str:
    t, s, k = results.get("teacher"), results.get("student"), results.get("student_kd")
    lines = ["KNOWLEDGE DISTILLATION SUMMARY"]
    if t:
        K = t.get("copies", 1)
        per = t["stage_a"]["total_params"]
        lines.append(
            f"teacher: average of {K} copies of the whole Phase 1 model + {t['stage_a']['method']} + BiGRU; "
            f"{per / 1e6:.1f}M weights each, {K * per / 1e6:.1f}M in total"
        )
        o = t.get("oof_train", {})
        if o:
            lines.append(
                f"teacher answers on its unseen training judgments: {_short(o['stage_b'])}; "
                f"average probability of the gold role {o['gold_prob']:.3f}, of the top role "
                f"{o['top_prob']:.3f}"
            )
    if student_info:
        lines.append(
            f"student: first {student_info['layers_student']} of {student_info['layers_teacher']} blocks "
            f"('{student_info['blocks']}'), {student_info['params_student'] / 1e6:.1f}M weights "
            f"({student_info['params_student'] / student_info['params_teacher']:.0%} of one teacher copy)"
        )
    if s and k:
        lines.append("effect of the teacher term (student_kd minus student), dev 13-role avg:")
        lines.append(f"  Stage A: {_delta(s, k, 'stage_a')}")
        lines.append(f"  Stage B: {_delta(s, k, 'stage_b')}")
    ts = (t or {}).get("stage_a", {}).get("dev_infer_seconds")
    ss = (s or {}).get("stage_a", {}).get("dev_infer_seconds")
    if ts is not None and ss is not None:
        lines.append(f"dev pass (Stage A over every dev sentence): teacher {ts:.1f}s for all copies, student {ss:.1f}s")
    return "\n".join(lines)


def run_kd(
    bb,
    tokenizer,
    prefix: str = "kd",
    keep=0.5,
    folds: int = 5,
    method: str = "lora",
    skip_done: bool = True,
    **settings,
) -> dict:
    """The whole experiment: teacher (out-of-fold) -> smaller student without and with the teacher term -> comparison
    (<prefix>_comparison.txt/.json)."""
    for bad in ("run_name", "kd_teacher", "method", "folds"):
        if bad in settings:
            raise ValueError(f"run_kd sets {bad} itself")
    cfg0 = T.resolve_cfg(bb, {**settings, "method": method})
    out_dir = cfg0["out_dir"]
    results: dict = {}

    tname = f"{prefix}_teacher"
    tpath = os.path.join(out_dir, tname, "teacher.pt")
    tmet = os.path.join(out_dir, tname, "metrics.json")
    if skip_done and os.path.exists(tpath) and os.path.exists(tmet):
        with open(tmet) as f:
            results["teacher"] = json.load(f)
        print(f"===== {tname}: already done, using {tpath}")
    else:
        results["teacher"] = teacher_oof(bb, run_name=tname, method=method, folds=folds, **settings)["metrics"]
    _free(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    sbb, info = make_student(bb, tokenizer, keep, verbose=False)
    print(
        f"[student] first {info['layers_student']} of {info['layers_teacher']} blocks of '{info['blocks']}': "
        f"{info['params_student'] / 1e6:.1f}M weights (teacher model {info['params_teacher'] / 1e6:.1f}M); "
        f"hidden states: {info['route']}"
        + (f"; layer count changed in {info['layer_count_fields']}" if info["layer_count_fields"] else "")
    )
    for key, extra in (("student", {}), ("student_kd", {"kd_teacher": tpath})):
        name = f"{prefix}_{key}"
        path = os.path.join(out_dir, name, "metrics.json")
        if skip_done and os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)
            print(f"===== {name}: already done, using {path}")
            continue
        res = None
        try:
            res = T.run(sbb, run_name=name, method=method, **extra, **settings)
            results[key] = res["metrics"]
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"===== {name} FAILED (error above)")
            results[key] = None
        finally:
            del res
            _free(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for key in ("student", "student_kd"):
        if results.get(key):
            results[key]["kd"] = dict(student=info, teacher=tpath if key == "student_kd" else None)
    P.save_comparison(results, out_dir, prefix, extra_text=kd_summary(results, info))
    return results
