"""Phase 7 runs on Kaggle: fine-tuning techniques of one Phase 1 model (one process per GPU), the knowledge
distillation pair (half-size student with / without the teacher), extra experiments, ensembles, the results table."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p7_paths  # noqa: E402,F401
import pcr_data as PD  # noqa: E402

OUT_DIR = "/kaggle/working/pcr_runs"
CACHE_DIR = "/kaggle/working/pcr_cache"
COST = {
    "frozen": 1.5,
    "full": 3.0,
    "partial": 2.2,
    "unfreeze": 2.4,
    "bitfit": 2.0,
    "lorma": 2.6,
    "adalora": 2.6,
    "adapter": 2.4,
    "lora": 2.6,
    "qlora": 2.9,
    "ia3": 2.3,
}


def _tail(path, n=1):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


def _done(out_dir, name):
    return os.path.exists(os.path.join(out_dir, name, "metrics.json"))


def prepare(settings: dict | None = None, log=print) -> str:
    """Tokens of every case (our tokenizer) and the BM25 matrices, computed once and cached; returns the path of the
    lexical file."""
    import pcr_train
    from cj_models import load_tokenizer
    from rr_backbone import make_encode

    cfg = pcr_train.resolve_cfg(
        {k: v for k, v in (settings or {}).items() if k not in ("technique", "run_name", "kd_teacher")}
    )
    pcr_train.prepare(make_encode(load_tokenizer(log=lambda *a: None)), cfg, log=log)
    return cfg["lexical_path"] or os.path.join(cfg["cache_dir"], "lexical.pkl")


def run_jobs(model: str, jobs: list[dict], ckpt: str, out_dir: str, gpus: int | None = None, poll: int = 60) -> None:
    import torch

    n_gpu = torch.cuda.device_count() if gpus is None else gpus
    n = max(1, n_gpu)
    loads, per = [0.0] * n, [[] for _ in range(n)]
    for j in sorted(jobs, key=lambda j: -COST.get(j["technique"], 2.5)):
        i = loads.index(min(loads))
        per[i].append(j)
        loads[i] += COST.get(j["technique"], 2.5)
    procs = []
    os.makedirs(out_dir, exist_ok=True)
    for i, js in enumerate(per):
        if not js:
            continue
        spec = os.path.join(out_dir, f"_{model}_worker{i}.json")
        with open(spec, "w") as f:
            json.dump(dict(model=model, ckpt=ckpt, jobs=js), f)
        log = os.path.join(out_dir, f"_{model}_worker{i}.log")
        env = dict(os.environ)
        if n_gpu > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        fh = open(log, "w")
        procs.append(
            (
                subprocess.Popen(
                    [sys.executable, "-u", os.path.join(HERE, "pcr_grid.py"), "--worker", spec],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=HERE,
                ),
                fh,
                log,
            )
        )
        print(
            f"[pcr] worker {i} ({'GPU ' + str(i) if n_gpu else 'CPU'}): {', '.join(j['run_name'] for j in js)}  "
            f"(log: {log})",
            flush=True,
        )
    t0 = time.time()
    while any(p.poll() is None for p, *_ in procs):
        time.sleep(poll)
        for i, (p, _, log) in enumerate(procs):
            state = "running" if p.poll() is None else f"finished (exit {p.returncode})"
            last = _tail(log)
            print(
                f"[pcr {(time.time() - t0) / 60:5.1f} min] worker {i} {state}: {last[0][:160] if last else ''}",
                flush=True,
            )
    for i, (p, fh, log) in enumerate(procs):
        fh.close()
        if p.returncode != 0:
            print(f"[pcr] worker {i} stopped with an error; last lines of {log}:")
            print("\n".join(_tail(log, 30)))


def _table(out_dir, names, title):
    import pcr_train

    rows = [m for n in names if (m := pcr_train.load_metrics(out_dir, n)) is not None]
    missing = [n for n in names if pcr_train.load_metrics(out_dir, n) is None]
    text = pcr_train.comparison(rows) if rows else "no finished runs"
    if missing:
        text += f"\nnot finished: {', '.join(missing)} (see their log.txt)"
    text = f"\n===== {title}\n{text}"
    print(text)
    return text


def run_variants(
    model: str,
    variants: dict,
    ckpt: str | None = None,
    out_dir: str = OUT_DIR,
    gpus: int | None = None,
    skip_done: bool = True,
    poll: int = 60,
    **settings,
) -> str:
    """Runs {run name: settings (incl. technique)} of one model; `settings` are shared by all of them."""
    import pcr_train

    allk = set(settings) | {k for v in variants.values() for k in v}
    unknown = allk - set(pcr_train.DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    names = list(variants)
    todo = [n for n in names if not (skip_done and _done(out_dir, n))]
    if todo:
        if not ckpt:
            from cj_models import find_checkpoint

            ckpt = find_checkpoint(model)
        prepare({**settings, **variants[todo[0]]})
        jobs = [
            dict(
                run_name=n,
                technique=variants[n].get("technique", settings.get("technique", "lorma")),
                settings=dict(settings, **variants[n], out_dir=out_dir),
            )
            for n in todo
        ]
        run_jobs(model, jobs, ckpt, out_dir, gpus, poll)
    return _table(out_dir, names, f"{model}: {', '.join(names)}")


def run_grid(
    model: str,
    techniques=("full", "lorma"),
    ckpt: str | None = None,
    out_dir: str = OUT_DIR,
    gpus: int | None = None,
    skip_done: bool = True,
    poll: int = 60,
    **settings,
) -> str:
    """Every technique on one Phase 1 model; runs are named <model>_<technique>."""
    return run_variants(
        model, {f"{model}_{t}": dict(technique=t) for t in techniques}, ckpt, out_dir, gpus, skip_done, poll, **settings
    )


def best_run(out_dir: str, names, need_vectors: bool = True) -> str | None:
    """The finished run (folder) with the best dev micro-F1@7 of the model alone ('mean'), among names."""
    import pcr_train

    rows = [
        (m["dev_dense"]["mean"]["f1"], os.path.join(out_dir, n))
        for n in names
        if (m := pcr_train.load_metrics(out_dir, n)) is not None
        and (not need_vectors or os.path.exists(os.path.join(out_dir, n, "teacher_vectors.pt")))
    ]
    return max(rows)[1] if rows else None


def run_kd(
    teacher_dir: str,
    student_layers=0.5,
    kd_alpha: float = 0.5,
    kd_T: float = 1.0,
    kd_loss: str = "listwise",
    technique: str = "full",
    plain: bool = True,
    out_dir: str = OUT_DIR,
    gpus: int | None = None,
    skip_done: bool = True,
    poll: int = 60,
    **settings,
) -> str:
    """Knowledge distillation: teacher = a finished full-size run (with teacher_vectors.pt); student = the same
    starting model with student_layers of its blocks (evenly spaced), trained with the teacher's similarities."""
    with open(os.path.join(teacher_dir, "config.json")) as f:
        tcfg = json.load(f)
    with open(os.path.join(teacher_dir, "metrics.json")) as f:
        tm = json.load(f)
    model, ckpt = tm["model"], tcfg["ckpt_path"]
    base = {
        k: tcfg[k]
        for k in (
            "data_root",
            "cache_dir",
            "lexical_path",
            "max_len",
            "chunk_len",
            "dim",
            "pooling",
            "loss",
            "tau",
            "hard_negs",
            "neg_top",
            "q_chunks",
            "d_chunks",
            "batch_queries",
            "max_chunks",
            "val_chunks",
            "seed",
            "fetch_data",
            "limit_train",
            "enc_batch",
            "eval_batch",
            "epochs",
            "val_every",
        )
        if k in tcfg
    }
    base.update(settings)
    base.update(student_layers=student_layers, technique=technique)
    variants = {
        f"{model}_student_kd": dict(base, kd_teacher=teacher_dir, kd_alpha=kd_alpha, kd_T=kd_T, kd_loss=kd_loss)
    }
    if plain:
        variants[f"{model}_student"] = dict(base)
    print(
        f"[kd] teacher {teacher_dir} ({tcfg['technique']}): dev micro-F1@7 alone {tm['dev_dense']['mean']['f1']:.4f}; "
        f"student: {model} with {student_layers} of its blocks, {technique}; teacher term {kd_loss} (alpha "
        f"{kd_alpha}, T {kd_T})"
    )
    text = run_variants(model, variants, ckpt, out_dir, gpus, skip_done, poll)
    text = text.replace(f"===== {model}:", f"===== knowledge distillation (teacher {os.path.basename(teacher_dir)}):")
    with open(os.path.join(out_dir, f"{model}_kd.txt"), "w") as f:
        f.write(text + "\n")
    return text


def _zrow(M):
    mu, sd = M.mean(1, keepdims=True), M.std(1, keepdims=True)
    return (M - mu) / np.where(sd > 1e-9, sd, 1.0)


def ensemble(
    run_dirs: list[str], out_dir: str, lexical_path: str | None = None, aggs=("mean", "maxsim"), log=print
) -> dict | None:
    """Several finished runs together (CPU): for each similarity, the average over runs of the per-query standardised
    scores is one feature next to the lexical ones; the ranker is cross-fitted / fitted on dev as for single runs."""
    lexical_path = lexical_path or os.path.join(CACHE_DIR, "lexical.pkl")
    L = PD.load_lexical(lexical_path)
    ok = [r for r in run_dirs if os.path.exists(os.path.join(r, "scores_dev.npz"))]
    if not ok:
        log("[ensemble] no finished runs with scores")
        return None
    dev, test = {}, {}
    for a in aggs:
        zd, zt = [], []
        for r in ok:
            with open(os.path.join(r, "ids.json")) as f:
                ids = json.load(f)
            if ids["dev"]["qids"] != L["dev"]["qids"] or ids["test"]["pids"] != L["test"]["pids"]:
                raise RuntimeError(f"{r} was scored on other data")
            sd, st = np.load(os.path.join(r, "scores_dev.npz")), np.load(os.path.join(r, "scores_test.npz"))
            if a in sd:
                zd.append(_zrow(sd[a]))
                zt.append(_zrow(st[a]))
        if zd:
            dev[f"ens_{a}"], test[f"ens_{a}"] = np.mean(zd, 0), np.mean(zt, 0)
    log(f"[ensemble] {len(ok)} runs: {', '.join(os.path.basename(r) for r in ok)}")
    res = PD.fused_submission(L, dev, test, out_dir, log=log)
    res["runs"] = ok
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ensemble.json"), "w") as f:
        json.dump({k: v for k, v in res.items()}, f, indent=1, default=float)
    return res


def report(base_dirs, out_path: str | None = None) -> str:
    """One table of every finished run under base_dirs (best fused dev score first), the lexical baselines, the
    ensembles and the submission files."""
    import pcr_train

    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    rows, lex, ens = [], [], []
    for base in base_dirs:
        for root, _, files in os.walk(base):
            rel = os.path.relpath(root, base) if len(base_dirs) == 1 else root
            if "metrics.json" in files:
                with open(os.path.join(root, "metrics.json")) as f:
                    m = json.load(f)
                if "dev_dense" in m:
                    m["run_name"] = rel
                    rows.append(m)
            if "lexical_results.json" in files:
                with open(os.path.join(root, "lexical_results.json")) as f:
                    lex.append((root, json.load(f)))
            if "ensemble.json" in files:
                with open(os.path.join(root, "ensemble.json")) as f:
                    ens.append((rel, json.load(f)))
    parts = []
    for root, r in lex:
        parts.append(
            "lexical baselines (no neural model), dev micro-F1@7:\n"
            + "\n".join(
                f"  BM25 {'averaged over the orders' if k.endswith('_sum') else k.split('_')[-1] + '-grams'} alone: "
                f"{v['f1']:.4f}"
                for k, v in r["single"].items()
                if k.startswith("dev_")
            )
            + f"\n  ranker on BM25 + year + citation counts (fitted on train): "
            f"{r['lexical_ltr']['dev']['f1']:.4f} -> {r['lexical_ltr']['submission']}\n  BM25 averaged: "
            f"{r['bm25_avg']['dev']['f1']:.4f} -> {r['bm25_avg']['submission']}"
        )
    parts.append(pcr_train.comparison(rows) if rows else "no finished runs")
    for name, r in ens:
        parts.append(
            f"ensemble {name} ({len(r.get('runs', []))} runs): dev fused (cross-fitted) {r['dev']['f1']:.4f}"
            f" -> {r.get('submission')}"
        )
    subs = sorted(
        [((m.get("dev_fused") or {}).get("f1") or 0.0, m.get("submission") or m.get("dense_submission")) for m in rows]
        + [(r["dev"]["f1"], r.get("submission")) for _, r in ens],
        key=lambda t: -t[0],
    )
    parts.append("submission files (best dev first):\n" + "\n".join(f"{f:.4f}  {s}" for f, s in subs if s))
    text = "\n\n".join(parts)
    print(text)
    out_path = out_path or os.path.join(base_dirs[0], "all_results.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    return text


def _worker(spec_path: str) -> int:
    with open(spec_path) as f:
        spec = json.load(f)
    import pcr_train
    from cj_models import load_backbone

    max_len = spec["jobs"][0]["settings"].get("max_len", pcr_train.DEFAULTS["max_len"])
    bb, info = load_backbone(spec["model"], max_len=max_len, path=spec["ckpt"])
    failed = []
    for j in spec["jobs"]:
        s = dict(
            j["settings"],
            ckpt_path=spec["ckpt"],
            run_name=j["run_name"],
            technique=j["technique"],
            max_len=info["max_len"],
        )
        try:
            pcr_train.run(bb, name=spec["model"], **s)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[worker] {j['run_name']} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(j["run_name"])
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    sys.exit(_worker(ap.parse_args().worker))
