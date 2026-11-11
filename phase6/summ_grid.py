"""Phase 6 runs on Kaggle: fine-tuning techniques of one Phase 1 model (one process per GPU), the knowledge
distillation pair (half-size student with / without the teacher) and the results table."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import p6_paths  # noqa: E402,F401

OUT_DIR = "/kaggle/working/summ_runs"
COST = {
    "frozen": 1.0,
    "full": 3.0,
    "partial": 2.0,
    "unfreeze": 2.4,
    "bitfit": 2.0,
    "lorma": 2.3,
    "adalora": 2.4,
    "adapter": 2.0,
    "lora": 2.3,
    "qlora": 2.6,
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


def _prepare(model, ckpt, settings):
    """Windows and pairs prepared (and cached) once, so the workers do not tokenise at the same time."""
    import summ_train
    from cj_mid import is_encoder_decoder
    from cj_models import load_backbone

    cfg = summ_train.resolve_cfg(
        {k: v for k, v in settings.items() if k not in ("technique", "run_name", "kd_teacher")}
    )
    bb, info = load_backbone(model, max_len=cfg["max_len"], path=ckpt, log=lambda *a: None)
    cfg["max_len"] = info["max_len"]
    summ_train.prepare(bb.encode, cfg, is_encoder_decoder(bb.model), log=print)
    return info["max_len"]


def run_jobs(model: str, jobs: list[dict], ckpt: str, out_dir: str, gpus: int | None = None, poll: int = 60) -> None:
    import torch

    n_gpu = torch.cuda.device_count() if gpus is None else gpus
    n = max(1, n_gpu)
    loads, per = [0.0] * n, [[] for _ in range(n)]
    for j in sorted(jobs, key=lambda j: -COST.get(j["technique"], 2.0)):
        i = loads.index(min(loads))
        per[i].append(j)
        loads[i] += COST.get(j["technique"], 2.0)
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
                    [sys.executable, "-u", os.path.join(HERE, "summ_grid.py"), "--worker", spec],
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
            f"[summ] worker {i} ({'GPU ' + str(i) if n_gpu else 'CPU'}): {', '.join(j['run_name'] for j in js)}  "
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
                f"[summ {(time.time() - t0) / 60:5.1f} min] worker {i} {state}: {last[0][:160] if last else ''}",
                flush=True,
            )
    for i, (p, fh, log) in enumerate(procs):
        fh.close()
        if p.returncode != 0:
            print(f"[summ] worker {i} stopped with an error; last lines of {log}:")
            print("\n".join(_tail(log, 30)))


def _table(out_dir, names, title):
    import summ_train

    rows = [m for n in names if (m := summ_train.load_metrics(out_dir, n)) is not None]
    missing = [n for n in names if summ_train.load_metrics(out_dir, n) is None]
    text = summ_train.comparison(rows) if rows else "no finished runs"
    if missing:
        text += f"\nnot finished: {', '.join(missing)} (see their log.txt)"
    text = f"\n===== {title}\n{text}"
    print(text)
    return text


def run_grid(
    model: str,
    techniques=("lora", "lorma"),
    ckpt: str | None = None,
    out_dir: str = OUT_DIR,
    gpus: int | None = None,
    skip_done: bool = True,
    poll: int = 60,
    **settings,
) -> str:
    """Every technique on one Phase 1 model; runs are named <model>_<technique>."""
    import summ_train

    unknown = set(settings) - set(summ_train.DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    names = [f"{model}_{t}" for t in techniques]
    todo = [t for t, n in zip(techniques, names) if not (skip_done and _done(out_dir, n))]
    if todo:
        if not ckpt:
            from cj_models import find_checkpoint

            ckpt = find_checkpoint(model)
        settings = dict(settings, max_len=_prepare(model, ckpt, settings))
        jobs = [dict(run_name=f"{model}_{t}", technique=t, settings=dict(settings, out_dir=out_dir)) for t in todo]
        run_jobs(model, jobs, ckpt, out_dir, gpus, poll)
    return _table(out_dir, names, f"{model}: {', '.join(techniques)}")


def best_run(out_dir: str, names) -> str | None:
    """The finished run (folder) with the best held-out (ROUGE-1 + ROUGE-L) / 2 among names."""
    import summ_train

    rows = [
        (m["held"]["rouge_mean"], os.path.join(out_dir, n))
        for n in names
        if (m := summ_train.load_metrics(out_dir, n)) is not None
        and os.path.exists(os.path.join(out_dir, n, "trained.pt"))
    ]
    return max(rows)[1] if rows else None


def run_kd(
    teacher_dir: str,
    student_layers=0.5,
    kd_alpha: float = 0.5,
    kd_T: float = 2.0,
    technique: str = "full",
    plain: bool = True,
    out_dir: str = OUT_DIR,
    gpus: int | None = None,
    skip_done: bool = True,
    poll: int = 60,
    **settings,
) -> str:
    """Knowledge distillation: teacher = a finished run (full size); student = the same starting model with
    student_layers of its blocks (evenly spaced), trained with the teacher (<model>_student_kd) and, for comparison,
    without it."""
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
            "max_len",
            "tgt_tokens",
            "enc_window",
            "empty_frac",
            "heldout_frac",
            "seed",
            "fetch_data",
            "beams",
            "no_repeat",
            "len_pen",
            "eval_docs",
            "batch_size",
            "grad_accum",
            "gen_batch",
            "limit_train",
        )
        if k in tcfg
    }
    base.update(settings)
    base.update(out_dir=out_dir, student_layers=student_layers)
    jobs = [
        dict(
            run_name=f"{model}_student_kd",
            technique=technique,
            settings=dict(base, kd_teacher=teacher_dir, kd_alpha=kd_alpha, kd_T=kd_T),
        )
    ]
    if plain:
        jobs.append(dict(run_name=f"{model}_student", technique=technique, settings=dict(base)))
    print(
        f"[kd] teacher {teacher_dir} ({tcfg['technique']}): held-out ROUGE-1 {tm['held']['rouge1']:.4f}, "
        f"ROUGE-L {tm['held']['rougeL']:.4f}; student: {model} with {student_layers} of its blocks, {technique}"
    )
    names = [j["run_name"] for j in jobs]
    todo = [j for j in jobs if not (skip_done and _done(out_dir, j["run_name"]))]
    if todo:
        run_jobs(model, todo, ckpt, out_dir, gpus, poll)
    text = _table(out_dir, names, f"knowledge distillation (teacher {os.path.basename(teacher_dir)})")
    with open(os.path.join(out_dir, f"{model}_kd.txt"), "w") as f:
        f.write(text + "\n")
    return text


def report(base_dirs, out_path: str | None = None) -> str:
    """One table of every finished run under base_dirs (best held-out ROUGE first) + the submission files."""
    import summ_train

    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    rows = []
    for base in base_dirs:
        for root, _, files in os.walk(base):
            if "metrics.json" in files:
                with open(os.path.join(root, "metrics.json")) as f:
                    m = json.load(f)
                if "held" in m:
                    m["run_name"] = os.path.relpath(root, base) if len(base_dirs) == 1 else root
                    rows.append(m)
    if not rows:
        text = "no finished runs"
    else:
        text = summ_train.comparison(rows)
        text += "\n\nsubmission files (best held-out first):\n" + "\n".join(
            f"{m['held']['rouge_mean']:.4f}  {m['submission']}"
            for m in sorted(rows, key=lambda m: -m["held"]["rouge_mean"])
            if m.get("submission")
        )
    print(text)
    out_path = out_path or os.path.join(base_dirs[0], "all_results.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    return text


def _worker(spec_path: str) -> int:
    with open(spec_path) as f:
        spec = json.load(f)
    import summ_train
    from cj_models import load_backbone

    max_len = spec["jobs"][0]["settings"].get("max_len", summ_train.DEFAULTS["max_len"])
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
            summ_train.run(bb, name=spec["model"], **s)
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
