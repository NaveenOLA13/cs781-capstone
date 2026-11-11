"""Run every fine-tuning technique on one Phase 1 model; with 2 GPUs, two runs at a time (one per GPU)."""

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

import cj_tech as T  # noqa: E402

COST = {
    "frozen": 1.0,
    "full": 3.0,
    "partial": 2.0,
    "unfreeze": 2.4,
    "bitfit": 2.0,
    "lorma": 2.3,
    "adalora": 2.4,
    "adapter": 2.0,
}


def split_jobs(techniques, n: int) -> list[list[str]]:
    """Techniques shared between n workers so the expected time is about equal (largest first)."""
    loads = [0.0] * n
    jobs: list[list[str]] = [[] for _ in range(n)]
    for t in sorted(techniques, key=lambda t: -COST.get(t, 2.0)):
        i = loads.index(min(loads))
        jobs[i].append(t)
        loads[i] += COST.get(t, 2.0)
    return [j for j in jobs if j]


def _out_dir(settings):
    import cj_train

    return settings.get("out_dir", cj_train.DEFAULTS["out_dir"])


def _tail(path, n=1):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


def run_grid(
    model: str,
    techniques=T.TECHNIQUES,
    gpus: int | None = None,
    ckpt: str | None = None,
    skip_done: bool = True,
    poll: int = 60,
    **settings,
) -> dict:
    import torch

    import cj_train
    from cj_models import find_checkpoint, load_tokenizer
    from rr_backbone import make_encode

    unknown = set(settings) - set(cj_train.DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    bad = [t for t in techniques if t not in T.TECHNIQUES]
    if bad:
        raise ValueError(f"unknown techniques {bad}; choose from {T.TECHNIQUES}")
    out_dir = _out_dir(settings)
    os.makedirs(out_dir, exist_ok=True)
    todo = [t for t in techniques if not (skip_done and cj_train.load_metrics(out_dir, f"{model}_{t}") is not None)]
    done = [t for t in techniques if t not in todo]
    if done:
        print(f"[grid] already finished (skipped): {', '.join(done)}")
    if todo:
        ckpt = ckpt or find_checkpoint(model)
        cfg = cj_train.resolve_cfg({k: v for k, v in settings.items() if k != "technique"})
        cj_train.prepare_tokens(make_encode(load_tokenizer()), cfg)
        n_gpu = torch.cuda.device_count() if gpus is None else gpus
        jobs = split_jobs(todo, max(1, n_gpu))
        procs = []
        for i, job in enumerate(jobs):
            spec = os.path.join(out_dir, f"_{model}_worker{i}.json")
            with open(spec, "w") as f:
                json.dump(dict(model=model, ckpt=ckpt, techniques=job, settings=settings), f)
            log = os.path.join(out_dir, f"_{model}_worker{i}.log")
            env = dict(os.environ)
            if n_gpu > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(i % n_gpu)
            fh = open(log, "w")
            procs.append(
                (
                    subprocess.Popen(
                        [sys.executable, os.path.join(HERE, "cj_grid.py"), "--worker", spec],
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=HERE,
                    ),
                    fh,
                    log,
                    job,
                )
            )
            print(f"[grid] worker {i} ({'GPU ' + str(i % n_gpu) if n_gpu else 'CPU'}): {', '.join(job)}  (log: {log})")
        t0 = time.time()
        while any(p.poll() is None for p, *_ in procs):
            time.sleep(poll)
            for i, (p, _, log, _) in enumerate(procs):
                state = "running" if p.poll() is None else f"finished (exit {p.returncode})"
                last = _tail(log)
                print(f"[grid {(time.time() - t0) / 60:5.1f} min] worker {i} {state}: {last[0][:150] if last else ''}")
        for i, (p, fh, log, _job) in enumerate(procs):
            fh.close()
            if p.returncode != 0:
                print(f"[grid] worker {i} stopped with an error; last lines of {log}:")
                print("\n".join(_tail(log, 25)))
    rows = [m for t in techniques if (m := cj_train.load_metrics(out_dir, f"{model}_{t}")) is not None]
    missing = [t for t in techniques if cj_train.load_metrics(out_dir, f"{model}_{t}") is None]
    table = cj_train.comparison(rows) if rows else "no finished runs"
    if missing:
        table += f"\nnot finished: {', '.join(missing)} (see their log.txt)"
    if rows:
        best = max(rows, key=lambda m: m["dev_tuned"]["macro_f1"])
        table += (
            f"\nbest on dev: {best['run_name']} (macro-F1 {best['dev_tuned']['macro_f1']:.4f}) -> submit "
            f"{best['submissions']['tuned']}"
        )
    print(f"\n===== {model}: {len(rows)} of {len(techniques)} techniques\n{table}")
    with open(os.path.join(out_dir, f"{model}_grid.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out_dir, f"{model}_grid.json"), "w") as f:
        json.dump(
            {
                m["run_name"]: {
                    k: m[k]
                    for k in (
                        "technique",
                        "dev_050",
                        "dev_tuned",
                        "threshold",
                        "train_fit",
                        "trainable_params",
                        "total_params",
                        "trainable_share",
                        "peak_gpu_gb",
                        "train_seconds",
                        "dev_infer_seconds",
                        "saved_mb",
                        "best_epoch",
                        "technique_info",
                        "submissions",
                    )
                }
                for m in rows
            },
            f,
            indent=1,
            default=str,
        )
    return {m["technique"]: m for m in rows}


def compare_all(out_dir: str | None = None) -> str:
    """One table of every finished run in out_dir (all models), best first; saved as all_runs.txt."""
    import cj_train

    out_dir = out_dir or cj_train.DEFAULTS["out_dir"]
    rows = []
    for name in sorted(os.listdir(out_dir)):
        m = cj_train.load_metrics(out_dir, name)
        if m is not None:
            rows.append(m)
    table = cj_train.comparison(rows) if rows else "no finished runs"
    with open(os.path.join(out_dir, "all_runs.txt"), "w") as f:
        f.write(table + "\n")
    print(table)
    return table


def _worker(spec_path: str) -> int:
    with open(spec_path) as f:
        spec = json.load(f)
    import cj_train
    from cj_models import load_backbone

    settings = {**spec["settings"], "ckpt_path": spec["ckpt"]}
    bb, info = load_backbone(
        spec["model"], max_len=settings.get("max_len", cj_train.DEFAULTS["max_len"]), path=spec["ckpt"]
    )
    if info["max_len"] != settings.get("max_len", cj_train.DEFAULTS["max_len"]):
        settings = {**settings, "max_len": info["max_len"]}
    failed = []
    for t in spec["techniques"]:
        try:
            cj_train.run(bb, name=spec["model"], technique=t, **settings)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[worker] {spec['model']} + {t} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(t)
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    sys.exit(_worker(ap.parse_args().worker))
