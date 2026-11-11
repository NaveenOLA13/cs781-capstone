"""Phase 4 knowledge distillation: a stronger teacher -> a smaller, cheaper student."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COPY_KEYS = (
    "data_root",
    "cache_dir",
    "max_len",
    "window",
    "train_chunks",
    "chunk_stride",
    "epochs",
    "batch_size",
    "class_weight_power",
    "seed",
    "fetch_data",
    "dropout",
    "head",
    "pooling",
)


def _read(path):
    with open(path) as f:
        return json.load(f)


def find_teacher(path: str) -> tuple[str, dict]:
    """(train_probs.json, its metrics) of the best-on-dev version under path; path may also be the file itself."""
    if os.path.isfile(path):
        cands = [os.path.dirname(path)]
    else:
        cands = [root for root, _, files in os.walk(path) if "train_probs.json" in files and "metrics.json" in files]
    if not cands:
        raise FileNotFoundError(f"no train_probs.json (+ metrics.json) under {path}: run cj_hier on the teacher first")
    scored = [(c, _read(os.path.join(c, "metrics.json"))) for c in cands]
    best_dir, best = max(scored, key=lambda cm: cm[1]["dev_tuned"]["macro_f1"])
    return os.path.join(best_dir, "train_probs.json"), best


def sft_run_of(teacher_file: str) -> str:
    """The SFT run folder above <run>/hier_k8/<version>/train_probs.json (the first one with config.json)."""
    d = os.path.dirname(os.path.abspath(teacher_file))
    for _ in range(5):
        if os.path.exists(os.path.join(d, "config.json")) and os.path.exists(os.path.join(d, "metrics.json")):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError(f"no SFT run (config.json + metrics.json) above {teacher_file}")


def _tail(path, n=1):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


def plan(
    teacher: str,
    model: str | None = None,
    ckpt: str | None = None,
    out_dir: str = "/kaggle/working/cj_kd",
    technique: str = "full",
    student_layers=0.5,
    kd_alpha: float = 0.5,
    kd_T: float = 2.0,
    plain: bool = True,
    **settings,
) -> dict:
    """Teacher, starting checkpoint and the worker jobs (no training)."""
    t_file, t_metrics = find_teacher(teacher)
    sft = sft_run_of(t_file)
    scfg, smet = _read(os.path.join(sft, "config.json")), _read(os.path.join(sft, "metrics.json"))
    model = model or smet["model"]
    ckpt = ckpt or scfg.get("ckpt_path")
    if not ckpt:
        from cj_models import find_checkpoint

        ckpt = find_checkpoint(model)
    base = {k: scfg[k] for k in COPY_KEYS if k in scfg}
    base.update(settings)
    base.update(out_dir=out_dir, student_layers=student_layers)
    jobs = [
        dict(
            run_name=f"{model}_student_kd",
            settings=dict(base, run_name=f"{model}_student_kd", kd_teacher=t_file, kd_alpha=kd_alpha, kd_T=kd_T),
        )
    ]
    if plain:
        jobs.append(dict(run_name=f"{model}_student", settings=dict(base, run_name=f"{model}_student")))
    return dict(
        teacher_file=t_file,
        teacher=t_metrics,
        sft_dir=sft,
        sft=smet,
        model=model,
        ckpt=ckpt,
        technique=technique,
        out_dir=out_dir,
        jobs=jobs,
    )


def run_kd(
    teacher: str,
    model: str | None = None,
    ckpt: str | None = None,
    out_dir: str = "/kaggle/working/cj_kd",
    technique: str = "full",
    student_layers=0.5,
    kd_alpha: float = 0.5,
    kd_T: float = 2.0,
    plain: bool = True,
    gpus: int | None = None,
    skip_done: bool = True,
    poll: int = 60,
    **settings,
) -> str:
    p = plan(teacher, model, ckpt, out_dir, technique, student_layers, kd_alpha, kd_T, plain, **settings)
    os.makedirs(out_dir, exist_ok=True)
    t = p["teacher"]
    print(
        f"[kd] teacher: {p['teacher_file']}\n     version '{t.get('version')}' of SFT run {p['sft_dir']}, "
        f"dev macro-F1 {t['dev_tuned']['macro_f1']:.4f} (tuned), {t['dev_050']['macro_f1']:.4f} at 0.5"
    )
    print(
        f"[kd] student: {p['model']} from {p['ckpt']}, first {student_layers} of its blocks, technique "
        f"{technique}; loss = {1 - kd_alpha:.2f} x gold + {kd_alpha:.2f} x teacher (temperature {kd_T})"
        + ("; also the same student without the teacher" if plain else "")
    )
    todo = [
        j for j in p["jobs"] if not (skip_done and os.path.exists(os.path.join(out_dir, j["run_name"], "metrics.json")))
    ]
    for j in p["jobs"]:
        if j not in todo:
            print(f"[kd] already finished (skipped): {j['run_name']}")
    if gpus is None:
        import torch

        gpus = torch.cuda.device_count()
    queue, procs, t0 = list(todo), [], time.time()
    while queue or procs:
        while queue and len(procs) < max(1, gpus):
            busy = {g for _, g, _, _ in procs}
            gpu = next(g for g in range(max(1, gpus)) if g not in busy)
            job = queue.pop(0)
            spec = os.path.join(out_dir, f"_{job['run_name']}_worker.json")
            with open(spec, "w") as f:
                json.dump(dict(model=p["model"], ckpt=p["ckpt"], techniques=[technique], settings=job["settings"]), f)
            log = os.path.join(out_dir, f"_{job['run_name']}_worker.log")
            env = dict(os.environ)
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            fh = open(log, "w")
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.join(HERE, "cj_grid.py"), "--worker", spec],
                stdout=fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=HERE,
            )
            procs.append((proc, gpu, fh, job["run_name"]))
            print(f"[kd] {job['run_name']} on {'GPU ' + str(gpu) if gpus else 'CPU'} (log: {log})", flush=True)
        time.sleep(poll)
        for proc, gpu, fh, name in list(procs):
            state = "running" if proc.poll() is None else f"finished (exit {proc.returncode})"
            last = _tail(os.path.join(out_dir, f"_{name}_worker.log"))
            print(
                f"[kd {(time.time() - t0) / 60:5.1f} min] {name} {state}: {last[0][:150] if last else ''}", flush=True
            )
            if proc.poll() is not None:
                fh.close()
                procs.remove((proc, gpu, fh, name))
                if proc.returncode != 0:
                    print("\n".join(_tail(os.path.join(out_dir, f"_{name}_worker.log"), 25)))
    return summary(p)


def summary(p: dict) -> str:
    """Students vs teacher: quality, size, how much of each case is read, dev prediction time; saved as kd.txt."""
    t, s = p["teacher"], p["sft"]
    sft_cfg = s.get("config", {})
    k = None
    for part in p["teacher_file"].split(os.sep):
        if part.startswith("hier_k") and part[6:].isdigit():
            k = int(part[6:])
    hdr = (
        f"{'model':24s} {'weights':>9s} {'layers':>7s} {'reads':>13s} {'dev F1@0.5':>11s} {'dev F1@t':>9s} "
        f"{'t':>5s} {'dev s':>7s} {'train':>7s}"
    )
    lines = [f"Knowledge distillation (teacher {p['teacher_file']})", hdr, "-" * len(hdr)]
    stage_b = t.get("params") or 0
    t_reads = f"{k} chunks" if k else "whole case"
    t_secs = s["dev_infer_seconds"] * (k or 1)
    lines.append(
        f"{'teacher (' + str(t.get('version')) + ')':24s} {(s['total_params'] + stage_b) / 1e6:8.1f}M "
        f"{'all':>7s} {t_reads:>13s} {t['dev_050']['macro_f1']:11.4f} {t['dev_tuned']['macro_f1']:9.4f} "
        f"{t['threshold']:5.2f} {'~' + format(t_secs, '.0f'):>7s} {s['train_seconds'] / 60:6.1f}m"
    )
    lines.append(
        f"{'teacher, last chunk only':24s} {s['total_params'] / 1e6:8.1f}M {'all':>7s} "
        f"{str(sft_cfg.get('max_len', '')) + ' tokens':>13s} {s['dev_050']['macro_f1']:11.4f} "
        f"{s['dev_tuned']['macro_f1']:9.4f} {s['threshold']:5.2f} {s['dev_infer_seconds']:7.1f} "
        f"{s['train_seconds'] / 60:6.1f}m"
    )
    for j in p["jobs"]:
        path = os.path.join(p["out_dir"], j["run_name"], "metrics.json")
        if not os.path.exists(path):
            lines.append(f"{j['run_name']:24s} not finished (see {p['out_dir']}/_{j['run_name']}_worker.log)")
            continue
        m = _read(path)
        st = m.get("student") or {}
        layers = f"{st.get('layers_student', '?')}/{st.get('layers_full', '?')}"
        lines.append(
            f"{j['run_name']:24s} {m['total_params'] / 1e6:8.1f}M {layers:>7s} "
            f"{str(m['max_len']) + ' tokens':>13s} {m['dev_050']['macro_f1']:11.4f} "
            f"{m['dev_tuned']['macro_f1']:9.4f} {m['threshold']:5.2f} {m['dev_infer_seconds']:7.1f} "
            f"{m['train_seconds'] / 60:6.1f}m"
        )
    lines.append(
        "weights: all weights used to predict (teacher: model + Stage B); dev s: seconds to predict the 994 "
        "dev cases (teacher: ~ one pass per chunk); dev F1@t: at the threshold tuned on dev (t)."
    )
    done = [j for j in p["jobs"] if os.path.exists(os.path.join(p["out_dir"], j["run_name"], "metrics.json"))]
    if done:
        best = max(
            done,
            key=lambda j: _read(os.path.join(p["out_dir"], j["run_name"], "metrics.json"))["dev_tuned"]["macro_f1"],
        )
        m = _read(os.path.join(p["out_dir"], best["run_name"], "metrics.json"))
        lines.append(
            f"best student on dev: {best['run_name']} ({m['dev_tuned']['macro_f1']:.4f}) -> submit "
            f"{m['submissions']['tuned']}"
        )
    text = "\n".join(lines)
    with open(os.path.join(p["out_dir"], "kd.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return text
