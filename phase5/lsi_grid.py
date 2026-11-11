"""Phase 5 runs on Kaggle: several fine-tuning techniques of one Phase 1 model (one process per GPU), the knowledge
distillation pair (student with / without the teacher) and an ensemble of finished runs."""

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
import p5_paths  # noqa: E402,F401  (Phase 4 building blocks)

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
OUT_DIR = "/kaggle/working/lsi_runs"


def _tail(path, n=1):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]
        return lines[-n:]
    except OSError:
        return []


def _done(out_dir, name):
    return os.path.exists(os.path.join(out_dir, name, "metrics.json"))


def _pretokenise(settings):
    """Tokenise every split once here, so the workers find it in the cache (and never write it at the same time)."""
    import lsi_train
    from cj_models import load_tokenizer
    from rr_backbone import make_encode

    cfg = lsi_train.resolve_cfg({k: v for k, v in settings.items() if k not in ("technique", "run_name", "kd_teacher")})
    lsi_train.prepare(make_encode(load_tokenizer()), cfg, log=lambda *a: None)


def run_jobs(model: str, jobs: list[dict], ckpt: str, out_dir: str, gpus: int | None = None, poll: int = 60) -> None:
    """jobs: [{"run_name", "technique", "settings"}]; split over the GPUs by expected cost, one process per GPU."""
    import torch

    n_gpu = torch.cuda.device_count() if gpus is None else gpus
    n = max(1, n_gpu)
    loads, per = [0.0] * n, [[] for _ in range(n)]
    for j in sorted(jobs, key=lambda j: -COST.get(j["technique"], 2.0)):
        i = loads.index(min(loads))
        per[i].append(j)
        loads[i] += COST.get(j["technique"], 2.0)
    procs = []
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
                    [sys.executable, "-u", os.path.join(HERE, "lsi_grid.py"), "--worker", spec],
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
            f"[lsi] worker {i} ({'GPU ' + str(i) if n_gpu else 'CPU'}): "
            f"{', '.join(j['run_name'] for j in js)}  (log: {log})",
            flush=True,
        )
    t0 = time.time()
    while any(p.poll() is None for p, *_ in procs):
        time.sleep(poll)
        for i, (p, _, log) in enumerate(procs):
            state = "running" if p.poll() is None else f"finished (exit {p.returncode})"
            last = _tail(log)
            print(
                f"[lsi {(time.time() - t0) / 60:5.1f} min] worker {i} {state}: {last[0][:160] if last else ''}",
                flush=True,
            )
    for i, (p, fh, log) in enumerate(procs):
        fh.close()
        if p.returncode != 0:
            print(f"[lsi] worker {i} stopped with an error; last lines of {log}:")
            print("\n".join(_tail(log, 25)))


def _table(out_dir, names, title):
    import lsi_train

    rows = [m for n in names if (m := lsi_train.load_metrics(out_dir, n)) is not None]
    missing = [n for n in names if lsi_train.load_metrics(out_dir, n) is None]
    table = lsi_train.comparison(rows) if rows else "no finished runs"
    if missing:
        table += f"\nnot finished: {', '.join(missing)} (see their log.txt)"
    if rows:
        best = max(rows, key=lambda m: max(m["cv_global"], m["cv_per_label"]))
        table += f"\nbest on dev (cross-fitted): {best['run_name']} -> submit {best['submissions']['chosen']}"
    text = f"\n===== {title}\n{table}"
    print(text)
    return text


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
    import lsi_train
    from cj_models import find_checkpoint

    unknown = set(settings) - set(lsi_train.DEFAULTS)
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    os.makedirs(out_dir, exist_ok=True)
    names = [f"{model}_{t}" for t in techniques]
    todo = [t for t, n in zip(techniques, names) if not (skip_done and _done(out_dir, n))]
    if len(todo) < len(techniques):
        print(f"[lsi] already finished (skipped): {', '.join(n for t, n in zip(techniques, names) if t not in todo)}")
    if todo:
        ckpt = ckpt or find_checkpoint(model)
        _pretokenise(settings)
        jobs = [dict(run_name=f"{model}_{t}", technique=t, settings=dict(settings, out_dir=out_dir)) for t in todo]
        run_jobs(model, jobs, ckpt, out_dir, gpus, poll)
    text = _table(out_dir, names, f"{model}: {len(techniques)} techniques")
    with open(os.path.join(out_dir, f"{model}_grid.txt"), "w") as f:
        f.write(text + "\n")
    return text


KD_COPY = (
    "data_root",
    "cache_dir",
    "max_len",
    "doc_tokens",
    "train_chunks",
    "eval_chunks",
    "chunk_pool",
    "epochs",
    "batch_size",
    "eval_batch_size",
    "loss",
    "pos_weight_power",
    "pos_weight_max",
    "asl_gamma_neg",
    "asl_gamma_pos",
    "asl_clip",
    "seed",
    "fetch_data",
    "dropout",
    "head",
    "pooling",
    "min_pos",
    "limit_train",
)


def _short_name(root: str, base: str, base_dirs) -> str:
    """A result's name: relative to its base folder, or (several bases) to their common parent folder."""
    if len(base_dirs) == 1:
        return os.path.relpath(root, base)
    common = os.path.commonpath([os.path.abspath(b) for b in base_dirs])
    return os.path.relpath(os.path.abspath(root), common)


def _sft_run_of(path: str) -> str:
    """The fine-tuned run folder at or above path (the first one with config.json + metrics.json)."""
    d = os.path.abspath(path)
    for _ in range(4):
        if os.path.exists(os.path.join(d, "config.json")) and os.path.exists(os.path.join(d, "metrics.json")):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError(f"no fine-tuned run (config.json + metrics.json) at or above {path}")


def plan_kd(
    teacher_dir: str,
    student_layers=0.5,
    kd_alpha: float = 0.5,
    kd_T: float = 2.0,
    technique: str = "full",
    plain: bool = True,
    out_dir: str = OUT_DIR,
    sft_run: str | None = None,
    **settings,
) -> dict:
    """Teacher, student checkpoint and the worker jobs (no training)."""
    t_file = os.path.join(teacher_dir, "train_probs.json")
    if not os.path.exists(t_file):
        raise FileNotFoundError(
            f"{t_file} missing: use a run trained with save_train_probs / save_chunks, or a Stage B version"
        )
    sft = _sft_run_of(sft_run or teacher_dir)
    with open(os.path.join(sft, "config.json")) as f:
        tcfg = json.load(f)
    with open(os.path.join(sft, "metrics.json")) as f:
        sm = json.load(f)
    tm = sm
    if os.path.abspath(teacher_dir) != sft and os.path.exists(os.path.join(teacher_dir, "metrics.json")):
        with open(os.path.join(teacher_dir, "metrics.json")) as f:
            tm = json.load(f)
    model = sm["model"]
    ckpt = tcfg.get("ckpt_path")
    if not ckpt:
        from cj_models import find_checkpoint

        ckpt = find_checkpoint(model)
    base = {k: tcfg[k] for k in KD_COPY if k in tcfg}
    base.update(settings)
    base.update(out_dir=out_dir, student_layers=student_layers, save_chunks=False, save_train_probs=False)
    jobs = [
        dict(
            run_name=f"{model}_student_kd",
            technique=technique,
            settings=dict(base, kd_teacher=t_file, kd_alpha=kd_alpha, kd_T=kd_T),
        )
    ]
    if plain:
        jobs.append(dict(run_name=f"{model}_student", technique=technique, settings=dict(base)))
    return dict(
        teacher_dir=teacher_dir,
        teacher_file=t_file,
        teacher=tm,
        sft_dir=sft,
        sft=sm,
        model=model,
        ckpt=ckpt,
        base=base,
        jobs=jobs,
        technique=technique,
        out_dir=out_dir,
    )


def run_kd(
    teacher_dir: str,
    student_layers=0.5,
    kd_alpha: float = 0.5,
    kd_T: float = 2.0,
    technique: str = "full",
    plain: bool = True,
    out_dir: str = OUT_DIR,
    sft_run: str | None = None,
    gpus: int | None = None,
    skip_done: bool = True,
    poll: int = 60,
    **settings,
) -> str:
    """Knowledge distillation: teacher = a finished run or Stage B version with train_probs.json; student = the same
    Phase 1 model (same starting checkpoint) with only its first student_layers blocks."""
    p = plan_kd(teacher_dir, student_layers, kd_alpha, kd_T, technique, plain, out_dir, sft_run, **settings)
    tm, sm = p["teacher"], p["sft"]
    print(
        f"[kd] teacher {teacher_dir}: dev macro-F1 {tm['dev_chosen']['macro_f1']:.4f} (cross-fitted "
        f"{max(tm['cv_global'], tm['cv_per_label']):.4f}); fine-tuned run {sm['run_name']}, "
        f"{sm['total_params'] / 1e6:.1f}M weights" + (f" + Stage B {tm['params'] / 1e6:.2f}M" if "params" in tm else "")
    )
    size = f"first {student_layers} of its blocks" if student_layers else "same size as the teacher's model"
    print(
        f"[kd] student: {p['model']} from {p['ckpt']}, {size}, {technique}; loss = "
        f"{1 - kd_alpha:.2f} x gold + {kd_alpha:.2f} x teacher (temperature {kd_T})"
        + ("; also the same student without the teacher" if plain else "")
    )
    names = [j["run_name"] for j in p["jobs"]]
    todo = [j for j in p["jobs"] if not (skip_done and _done(out_dir, j["run_name"]))]
    if todo:
        _pretokenise(p["base"])
        run_jobs(p["model"], todo, p["ckpt"], out_dir, gpus, poll)
    text = _table(out_dir, names, f"knowledge distillation (teacher {teacher_dir})")
    text += (
        f"\nteacher: dev macro-F1 {tm['dev_chosen']['macro_f1']:.4f}, cross-fitted "
        f"{max(tm['cv_global'], tm['cv_per_label']):.4f}"
    )
    print(text.splitlines()[-1])
    with open(os.path.join(out_dir, f"{p['model']}_kd.txt"), "w") as f:
        f.write(text + "\n")
    return text


def report(base_dirs, out_path: str | None = None) -> str:
    """One table of every result under base_dirs (fine-tuned runs, Stage B versions, KD students, ensembles), best
    cross-fitted dev macro-F1 first; saved as out_path (default <first base>/all_results.txt)."""
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    rows = []
    for base in base_dirs:
        for root, _, files in os.walk(base):
            if "metrics.json" not in files:
                continue
            with open(os.path.join(root, "metrics.json")) as f:
                m = json.load(f)
            if "dev_chosen" not in m:
                continue
            name = _short_name(root, base, base_dirs)
            kind = (
                "ensemble"
                if "members" in m
                else "Stage B"
                if "version" in m
                else "KD student"
                if m.get("student") or m.get("kd")
                else "fine-tuned"
            )
            size = (
                f"{m['trainable_params'] / 1e6:.2f}M/{m['total_params'] / 1e6:.0f}M"
                if "trainable_params" in m
                else f"+{m['params'] / 1e6:.2f}M"
                if "params" in m
                else f"{len(m.get('members', []))} runs"
            )
            rows.append((max(m["cv_global"], m["cv_per_label"]), name, kind, size, m))
    rows.sort(key=lambda r: -r[0])
    hdr = (
        f"{'result':44s} {'kind':11s} {'trained/size':>14s} {'macro@t':>8s} {'macro':>7s} {'cv':>7s} "
        f"{'micro':>7s} {'subset':>7s}"
    )
    lines = [hdr, "-" * len(hdr)]
    for cv, name, kind, size, m in rows:
        lines.append(
            f"{name[-44:]:44s} {kind:11s} {size:>14s} {m['dev_global']['macro_f1']:8.4f} "
            f"{m['dev_chosen']['macro_f1']:7.4f} {cv:7.4f} {m['dev_chosen']['micro_f1']:7.4f} "
            f"{m['dev_chosen']['subset_acc']:7.4f}"
        )
    lines.append(
        "macro@t: dev macro-F1 at one global threshold; macro: with the chosen thresholds; cv: cross-fitted "
        "(thresholds tuned on the other half of dev; the honest estimate, used for ranking)."
    )
    if rows:
        best = rows[0][4]
        sub = best.get("submission") or best.get("submissions", {}).get("chosen")
        lines.append(f"best: {rows[0][1]} (cv {rows[0][0]:.4f}) -> submit {sub}")
    text = "\n".join(lines)
    out_path = out_path or os.path.join(base_dirs[0], "all_results.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(text)
    return text


def compare_all(out_dir: str = OUT_DIR) -> str:
    """One table of every finished run in out_dir, best first; saved as all_runs.txt."""
    import lsi_train

    rows = [m for n in sorted(os.listdir(out_dir)) if (m := lsi_train.load_metrics(out_dir, n)) is not None]
    table = lsi_train.comparison(rows) if rows else "no finished runs"
    with open(os.path.join(out_dir, "all_runs.txt"), "w") as f:
        f.write(table + "\n")
    print(table)
    return table


def ensemble(
    base_dirs,
    out_dir: str = "/kaggle/working/lsi_ensemble",
    data_root: str | None = None,
    max_members: int = 5,
    min_gain: float = 0.001,
    fetch: bool = True,
    log=print,
) -> dict:
    """Average the probabilities of finished runs (greedy forward selection on dev macro-F1 at a global threshold),
    then thresholds as in a single run."""
    import numpy as np

    import lsi_data as L

    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    runs = {}
    for base in base_dirs:
        for root, _, files in os.walk(base):
            if os.path.abspath(root).startswith(os.path.abspath(out_dir)):
                continue
            if {"dev_probs.json", "test_probs.json", "labels.json"} <= set(files):
                runs[_short_name(root, base, base_dirs)] = root
    if not runs:
        raise ValueError(f"no finished runs under {base_dirs}")
    data = L.load_data(data_root or L.DEFAULT_ROOT, fetch=fetch)
    labels = L.label_space(data)
    Y = L.multi_hot(data["dev"], labels)
    ids_dev, ids_test = [d["id"] for d in data["dev"]], [d["id"] for d in data["test"]]
    loaded, score = {}, {}
    for name, folder in sorted(runs.items()):
        with open(os.path.join(folder, "labels.json")) as f:
            if json.load(f) != labels:
                log(f"[ensemble] skipped {name}: other statute list")
                continue
        try:
            loaded[name] = (
                L.load_probs(os.path.join(folder, "dev_probs.json"), ids_dev),
                L.load_probs(os.path.join(folder, "test_probs.json"), ids_test),
            )
        except ValueError as e:
            log(f"[ensemble] skipped {name}: {e}")
            continue
        score[name] = L.tune_global(Y, loaded[name][0])[1]
    ranked = sorted(score, key=lambda n: -score[n])
    log("[ensemble] runs by dev macro-F1 (global threshold):\n" + "\n".join(f"  {score[n]:.4f}  {n}" for n in ranked))
    members, cur = [ranked[0]], score[ranked[0]]
    steps = [dict(added=ranked[0], dev_f1=cur)]
    while len(members) < max_members:
        best = None
        for n in ranked:
            if n in members:
                continue
            f1 = L.tune_global(Y, np.mean([loaded[m][0] for m in members + [n]], 0))[1]
            if best is None or f1 > best[1]:
                best = (n, f1)
        if best is None or best[1] < cur + min_gain:
            break
        members.append(best[0])
        cur = best[1]
        steps.append(dict(added=best[0], dev_f1=cur))
    P_dev = np.mean([loaded[m][0] for m in members], 0)
    P_test = np.mean([loaded[m][1] for m in members], 0)
    th = L.choose_thresholds(Y, P_dev)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    L.save_probs(os.path.join(out_dir, "dev_probs.json"), ids_dev, P_dev)
    L.save_probs(os.path.join(out_dir, "test_probs.json"), ids_test, P_test)
    sub = L.write_submission(data["test"], labels, P_test, th["thresholds"], os.path.join(out_dir, "submission"))
    tr_paths = [os.path.join(runs[m], "train_probs.json") for m in members]
    teacher = False
    if all(os.path.exists(p) for p in tr_paths):
        try:
            ids_tr = [d["id"] for d in data["train"]]
            L.save_probs(
                os.path.join(out_dir, "train_probs.json"),
                ids_tr,
                np.mean([L.load_probs(p, ids_tr) for p in tr_paths], 0),
            )
            teacher = True
        except ValueError as e:
            log(f"[ensemble] no train_probs.json: {e}")
    res = dict(
        members=members,
        steps=steps,
        threshold_kind=th["kind"],
        t_global=th["t_global"],
        cv_global=th["cv_global"],
        cv_per_label=th["cv_per_label"],
        dev_global=L.short(th["dev_global"]),
        dev_chosen=L.short(th["dev_chosen"]),
        single_best=ranked[0],
        single_best_dev_f1=score[ranked[0]],
        submission=sub,
        train_probs=teacher,
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("[ensemble] members: " + " + ".join(f"{s['added']} ({s['dev_f1']:.4f})" for s in steps))
    log(L.fmt(th["dev_chosen"], f"[ensemble] dev ({th['kind']} thresholds): "))
    log(
        f"[ensemble] cross-fitted macro-F1: global {th['cv_global']:.4f}, per statute {th['cv_per_label']:.4f}; "
        f"best single run {ranked[0]} {score[ranked[0]]:.4f}; submit {sub}"
    )
    return res


def _worker(spec_path: str) -> int:
    with open(spec_path) as f:
        spec = json.load(f)
    import lsi_train
    from cj_models import load_backbone

    max_len = spec["jobs"][0]["settings"].get("max_len", lsi_train.DEFAULTS["max_len"])
    bb, info = load_backbone(spec["model"], max_len=max_len, path=spec["ckpt"])
    failed = []
    for j in spec["jobs"]:
        s = dict(j["settings"], ckpt_path=spec["ckpt"], run_name=j["run_name"], technique=j["technique"])
        if info["max_len"] != s.get("max_len", lsi_train.DEFAULTS["max_len"]):
            s["max_len"] = info["max_len"]
        try:
            lsi_train.run(bb, name=spec["model"], **s)
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
