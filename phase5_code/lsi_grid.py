"""Phase 5 runs on Kaggle: several fine-tuning techniques of one Phase 1 model (one process per GPU), the knowledge
distillation pair (student with / without the teacher) and an ensemble of finished runs.

    from lsi_grid import run_grid, run_kd, ensemble, compare_all
    run_grid("llama", techniques=("full", "lorma"), save_train_probs=True)   # 2 GPUs: one run each
    run_kd("/kaggle/working/lsi_runs/llama_full")                            # teacher = that run
    ensemble(["/kaggle/working/lsi_runs"])                                   # best mix of finished runs on dev
    compare_all()                                                            # table of every run

Each GPU gets its own process (python -u lsi_grid.py --worker job.json) that loads the model once and runs its jobs
one after another; progress lines are printed here every minute, full logs are in <out_dir>/<run>/log.txt.
Finished runs (metrics.json present) are skipped, so a cell can be run again after an interruption.
"""
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

COST = {"frozen": 1.0, "full": 3.0, "partial": 2.0, "unfreeze": 2.4, "bitfit": 2.0, "lorma": 2.3,
        "adalora": 2.4, "adapter": 2.0}
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
    cfg = lsi_train.resolve_cfg({k: v for k, v in settings.items() if k not in ("technique", "run_name",
                                                                                  "kd_teacher")})
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
        procs.append((subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "lsi_grid.py"), "--worker", spec],
                                       stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE), fh, log))
        print(f"[lsi] worker {i} ({'GPU ' + str(i) if n_gpu else 'CPU'}): "
              f"{', '.join(j['run_name'] for j in js)}  (log: {log})", flush=True)
    t0 = time.time()
    while any(p.poll() is None for p, *_ in procs):
        time.sleep(poll)
        for i, (p, _, log) in enumerate(procs):
            state = "running" if p.poll() is None else f"finished (exit {p.returncode})"
            last = _tail(log)
            print(f"[lsi {(time.time() - t0) / 60:5.1f} min] worker {i} {state}: {last[0][:160] if last else ''}",
                  flush=True)
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
        table += (f"\nbest on dev (cross-fitted): {best['run_name']} -> submit {best['submissions']['chosen']}")
    text = f"\n===== {title}\n{table}"
    print(text)
    return text


def run_grid(model: str, techniques=("full", "lorma"), ckpt: str | None = None, out_dir: str = OUT_DIR,
             gpus: int | None = None, skip_done: bool = True, poll: int = 60, **settings) -> str:
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


KD_COPY = ("data_root", "cache_dir", "max_len", "doc_tokens", "train_chunks", "eval_chunks", "chunk_pool", "epochs",
           "batch_size", "eval_batch_size", "pos_weight_power", "pos_weight_max", "seed", "fetch_data", "dropout",
           "head", "pooling", "min_pos")


def run_kd(teacher_dir: str, student_layers=0.5, kd_alpha: float = 0.5, kd_T: float = 2.0, technique: str = "full",
           plain: bool = True, out_dir: str = OUT_DIR, gpus: int | None = None, skip_done: bool = True,
           poll: int = 60, **settings) -> str:
    """Knowledge distillation: teacher = a finished run with train_probs.json (run it with save_train_probs=True);
    student = the same Phase 1 model with only its first student_layers blocks (a share or a number), trained with
    the teacher (<model>_student_kd) and, for comparison, without it (<model>_student). Settings are copied from
    the teacher run unless given."""
    t_file = os.path.join(teacher_dir, "train_probs.json")
    if not os.path.exists(t_file):
        raise FileNotFoundError(f"{t_file} missing: run the teacher with save_train_probs=True")
    with open(os.path.join(teacher_dir, "config.json")) as f:
        tcfg = json.load(f)
    with open(os.path.join(teacher_dir, "metrics.json")) as f:
        tm = json.load(f)
    model = tm["model"]
    ckpt = tcfg.get("ckpt_path")
    if not ckpt:
        from cj_models import find_checkpoint
        ckpt = find_checkpoint(model)
    base = {k: tcfg[k] for k in KD_COPY if k in tcfg}
    base.update(settings)
    base.update(out_dir=out_dir, student_layers=student_layers)
    print(f"[kd] teacher {teacher_dir}: {tm['run_name']}, dev macro-F1 {tm['dev_chosen']['macro_f1']:.4f} "
          f"(cross-fitted {max(tm['cv_global'], tm['cv_per_label']):.4f}), {tm['total_params'] / 1e6:.1f}M weights")
    print(f"[kd] student: {model} from {ckpt}, first {student_layers} of its blocks, {technique}; loss = "
          f"{1 - kd_alpha:.2f} x gold + {kd_alpha:.2f} x teacher (temperature {kd_T})"
          + ("; also the same student without the teacher" if plain else ""))
    jobs = [dict(run_name=f"{model}_student_kd", technique=technique,
                 settings=dict(base, kd_teacher=t_file, kd_alpha=kd_alpha, kd_T=kd_T))]
    if plain:
        jobs.append(dict(run_name=f"{model}_student", technique=technique, settings=dict(base)))
    names = [j["run_name"] for j in jobs]
    todo = [j for j in jobs if not (skip_done and _done(out_dir, j["run_name"]))]
    if todo:
        _pretokenise(base)
        run_jobs(model, todo, ckpt, out_dir, gpus, poll)
    text = _table(out_dir, [tm["run_name"]] + names if os.path.dirname(teacher_dir.rstrip("/")) == out_dir.rstrip("/")
                  else names, f"knowledge distillation (teacher {tm['run_name']})")
    with open(os.path.join(out_dir, f"{model}_kd.txt"), "w") as f:
        f.write(text + "\n")
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


def ensemble(base_dirs, out_dir: str = "/kaggle/working/lsi_ensemble", data_root: str | None = None,
             max_members: int = 5, min_gain: float = 0.001, fetch: bool = True, log=print) -> dict:
    """Average the probabilities of finished runs (greedy forward selection on dev macro-F1 at a global threshold),
    then thresholds as in a single run. Runs must share the statute list (labels.json)."""
    import numpy as np

    import lsi_data as L
    if isinstance(base_dirs, str):
        base_dirs = [base_dirs]
    runs = {}
    for base in base_dirs:
        for root, _, files in os.walk(base):
            if {"dev_probs.json", "test_probs.json", "labels.json"} <= set(files):
                runs[os.path.relpath(root, base) if len(base_dirs) == 1 else root] = root
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
            loaded[name] = (L.load_probs(os.path.join(folder, "dev_probs.json"), ids_dev),
                            L.load_probs(os.path.join(folder, "test_probs.json"), ids_test))
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
    res = dict(members=members, steps=steps, threshold_kind=th["kind"], t_global=th["t_global"],
               cv_global=th["cv_global"], cv_per_label=th["cv_per_label"], dev_global=L.short(th["dev_global"]),
               dev_chosen=L.short(th["dev_chosen"]), single_best=ranked[0], single_best_dev_f1=score[ranked[0]],
               submission=sub)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("[ensemble] members: " + " + ".join(f"{s['added']} ({s['dev_f1']:.4f})" for s in steps))
    log(L.fmt(th["dev_chosen"], f"[ensemble] dev ({th['kind']} thresholds): "))
    log(f"[ensemble] cross-fitted macro-F1: global {th['cv_global']:.4f}, per statute {th['cv_per_label']:.4f}; "
        f"best single run {ranked[0]} {score[ranked[0]]:.4f}; submit {sub}")
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
