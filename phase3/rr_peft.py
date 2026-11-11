"""Phase 3: parameter-efficient fine-tuning (PEFT) of Stage A with the Hugging Face `peft` library."""

from __future__ import annotations

import gc
import json
import os
import re
import traceback

import torch
import torch.nn as nn

import rr_data as D
from rr_head import find_lm_head

PEFT_METHODS = ("lora", "rslora", "adalora", "ia3")
ALL_METHODS = ("full", "frozen") + PEFT_METHODS
DEFAULT_LR = {"lora": 5e-4, "rslora": 5e-4, "adalora": 5e-4, "ia3": 3e-3}

K_NAMES = {"k_proj", "k", "key", "wk", "w_k", "to_k", "k_lin", "key_proj"}
V_NAMES = {"v_proj", "v", "value", "wv", "w_v", "to_v", "v_lin", "value_proj"}
FUSED_NAMES = {
    "qkv",
    "c_attn",
    "qkv_proj",
    "wqkv",
    "w_qkv",
    "in_proj",
    "attn_proj",
    "query_key_value",
    "kv",
    "kv_proj",
    "wkv",
    "w_kv",
    "to_kv",
}


# --- which layers
def linear_layers(model: nn.Module, vocab: int | None = None) -> dict[str, nn.Linear]:
    """Every nn.Linear of the Phase 1 model except the LM head (the projection to the vocabulary)."""
    head = find_lm_head(model)
    return {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n != head and not (vocab and m.out_features == vocab)
    }


def compact(names) -> list[str]:
    """['layers.0.attn.wq', 'layers.1.attn.wq'] -> ['layers.*.attn.wq (2)'] for printing."""
    counts: dict[str, int] = {}
    for n in names:
        p = re.sub(r"(^|\.)\d+(\.|$)", r"\1*\2", n, count=1)
        counts[p] = counts.get(p, 0) + 1
    return [f"{p} ({c})" for p, c in counts.items()]


def _match(names, wanted):
    return [n for n in names if any(n == t or n.endswith("." + t) for t in wanted)]


def choose_targets(
    model: nn.Module, method: str, d_model: int, vocab: int | None = None, targets=None, ff=None
) -> tuple[list[str], list[str] | None, str]:
    """(target layer names, IA3 feed-forward layer names, description)."""
    lins = linear_layers(model, vocab)
    if not lins:
        raise ValueError(
            "the model has no nn.Linear layers that peft can adapt (peft wraps nn.Linear; "
            "recurrent layers such as nn.GRU / nn.LSTM cannot take LoRA or IA3)"
        )
    order = list(lins)
    if targets:
        chosen = _match(order, targets)
        missing = [t for t in targets if not _match(order, [t])]
        if missing:
            raise ValueError(f"peft_targets not found among the Linear layers: {missing}; available: {compact(order)}")
        targets = chosen
    if method != "ia3":
        targets = targets or order
        return targets, None, "given layers" if targets != order else "every Linear layer except the LM head"
    if ff:
        ff = _match(order, ff)
    else:
        ff = [n for n, m in lins.items() if m.out_features == d_model and m.in_features > d_model]
    note = "given layers"
    if not targets:
        last = {n: n.split(".")[-1].lower() for n in order}
        k = [n for n in order if last[n] in K_NAMES]
        v = [n for n in order if last[n] in V_NAMES]
        if k and v:
            attn, note = k + v, "key and value projections (output scaled)"
        else:
            attn = [n for n in order if last[n] in FUSED_NAMES]
            if not attn:
                raise ValueError(
                    "IA3: could not find the key/value projections by name; pass "
                    f"peft_targets=[...] and ia3_ff=[...]. Linear layers: {compact(order)}"
                )
            note = "fused query/key/value projections (output scaled; the query part is scaled too)"
        note += " + feed-forward down-projection (input scaled)" if ff else " (no feed-forward layer found)"
        targets = sorted(set(attn) | set(ff), key=order.index)
    ff = [n for n in ff if n in targets]
    return targets, ff, note


# --- peft config
def make_config(method: str, targets: list[str], ff, cfg: dict, total_steps: int):
    import peft

    if method in ("lora", "rslora"):
        return peft.LoraConfig(
            r=cfg["peft_r"],
            lora_alpha=cfg["peft_alpha"],
            lora_dropout=cfg["peft_dropout"],
            target_modules=targets,
            bias="none",
            use_rslora=(method == "rslora"),
        )
    if method == "adalora":
        tinit, tfinal = int(cfg["adalora_tinit"] * total_steps), int(cfg["adalora_tfinal"] * total_steps)
        return peft.AdaLoraConfig(
            init_r=cfg["adalora_init_r"],
            target_r=cfg["peft_r"],
            lora_alpha=cfg["peft_alpha"],
            lora_dropout=cfg["peft_dropout"],
            target_modules=targets,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=cfg["adalora_deltaT"],
            orth_reg_weight=cfg["adalora_orth"],
            total_step=total_steps,
        )
    if method == "ia3":
        return peft.IA3Config(target_modules=targets, feedforward_modules=ff)
    raise ValueError(f"unknown PEFT method {method!r}; choose from {PEFT_METHODS}")


class PeftHooks:
    """The extra steps a PEFT method needs in the training loop (only AdaLoRA needs any)."""

    def __init__(self, peft_model, method: str, cfg: dict, targets, ff, note: str, total_steps: int):
        self.peft_model, self.method = peft_model, method
        self.tuner = peft_model.base_model
        self.adapter = "default"
        self.targets, self.ff, self.note, self.total_steps = list(targets), list(ff or []), note, total_steps
        self.step = 0
        self.orth_w = float(cfg["adalora_orth"]) if method == "adalora" else 0.0
        model = self.tuner.model
        self._lora = [(n, p) for n, p in model.named_parameters() if "lora_" in n and f".{self.adapter}" in n]
        self._orth = [("lora_A" in n, p) for n, p in self._lora if "lora_A" in n or "lora_B" in n]

    def extra_loss(self):
        if self.method != "adalora" or self.orth_w <= 0:
            return None
        ps = [(a, p) for a, p in self._orth if p.requires_grad]
        if not ps:
            return None
        reg = 0.0
        for is_a, p in ps:
            cov = p @ p.T if is_a else p.T @ p
            reg = reg + torch.linalg.norm(cov - torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype))
        return self.orth_w * reg / len(ps)

    def after_step(self, stepped: bool = True) -> None:
        if self.method != "adalora" or not stepped:
            return
        for _, p in self._lora:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        self.tuner.update_and_allocate(self.step)
        self.step += 1

    def summary(self) -> dict:
        out: dict = dict(
            method=self.method,
            layers=len(self.targets),
            targets=compact(self.targets),
            note=self.note,
            feedforward=compact(self.ff),
        )
        if self.method == "adalora":
            pat = self.tuner.peft_config[self.adapter].rank_pattern
            out["optimizer_steps"] = self.step
            if pat:
                ranks = {
                    re.sub(r"\.lora_E(\.\w+)?$", "", k): int(
                        sum(bool(x) for x in (v.view(-1).tolist() if torch.is_tensor(v) else v))
                    )
                    for k, v in pat.items()
                }
                groups: dict[str, list[int]] = {}
                for k, r in ranks.items():
                    groups.setdefault(compact([k])[0].rsplit(" (", 1)[0], []).append(r)
                out["final_rank_total"] = sum(ranks.values())
                out["final_rank_mean_by_layer_type"] = {g: round(sum(v) / len(v), 2) for g, v in groups.items()}
                out["final_rank_by_layer"] = ranks
        return out


def apply_peft(model: nn.Module, cfg: dict, d_model: int, vocab: int | None, total_steps: int, log=print) -> PeftHooks:
    """Adds the adapters of cfg["method"] to `model` IN PLACE (call it on a fresh copy) and freezes every Phase 1
    weight."""
    try:
        import peft
    except ImportError as e:  # noqa: BLE001
        raise ImportError("the `peft` library is needed for PEFT methods: pip install peft") from e
    method = cfg["method"]
    if not d_model:
        head = dict(model.named_modules()).get(find_lm_head(model) or "")
        d_model = getattr(head, "in_features", 0)
    targets, ff, note = choose_targets(model, method, d_model, vocab, cfg["peft_targets"], cfg["ia3_ff"])
    ff = ff or []
    config = make_config(method, targets, ff, cfg, total_steps)
    peft_model = peft.get_peft_model(model, config)
    if peft_model.base_model.model is not model:
        raise RuntimeError("peft did not modify the model in place")
    mods = dict(model.named_modules())
    missing = [n for n in targets if not hasattr(mods.get(n), "base_layer")]
    if missing:
        raise RuntimeError(f"peft did not wrap these layers: {missing[:5]}")
    if method == "ia3":
        wrong = [n for n in targets if bool(getattr(mods[n], "is_feedforward", False)) != (n in ff)]
        if wrong:
            raise RuntimeError(f"IA3 feed-forward flag differs from the plan for: {wrong[:5]}")
    if method == "adalora" and not hasattr(peft_model.base_model, "update_and_allocate"):
        raise RuntimeError("this peft version has no AdaLoRA update_and_allocate")
    r, a = cfg["peft_r"], cfg["peft_alpha"]
    detail = {
        "lora": f"rank {r}, alpha {a}, scale alpha/r = {a / r:.3g}, dropout {cfg['peft_dropout']}",
        "rslora": f"rank {r}, alpha {a}, scale alpha/sqrt(r) = {a / r**0.5:.3g}, dropout {cfg['peft_dropout']}",
        "adalora": f"rank {cfg['adalora_init_r']} per layer at the start -> average {r} at the end; "
        f"{total_steps} steps: first {int(cfg['adalora_tinit'] * total_steps)} without changes, "
        f"last {int(cfg['adalora_tfinal'] * total_steps)} fixed, budget update every "
        f"{cfg['adalora_deltaT']} steps; orthogonality weight {cfg['adalora_orth']}",
        "ia3": "scaling vectors start at 1",
    }[method]
    log(f"[peft] {method} with peft {peft.__version__}: {detail}; learning rate {cfg['lr_peft']}")
    log(f"[peft] {len(targets)} layers: {note}: " + ", ".join(compact(targets)))
    if method == "ia3" and ff:
        log("[peft] feed-forward (input scaled): " + ", ".join(compact(ff)))
    return PeftHooks(peft_model, method, cfg, targets, ff, note, total_steps)


def attach(clf: nn.Module, hooks: PeftHooks) -> None:
    """Keep the hooks on the classifier without registering the peft wrapper as a submodule (that would list every
    weight twice in state_dict)."""
    object.__setattr__(clf, "peft_hooks", hooks)


def trained_state(clf: nn.Module) -> dict:
    """Only the trained tensors (adapters + head) and AdaLoRA's fixed rank counters."""
    names = {n for n, p in clf.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in clf.state_dict().items() if k in names or ".ranknum." in k}


def save_adapter(clf: nn.Module, path: str, labels, cfg: dict) -> None:
    hooks = getattr(clf, "peft_hooks", None)
    pc = hooks.tuner.peft_config[hooks.adapter].to_dict() if hooks is not None else None
    torch.save(
        {
            "trained_weights": trained_state(clf),
            "peft_config": pc,
            "labels": labels,
            "config": cfg,
            "note": "only the trained tensors (adapters and head); every other weight equals the Phase 1 "
            "checkpoint. To use: build the Phase 1 model, apply the same peft config, build the "
            "classifier, load_state_dict(trained_weights, strict=False).",
        },
        path,
    )


# --- several methods
def run_methods(bb, methods=ALL_METHODS, prefix: str = "run", skip_done: bool = True, **settings) -> dict:
    """rr_train.run once per method (run name f"{prefix}_{method}", same settings otherwise), then a comparison: cost,
    dev 13-role average for all / CL / IT, memorising, closely related roles."""
    import rr_train as T

    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {ALL_METHODS}")
    if "method" in settings or "run_name" in settings:
        raise ValueError("run_methods sets method and run_name itself; use methods=... and prefix=...")
    cfg0 = T.resolve_cfg(bb, settings)
    results: dict[str, dict | None] = {}
    for m in methods:
        name = f"{prefix}_{m}"
        path = os.path.join(cfg0["out_dir"], name, "metrics.json")
        if skip_done and os.path.exists(path):
            with open(path) as f:
                results[m] = json.load(f)
            print(f"===== {name}: already done, using {path}")
            continue
        res = None
        try:
            res = T.run(bb, run_name=name, method=m, **settings)
            results[m] = res["metrics"]
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"===== {name} FAILED (error above); continuing with the next method")
            results[m] = None
        finally:
            del res
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save_comparison(results, cfg0["out_dir"], prefix)
    return results


def save_comparison(results: dict, out_dir: str, prefix: str, extra_text: str = "") -> str:
    """Print the comparison of finished runs and save it as <prefix>_comparison.txt / .json in out_dir."""
    text = D.format_comparison(results) + (("\n\n" + extra_text) if extra_text else "")
    print("\n" + "=" * 78 + f"\nCOMPARISON '{prefix}' (dev)\n" + text)
    rows = {}
    for m, met in results.items():
        if not met:
            rows[m] = None
            continue
        a, b = met["stage_a"], met["stage_b"]
        labels = list(a["dev"]["all"]["per_label"])
        rows[m] = dict(
            run=met["run_name"],
            trainable=a["trainable_params"],
            total=a.get("total_params"),
            saved_mb=a.get("saved_mb"),
            peak_gpu_gb=a.get("peak_gpu_gb"),
            stage_a_minutes=a["train_seconds"] / 60,
            best_epoch=a["best_epoch"],
            dev_infer_seconds=a.get("dev_infer_seconds"),
            dev_13avg={st: {k: v["macro_f1_all13"] for k, v in s["dev"].items()} for st, s in (("A", a), ("B", b))},
            train_fit_acc=(a.get("train_fit") or {}).get("all", {}).get("accuracy"),
            close_pairs={st: D.pair_confusions(s["dev"], labels) for st, s in (("A", a), ("B", b))},
            peft=a.get("peft"),
            kd=met.get("kd"),
            submission=os.path.join(out_dir, met["run_name"], met["best"], "submission.zip"),
        )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{prefix}_comparison.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, f"{prefix}_comparison.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\nsubmission files (best stage on dev):")
    for m, r in rows.items():
        print(f"  {m:12s} " + ("FAILED" if r is None else r["submission"]))
    print(f"saved {prefix}_comparison.txt and {prefix}_comparison.json in {out_dir}")
    return text
