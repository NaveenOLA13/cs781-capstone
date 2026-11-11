"""Phase 5 best submission: macro-F1 0.3678 (Codabench ID 1016762, rank 79).

Model: the Phase 1 Llama (85M), mid-trained on the LSI train + dev text (20M tokens, never test text). The steps are
those of `phase5_run.ipynb` with MODEL = "llama":
1. LoRA and LoRMA (1 epoch; training reads 3 random chunks of 512 tokens per document, prediction reads up to 16
   chunks of the first 8,192 tokens; a statute's document score is its maximum over the chunks);
2. the whole-document stage on both: statute-wise attention over all chunk vectors (`gru_attn_res`, `gru_attn`);
3. knowledge distillation: teacher = the best result so far; student = the mid-trained model with half its blocks,
   trained with and without the teacher;
4. QLoRA (own 4-bit NF4) and adapters;
5. the ensemble: greedy forward selection of these results on dev, thresholds by 2-fold cross-fitting on dev.

Run from this folder on two GPUs (HF_TOKEN set, the dataset is gated):
    python best_submission.py --ckpt /path/to/llama_85m_v1/best.pt
Output: /kaggle/working/lsi_ensemble_llama/submission/submission.zip
"""

import argparse
import glob
import json
import os

import p5_paths  # noqa: F401  (Phase 4 building blocks)
from lsi_grid import ensemble, report, run_grid, run_kd
from lsi_hier import run_stage_b
from lsi_mid import mid_train

MODEL = "llama"
OUT = "/kaggle/working/lsi_runs"
ENS = f"/kaggle/working/lsi_ensemble_{MODEL}"
COMMON = dict(epochs=1, train_chunks=3, eval_chunks=16, doc_tokens=8192, batch_size=8, eval_batch_size=32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Phase 1 Llama checkpoint (best.pt)")
    a = ap.parse_args()
    start = mid_train(MODEL, path=a.ckpt, tokens=20_000_000, batch_size=16)
    run_grid(MODEL, techniques=("lora", "lorma"), ckpt=start, out_dir=OUT, save_chunks=True, **COMMON)
    for tech in ("lora", "lorma"):
        run_stage_b(f"{OUT}/{MODEL}_{tech}", versions=("gru_attn_res", "gru_attn"), epochs=8, patience=2)
    cands = []
    for tech in ("lora", "lorma"):
        run_dir = f"{OUT}/{MODEL}_{tech}"
        files = glob.glob(f"{run_dir}/stageB/*/metrics.json")
        if os.path.exists(f"{run_dir}/train_probs.json"):
            files.append(f"{run_dir}/metrics.json")
        for f in files:
            m = json.load(open(f))
            cands.append((max(m["cv_global"], m["cv_per_label"]), os.path.dirname(f)))
    run_kd(max(cands)[1], student_layers=0.5, plain=True, out_dir=OUT, **COMMON)
    run_grid(MODEL, techniques=("qlora", "adapter"), ckpt=start, out_dir=OUT, **COMMON)
    ensemble([OUT], out_dir=ENS)
    report([OUT, ENS], out_path="/kaggle/working/all_results.txt")
    print("submit:", os.path.join(ENS, "submission", "submission.zip"))


if __name__ == "__main__":
    main()
