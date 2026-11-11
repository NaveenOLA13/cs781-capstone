"""Phase 4 best submission: macro-F1 0.7765 (Codabench ID 1010315, rank 28).

Model: the Phase 1 Llama (85M), mid-trained by next-token prediction on the cleaned CJPE train + dev text (40M tokens,
never test text).
Stage A: full fine-tuning for 1 epoch; each training case gives its last 3 chunks of 512 tokens (stride 448) as
separate examples; last-token pooling and a linear layer with 2 outputs, class-weighted cross-entropy.
Stage B (whole case): Stage A's encoder, unchanged, encodes the last 8 chunks; `gru_attn` (BiGRU + attention pooling +
the last chunk) decides; the threshold is tuned on the cleaned dev set (0.22).

Run from this folder on two GPUs (HF_TOKEN set, the dataset is gated):
    python best_submission.py --ckpt /path/to/llama_85m_v1/best.pt
Output: /kaggle/working/cj_runs/llama_full/hier_k8/gru_attn/submission_tuned/submission.zip
"""

import argparse
import os

from cj_grid import run_grid
from cj_hier import run_hier_many
from cj_mid import mid_train

OUT = "/kaggle/working/cj_runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Phase 1 Llama checkpoint (best.pt)")
    a = ap.parse_args()
    start = mid_train("llama", path=a.ckpt, tokens=40_000_000)
    run_grid("llama", techniques=("full",), ckpt=start, train_chunks=3, epochs=1, out_dir=OUT)
    run_hier_many([os.path.join(OUT, "llama_full")], k=8)
    print("submit:", os.path.join(OUT, "llama_full", "hier_k8", "gru_attn", "submission_tuned", "submission.zip"))


if __name__ == "__main__":
    main()
