"""Phase 1 best submission: BERTScore 0.8288 (Codabench, rank 92).

Model: the Llama-style model (85M; 12 blocks, d 768, grouped-query attention, RoPE, RMSNorm, SwiGLU), pretrained from
scratch with `configs/llama_85m.yaml` (run `llama_85m_v1`). The submission holds 128-word continuations of the 1,000
test prompts; this script writes them with `generate.py` and its default settings (top-p 0.9, temperature 0.8,
repetition penalty 1.1, <eos> banned so every continuation reaches 128 words).

Run from this folder (HF_TOKEN set, the dataset is gated):
    python best_submission.py --ckpt runs/llama_85m_v1/best.pt              # generation from the trained model
    python best_submission.py --pretrain --ckpt runs/llama_85m_v1/best.pt   # pretraining first (set data.dir)
Output: sub/llama/submission.zip
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/llama_85m_v1/best.pt")
    ap.add_argument("--out", default="sub/llama")
    ap.add_argument("--pretrain", action="store_true", help="pretrain the model first (about 6 h on two T4 GPUs)")
    a = ap.parse_args()
    if a.pretrain:
        # stops before Kaggle's 12-hour limit; continue with: train.py --config ... --resume runs/llama_85m_v1/last.pt
        run("train.py", "--config", "configs/llama_85m.yaml")
    run("generate.py", "--ckpt", a.ckpt, "--out", a.out)


if __name__ == "__main__":
    main()
