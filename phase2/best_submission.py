"""Phase 2 best submission: strict macro-F1 0.4715 (Codabench, rank 90).

Model: the Phase 1 encoder-decoder (`encdec_86m_v1`); its bidirectional GRU encoder sees both sides of every token.
Method: full fine-tuning with a linear BIO tag head (12 labels -> 25 tags), learning rate 2e-4, 6 epochs, windows of
512 tokens (stride 384), learned in-token boundary repair and dev-selected post-processing; dev = 15% of the 70
training documents, split by document. This is the `encdec_full` line of `run_phase2.sh`.

Run from the repository root (HF_TOKEN set, the dataset is gated):
    python phase2/best_submission.py --phase1 /path/to/encdec_86m_v1/best.pt
Output: runs_ner/encdec_full/sub/submission.zip (prediction.json for every fold-3 document)
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", required=True, help="Phase 1 encoder-decoder checkpoint (best.pt)")
    a = ap.parse_args()
    cmd = [
        sys.executable,
        os.path.join(HERE, "train_ner.py"),
        "--phase1", a.phase1,
        "--run", "encdec_full",
        "--method", "full",
        "--head", "linear",
        "--lr", "2e-4",
        "--epochs", "6",
    ]  # fmt: skip
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
