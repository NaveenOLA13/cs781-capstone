"""Phase 3 best submission: macro-F1 0.5079 over the 13 roles (Codabench ID 993279, rank 54).

Model: the Phase 1 Llama (85M) + LoRA (rank 8, alpha 16, dropout 0.05, every Linear layer except the LM head,
learning rate 5e-4; `peft` library, allowed in this phase).
Stage A labels one sentence (up to 128 tokens) with the learned vector of its position bucket; Stage B, a BiGRU, reads
the Stage A vectors of all sentences of a judgment. Five copies are trained on train + dev (72 of the 90 judgments
each), their probabilities averaged, then Viterbi smoothing with role-change probabilities per domain; the smoothing
weight is chosen out of fold (0.75).

Run from this folder on a GPU (HF_TOKEN set, the dataset is gated):
    python best_submission.py --ckpt /path/to/llama_85m_v1/best.pt
Output: the path printed at the end (`final_subs/<Stage B version>/smoothed/submission.zip`).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase4"))  # cj_models: loads a Phase 1 model

from cj_models import load_model, load_tokenizer  # noqa: E402  (also puts phase1/ and phase3/ on the path)
from rr_backbone import make_backbone  # noqa: E402
from rr_final import run_final  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Phase 1 Llama checkpoint (best.pt)")
    a = ap.parse_args()
    model, _, _, _ = load_model("llama", a.ckpt)
    bb = make_backbone(model, load_tokenizer())
    res = run_final(bb, method="lora")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
