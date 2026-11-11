"""Phase 6 best submission: BERTScore F1 0.8518, ROUGE-1 0.5905, ROUGE-L 0.2977 (Codabench ID 1030894, rank 18).

Model: the Phase 1 decoder-only model (80M). The steps are those of `phase6_round2.ipynb` with MODEL = "decoder":
1. mid-training on the training documents and summaries (30M tokens, at most 30 minutes);
2. two full fine-tuning runs, one per GPU (all weights, learning rate 1e-4, 4 epochs, batch 8 x 2 accumulation, best
   epoch on the held-out loss kept): `decoder_full` (some windows have an empty target) and `decoder_full_noempty`
   (every window has a target);
3. each run writes the held-out summaries with three decodings (beam search with 2 beams, top-p sampling, beam search
   with a 1.3 times longer length budget) and the test file with the decoding that scores best on held-out ROUGE.
The 0.8518 file is the test file of one of the two runs; `report` lists both, best held-out score first.

Run from this folder on two GPUs (HF_TOKEN set, the dataset is gated):
    python best_submission.py --ckpt /path/to/decoder_80m_v1/best.pt
Output: /kaggle/working/summ_runs/decoder_full*/.../submission.zip (listed in /kaggle/working/all_results.txt)
"""

import argparse
import os
import time

import p6_paths  # noqa: F401  (Phase 4 / 5 building blocks)
from summ_grid import _prepare, report, run_jobs
from summ_mid import mid_train

MODEL = "decoder"
OUT = "/kaggle/working/summ_runs"
COMMON = dict(
    max_len=1024,
    epochs=4,
    batch_size=8,
    grad_accum=2,
    gen_batch=8,
    eval_docs=25,
    beams=2,
    no_repeat=3,
    log_every=200,
    lr_body=1e-4,
    decodings=("beam", "top_p", "beam_long"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Phase 1 decoder-only checkpoint (best.pt)")
    a = ap.parse_args()
    start = mid_train(MODEL, path=a.ckpt, tokens=30_000_000, batch_size=16, finish_by=time.time() + 30 * 60)
    base = dict(COMMON, out_dir=OUT)
    jobs = [
        dict(run_name=f"{MODEL}_full", technique="full", settings=dict(base)),
        dict(run_name=f"{MODEL}_full_noempty", technique="full", settings=dict(base, empty_frac=0.0)),
    ]
    for j in jobs:
        j["settings"]["max_len"] = _prepare(MODEL, start, j["settings"])
    todo = [j for j in jobs if not os.path.exists(f"{OUT}/{j['run_name']}/metrics.json")]
    run_jobs(MODEL, todo, start, OUT)
    report([OUT], out_path="/kaggle/working/all_results.txt")
    print(open("/kaggle/working/all_results.txt").read())


if __name__ == "__main__":
    main()
