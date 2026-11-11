"""Phase 7 best submission: micro-F1@7 0.4914 (Codabench ID 1039837, rank 11; dev 0.4825, 5-fold cross-fitted).

Input: the saved dev / test scores of the Phase 7 Llama runs (`pcr_results_llama.zip`, made with `pcr_grid`: the
mid-trained Phase 1 Llama fine-tuned as a chunk-level bi-encoder with several techniques, see README). No training
happens here; this step runs on a CPU.
`pcr_final.main` builds, per query, candidates from BM25 (word 1- to 5-grams), the votes of the 20 most similar
labelled queries, BM25 on the 40 words around each <CITATION> mask and the models' scores; removes candidates decided
after the query (judgment dates from the header); adds the citator matches (back, forward, co-cited); fits every
variant's ranker by 5-fold cross-fitting on dev and writes the best ones. The submitted file is the best variant,
`final_all+models_citator_datefilter_coord__dev0.4825` (ranker: coordinate ascent).

Run from this folder (HF_TOKEN set, the dataset is gated):
    python best_submission.py --results /path/to/folder/with/pcr_results_llama.zip
Output: /kaggle/working/final/final_all+models_citator_datefilter_coord__dev*.zip (the top line of final_results.txt)
"""

import argparse

import p7_paths  # noqa: F401  (Phase 4 building blocks)
from pcr_final import main as final_ranking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder holding the Llama runs' results zip (or unzipped runs)")
    ap.add_argument("--out", default="/kaggle/working/final")
    a = ap.parse_args()
    rows = final_ranking(results_roots=(a.results,), out_dir=a.out)
    print("submit:", rows[0].get("submission"))


if __name__ == "__main__":
    main()
