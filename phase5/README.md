# Phase 5: Legal Statute Identification (LSI)

Multi-label task: for every legal document, predict the set of Indian Penal Code statute ids (100 ids, about 3.8 per
document). Metric: macro-F1 over statutes (primary), micro-F1, subset accuracy. All code is ours; every experiment
starts from our Phase 1 models; only the competition data is used (statute labels of train/dev; test text only for
prediction, never for training or mid-training).

## Pipeline (per Phase 1 model)

| stage | what | file |
|---|---|---|
| pretraining | our Phase 1 model (decoder-only 80M, encoder-decoder 86M, Llama 85M, MoE 214M) | (Phase 1) |
| mid-training | next-token prediction on the LSI train + dev text (40M tokens; MoE 20M); not for the encoder-decoder | `lsi_mid.py` |
| fine-tuning (SFT) | 10 techniques (LoRA, LoRMA, QLoRA (own 4-bit NF4), adapters, full, partial, gradual unfreezing, BitFit, AdaLoRA, frozen; Phase 5 runs LoRA, LoRMA, QLoRA, adapter per model), all our own code (Phase 4 `cj_tech.py`) | `lsi_train.py`, `lsi_grid.py` |
| long documents | the first 8,192 tokens are cut into 512-token chunks; training reads 3 random chunks spread over the document, prediction reads all of them; document score of a statute = maximum over its chunks (multiple-instance view: one part of the document is enough evidence) | `lsi_train.py` |
| whole-document stage (Stage B) | statute-wise attention over all chunk vectors of a document (+ BiGRU over the chunks, optional statute co-occurrence layer) | `lsi_hier.py` |
| thresholds | one global threshold or one per statute, chosen by 2-fold cross-fitting on dev; every document gets at least its top statute | `lsi_data.py` |
| knowledge distillation | teacher = best Stage B version of llama_full (whole document, full size); student = the mid-trained Llama with half its blocks, reading chunks; same student without the teacher for comparison | `lsi_grid.run_kd` |
| deadline | `finish_by` (a clock time): every run measures its own speed (prediction before training, training steps after 20 and every 100 steps) and trains only the steps that fit before it, then predicts and writes everything; `phase5_run.ipynb` gives each stage its share of the time until `FINISH_IST` | `lsi_train.py`, `lsi_data.plan_steps` |
| ensemble | greedy forward selection of results on dev (any models / techniques / Stage B versions), cross-fitted thresholds | `lsi_grid.ensemble` |

Imbalance: BCE with pos_weight = (negatives/positives)^0.5 (capped at 30) and head bias = statute frequency; the
asymmetric loss (`loss="asl"`) is available as an alternative.

## Files

- `lsi_data.py` data loading and cleaning (test documents removed from train/dev), statute list, exploration, metrics,
  thresholds, submission writer (valid unique integer ids, one row per test document). No torch.
- `lsi_train.py` one model + one technique; options: chunks, loss, KD student (`student_layers`, `kd_teacher`),
  `save_chunks` (vector and scores of every chunk, for Stage B).
- `lsi_grid.py` `run_grid` (techniques on both GPUs), `run_kd`, `ensemble`, `compare_all`, `report`.
- `lsi_hier.py` Stage B; `lsi_mid.py` mid-training; `p5_paths.py` reuses the Phase 4 building blocks from `../phase4`.
- `phase5_run.ipynb` the Kaggle notebook for one model (set `MODEL`); `phase5_final.ipynb` the cross-model ensemble.
- Tests: `test_lsi_data.py` (no torch), `test_lsi_mid.py`, `test_lsi_train.py`, `test_lsi_hier.py` (tiny stand-in
  models; the notebook runs them first).

## How to run (Kaggle, GPU T4 x2, internet on, secrets GH_TOKEN + HF_TOKEN)

1. Inputs: dataset `checkpoints-models` (decoder, encoder-decoder, Llama) and `moe-checkpoint`.
2. `phase5_run.ipynb` (set `MODEL` and `FINISH_IST`; each model: 2 main techniques + whole-document stage + KD + 2 more techniques, spares if one fails) with `MODEL = "llama"`, then `"decoder"` (your account), `"moe"`, then `"encdec"` (friend's
   account), each as Save Version -> Save & Run All. A notebook stops starting new stages after 10.5 h; to continue,
   attach its output as input and set `RESUME = "auto"`.
3. `phase5_final.ipynb` with the four outputs attached: best mix of everything, its submission, `all_results.txt`.

Outputs per notebook: `lsi_runs/<model>_<technique>/` (log, metrics, per-statute F1, probabilities, `submission/`),
`lsi_runs/<model>_<technique>/stageB/<version>/`, KD students `lsi_runs/llama_student_kd`, `llama_student`,
`lsi_ensemble_<model>/`, `all_results.txt`, `lsi_results_<model>.zip` (everything except weights).

## Research basis

- The data is the ILSI dataset (Paul et al., AAAI 2022, LeSICiN): 66,090 fact descriptions, 100 IPC sections,
  42,884 / 10,203 / 13,043 split, 3.78 statutes per document, names masked. Published macro-F1 on ILSI: LeSICiN
  28.45; hierarchical BERT baselines and other neural baselines 7.6-23.9 (threshold choice matters a lot: LeSICiN with
  another threshold 19.77). IL-TUR (Joshi et al., ACL 2024), same task with statute references masked: InLegalBERT
  (hierarchical, BiLSTM over sentence vectors) 26.23 macro-F1, LeSICiN 28.08, GPT-4 zero-shot 23.99.
  -> every strong system reads the document hierarchically; thresholds are tuned; the score range is low (rare
  statutes).
- Long legal documents: hierarchical encoders (chunk/paragraph encoder + document-level layer) are the standard
  strong choice for multi-label legal classification (LexGLUE's ECtHR tasks; Dai et al. 2022 on long-document
  classification).
- Many labels, each with its own evidence: label-wise attention (each label attends over the text with its own query)
  is the strongest family for large multi-label document tasks (CAML, Mullenbach et al. 2018; Chalkidis et al. 2019 on
  EU legislation) -> Stage B's statute-wise attention over chunks.
- Imbalance and many negatives: weighted BCE (LeSICiN uses frequency-based class weights capped at 10) and the
  asymmetric loss (Ridnik et al., ICCV 2021).
- Mid-training on in-domain text before fine-tuning: Phase 4 showed it (held-out loss 3.46 -> 2.60, dev macro-F1
  +5.7 points for the same fine-tuning).

Sources: LeSICiN https://arxiv.org/abs/2112.14731 ; IL-TUR https://aclanthology.org/2024.acl-long.618 ;
LexGLUE https://arxiv.org/abs/2110.00976 ; long-document classification https://arxiv.org/abs/2204.06683 ;
label-wise attention on EU legislation https://arxiv.org/abs/1906.02192 ; asymmetric loss https://arxiv.org/abs/2009.14119
