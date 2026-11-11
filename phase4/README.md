# Phase 4: court judgment prediction (CJPE)

Predict from the text of a Supreme Court appeal whether it was accepted (1) or rejected (0). Metric: macro-F1. Best
test score: 0.7765 (mid-trained Llama, whole case read in 8 chunks with a BiGRU and attention pooling).

## Method

| Stage | What | File |
|---|---|---|
| Data cleaning | the dev file contained the whole test set with labels, and train / dev had repeated texts; test cases are removed from dev and repeated texts from both (train 37,387 -> 31,677, dev 3,505 -> 994) | `cj_data.py` |
| Model loading | builds a Phase 1 model from its checkpoint and the bpe12k tokenizer from `phase1/`; shared by Phases 4 to 7 | `cj_models.py` |
| Mid-training | next-token prediction on the train + dev judgments (40M tokens for Llama), never test text; not for the encoder-decoder | `cj_mid.py` |
| Fine-tuning techniques (own code) | frozen, full, partial (top third), gradual unfreezing, BitFit, LoRMA, AdaLoRA, adapters, LoRA, QLoRA (4-bit NF4), (IA)^3; low-rank updates also work on GRU weights | `cj_tech.py` |
| Stage A | last 512 tokens of a case (training: last 3 chunks of 512, stride 448, as separate examples) -> last-token or mean pooling -> linear layer with 2 outputs; class-weighted cross-entropy | `cj_train.py` |
| Stage B (whole case) | Stage A's encoder, unchanged, encodes the last 8 chunks; `gru_attn` (BiGRU + attention pooling + last chunk), `transformer` (two attention blocks, positions from the end) or `mean` decides | `cj_hier.py` |
| Threshold | chosen on dev for macro-F1 (Stage B probabilities on new cases are low) | `cj_data.py` |
| Distillation | teacher: the whole-case `gru_attn` system; student: the mid-trained model with its first half of blocks reading the last 512 tokens; the same student without the teacher for comparison | `cj_kd.py`, `cj_train.py` |
| Ensemble | greedy forward selection of finished runs on dev, one tuned threshold | `cj_ensemble.py` |

## Files

- `cj_grid.py` runs several techniques of one model, one process per GPU.
- `cj_explore.py` data statistics: fields, label balance, lengths in words and tokens, duplicates across splits.
- `test_*.py` self-tests (data and ensemble without torch; the rest on tiny stand-in models).

## Running (Kaggle, two GPUs)

```python
from cj_grid import run_grid
from cj_hier import run_hier_many
from cj_kd import run_kd
from cj_mid import mid_train

START = mid_train("llama")
run_grid("llama", techniques=("full", "lorma"), ckpt=START, train_chunks=3)
run_hier_many(["/kaggle/working/cj_runs/llama_full"], k=8)
run_kd("/kaggle/working/cj_runs/llama_full/hier_k8", ckpt=START)
```
