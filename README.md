# CS781 Capstone: Legal Language Modelling from Scratch

Four language models pretrained from scratch on Indian legal text (Phase 1), then adapted to six legal NLP tasks
(Phases 2 to 7). Every experiment in this repository starts from these models: no externally pretrained model,
tokenizer or embedding is used. The fine-tuning techniques, BM25, rankers and losses are our own code (the `peft`
library is used only in Phase 3, where it was allowed). Test labels are never used, and test text is never used for
training.

| Phase | Task | Metric | Best test score |
|---|---|---|---|
| 1 | Pretraining: decoder-only (80M), Llama-style (85M), mixture of experts (214M), RNN encoder-decoder (86M); test = 128-word continuations of 1,000 prompts | BERTScore | 0.8288 |
| 2 | Legal named entity recognition | strict macro-F1 | 0.4715 |
| 3 | Rhetorical role labelling | macro-F1 (13 roles) | 0.5079 |
| 4 | Court judgment prediction | macro-F1 | 0.7765 |
| 5 | Legal statute identification | macro-F1 | 0.3678 |
| 6 | Judgment summarization | BERTScore F1 | 0.8518 |
| 7 | Prior case retrieval | micro-F1@7 | 0.4914 |

## Best submissions

Each phase folder has a `best_submission.py` that reruns its best submission with the exact settings of that run
(its header gives the score, the Codabench submission ID, the model and the settings).

| Phase | File | Model | Codabench |
|---|---|---|---|
| 1 | `phase1/best_submission.py` | Llama-style 85M, `generate.py` | 0.8288 |
| 2 | `phase2/best_submission.py` | encoder-decoder, full fine-tuning, linear head | 0.4715 |
| 3 | `phase3/best_submission.py` | Llama + LoRA + BiGRU, 5 copies, Viterbi smoothing | 0.5079 |
| 4 | `phase4/best_submission.py` | mid-trained Llama, full fine-tuning, whole case (`gru_attn`) | 0.7765 |
| 5 | `phase5/best_submission.py` | mid-trained Llama runs, statute-wise attention, ensemble | 0.3678 |
| 6 | `phase6/best_submission.py` | mid-trained decoder-only, full fine-tuning, best of three decodings | 0.8518 |
| 7 | `phase7/best_submission.py` | final ranking over the Llama runs (dates, votes, citation contexts, citator) | 0.4914 |

## Layout

```
phase1/   tokenizer, corpus preparation, the four architectures (models/), pretraining, configs, tokenizer file
phase2/   named entity recognition (BIO tagging, own LoRA / LoRMA)
phase3/   rhetorical roles; also the shared backbone wrapper (rr_backbone.py) and heads (rr_head.py)
phase4/   judgment prediction; also the model loader (cj_models.py) and all fine-tuning techniques (cj_tech.py)
phase5/   statute identification
phase6/   summarization
phase7/   prior case retrieval
```

Later phases reuse earlier code: Phases 4 to 7 load models with `phase4/cj_models.py` (which puts `phase1/` and
`phase3/` on the import path) and fine-tune them with `phase4/cj_tech.py`; each phase's `p*_paths.py` makes those
folders importable.

## Setup

```
pip install -r requirements.txt
export HF_TOKEN=...            # the competition data is read from the Hugging Face dataset Exploration-Lab/CS-781-Capstone
```

All runs used Kaggle notebooks with two T4 GPUs. Phase 1 checkpoints (`best.pt` of each model) are attached to the
notebooks as a dataset. Each phase folder has a README with its method, files and how it was run.

## Tests

Every phase has self-tests (`test_*.py`) that run on tiny stand-in models or fake data, for example
`cd phase7 && python test_pcr_final.py`. The tests of data and ranking code need no torch; the others need torch.
