# Phase 3: rhetorical role labelling

Each sentence of a judgment (Competition Law or Income Tax) gets one of 13 rhetorical roles. Metric: macro-F1 over
all 13 roles. Best test score: 0.5079 (five copies of Llama + LoRA + BiGRU, Viterbi smoothing).

## Method

| Stage | What | File |
|---|---|---|
| Backbone | wraps a Phase 1 model: finds its hidden states, pools by the last token (left-to-right models) or the mean (encoder-decoder), batches of one length for the RNN encoder | `rr_backbone.py`, `rr_head.py` |
| Stage A (one sentence) | sentence (up to 128 tokens) -> pooled vector + learned vector of its position bucket (10 buckets) -> linear layer with 13 outputs; class weights `min(10, sqrt(n_max / n_c))` | `rr_head.py`, `rr_train.py` |
| Stage B (whole judgment) | Stage A frozen; a BiGRU (128 per direction) reads the Stage A vectors of all sentences of a judgment | `rr_head.py`, `rr_train.py` |
| Fine-tuning techniques | full, frozen, LoRA, rsLoRA, AdaLoRA, (IA)^3 on Llama with the `peft` library (allowed in this phase) | `rr_peft.py` |
| Out-of-fold stacking | Stage B trained on probabilities of Stage A models that did not see the judgment (tried; worse) | `rr_cv.py` |
| Distillation | teacher: five copies of Llama + LoRA + BiGRU, each labels the judgments it did not see; student: first 6 of Llama's 12 blocks, `0.5 CE + 0.5 T^2 KL`, T = 2; the same student without the teacher for comparison | `rr_kd.py` |
| Final submission | five copies trained on train + dev (72 of 90 judgments each), averaged, then Viterbi smoothing with role-change probabilities per domain, weight chosen out of fold (0.75) | `rr_final.py` |

## Files

- `rr_data.py` data, tokenisation, batching, class weights, metrics, smoothing, submission writer (no torch).
- `explore_rr.py` data statistics: sizes, label balance, sentence lengths, where each role occurs, role transitions.
- `test_*.py` self-tests on tiny stand-in models and fake data.

## Running

```python
from rr_backbone import make_backbone
from rr_final import run_final
from rr_kd import run_kd
from rr_peft import run_methods
from rr_train import run

bb = make_backbone(model, tokenizer)  # a Phase 1 model and the bpe12k tokenizer
run(bb, run_name="llama_lora", method="lora")
run_methods(bb, methods=("lora", "rslora", "adalora", "ia3"))
run_kd(bb, tokenizer, method="lora")
run_final(bb, method="lora")
```
