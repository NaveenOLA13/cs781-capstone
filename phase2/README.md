# Phase 2 — Legal NER (L-NER)

Fine-tunes the Phase 1 backbones (decoder / llama / moe / encdec) as BIO token taggers and
writes character-span predictions for fold3.

| file | what |
|---|---|
| `ner_data.py` | load folds from HF (or `--data_dir`), BPE token ↔ char offsets, BIO tags, sliding windows, span decoding, learned boundary repair, strict macro P/R/F1, submission writer |
| `check_offsets.py` | **run first**: gold → BIO → decoded spans round trip (must be ≈1.0) + dataset statistics |
| `peft.py` | LoRA, **LoRMA₊** `h=(I+(α/r)BA)W₀x`, **LoRMA_π** `h=I_π(BA)W₀x` (identity at init) |
| `ner_model.py` | backbone features + tag head (`linear` or `bigru`), echo input option, KD loss |
| `train_ner.py` | train / eval (dev strict macro-F1 each epoch) / predict fold3 → `runs_ner/<run>/sub/submission.zip` |
| `run_phase2.sh` | the experiment plan (4 backbones, full/LoRA/LoRMA₊/LoRMA_π/head-only, KD + control, ablations) |

## Kaggle quick start
1. Attach: this repo (with `tok/bpe12k/tokenizer.json`), your Phase 1 `best.pt` files as a dataset, GPU T4.
2. Add `HF_TOKEN` as a Kaggle secret (same as Phase 1).
3. `cd /kaggle/working/cs-781 && python phase2/check_offsets.py`
4. Smoke test: `python phase2/train_ner.py --phase1 <best.pt> --run smoke --max_docs 8 --epochs 1 --no_test`
5. Run experiments from `run_phase2.sh` (several Kaggle notebooks in parallel to save time).
6. Upload `runs_ner/<run>/sub/submission.zip` (prediction.json at zip root, every fold3 id).

## Design notes (things to explain in the viva)
- **Offsets.** Pre-tokens carry a leading space (`" STATE"`), so token char spans are whitespace-stripped;
  whitespace-only tokens get label -100 and are skipped in decoding. Multi-byte UTF-8 handled byte→char.
  End offsets are exclusive (`[137,153,"RESP","STATE OF HARYANA"]` = 16 chars).
- **In-token boundaries.** The BPE regex merges punctuation runs (`Ltd.` + `,` → token `.,`), so an entity can
  end inside a token. `learn_boundary_table` learns from TRAIN docs how many chars of such tokens belong to the
  entity (`--boundary_fix 1`). `check_offsets.py` prints the oracle F1 with and without it.
- **Long documents.** Windows of `--window` tokens with `--stride`; at inference each token's tag distribution
  is a triangular-weighted average over all windows covering it (edges have the least context).
- **Causal backbones.** Decoder/llama/moe only see left context. `--head bigru` adds right context;
  `--echo 1` feeds `[w ; w]` and reads the second copy. EncDec uses its BiGRU encoder (bidirectional).
- **LoRMA.** Multiplicative update on the output side of each frozen Linear, computed as `M (W₀x)` so `MW₀`
  is never formed. LoRMA₊ starts at `M=I` via `B=0`; LoRMA_π starts at `M=I` via the rotation trick.
  Not applicable to GRU weights → encdec uses full/head fine-tuning.
- **KD.** Teacher = fine-tuned NER checkpoint; loss = (1-α)·CE + α·T²·KL(teacher_T ‖ student_T) per token.
  Same tokenizer across all Phase 1 models → token-level logits align 1:1.
- **Metric.** `strict_scores` = exact (start, end, label) match, macro over the 12 labels.
