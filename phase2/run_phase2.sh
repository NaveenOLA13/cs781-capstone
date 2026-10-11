#!/usr/bin/env bash
# Phase 2 (L-NER) experiment plan. Run from the REPO ROOT on a Kaggle GPU (T4/P100).
# Each run prints dev strict macro-F1 and writes runs_ner/<run>/sub/submission.zip for the leaderboard.
# Hyperparameters below are only starting values -- choose/tune them yourself (course LLM policy).
set -e
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN (Kaggle secret) to stream Phase_2_LNER}

P1=/kaggle/input/cs781-phase1-ckpts          # <-- folder holding your Phase 1 best.pt files
DEC=$P1/decoder_80m_v1/best.pt
LLAMA=$P1/llama_85m_v1/best.pt
MOE=$P1/moe_214m_v1/best.pt
ENCDEC=$P1/encdec_86m_v1/best.pt
T=phase2/train_ner.py

# 0) BEFORE anything: offsets round trip must be ~1.0 on every label that occurs
python phase2/check_offsets.py

# 0b) 2-minute smoke test of the full train->eval->predict path
python $T --phase1 $LLAMA --run smoke --max_docs 8 --epochs 1 --no_test

# ---- 4 different Phase 1 backbones x different adaptation methods -------------------------
# E1  Llama-85M  + LoRMA_+ (compulsory)  + BiGRU head (gives right context to a causal LM)
python $T --phase1 $LLAMA --run llama_lorma_plus --method lorma_plus --r 16 --alpha 32 \
          --head bigru --lr 5e-4 --epochs 6
# E2  Llama-85M  + LoRMA_pi (permutation rank inflation variant)
python $T --phase1 $LLAMA --run llama_lorma_pi --method lorma_pi --r 16 \
          --head bigru --lr 5e-4 --epochs 6
# E3  Decoder-80M + LoRA, echo input (each token sees the whole window)
python $T --phase1 $DEC --run dec_lora_echo --method lora --r 16 --alpha 32 --echo 1 \
          --head linear --lr 5e-4 --epochs 6
# E4  MoE-214M   + full fine-tuning (also the KD TEACHER)
python $T --phase1 $MOE --run moe_full --method full --head bigru --lr 5e-5 --epochs 6 --batch_size 4 --grad_accum 2
# E5  EncDec-86M (BiGRU encoder is natively bidirectional) + full fine-tuning
python $T --phase1 $ENCDEC --run encdec_full --method full --head linear --lr 2e-4 --epochs 6
# E6  Knowledge distillation: MoE-214M teacher (E4) -> 6-layer Llama student (~half the blocks)
python $T --phase1 $LLAMA --run kd_moe_to_llama6 --keep_layers 6 --method full --head bigru \
          --teacher runs_ner/moe_full/best_ner.pt --kd_alpha 0.5 --kd_T 2 --lr 1e-4 --epochs 6
#     KD control: same student WITHOUT the teacher (needed to show KD actually helps)
python $T --phase1 $LLAMA --run llama6_noKD --keep_layers 6 --method full --head bigru --lr 1e-4 --epochs 6

# ---- ablations for the report ------------------------------------------------------------
# frozen backbone, head only (how much does adaptation of the backbone matter?)
python $T --phase1 $LLAMA --run llama_headonly --method head --head bigru --head_lr 1e-3 --epochs 6
# no pretraining (same arch, random init) -> value of Phase 1 pretraining
python $T --phase1 $LLAMA --run llama_scratch --scratch 1 --method full --head bigru --lr 3e-4 --epochs 6

# ---- final model: retrain the best config on ALL labelled docs, submit its zip -----------
# python $T --phase1 $LLAMA --run llama_lorma_plus_ALL --method lorma_plus ... --train_on_all 1
