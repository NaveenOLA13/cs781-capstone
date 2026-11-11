#!/usr/bin/env bash
# Phase 2 runs (from the repository root, Kaggle GPU). Each run writes runs_ner/<run>/sub/submission.zip.
set -e
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN to read Phase_2_LNER}

P1=/kaggle/input/cs781-phase1-ckpts
DEC=$P1/decoder_80m_v1/best.pt
LLAMA=$P1/llama_85m_v1/best.pt
MOE=$P1/moe_214m_v1/best.pt
ENCDEC=$P1/encdec_86m_v1/best.pt
T=phase2/train_ner.py

python phase2/check_offsets.py
python $T --phase1 $LLAMA --run smoke --max_docs 8 --epochs 1 --no_test

# backbones x techniques
python $T --phase1 $LLAMA --run llama_lorma_plus --method lorma_plus --r 16 --alpha 32 \
          --head bigru --lr 5e-4 --epochs 6
python $T --phase1 $LLAMA --run llama_lorma_pi --method lorma_pi --r 16 \
          --head bigru --lr 5e-4 --epochs 6
python $T --phase1 $DEC --run dec_lora_echo --method lora --r 16 --alpha 32 --echo 1 \
          --head linear --lr 5e-4 --epochs 6
python $T --phase1 $MOE --run moe_full --method full --head bigru --lr 5e-5 --epochs 6 --batch_size 4 --grad_accum 2
python $T --phase1 $ENCDEC --run encdec_full --method full --head linear --lr 2e-4 --epochs 6

# distillation (MoE teacher -> 6-block Llama) and its control without the teacher
python $T --phase1 $LLAMA --run kd_moe_to_llama6 --keep_layers 6 --method full --head bigru \
          --teacher runs_ner/moe_full/best_ner.pt --kd_alpha 0.5 --kd_T 2 --lr 1e-4 --epochs 6
python $T --phase1 $LLAMA --run llama6_noKD --keep_layers 6 --method full --head bigru --lr 1e-4 --epochs 6

# ablations: frozen backbone, no pretraining
python $T --phase1 $LLAMA --run llama_headonly --method head --head bigru --head_lr 1e-3 --epochs 6
python $T --phase1 $LLAMA --run llama_scratch --scratch 1 --method full --head bigru --lr 3e-4 --epochs 6
