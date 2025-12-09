#!/bin/bash

# Script to run MeanFlow finetuning on MNIST with BASELINE hyperparameters
# Loads pretrained model and trains for 5k more steps with smaller LR

echo "Starting MeanFlow finetuning on MNIST (Baseline - Finetune)..."

# Baseline Parameters (explicitly stated, though defaults in script match some)
# 'stage0_t_method': 'lognorm'
# 'stage0_t_mu': -0.4
# 'stage0_t_sigma': 1.0
# 'stage0_r_method': 'lognorm'
# 'stage0_r_lognorm_mu': -0.4
# 'stage0_r_lognorm_sigma': 1.0
# 'stage0_instant_prob': 0.75
# 'stage0_resample': False

EXP_NAME="baseline_10k_finetune_5k"

python train_mnist.py \
    --exp_name "$EXP_NAME" \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --phase_configs "configs/baseline.yaml" \
    --resume_from "pretrained/baseline_10k_28x28.pt" \
    --n_steps 5000 \
    --lr 1e-5

python evaluate.py --exp_name "$EXP_NAME" --batch_size 100 --ckpt_all --limit_batches 10
