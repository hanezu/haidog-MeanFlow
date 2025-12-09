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

python train_mnist.py \
    --exp_name "baseline_10k_finetune_5k" \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --phase_configs "configs/baseline.yaml" \
    --resume_from "pretrained/baseline_10k.pt" \
    --n_steps 5000 \
    --lr 1e-5
