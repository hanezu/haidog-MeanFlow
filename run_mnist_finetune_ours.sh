#!/bin/bash

# Script to run MeanFlow finetuning on MNIST with BEST (OURS) hyperparameters
# Loads pretrained model and trains for 5k more steps with smaller LR

echo "Starting MeanFlow finetuning on MNIST (Ours - Finetune)..."

# Best Parameters (from trial_chamfer_0_00176)
# 'stage0_t_method': 'uniform'
# 'stage0_r_method': 'lognorm'
# 'stage0_r_lognorm_mu': -2.710608627189448
# 'stage0_r_lognorm_sigma': 1.3670615236037662
# 'stage0_instant_prob': 0.44800769613418734
# 'stage0_resample': True

python train_mnist.py \
    --exp_name "ours_1_phase_finetune_5k" \
    --batch_size 64 \
    --gradient_accumulation_steps 2 \
    --phase_configs "configs/ours_1_phase.yaml" \
    --resume_from "pretrained/baseline_10k_28x28.pt" \
    --n_steps 5000 \
    --lr 1e-5
