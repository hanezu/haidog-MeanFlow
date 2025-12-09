#!/bin/bash

# Script to run MeanFlow training on MNIST with BEST (OURS) hyperparameters

echo "Starting MeanFlow training on MNIST (Ours - Best Trial)..."

# Best Parameters (from trial_chamfer_0_00176)
# 'stage0_t_method': 'uniform'
# 'stage0_r_method': 'lognorm'
# 'stage0_r_lognorm_mu': -2.710608627189448
# 'stage0_r_lognorm_sigma': 1.3670615236037662
# 'stage0_instant_prob': 0.44800769613418734
# 'stage0_resample': True

python train_mnist.py \
    --exp_name "ours_1_phase" \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --phase_configs "configs/ours_1_phase.yaml"
