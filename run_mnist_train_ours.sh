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
    --exp_name "t_uniform_r_lognorm_instant_prob_45_resample" \
    --stage0_t_method "uniform" \
    --stage0_r_method "lognorm" \
    --stage0_r_lognorm_mu -2.7106086 \
    --stage0_r_lognorm_sigma 1.3670615 \
    --stage0_instant_prob 0.4480077 \
    --stage0_resample
