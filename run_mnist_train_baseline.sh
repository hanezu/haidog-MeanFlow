#!/bin/bash

# Script to run MeanFlow training on MNIST with BASELINE hyperparameters

echo "Starting MeanFlow training on MNIST (Baseline)..."

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
    --exp_name "baseline" \
    --stage0_t_method "lognorm" \
    --stage0_t_mu -0.4 \
    --stage0_t_sigma 1.0 \
    --stage0_r_method "lognorm" \
    --stage0_r_lognorm_mu -0.4 \
    --stage0_r_lognorm_sigma 1.0 \
    --stage0_instant_prob 0.75 \
    # --stage0_resample is False by default