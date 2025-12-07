#!/bin/bash

# Script to run MeanFlow training on MNIST with BASELINE hyperparameters AND NO CFG

echo "Starting MeanFlow training on MNIST (Baseline - No CFG)..."

python train_mnist.py \
    --exp_name "baseline_no_cfg" \
    --n_steps 6000 \
    --cfg_scale 1.0 \
    --cfg_ratio 0.0 \
    --sample_steps 1 \
    --stage0_t_method "lognorm" \
    --stage0_t_mu -0.4 \
    --stage0_t_sigma 1.0 \
    --stage0_r_method "lognorm" \
    --stage0_r_lognorm_mu -0.4 \
    --stage0_r_lognorm_sigma 1.0 \
    --stage0_instant_prob 0.75 \
    # --stage0_resample is False by default
