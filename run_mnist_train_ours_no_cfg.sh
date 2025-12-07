#!/bin/bash

# Script to run MeanFlow training on MNIST with BEST (OURS) hyperparameters AND NO CFG

echo "Starting MeanFlow training on MNIST (Ours - No CFG)..."

python train_mnist.py \
    --exp_name "ours_no_cfg" \
    --n_steps 6000 \
    --cfg_scale 1.0 \
    --cfg_ratio 0.0 \
    --sample_steps 1 \
    --stage0_t_method "uniform" \
    --stage0_r_method "lognorm" \
    --stage0_r_lognorm_mu -2.7106086 \
    --stage0_r_lognorm_sigma 1.3670615 \
    --stage0_instant_prob 0.4480077 \
    --stage0_resample
