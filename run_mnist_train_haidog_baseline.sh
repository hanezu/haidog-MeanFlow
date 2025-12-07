#!/bin/bash

# Script to run MeanFlow training on MNIST with Haidog's parameters (matches original script)
# - Flow Ratio: 0.50
# - CFG: Yes
# - Batch Size: 32 (with grad accum 4 => 128 effective)

echo "Starting MeanFlow training on MNIST (Haidog's Baseline)..."

python train_mnist.py \
    --exp_name "haidog_baseline" \
    --n_steps 6000 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --cfg_scale 2.0 \
    --cfg_ratio 0.1 \
    --sample_steps 5 \
    --stage0_t_method "lognorm" \
    --stage0_t_mu -0.4 \
    --stage0_t_sigma 1.0 \
    --stage0_r_method "lognorm" \
    --stage0_r_lognorm_mu -0.4 \
    --stage0_r_lognorm_sigma 1.0 \
    --stage0_instant_prob 0.50 \
    # --stage0_resample is False by default
