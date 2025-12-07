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
    --stage0_phase_configs "configs/haidog_baseline.yaml"
