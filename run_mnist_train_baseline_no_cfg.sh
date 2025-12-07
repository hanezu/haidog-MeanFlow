#!/bin/bash

# Script to run MeanFlow training on MNIST with BASELINE hyperparameters AND NO CFG

echo "Starting MeanFlow training on MNIST (Baseline - No CFG)..."

python train_mnist.py \
    --exp_name "baseline_no_cfg" \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --cfg_scale 1.0 \
    --cfg_ratio 0.0 \
    --phase_configs "configs/baseline.yaml"
