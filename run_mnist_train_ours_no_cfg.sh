#!/bin/bash

# Script to run MeanFlow training on MNIST with BEST (OURS) hyperparameters AND NO CFG

echo "Starting MeanFlow training on MNIST (Ours 1-phase - No CFG)..."

python train_mnist.py \
    --exp_name "ours_no_cfg" \
    --batch_size 64 \
    --gradient_accumulation_steps 2 \
    --cfg_scale 1.0 \
    --cfg_ratio 0.0 \
    --phase_configs "configs/ours_1_phase.yaml"
