#!/bin/bash

# Script to run MeanFlow training on MNIST dataset

# Ensure dependencies are installed (optional check, assuming environment is set)
# pip install -r requirements.txt 

echo "Starting MeanFlow training on MNIST..."

# Run the training script
# Using python directly as Accelerator is initialized inside the script. 
# For multi-gpu, you might use: accelerate launch train_mnist.py
python train_mnist.py
