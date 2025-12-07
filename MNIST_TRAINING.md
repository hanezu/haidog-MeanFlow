# Training MeanFlow on MNIST

This document outlines how to train the MeanFlow model using the MNIST dataset, with support for different hyperparameter configurations derived from ablation studies.

## Prerequisites

Ensure you have the required dependencies installed. This typically includes `torch`, `torchvision`, `accelerate`, `timm`, `einops`, etc.

## Files

- **`train_mnist.py`**: The main Python script for training. It now accepts command-line arguments to control the sampling strategy and other hyperparameters.
- **`run_mnist_train_baseline.sh`**: Launches training with **Baseline** hyperparameters (standard Lognorm sampling for both T and R).
- **`run_mnist_train_ours.sh`**: Launches training with **Ours (Best)** hyperparameters (Uniform T, specific Lognorm R, Resampling enabled).

## How to Run Training

### 1. Baseline Configuration
To run the baseline configuration:
```bash
./run_mnist_train_baseline.sh
```

### 2. Ours (Best) Configuration
To run the optimized configuration found via Optuna:
```bash
./run_mnist_train_ours.sh
```

The script will:
1.  Download the MNIST dataset (if not present) into a `mnist` directory.
2.  Create `images_mnist` to save sample generations during training.
3.  Create `checkpoints_mnist` to save model checkpoints.
4.  Log training progress to `log_mnist.txt`.

## Hyperparameter Adjustment

You can adjust training hyperparameters via command-line arguments passed to `train_mnist.py`.

### Key Arguments
- `--n_steps`: Total training steps (default: 10000).
- `--batch_size`: Batch size (default: 128).
- `--stage0_instant_prob`: The "flow ratio" or probability of instant flow (default: 0.5).

### Sampling Strategy Arguments
- `--stage0_t_method`: Method for sampling T (`lognorm` or `uniform`).
- `--stage0_r_method`: Method for sampling R (`lognorm` or `uniform`).
- `--stage0_resample`: Flag to enable resampling to strictly enforce ordering constraints (used in "Ours").

Example custom run:
```bash
python train_mnist.py --stage0_t_method uniform --stage0_r_method uniform --stage0_instant_prob 0.2
```

## Output

All outputs are now organized by experiment name under the `results/` directory.

- **Logs**: Saved to `results/{exp_name}/log.txt`.
- **Samples**: Saved to `results/{exp_name}/images/` for generated images.
- **Checkpoints**: Saved to `results/{exp_name}/checkpoints/` at the end of training.

For example, running the baseline script will produce outputs in `results/baseline/`.
