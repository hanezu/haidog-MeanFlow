# Training MeanFlow on MNIST

This document outlines how to train the MeanFlow model using the MNIST dataset, with support for different hyperparameter configurations derived from ablation studies.

## Prerequisites

Ensure you have the required dependencies installed. This typically includes `torch`, `torchvision`, `accelerate`, `timm`, `einops`, etc.

## Files

- **`train_mnist.py`**: The main Python script for training. It now accepts command-line arguments to control the sampling strategy, CFG settings, and other hyperparameters.
- **`run_mnist_train_baseline.sh`**: Baseline hyperparameters (Lognorm T/R) with CFG ($w=2.0$) and Optuna flow ratio (0.75).
- **`run_mnist_train_ours.sh`**: Ours hyperparameters (Uniform T) with CFG ($w=2.0$) and Optuna flow ratio (0.75).
- **`run_mnist_train_baseline_no_cfg.sh`**: Baseline hyperparameters without CFG ($w=1.0$).
- **`run_mnist_train_ours_no_cfg.sh`**: Ours hyperparameters without CFG ($w=1.0$).
- **`run_mnist_train_haidog_baseline.sh`**: Original "Haidog" stable parameters (Flow ratio 0.50) with Baseline sampling.

## How to Run Training

All scripts are configured to run for 6,000 steps with an effective batch size of 128 (using gradient accumulation on smaller GPUs).

### 1. Optuna-Derived Configurations (Flow Ratio 0.75)
These configurations use the hyperparameters found via Optuna optimization on CIFAR/2D tasks.

*   **Baseline (with CFG)**:
    ```bash
    ./run_mnist_train_baseline.sh
    ```
*   **Ours (with CFG)**:
    ```bash
    ./run_mnist_train_ours.sh
    ```
*   **Baseline (No CFG)**:
    ```bash
    ./run_mnist_train_baseline_no_cfg.sh
    ```
*   **Ours (No CFG)**:
    ```bash
    ./run_mnist_train_ours_no_cfg.sh
    ```

### 2. Haidog Stable Configurations (Flow Ratio 0.50)
These configurations use the original `flow_ratio=0.50` which is known to be stable for this MNIST implementation.

*   **Haidog Baseline**:
    ```bash
    ./run_mnist_train_haidog_baseline.sh
    ```

The script will:
1.  Download the MNIST dataset (if not present) into a `mnist` directory.
2.  Create `images_mnist` to save sample generations during training.
3.  Create `checkpoints_mnist` to save model checkpoints.
4.  Log training progress to `log_mnist.txt`.

## Evaluation

You can evaluate the trained models using `evaluate.py`. This script calculates the **Approximate Negative Log Likelihood (NLL)** and **Bits Per Dimension (BPD)** on the MNIST test set.

It integrates the ODE (from data to noise) and uses the **Hutchinson Estimator** to efficiently approximate the divergence of the vector field. This is significantly faster than exact NLL computation.

### Usage

To evaluate the "baseline" experiment (latest checkpoint):
```bash
python evaluate.py --exp_name baseline
```

To evaluate **all checkpoints** for the "baseline" experiment:
```bash
python evaluate.py --exp_name baseline --ckpt_all
```

To evaluate the "ours" experiment:
```bash
python evaluate.py --exp_name t_uniform_r_lognorm_instant_prob_45_resample
```

### Arguments
- `--exp_name`: (Required) The name of the experiment folder in `results/`.
- `--ckpt_step`: (Optional) Specific checkpoint step to evaluate. Defaults to the latest checkpoint.
- `--ckpt_all`: (Flag) If set, evaluates ALL checkpoints found in the experiment directory.
- `--batch_size`: (Default: 16) Batch size for evaluation.
- `--limit_batches`: (Default: 10) Number of batches to evaluate.

## Hyperparameter Adjustment

You can adjust training hyperparameters via command-line arguments passed to `train_mnist.py`.

### Key Arguments
- `--n_steps`: Total training steps (default: 10000).
- `--batch_size`: Batch size per device (default: 32).
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 1).
- `--stage0_instant_prob`: The "flow ratio" or probability of instant flow (default: 0.5).

### Sampling Strategy Arguments
- `--stage0_t_method`: Method for sampling T (`lognorm` or `uniform`).
- `--stage0_r_method`: Method for sampling R (`lognorm` or `uniform`).
- `--stage0_resample`: Flag to enable resampling to strictly enforce ordering constraints (used in "Ours").

### CFG & Generation
- `--cfg_scale`: Classifier-Free Guidance scale (default: 2.0). Set to 1.0 for no guidance.
- `--cfg_ratio`: Probability of dropping labels during training (default: 0.10). Set to 0.0 for no CFG training.
- `--sample_steps`: Number of steps used for generation during training visualization (default: 5).

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
