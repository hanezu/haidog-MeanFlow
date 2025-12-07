# Training MeanFlow on MNIST

This document outlines how to train the MeanFlow model using the MNIST dataset, with support for different hyperparameter configurations derived from ablation studies.

## Prerequisites

Ensure you have the required dependencies installed. This typically includes `torch`, `torchvision`, `accelerate`, `timm`, `einops`, etc.

## Files

- **`train_mnist.py`**: The main Python script for training. It now accepts command-line arguments to control the sampling strategy, CFG settings, and other hyperparameters.
- **`run_mnist_train_baseline.sh`**: Baseline hyperparameters (lognorm t/r) with CFG ($w=2.0$) and original flow ratio (0.75).
- **`run_mnist_train_ours.sh`**: Ours hyperparameters (Uniform t, lognorm r) with CFG ($w=2.0$) and Optuna-optimized flow ratio (0.45).
- **`run_mnist_train_baseline_no_cfg.sh`**: Baseline hyperparameters without CFG ($w=1.0$).
- **`run_mnist_train_ours_no_cfg.sh`**: Ours hyperparameters without CFG ($w=1.0$).
- **`run_mnist_train_haidog_baseline.sh`**: Original "Haidog" stable parameters (Flow ratio 0.50) with Baseline sampling.

## How to Run Training

All scripts are configured to run for 10,000 steps with an effective batch size of 128 (using gradient accumulation on smaller GPUs).

### 1. Baseline Configurations

*   **Baseline (with CFG)**:
    ```bash
    ./run_mnist_train_baseline.sh
    ```
*   **Baseline (No CFG)**:
    ```bash
    ./run_mnist_train_baseline_no_cfg.sh
    ```

### 2. Ours Configurations

*   **Ours (with CFG)**:
    ```bash
    ./run_mnist_train_ours.sh
    ```
*   **Ours (No CFG)**:
    ```bash
    ./run_mnist_train_ours_no_cfg.sh
    ```

### 3. Haidog Stable Configurations (Flow Ratio 0.50)
These configurations use the original `flow_ratio=0.50` which is known to be stable for this MNIST implementation.

*   **Haidog Baseline**:
    ```bash
    ./run_mnist_train_haidog_baseline.sh
    ```

The script will:
1.  Download the MNIST dataset (if not present) into a `mnist` directory.
2.  Create `results/EXP_NAME` to save sample generations and model checkpoints during training.
4.  Log training progress to `log.txt`.

## Evaluation

For details on evaluating trained models (calculating NLL and BPD), please refer to [MNIST_EVALUATION.md](MNIST_EVALUATION.md).

## Hyperparameter Adjustment

You can adjust training hyperparameters via command-line arguments passed to `train_mnist.py`.

### Key Arguments
- `--n_steps`: Total training steps (default: 10000).
- `--batch_size`: Batch size per device (default: 32).
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default: 4).
- `--stage0_instant_prob`: The "flow ratio" or probability of instant flow (default: 0.5).

### Sampling Strategy Arguments
- `--stage0_t_method`: Method for sampling T (`lognorm` or `uniform`).
- `--stage0_r_method`: Method for sampling R (`lognorm` or `uniform`).
- `--stage0_resample`: Flag to enable resampling to strictly enforce ordering constraints (used in "Ours").

### CFG & Generation
- `--cfg_scale`: Classifier-Free Guidance scale (default: 2.0). Set to 1.0 for no guidance.
- `--cfg_ratio`: Probability of dropping labels during training (default: 0.10). Set to 0.0 for no CFG training.
- `--sample_steps`: Space-separated list of steps used for generation during training visualization (default: `1 5`). Each value produces its own grid (e.g., `1-step_XXXX.png`, `5-step_XXXX.png`).

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
