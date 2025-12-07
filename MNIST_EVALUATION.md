# Evaluating MeanFlow on MNIST

This document outlines how to evaluate the trained MeanFlow models using `evaluate.py`. This script calculates the **Approximate Negative Log Likelihood (NLL)** and **Bits Per Dimension (BPD)** on the MNIST test set.

It integrates the ODE (from data to noise) and uses the **Hutchinson Estimator** to efficiently approximate the divergence of the vector field. This is significantly faster than exact NLL computation.

## Usage

You can evaluate a specific checkpoint or all checkpoints for a given experiment.

### Representative Example

To evaluate all checkpoints for a specific experiment with a larger batch size and a limit on batches (useful for quick checks):

```bash
python evaluate.py --exp_name baseline_Dec_06-21_18 --batch_size 100 --limit_batches 1 --ckpt_all
```

### Common Commands

To evaluate the **latest checkpoint** of the "baseline" experiment:
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

## Arguments

- `--exp_name`: (Required) The name of the experiment folder in `results/`.
- `--ckpt_step`: (Optional) Specific checkpoint step to evaluate. Defaults to the latest checkpoint.
- `--ckpt_all`: (Flag) If set, evaluates ALL checkpoints found in the experiment directory.
- `--batch_size`: (Default: 64) Batch size for evaluation. Increasing this speeds up evaluation significantly if GPU memory permits.
- `--limit_batches`: (Default: None) Number of batches to evaluate. If not set, evaluates the entire test set (approx 10k images). Set this to a small number (e.g., 10) for quicker estimates.

## Output

Results are printed to the console and appended to a file named `evaluation_results.txt` within the experiment's directory (e.g., `results/{exp_name}/evaluation_results.txt`).

The output includes:
- **Average NLL**: Negative Log Likelihood in nats.
- **Average BPD**: Bits Per Dimension, comparable to standard literature results.
