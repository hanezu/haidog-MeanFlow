# Training MeanFlow on MNIST

This document outlines how to train the MeanFlow model using the MNIST dataset.

## Prerequisites

Ensure you have the required dependencies installed. This typically includes `torch`, `torchvision`, `accelerate`, `timm`, `einops`, etc.

## Files

- **`train_mnist.py`**: The main Python script created specifically for training on MNIST. It loads the dataset, initializes the `MFDiT` model (configured for 1-channel input), and runs the training loop using `MeanFlow` logic.
- **`run_mnist_train.sh`**: A helper shell script to execute the training.

## How to Run Training

You can start the training process by running the shell script:

```bash
./run_mnist_train.sh
```

Or directly via Python:

```bash
python train_mnist.py
```

The script will:
1.  Download the MNIST dataset (if not present) into a `mnist` directory.
2.  Create `images_mnist` to save sample generations during training.
3.  Create `checkpoints_mnist` to save model checkpoints.
4.  Log training progress to `log_mnist.txt`.

## Hyperparameter Adjustment

You can adjust training hyperparameters directly in `train_mnist.py`. Key parameters include:

### Training Configuration
- `n_steps`: Total number of training steps (default: 200,000).
- `batch_size`: Batch size for training (default: 128).
- `image_size`: Size to resize MNIST images to (default: 32x32).

### Model Configuration (`MFDiT`)
- `dim`: Model dimension width (default: 384).
- `depth`: Number of DiT blocks (default: 12).
- `num_heads`: Number of attention heads (default: 6).

### MeanFlow Configuration (`MeanFlow`)
- `flow_ratio`: Ratio for flow matching (default: 0.50).
- `time_dist`: Time distribution parameters (default: `['lognorm', -0.4, 1.0]`).
- `cfg_scale`: Classifier-Free Guidance scale (default: 2.0).

## Output

- **Logs**: Check `log_mnist.txt` or the console output for loss values.
- **Samples**: Check `images_mnist/` for generated images saved every `sample_step` (default: 1000 steps).
- **Checkpoints**: Saved in `checkpoints_mnist/` at the end of training.
