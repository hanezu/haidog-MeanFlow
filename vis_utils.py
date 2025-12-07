import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import pandas as pd
import subprocess
from IPython.display import display

def parse_log_file(filepath):
    """Parses the training log file to extract steps, losses, and MSE losses."""
    steps = []
    losses = []
    mse_losses = []
    
    if not os.path.exists(filepath):
        print(f"Log file not found: {filepath}")
        return steps, losses, mse_losses
        
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Pattern: Global Step: 500    Loss: 0.123456    MSE_Loss: 0.012345
    pattern = r"Global Step: (\d+)\s+Loss: ([0-9.]+)\s+MSE_Loss: ([0-9.]+)"
    matches = re.findall(pattern, content)
    
    for match in matches:
        steps.append(int(match[0]))
        losses.append(float(match[1]))
        mse_losses.append(float(match[2]))
        
    return steps, losses, mse_losses

def plot_training_curves(log_file):
    """Plots training loss and MSE loss from the log file."""
    steps, losses, mse_losses = parse_log_file(log_file)

    if steps:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, label='Total Loss')
        plt.xlabel('Global Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(steps, mse_losses, label='MSE Loss', color='orange')
        plt.xlabel('Global Step')
        plt.ylabel('MSE')
        plt.title('MSE Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    else:
        print("No log data found to plot.")

def parse_eval_results(filepath):
    """Parses the evaluation results text file into a DataFrame."""
    data = []
    if not os.path.exists(filepath):
        print(f"Evaluation results file not found: {filepath}")
        return pd.DataFrame()
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    current_step = None
    current_nll = None
    current_bpd = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Step"):
            current_step = int(line.split()[1].replace(':', ''))
        elif line.startswith("NLL"):
            current_nll = float(line.split(':')[1])
        elif line.startswith("BPD"):
            current_bpd = float(line.split(':')[1])
            
            if current_step is not None:
                data.append({"Step": current_step, "NLL": current_nll, "BPD": current_bpd})
                current_step = None
                
    return pd.DataFrame(data)

def show_eval_results(eval_results_file, exp_name):
    """
    Displays evaluation results table and plots BPD curve.
    If results file doesn't exist, it runs 'evaluate.py --ckpt_all' automatically.
    """
    if not os.path.exists(eval_results_file):
        print(f"Evaluation results not found at {eval_results_file}.")
        print(f"Running evaluation for experiment '{exp_name}' (all checkpoints)...")
        try:
            # Run evaluate.py with --ckpt_all to get full history
            subprocess.run(["python", "evaluate.py", "--exp_name", exp_name, "--ckpt_all"], check=True)
            print("Evaluation complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            return

    df_eval = parse_eval_results(eval_results_file)

    if not df_eval.empty:
        # Sort by step
        df_eval = df_eval.sort_values("Step")
        print("Evaluation Results:")
        display(df_eval)
        
        # Plot BPD over steps if multiple checkpoints evaluated
        if len(df_eval) > 1:
            plt.figure(figsize=(8, 5))
            plt.plot(df_eval["Step"], df_eval["BPD"], marker='o', linestyle='-', color='green')
            plt.xlabel("Checkpoint Step")
            plt.ylabel("Bits Per Dimension (BPD)")
            plt.title("BPD over Training")
            plt.grid(True)
            plt.show()
    else:
        print("No evaluation results found even after attempting to run evaluation.")

def show_generated_samples(images_dir):
    """Displays the generated sample image from the latest checkpoint."""
    if os.path.exists(images_dir):
        files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        if files:
            # Find the file with the highest step number
            # Format: step_1000.png
            steps = [int(f.split('_')[1].split('.')[0]) for f in files]
            max_step = max(steps)
            latest_img_path = os.path.join(images_dir, f"step_{max_step}.png")
            
            print(f"Displaying samples from step {max_step}:")
            img = mpimg.imread(latest_img_path)
            plt.figure(figsize=(15, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            print("No image samples found.")
    else:
        print(f"Images directory not found: {images_dir}")
