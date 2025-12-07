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

def show_eval_results(eval_results_file, exp_name, ckpt_step=None, show_all=False):
    """
    Displays evaluation results table and plots BPD curve.
    
    Args:
        eval_results_file: Path to the results file.
        exp_name: Experiment name.
        ckpt_step: Specific checkpoint step to show/evaluate. If None, defaults to latest if show_all is False.
        show_all: If True, shows/evaluates all checkpoints. Overrides ckpt_step for evaluation scope.
    """
    
    # Check what we have currently
    df_eval = parse_eval_results(eval_results_file)
    
    # Determine what needs to be evaluated
    needs_eval = False
    eval_args = ["python", "evaluate.py", "--exp_name", exp_name]
    
    if show_all:
        # We want all. If file doesn't exist or is empty, we definitely need eval.
        # If file exists, we assume it's up to date? Or should we check if all ckpts are present?
        # For simplicity, if file missing/empty, run all.
        if df_eval.empty:
            needs_eval = True
            eval_args.append("--ckpt_all")
    else:
        # We want specific step or latest
        target_step = ckpt_step
        
        if target_step is None:
            # Find latest checkpoint step from files to know what to look for
            exp_dir = os.path.dirname(eval_results_file)
            ckpt_dir = os.path.join(exp_dir, "checkpoints")
            if os.path.exists(ckpt_dir):
                files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
                if files:
                    target_step = max([int(f.split("_")[1].split(".")[0]) for f in files])
        
        if target_step is not None:
            # Check if this step is in df_eval
            if df_eval.empty or target_step not in df_eval["Step"].values:
                needs_eval = True
                eval_args.extend(["--ckpt_step", str(target_step)])
        else:
            print("No checkpoints found to evaluate.")
            return

    if needs_eval:
        print(f"Running evaluation for experiment '{exp_name}'...")
        try:
            subprocess.run(eval_args, check=True)
            print("Evaluation complete.")
            # Reload results
            df_eval = parse_eval_results(eval_results_file)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            return

    if not df_eval.empty:
        # Sort by step
        df_eval = df_eval.sort_values("Step")
        
        # Filter display
        if not show_all:
            if ckpt_step is not None:
                df_display = df_eval[df_eval["Step"] == ckpt_step]
            else:
                # Show latest
                df_display = df_eval.iloc[[-1]]
        else:
            df_display = df_eval

        print("Evaluation Results:")
        display(df_display)
        
        # Plot BPD only if we have multiple points
        if len(df_eval) > 1 and show_all:
            plt.figure(figsize=(8, 5))
            plt.plot(df_eval["Step"], df_eval["BPD"], marker='o', linestyle='-', color='green')
            plt.xlabel("Checkpoint Step")
            plt.ylabel("Bits Per Dimension (BPD)")
            plt.title("BPD over Training")
            plt.grid(True)
            plt.show()
    else:
        print("No evaluation results found.")

from PIL import Image, ImageDraw, ImageFont
import math

def show_generated_samples(images_dir):
    """Concatenates all generated sample images into a single vertical image with labels."""
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return

    files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    if not files:
        print("No image samples found.")
        return

    # Parse files to get steps and types
    # Structure: List of (step, type, filepath)
    image_list = []

    for f in files:
        step = -1
        type_label = "default"
        
        if f.startswith("1-step_"):
            try:
                step = int(f.split('_')[1].split('.')[0])
                type_label = "1-step"
            except ValueError: pass
        elif f.startswith("5-step_"):
            try:
                step = int(f.split('_')[1].split('.')[0])
                type_label = "5-step"
            except ValueError: pass
        elif f.startswith("step_"):
            parts = f.split('_')
            if len(parts) == 2:
                try:
                    step = int(parts[1].split('.')[0])
                except ValueError: pass
            elif len(parts) >= 3:
                try:
                    step = int(parts[1])
                    type_label = parts[2].split('.')[0]
                except ValueError: pass
        
        if step != -1:
            image_list.append({'step': step, 'type': type_label, 'path': os.path.join(images_dir, f)})

    if not image_list:
        print("Could not parse valid images.")
        return

    # Sort by step, then type
    image_list.sort(key=lambda x: (x['step'], x['type']))

    # Load images and add labels
    loaded_images = []
    max_width = 0
    
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for item in image_list:
        try:
            img = Image.open(item['path'])
            
            # Add label to the left
            label_width = 80 # Adjust as needed
            new_img = Image.new('RGB', (img.width + label_width, img.height), color='white')
            new_img.paste(img, (label_width, 0))
            
            draw = ImageDraw.Draw(new_img)
            if item['type'] == 'default':
                label_text = f"Step: {item['step']}"
            else:
                label_text = f"Step: {item['step']}\nType: {item['type']}"
            
            # Center text vertically in the margin
            # Get text bbox
            bbox = draw.multiline_textbbox((0, 0), label_text, font=font)
            text_height = bbox[3] - bbox[1]
            y_text = (img.height - text_height) // 2
            
            draw.multiline_text((10, y_text), label_text, fill='black', font=font)
            
            loaded_images.append(new_img)
            max_width = max(max_width, new_img.width)
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")

    if not loaded_images:
        return

    # Concatenate vertically
    total_height = sum(img.height for img in loaded_images)
    
    final_image = Image.new('RGB', (max_width, total_height), color='white')
    
    y_offset = 0
    for img in loaded_images:
        final_image.paste(img, (0, y_offset))
        y_offset += img.height
        
    # Display
    # Adjust figsize to show detail. 
    # Assuming standard image is ~300px wide, label adds ~250px. Total ~550px.
    # Height depends on number of images.
    # Display with a fixed width in notebook, scroll for height.
    plt.figure(figsize=(12, len(loaded_images) * 4)) 
    plt.imshow(final_image)
    plt.axis('off')
    plt.show()
