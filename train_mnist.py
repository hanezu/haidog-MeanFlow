import argparse
from models.dit import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import time
import os
from datetime import datetime
import pytz


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MeanFlow on MNIST")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name for logging")
    parser.add_argument("--n_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--stage0_t_method", type=str, default="lognorm", help="Method for sampling t")
    parser.add_argument("--stage0_t_mu", type=float, default=-0.4, help="Mu for t lognorm")
    parser.add_argument("--stage0_t_sigma", type=float, default=1.0, help="Sigma for t lognorm")
    parser.add_argument("--stage0_r_method", type=str, default="lognorm", help="Method for sampling r")
    parser.add_argument("--stage0_r_lognorm_mu", type=float, default=-0.4, help="Mu for r lognorm")
    parser.add_argument("--stage0_r_lognorm_sigma", type=float, default=1.0, help="Sigma for r lognorm")
    parser.add_argument("--stage0_instant_prob", type=float, default=0.50, help="Probability of instant flow (flow_ratio)")
    parser.add_argument("--stage0_resample", action="store_true", help="Enable resampling for ordering")
    # CFG and Sampling arguments
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="Classifier-Free Guidance scale (1.0 for no guidance)")
    parser.add_argument("--cfg_ratio", type=float, default=0.10, help="Probability of dropping labels for CFG training")
    parser.add_argument("--sample_steps", type=int, default=5, help="Number of steps for sampling generation")
    
    args = parser.parse_args()

    # Hyperparameters for MNIST training
    n_steps = args.n_steps
    # device = "cuda" if torch.cuda.is_available() else "cpu" # Handled by Accelerator
    batch_size = args.batch_size # batch_size=32 fits in a T4: ~14GB GPU memory
    image_size = 32
    
    # Directories setup
    exp_dir = os.path.join("results", args.exp_name)
    images_dir = os.path.join(exp_dir, "images")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_file = os.path.join(exp_dir, "log.txt")
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Experiment Directory: {exp_dir}")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')

    # MNIST Dataset
    dataset = torchvision.datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            # T.Normalize((0.5,), (0.5,)) # MeanFlow seems to use 'minmax' normalization internally by default (x * 2 - 1), so just ToTensor [0,1] is likely expected input if normalizer is default.
        ]),
    )

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    train_dataloader = cycle(train_dataloader)

    # Model setup for MNIST (1 channel)
    model = MFDiT(
        input_size=image_size,
        patch_size=2,
        in_channels=1, # MNIST is grayscale
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    # Configure distributions from args
    t_dist = [args.stage0_t_method]
    if args.stage0_t_method == 'lognorm':
        t_dist.extend([args.stage0_t_mu, args.stage0_t_sigma])
        
    r_dist = [args.stage0_r_method]
    if args.stage0_r_method == 'lognorm':
        r_dist.extend([args.stage0_r_lognorm_mu, args.stage0_r_lognorm_sigma])

    if accelerator.is_main_process:
        print(f"MeanFlow Config: T={t_dist}, R={r_dist}, FlowRatio={args.stage0_instant_prob}, Resample={args.stage0_resample}")
        print(f"CFG Config: Scale={args.cfg_scale}, Ratio={args.cfg_ratio}, SampleSteps={args.sample_steps}")

    # MeanFlow setup
    meanflow = MeanFlow(
        channels=1, # MNIST is grayscale
        image_size=image_size,
        num_classes=10,
        flow_ratio=args.stage0_instant_prob,
        t_dist=t_dist,
        r_dist=r_dist,
        resample=args.stage0_resample,
        cfg_ratio=args.cfg_ratio,
        cfg_scale=args.cfg_scale,
        # experimental
        cfg_uncond='u')

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    losses = 0.0
    mse_losses = 0.0

    log_step = 500
    sample_step = 1000

    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description(f"Training {args.exp_name}")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            x = data[0].to(accelerator.device)
            c = data[1].to(accelerator.device)

            loss, mse_val = meanflow.loss(model, x, c)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            mse_losses += mse_val.item()

            if accelerator.is_main_process:
                if global_step % log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    batch_info = f'Global Step: {global_step}'
                    loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'

                    # Extract the learning rate from the optimizer
                    lr = optimizer.param_groups[0]['lr']
                    lr_info = f'Learning Rate: {lr:.6f}'

                    log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'
                    print(f"\n{batch_info} {loss_info}") # Print to console as well

                    with open(log_file, mode='a') as n:
                        n.write(log_message)

                    losses = 0.0
                    mse_losses = 0.0

            if global_step % sample_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, 'module') else model
                    # Sample digits 0-9
                    z = meanflow.sample_each_class(model_module, 1, classes=list(range(10)), sample_steps=args.sample_steps) 
                    log_img = make_grid(z, nrow=10)
                    img_save_path = os.path.join(images_dir, f"step_{global_step}.png")
                    save_image(log_img, img_save_path)

                    # Save checkpoint
                    ckpt_path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                    accelerator.save(model_module.state_dict(), ckpt_path)
                accelerator.wait_for_everyone()
                model.train()
                
