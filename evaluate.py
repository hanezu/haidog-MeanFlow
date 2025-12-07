import argparse
import os
import torch
import torchvision
import torchvision.transforms as T
from models.dit import MFDiT
from torchdiffeq import odeint
from tqdm import tqdm
import math

def div_fn(u, x):
    """
    Compute divergence of u with respect to x using Hutchinson's Trace Estimator.
    Approximate Tr(J) = E[epsilon^T * J * epsilon]
    """
    # epsilon ~ Rademacher distribution (random +1 or -1)
    epsilon = torch.randint(0, 2, x.shape, device=x.device).float() * 2 - 1
    
    # Compute vector-Jacobian product: epsilon^T * J
    # We calculate grad( (u * epsilon).sum(), x )
    vjp = torch.autograd.grad(u, x, grad_outputs=epsilon, create_graph=False)[0]
    
    # Compute trace approximation: epsilon^T * vjp
    # Sum over all dimensions except batch
    div = (vjp * epsilon).sum(dim=[1, 2, 3])
    
    return div

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt_step", type=int, default=None, help="Step number of checkpoint. If None, uses latest.")
    parser.add_argument("--ckpt_all", action="store_true", help="Evaluate all checkpoints found in the directory.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for evaluation (100 fits on a 16GB T4)")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches to evaluate (None for all)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    image_size = 32
    D = image_size * image_size * 1  # Dimensions
    
    # Paths
    exp_dir = os.path.join("results", args.exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    
    # Determine which steps to evaluate
    if not os.path.exists(ckpt_dir):
         raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        
    all_steps = sorted([int(f.split("_")[1].split(".")[0]) for f in files])
    
    if args.ckpt_all:
        steps_to_eval = all_steps
        print(f"Evaluating all checkpoints: {steps_to_eval}")
    elif args.ckpt_step is not None:
        if args.ckpt_step not in all_steps:
             print(f"Warning: Checkpoint step {args.ckpt_step} not found in directory. Available: {all_steps}")
        steps_to_eval = [args.ckpt_step]
    else:
        # Latest only
        steps_to_eval = [max(all_steps)]
        print(f"Evaluating latest checkpoint: {steps_to_eval[0]}")

    # Load Model Structure (once)
    model = MFDiT(
        input_size=image_size,
        patch_size=2,
        in_channels=1,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
    ).to(device)

    # Data (once)
    dataset = torchvision.datasets.MNIST(
        root="mnist",
        train=False, # Test set
        download=True,
        transform=T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluation Loop over checkpoints
    for step in steps_to_eval:
        ckpt_path = os.path.join(ckpt_dir, f"step_{step}.pt")
        print(f"\n--- Processing Checkpoint: Step {step} ---")
        
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Failed to load checkpoint {ckpt_path}: {e}")
            continue
            
        total_nll = 0.0
        total_bpd = 0.0
        count = 0
        
        print(f"Evaluating Approximate NLL (Hutchinson) on {args.limit_batches if args.limit_batches else 'all'} batches...")
        
        # Prior: Standard Normal
        def log_normal_standard(z):
            # z shape: [B, C, H, W]
            # D is passed or inferred
            return -0.5 * (z ** 2).sum(dim=[1,2,3]) - 0.5 * D * math.log(2 * math.pi)

        with torch.no_grad():
            total_batches = args.limit_batches if args.limit_batches is not None else len(dataloader)
            for i, (x, y) in enumerate(tqdm(dataloader, total=total_batches)):
                if args.limit_batches is not None and i >= args.limit_batches:
                    break
                    
                x = x.to(device)
                c_batch = y.to(device)
                
                # 1. Dequantization
                # x is in [0, 1], effectively discrete levels 0/255, 1/255...
                x_disc = (x * 255.0).floor()
                u = torch.rand_like(x_disc)
                x_deq = (x_disc + u) / 256.0 # in [0, 1)

                # 2. Normalization to [-1, 1] (match training)
                x_norm = x_deq * 2 - 1
                
                def ode_func(t, state):
                    x_t = state[0]
                    with torch.set_grad_enabled(True):
                        x_t.requires_grad_(True)
                        t_tensor = t.repeat(x_t.shape[0])
                        r_tensor = t_tensor.clone()
                        
                        # Model predicts vector field v(x_t)
                        v = model(x_t, t_tensor, r_tensor, y=c_batch)
                        
                        # Hutchinson Estimator Divergence
                        div = div_fn(v, x_t)
                            
                        return v, div

                # Integrate 0 (Data) -> 1 (Noise)
                log_det_0 = torch.zeros(x.shape[0], device=device)
                
                try:
                    # Using rk4 with step_size=0.05
                    options = {'step_size': 0.05}
                    out = odeint(ode_func, (x_norm, log_det_0), torch.tensor([0.0, 1.0], device=device), method='rk4', options=options)
                except Exception as e:
                    print(f"Integration failed: {e}")
                    continue
                    
                z_1 = out[0][-1] # Shape (B, C, H, W)
                delta_log_det = out[1][-1] # Shape (B,)
                
                # log p(z_1)
                log_pz = log_normal_standard(z_1)
                
                # log p(x_norm) = log p(z_1) + log |det dF/dx|
                # Note: delta_log_det accumulates div(v). For CNF: log p(x) = log p(z) - integral(div). 
                # But torchdiffeq integrates forward 0->1.
                # if dz/dt = v(z,t), then d(log p)/dt = -div(v).
                # log p(z1) - log p(z0) = integral_0^1 -div(v) dt
                # log p(z0) = log p(z1) + integral_0^1 div(v) dt
                # out[1] is integral of whatever correct quantity.
                # The code assumes delta_log_det is correct.
                
                log_px_norm = log_pz + delta_log_det # continuous density on [-1, 1]
                
                # Change of variables: x_norm = 2 * x_deq - 1
                # log p(x_deq) = log p(x_norm) + log |d(x_norm)/d(x_deq)|
                # |d(x_norm)/d(x_deq)| = 2^D
                log_px_deq = log_px_norm + D * math.log(2.0)
                
                # Discrete correction
                # log P_disc = log p(x_deq) - D * log(256)
                deq_const = D * math.log(256.0)
                log_px_disc = log_px_deq - deq_const
                
                nll = -log_px_disc.mean().item()
                bpd = nll / (D * math.log(2.0))
                
                total_nll += nll
                total_bpd += bpd
                count += 1
            
        if count > 0:
            avg_nll = total_nll / count
            avg_bpd = total_bpd / count
            print(f"Results for {args.exp_name} (Step {step}):")
            print(f"Average NLL: {avg_nll:.4f}")
            print(f"Average BPD: {avg_bpd:.4f}")
            
            # Save results
            res_file = os.path.join(exp_dir, "evaluation_results.txt")
            with open(res_file, "a") as f:
                f.write(f"Step {step}:\n")
                f.write(f"  NLL: {avg_nll:.4f}\n")
                f.write(f"  BPD: {avg_bpd:.4f}\n\n")
        else:
            print(f"No batches processed for step {step}.")
        
        torch.cuda.empty_cache()
