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
    vjp = torch.autograd.grad(u, x, grad_outputs=epsilon, create_graph=True)[0]
    
    # Compute trace approximation: epsilon^T * vjp
    # Sum over all dimensions except batch
    div = (vjp * epsilon).sum(dim=[1, 2, 3])
    
    return div


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt_step", type=int, default=None, help="Step number of checkpoint. If None, uses latest.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--limit_batches", type=int, default=10, help="Limit number of batches to evaluate")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 32
    
    # Paths
    exp_dir = os.path.join("results", args.exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    
    if args.ckpt_step is None:
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if not files:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        steps = [int(f.split("_")[1].split(".")[0]) for f in files]
        latest_step = max(steps)
        ckpt_path = os.path.join(ckpt_dir, f"step_{latest_step}.pt")
        print(f"Using latest checkpoint: {ckpt_path}")
    else:
        ckpt_path = os.path.join(ckpt_dir, f"step_{args.ckpt_step}.pt")
        
    # Load Model
    model = MFDiT(
        input_size=image_size,
        patch_size=2,
        in_channels=1,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Data
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
    
    # Evaluation Loop
    total_nll = 0.0
    total_bpd = 0.0
    count = 0
    
    print(f"Evaluating Approximate NLL (Hutchinson) on {args.limit_batches} batches...")
    
    # Prior: Standard Normal
    def log_normal_standard(z):
        D = z.view(z.shape[0], -1).shape[1]
        return -0.5 * (z ** 2).sum(dim=[1,2,3]) - 0.5 * D * math.log(2 * math.pi)

    for i, (x, y) in enumerate(tqdm(dataloader, total=args.limit_batches)):
        if i >= args.limit_batches:
            break
            
        x = x.to(device)
        c_batch = y.to(device)
        
        def ode_func(t, state):
            x_t = state[0]
            with torch.set_grad_enabled(True):
                x_t.requires_grad_(True)
                t_tensor = t.repeat(x_t.shape[0])
                r_tensor = t_tensor.clone()
                
                v = model(x_t, t_tensor, r_tensor, y=c_batch)
                
                # Hutchinson Estimator Divergence
                div = div_fn(v, x_t)
                    
                return v, div

        # Integrate 0 (Data) -> 1 (Noise)
        log_det_0 = torch.zeros(x.shape[0], device=device)
        
        try:
            # Using rk4 with step_size=0.05
            options = {'step_size': 0.05}
            out = odeint(ode_func, (x, log_det_0), torch.tensor([0.0, 1.0], device=device), method='rk4', options=options)
        except Exception as e:
            print(f"Integration failed: {e}")
            continue
            
        z_1 = out[0][-1] # Shape (B, C, H, W)
        delta_log_det = out[1][-1] # Shape (B,)
        
        # log p(x) = log p(z_1) + log |det dF/dx|
        log_pz = log_normal_standard(z_1)
        log_px = log_pz + delta_log_det
        
        nll = -log_px.mean().item()
        
        # BPD = NLL / (D * ln(2))
        D = 32*32*1 
        bpd = (nll / (D * math.log(2.0)))
        
        total_nll += nll
        total_bpd += bpd
        count += 1
        
    if count > 0:
        avg_nll = total_nll / count
        avg_bpd = total_bpd / count
        print(f"\nResults for {args.exp_name} (Step {latest_step if args.ckpt_step is None else args.ckpt_step}):")
        print(f"Average NLL: {avg_nll:.4f}")
        print(f"Average BPD: {avg_bpd:.4f}")
        
        # Save results
        res_file = os.path.join(exp_dir, "evaluation_results.txt")
        with open(res_file, "a") as f:
            f.write(f"Step {latest_step if args.ckpt_step is None else args.ckpt_step}:\n")
            f.write(f"  NLL: {avg_nll:.4f}\n")
            f.write(f"  BPD: {avg_bpd:.4f}\n\n")
            
    else:
        print("No batches processed.")