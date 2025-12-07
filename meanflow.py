import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from typing import List
import numpy as np


class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


class SamplerScheduler:
    def __init__(self, total_iterations: int, phase_configs: dict):
        self.total_iterations = max(int(total_iterations), 1)

        # Register initial phases
        self.phases = []
        for name, cfg in phase_configs.items():
            self.register_phase(name, cfg)

    def register_phase(self, name: str, phase_config: dict):
        """
        Register a phase.

        phase_config:
          - "interval": [start, end]
          - "t": {"method": str, "config": dict}
          - "r": {"method": str, "config": dict}
          - "instant_prob": float (optional): Probability to force r = t
          - "resample": bool (optional): If instant_prob specified, resample will determine wherther use a standard lognorm to resample the instant t and r
        """ 
        start, end = phase_config["interval"]

        assert 0.0 <= start < end <= 1.0, "Invalid interval"
        assert all(end <= phase['start'] or phase['end'] <= start for phase in self.phases), "Interval overlapped"

        self.phases.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "t_sampler": getattr(self, f"_construct_{phase_config['t']['method']}")(**phase_config['t']['config']),
                "r_sampler": getattr(self, f"_construct_{phase_config['r']['method']}")(**phase_config['r']['config']),
                "instant_prob": phase_config.get("instant_prob", 0.0),
                "resample": phase_config.get("resample", False),
            }
        )

        self.phases.sort(key=lambda s: s["start"])

    def _current_stage(self, iteration: int) -> dict:
        """
        Get current stage.
        """
        p = max(min(iteration, self.total_iterations - 1), 0) / (self.total_iterations - 1)
        for s in self.phases:
            if s["start"] <= p < s["end"]:
                return s
        return self.phases[-1]

    def sample(self, batch_size: int, iteration: int, device: str) -> List[float]:
        """
        Sample (r, t) from the corresponding stage according to iteration.
        """
        self.device = device
        stage = self._current_stage(iteration)

        t = stage["t_sampler"](batch_size)
        r = stage["r_sampler"](batch_size)

        return self._postprocess(r, t, stage)

    def _postprocess(self, r: torch.Tensor, t: torch.Tensor, stage: dict) -> List[float]:
        """
        Guarantee r <= t; Optional: Guarantee |t - r| >= min_delta or partially set r == t.
        """
        # Apply strict equality for a percentage of the batch
        if instant_prob := stage["instant_prob"]:
            n = int(t.shape[0] * instant_prob)
            instant_mask = torch.randperm(t.shape[0], device=self.device)[:n]
            if stage["resample"]:
                r[instant_mask] = t[instant_mask] = self._construct_lognorm(mu=-0.4, sigma=1)(n)
            else:
                r[instant_mask] = t[instant_mask]

        # Guarantee r <= t
        if (swap_mask := r > t).any():
            r[swap_mask], t[swap_mask] = t[swap_mask], r[swap_mask]

        return r, t

    def _construct_uniform(self, **kwargs) -> callable:
        """
        Uniform distribution.
        """
        return lambda x: torch.rand(x, device=self.device)
    
    def _construct_lognorm(self, **kwargs) -> callable:
        """
        Lognorm distribution.
        """
        return lambda x: torch.sigmoid(torch.randn(x, device=self.device) * kwargs['sigma'] + kwargs['mu'])


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # sampling configuration
        total_iterations=None,
        phase_configs=None,
        # t_dist=['lognorm', -0.4, 1.0],
        # r_dist=['lognorm', -0.4, 1.0],
        resample=False,
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=2.0,
        # experimental
        cfg_uncond='v',
        jvp_api='autograd',
    ):
        super().__init__()
        if total_iterations is None or phase_configs is None:
            raise ValueError("total_iterations and phase_configs must be provided via sampler config files.")
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None

        self.normer = Normalizer.from_list(normalizer)

        self.sampler = SamplerScheduler(total_iterations, phase_configs)
        self.flow_ratio = flow_ratio
        # self.t_dist = t_dist
        # self.r_dist = r_dist
        self.resample = resample
        
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale

        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    def _sample_val(self, dist, batch_size):
        if dist[0] == 'uniform':
             return np.random.rand(batch_size).astype(np.float32)
        elif dist[0] == 'lognorm':
             mu, sigma = dist[1], dist[2]
             normal_samples = np.random.randn(batch_size).astype(np.float32) * sigma + mu
             return 1 / (1 + np.exp(-normal_samples))
        else:
            raise ValueError(f"Unknown distribution: {dist[0]}")

    def sample_t_r(self, batch_size, device):
        raise NotImplementedError("Sampling t/r directly is removed; provide sampler settings via config files.")

    def loss(self, model, iteration, x, c=None):
        batch_size = x.shape[0]
        device = x.device

        # t, r = self.sample_t_r(batch_size, device)
        r, t = self.sampler.sample(batch_size, iteration, device)

        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x)
        x = self.normer.norm(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(z, t, t, uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    # offical JAX repo uses original v for unconditional items
                    cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                    v_hat = torch.where(cfg_mask, v, v_hat)
            else:
                v_hat = v

        # forward pass
        # u = model(z, t, r, y=c)
        model_partial = partial(model, y=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=5, device='cuda'):
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        z = torch.randn(c.shape[0], self.channels,
                        self.image_size, self.image_size,
                        device=device)

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        # print(t_vals)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            v = model(z, t, r, c)
            z = z - (t_-r_) * v

        z = self.normer.unnorm(z)
        return z
