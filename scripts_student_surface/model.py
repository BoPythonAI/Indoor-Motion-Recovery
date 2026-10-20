"""Same BC/temporal diffusion architectures as student_ema, new target contract."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, width, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width); self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.film = nn.Linear(cond_dim, width * 2)

    def forward(self, x, c):
        scale, bias = self.film(c).chunk(2, -1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1 + scale[:, :, None]) + bias[:, :, None]
        return x + self.conv2(F.silu(self.norm2(h)))


class ChunkPolicy(nn.Module):
    def __init__(self, obs_dim, kind='diffusion', horizon=6):
        super().__init__()
        if kind not in ('bc', 'diffusion') or horizon < 1:
            raise ValueError('Expected bc or diffusion and a positive chunk horizon')
        self.kind, self.horizon, self.obs_dim = kind, horizon, obs_dim
        self.register_buffer('action_mean', torch.zeros(69))
        self.register_buffer('action_std', torch.ones(69))
        self.condition = nn.Sequential(nn.Linear(obs_dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        if kind == 'bc':
            self.head = nn.Linear(256, horizon * 69)
        else:
            self.time_mlp = nn.Sequential(nn.Linear(64, 256), nn.SiLU(), nn.Linear(256, 256))
            self.input = nn.Conv1d(69, 128, 1)
            self.blocks = nn.ModuleList([TemporalBlock(128, 256) for _ in range(4)])
            self.output = nn.Conv1d(128, 69, 1)
            grid = torch.linspace(0, 1, 101)
            abar = torch.cos((grid + .008) / 1.008 * math.pi / 2).square()
            abar = abar / abar[0]
            beta = (1 - abar[1:] / abar[:-1]).clamp(.0001, .999)
            self.register_buffer('abar', torch.cumprod(1 - beta, 0))

    def noise_pred(self, noisy, t, c):
        freqs = torch.exp(torch.arange(32, device=t.device) * (-math.log(10000) / 31))
        phase = t[:, None] * freqs[None]
        condition = c + self.time_mlp(torch.cat([phase.sin(), phase.cos()], -1))
        h = self.input(noisy.transpose(1, 2))
        for block in self.blocks:
            h = block(h, condition)
        return self.output(h).transpose(1, 2)

    def loss(self, obs, target, mask):
        c = self.condition(obs)
        y = (target - self.action_mean) / self.action_std
        if self.kind == 'bc':
            pred = self.head(c).reshape(-1, self.horizon, 69)
            expected = y
        else:
            t = torch.randint(len(self.abar), (len(obs),), device=obs.device)
            alpha = self.abar[t, None, None]
            noise = torch.randn_like(y)
            noisy = alpha.sqrt() * y + (1 - alpha).sqrt() * noise
            expected = alpha.sqrt() * noise - (1 - alpha).sqrt() * y
            pred = self.noise_pred(noisy, t, c)
        return ((pred - expected).square() * mask[:, :, None]).sum() / (mask.sum() * 69).clamp_min(1)

    @torch.no_grad()
    def sample(self, obs, steps=8, generator=None, initial_noise=None):
        c = self.condition(obs)
        if self.kind == 'bc':
            out = self.head(c).reshape(-1, self.horizon, 69)
        else:
            if not 1 <= steps <= len(self.abar):
                raise ValueError('Inference steps must be within the diffusion training schedule')
            shape = (len(obs), self.horizon, 69)
            if initial_noise is not None and tuple(initial_noise.shape) != shape:
                raise ValueError('Initial noise must match current batch and action chunk')
            out = torch.randn(shape, device=obs.device, generator=generator) if initial_noise is None else initial_noise.clone()
            indices = torch.linspace(len(self.abar) - 1, 0, steps, device=obs.device).long()
            for i, t in enumerate(indices):
                alpha = self.abar[t]
                velocity = self.noise_pred(out, t.expand(len(obs)), c)
                clean = (alpha.sqrt() * out - (1 - alpha).sqrt() * velocity).clamp(-8, 8)
                noise = (1 - alpha).sqrt() * out + alpha.sqrt() * velocity
                previous = self.abar[indices[i + 1]] if i + 1 < len(indices) else torch.ones((), device=obs.device)
                out = previous.sqrt() * clean + (1 - previous).sqrt() * noise
        return out * self.action_std + self.action_mean


@torch.no_grad()
def update_ema(ema, online, step, max_decay=.9999, power=.75):
    decay = min(max_decay, 1 - (1 + step) ** (-power))
    for averaged, current in zip(ema.parameters(), online.parameters()):
        averaged.lerp_(current, 1 - decay)
    for averaged, current in zip(ema.buffers(), online.buffers()):
        averaged.copy_(current)
    return decay
