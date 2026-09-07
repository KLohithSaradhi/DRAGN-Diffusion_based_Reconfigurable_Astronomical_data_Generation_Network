"""Training objectives for flow matching and DDPM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from dragn.models import DiT


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_previous: Tensor
    posterior_variance: Tensor

    @classmethod
    def create(
        cls,
        timesteps: int,
        schedule: str,
        device: torch.device,
    ) -> "DiffusionSchedule":
        if schedule == "linear":
            scale = 1000.0 / timesteps
            betas = torch.linspace(scale * 1e-4, scale * 2e-2, timesteps, device=device)
            betas = betas.clamp(max=0.999)
        elif schedule == "cosine":
            offset = 0.008
            points = torch.linspace(0, timesteps, timesteps + 1, device=device)
            cumulative = torch.cos(((points / timesteps + offset) / (1 + offset)) * math.pi / 2).square()
            cumulative = cumulative / cumulative[0]
            betas = (1.0 - cumulative[1:] / cumulative[:-1]).clamp(1e-4, 0.999)
        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule}")
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = torch.cat((torch.ones(1, device=device), cumulative[:-1]))
        posterior_variance = betas * (1.0 - previous) / (1.0 - cumulative)
        return cls(betas, alphas, cumulative, previous, posterior_variance)

    @property
    def timesteps(self) -> int:
        return self.betas.numel()

    def extract(self, values: Tensor, timestep: Tensor, shape: torch.Size) -> Tensor:
        selected = values.gather(0, timestep)
        return selected.reshape(timestep.shape[0], *((1,) * (len(shape) - 1)))

    def normalized_time(self, timestep: Tensor) -> Tensor:
        denominator = max(self.timesteps - 1, 1)
        return timestep.float() / denominator


def flow_prediction_and_target(
    model: DiT,
    latents: Tensor,
    source_std: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    source = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    source = source * source_std
    time = torch.rand(latents.shape[0], device=latents.device, generator=generator)
    broadcast_time = time.reshape(-1, 1, 1, 1)
    path = (1.0 - broadcast_time) * source + broadcast_time * latents
    target_velocity = latents - source
    return model(path, time), target_velocity


def ddpm_prediction_and_target(
    model: DiT,
    latents: Tensor,
    schedule: DiffusionSchedule,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    timestep = torch.randint(
        0,
        schedule.timesteps,
        (latents.shape[0],),
        device=latents.device,
        generator=generator,
    )
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    alpha_bar = schedule.extract(schedule.alphas_cumprod, timestep, latents.shape)
    noisy = alpha_bar.sqrt() * latents + (1.0 - alpha_bar).sqrt() * noise
    return model(noisy, schedule.normalized_time(timestep)), noise
