"""Objective-matched samplers for DRAGN v2."""

from __future__ import annotations

import torch
from torch import Tensor

from dragn.models import DiT
from dragn.objectives import DiffusionSchedule


def seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def sample_flow(
    model: DiT,
    shape: tuple[int, int, int, int],
    source_std: float,
    steps: int,
    generator: torch.Generator,
) -> Tensor:
    device = next(model.parameters()).device
    latents = torch.randn(shape, device=device, generator=generator) * source_std
    step_size = 1.0 / steps
    for index in range(steps):
        time = torch.full((shape[0],), index / steps, device=device)
        latents = latents + step_size * model(latents, time)
    return latents


@torch.no_grad()
def sample_ddpm(
    model: DiT,
    shape: tuple[int, int, int, int],
    schedule: DiffusionSchedule,
    generator: torch.Generator,
) -> Tensor:
    device = next(model.parameters()).device
    latents = torch.randn(shape, device=device, generator=generator)
    for index in reversed(range(schedule.timesteps)):
        timestep = torch.full((shape[0],), index, device=device, dtype=torch.long)
        predicted_noise = model(latents, schedule.normalized_time(timestep))
        alpha = schedule.extract(schedule.alphas, timestep, latents.shape)
        alpha_bar = schedule.extract(schedule.alphas_cumprod, timestep, latents.shape)
        beta = schedule.extract(schedule.betas, timestep, latents.shape)
        mean = (latents - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
        if index > 0:
            variance = schedule.extract(schedule.posterior_variance, timestep, latents.shape)
            noise = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=generator)
            latents = mean + variance.clamp_min(1e-20).sqrt() * noise
        else:
            latents = mean
    return latents
