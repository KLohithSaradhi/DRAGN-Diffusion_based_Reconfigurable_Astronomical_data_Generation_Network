"""Loss and latent-statistics utilities for AutoencoderKL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from dragn.models.autoencoder_kl import AutoencoderOutput


@dataclass
class AutoencoderLossOutput:
    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    kl_weight: float


class AutoencoderKLLoss(nn.Module):
    def __init__(
        self,
        reconstruction: str = "l1",
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1e-6,
        kl_warmup_fraction: float = 0.1,
        perceptual_weight: float = 0.0,
    ):
        super().__init__()
        if reconstruction not in {"l1", "mse"}:
            raise ValueError("reconstruction must be 'l1' or 'mse'")
        if reconstruction_weight <= 0 or kl_weight < 0:
            raise ValueError("loss weights must be non-negative and reconstruction_weight positive")
        if not 0.0 <= kl_warmup_fraction <= 1.0:
            raise ValueError("kl_warmup_fraction must be between zero and one")
        if perceptual_weight != 0.0:
            raise NotImplementedError("Perceptual loss is reserved but intentionally disabled in Step 2")
        self.reconstruction = reconstruction
        self.reconstruction_weight = reconstruction_weight
        self.maximum_kl_weight = kl_weight
        self.kl_warmup_fraction = kl_warmup_fraction

    def current_kl_weight(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.kl_warmup_fraction == 0.0:
            return self.maximum_kl_weight
        warmup_steps = max(1, round(total_steps * self.kl_warmup_fraction))
        return self.maximum_kl_weight * min(max(step, 0) / warmup_steps, 1.0)

    def forward(
        self,
        output: AutoencoderOutput,
        target: Tensor,
        step: int,
        total_steps: int,
    ) -> AutoencoderLossOutput:
        if self.reconstruction == "l1":
            reconstruction_loss = F.l1_loss(output.reconstruction, target)
        else:
            reconstruction_loss = F.mse_loss(output.reconstruction, target)
        kl_loss = output.posterior.kl()
        current_kl_weight = self.current_kl_weight(step, total_steps)
        total = self.reconstruction_weight * reconstruction_loss + current_kl_weight * kl_loss
        return AutoencoderLossOutput(total, reconstruction_loss, kl_loss, current_kl_weight)


class LatentScaleEstimator:
    """Streaming scalar standard-deviation estimator for sampled dataset latents."""

    def __init__(self):
        self.count = 0
        self.total = torch.zeros((), dtype=torch.float64)
        self.square_total = torch.zeros((), dtype=torch.float64)

    @torch.no_grad()
    def update(self, latents: Tensor) -> None:
        values = latents.detach().to(device="cpu", dtype=torch.float64)
        self.count += values.numel()
        self.total += values.sum()
        self.square_total += values.square().sum()

    def standard_deviation(self) -> Tensor:
        if self.count == 0:
            raise RuntimeError("Cannot estimate latent scale before observing latents")
        mean = self.total / self.count
        variance = self.square_total / self.count - mean.square()
        return variance.clamp_min(0).sqrt().to(dtype=torch.float32)
