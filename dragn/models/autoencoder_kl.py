"""Convolution-only KL-regularized autoencoder used by DRAGN v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from dragn.config import AutoencoderSection


def _group_count(channels: int, maximum: int = 32) -> int:
    """Return the largest useful GroupNorm group count that divides channels."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("Every positive channel count is divisible by one")


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        num_res_blocks: int,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        stages: list[nn.Module] = []
        current_channels = base_channels
        for stage_index, multiplier in enumerate(channel_multipliers):
            stage_channels = base_channels * multiplier
            for _ in range(num_res_blocks):
                stages.append(ResBlock(current_channels, stage_channels))
                current_channels = stage_channels
            if stage_index < len(channel_multipliers) - 1:
                stages.append(Downsample(current_channels))
        self.stages = nn.Sequential(*stages)
        self.norm_out = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.conv_out = nn.Conv2d(current_channels, 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        x = self.stages(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        num_res_blocks: int,
    ):
        super().__init__()
        current_channels = base_channels * channel_multipliers[-1]
        self.conv_in = nn.Conv2d(latent_channels, current_channels, kernel_size=3, padding=1)

        stages: list[nn.Module] = []
        reversed_multipliers = list(reversed(channel_multipliers))
        for stage_index, multiplier in enumerate(reversed_multipliers):
            stage_channels = base_channels * multiplier
            for _ in range(num_res_blocks):
                stages.append(ResBlock(current_channels, stage_channels))
                current_channels = stage_channels
            if stage_index < len(reversed_multipliers) - 1:
                stages.append(Upsample(current_channels))
        self.stages = nn.Sequential(*stages)
        self.norm_out = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.conv_out = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.conv_in(z)
        z = self.stages(z)
        return torch.tanh(self.conv_out(F.silu(self.norm_out(z))))


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior represented by mean and log variance."""

    def __init__(self, moments: Tensor):
        self.mean, self.logvar = moments.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise

    def mode(self) -> Tensor:
        return self.mean

    def kl(self) -> Tensor:
        """Return KL(q(z|x) || N(0,I)), summed per sample then batch-averaged."""
        per_element = 0.5 * (self.mean.square() + self.var - 1.0 - self.logvar)
        return per_element.flatten(start_dim=1).sum(dim=1).mean()


@dataclass
class AutoencoderOutput:
    reconstruction: Tensor
    posterior: DiagonalGaussianDistribution
    latents: Tensor


class AutoencoderKL(nn.Module):
    """Residual convolutional VAE with an explicitly persisted latent scale."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        if len(channel_multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least two stages")
        if any(multiplier <= 0 for multiplier in channel_multipliers):
            raise ValueError("channel multipliers must all be positive")

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.downsample_factor = 2 ** (len(channel_multipliers) - 1)
        self.encoder = Encoder(
            in_channels,
            latent_channels,
            base_channels,
            channel_multipliers,
            num_res_blocks,
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = Decoder(
            in_channels,
            latent_channels,
            base_channels,
            channel_multipliers,
            num_res_blocks,
        )
        self.register_buffer("latent_scale", torch.ones((), dtype=torch.float32))

    def encode(self, images: Tensor) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(self.quant_conv(self.encoder(images)))

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(self.post_quant_conv(latents))

    def forward(
        self,
        images: Tensor,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> AutoencoderOutput:
        posterior = self.encode(images)
        latents = posterior.sample(generator) if sample_posterior else posterior.mode()
        return AutoencoderOutput(self.decode(latents), posterior, latents)

    def set_latent_scale(self, latent_std: Tensor | float, epsilon: float = 1e-6) -> None:
        """Persist the reciprocal dataset latent standard deviation in checkpoints."""
        std = torch.as_tensor(latent_std, device=self.latent_scale.device, dtype=torch.float32)
        if std.numel() != 1 or not torch.isfinite(std) or std <= 0:
            raise ValueError("latent_std must be one finite positive scalar")
        self.latent_scale.copy_(std.clamp_min(epsilon).reciprocal())

    def encode_for_diffusion(
        self,
        images: Tensor,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Encode images using the exact scaling convention expected by DiT."""
        posterior = self.encode(images)
        latents = posterior.sample(generator) if sample_posterior else posterior.mode()
        return latents * self.latent_scale.to(dtype=latents.dtype)

    def decode_from_diffusion(self, scaled_latents: Tensor) -> Tensor:
        """Invert latent scaling before decoding; this is the inference entry point."""
        scale = self.latent_scale.to(dtype=scaled_latents.dtype)
        return self.decode(scaled_latents / scale)


def build_autoencoder(config: "AutoencoderSection", in_channels: int) -> AutoencoderKL:
    """Construct AutoencoderKL solely from validated YAML configuration."""
    model = AutoencoderKL(
        in_channels=in_channels,
        latent_channels=config.latent_channels,
        base_channels=config.base_channels,
        channel_multipliers=config.channel_multipliers,
        num_res_blocks=config.num_res_blocks,
    )
    if model.downsample_factor != config.downsample_factor:
        raise ValueError(
            "Configured autoencoder.downsample_factor does not match the architecture implied "
            "by autoencoder.channel_multipliers"
        )
    return model
