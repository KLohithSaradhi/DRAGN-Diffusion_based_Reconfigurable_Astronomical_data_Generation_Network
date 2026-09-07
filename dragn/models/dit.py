"""Patch-based Diffusion Transformer for DRAGN v2."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from dragn.config import DiTSection


def fixed_2d_sincos_position_embedding(
    height: int,
    width: int,
    hidden_size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if hidden_size % 4:
        raise ValueError("hidden_size must be divisible by four")
    axis_size = hidden_size // 4
    frequencies = torch.arange(axis_size, device=device, dtype=torch.float32)
    frequencies = 1.0 / (10000.0 ** (frequencies / max(axis_size, 1)))
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )

    def encode(coordinates: Tensor) -> Tensor:
        angles = coordinates.reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat((angles.sin(), angles.cos()), dim=1)

    embedding = torch.cat((encode(y), encode(x)), dim=1)
    return embedding.unsqueeze(0).to(dtype=dtype)


class FourierTimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int, maximum_frequency: float = 1000.0):
        super().__init__()
        if hidden_size % 2:
            raise ValueError("hidden_size must be even")
        half = hidden_size // 2
        frequencies = torch.logspace(0, math.log10(maximum_frequency), half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1:
            raise ValueError("time must have shape [batch]")
        if torch.any(time < 0) or torch.any(time > 1):
            raise ValueError("time must be normalized to [0, 1]")
        angles = 2.0 * math.pi * time.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        return self.mlp(embedding)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_size)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.proj(attended)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, ratio: float, dropout: float):
        super().__init__()
        intermediate_size = int(hidden_size * ratio)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate="tanh")))


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio, dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.modulation(condition).chunk(6, dim=1)
        x = x + gate_attn.unsqueeze(1) * self.attention(_modulate(self.norm1(x), shift_attn, scale_attn))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        self.projection = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: int | tuple[int, int],
        patch_size: int = 2,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        input_height, input_width = (input_size, input_size) if isinstance(input_size, int) else input_size
        if input_height % patch_size or input_width % patch_size:
            raise ValueError("Both input dimensions must be divisible by patch_size")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.in_channels = in_channels
        self.input_size = (input_height, input_width)
        self.patch_size = patch_size
        self.grid_size = (input_height // patch_size, input_width // patch_size)
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.register_buffer(
            "position_embedding",
            fixed_2d_sincos_position_embedding(*self.grid_size, hidden_size),
            persistent=True,
        )
        self.time_embedding = FourierTimeEmbedding(hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

    def unpatchify(self, tokens: Tensor) -> Tensor:
        batch, token_count, _ = tokens.shape
        grid_height, grid_width = self.grid_size
        if token_count != grid_height * grid_width:
            raise ValueError("Token count does not match the configured patch grid")
        patch = self.patch_size
        x = tokens.reshape(batch, grid_height, grid_width, patch, patch, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(batch, self.in_channels, grid_height * patch, grid_width * patch)

    def forward(self, x: Tensor, time: Tensor) -> Tensor:
        if x.shape[1] != self.in_channels or x.shape[-2:] != self.input_size:
            raise ValueError(
                f"Expected input [B, {self.in_channels}, {self.input_size[0]}, {self.input_size[1]}], "
                f"received {tuple(x.shape)}"
            )
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        condition = self.time_embedding(time).to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.unpatchify(self.final_layer(tokens, condition))


def build_dit(config: "DiTSection", in_channels: int, input_size: int | tuple[int, int]) -> DiT:
    return DiT(
        in_channels=in_channels,
        input_size=input_size,
        patch_size=config.patch_size,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
    )
