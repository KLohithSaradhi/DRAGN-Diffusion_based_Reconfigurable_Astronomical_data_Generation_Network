"""Small, dependency-free LoRA adapters for DRAGN DiT linear layers."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from dragn.config import LoRASection


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank > min(base.in_features, base.out_features):
            raise ValueError(
                f"LoRA rank {rank} exceeds the smaller dimension of "
                f"Linear({base.in_features}, {base.out_features})"
            )
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank
        self.adapter_scale = 1.0
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        residual = torch.nn.functional.linear(self.dropout(inputs), self.lora_a)
        residual = torch.nn.functional.linear(residual, self.lora_b)
        return self.base(inputs) + residual * self.scaling * self.adapter_scale


def inject_lora(model: nn.Module, config: LoRASection) -> list[str]:
    """Freeze a DiT and replace the modules selected by the YAML preset."""
    model.requires_grad_(False)
    replaced: list[str] = []
    for block_index, block in enumerate(model.blocks):
        candidates = {
            "attention.qkv": (block.attention, "qkv"),
            "attention.proj": (block.attention, "proj"),
            "mlp.fc1": (block.mlp, "fc1"),
            "mlp.fc2": (block.mlp, "fc2"),
        }
        for target in config.targets:
            parent, attribute = candidates[target]
            base = getattr(parent, attribute)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected a linear layer at blocks.{block_index}.{target}")
            setattr(
                parent,
                attribute,
                LoRALinear(base, config.rank, config.alpha, config.dropout),
            )
            replaced.append(f"blocks.{block_index}.{target}")
    if not replaced:
        raise RuntimeError("No DiT modules matched the configured LoRA preset")
    return replaced


def adapter_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    for name, parameter in model.named_parameters():
        if name.endswith(".lora_a") or name.endswith(".lora_b"):
            yield parameter


def adapter_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }


def load_adapter_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    parameters = dict(model.named_parameters())
    expected = {
        name for name in parameters
        if name.endswith(".lora_a") or name.endswith(".lora_b")
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(f"LoRA state mismatch; missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value)


def set_adapter_scale(model: nn.Module, scale: float) -> None:
    if scale < 0:
        raise ValueError("LoRA adapter scale must be non-negative")
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.adapter_scale = scale


class AdapterEMA:
    """EMA over adapter tensors only, without duplicating the frozen DiT."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = adapter_state_dict(model)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = adapter_state_dict(model)
        for name, value in self.shadow.items():
            value.lerp_(current[name], 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "adapter": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        if state["decay"] != self.decay:
            raise ValueError("Adapter EMA decay does not match the YAML configuration")
        if set(state["adapter"]) != set(self.shadow):
            raise ValueError("Adapter EMA state does not match the injected LoRA modules")
        self.shadow = {
            name: value.detach().clone() for name, value in state["adapter"].items()
        }

    @contextmanager
    def apply(self, model: nn.Module):
        raw = adapter_state_dict(model)
        load_adapter_state_dict(model, self.shadow)
        try:
            yield model
        finally:
            load_adapter_state_dict(model, raw)
