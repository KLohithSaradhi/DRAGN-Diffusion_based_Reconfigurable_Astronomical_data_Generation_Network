"""Shared production training infrastructure."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dragn.config import ExperimentConfig


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, target_value in self.model.state_dict().items():
            source_value = source[name].detach()
            if target_value.is_floating_point():
                target_value.lerp_(source_value, 1.0 - self.decay)
            else:
                target_value.copy_(source_value)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if state["decay"] != self.decay:
            raise ValueError("EMA decay in checkpoint does not match the YAML configuration")
        self.model.load_state_dict(state["model"], strict=True)


def experiment_signature(config: ExperimentConfig, ae_checkpoint_hash: str | None = None) -> str:
    payload = {
        "schema_version": config.schema_version,
        "task": config.task,
        "data_geometry": {
            "layout": config.data.layout,
            "image_size": config.data.image_size,
            "channels": config.data.channels,
            "filter": config.data.filter.model_dump(mode="json") if config.data.filter else None,
        },
        "autoencoder": config.autoencoder.model_dump(mode="json", exclude={"checkpoint"}),
        "ae_checkpoint_hash": ae_checkpoint_hash,
        "model": config.model.model_dump(mode="json") if config.model else None,
        "objective": config.objective.model_dump(mode="json") if config.objective else None,
        "lora": config.lora.model_dump(mode="json") if config.lora else None,
        "optimization": {
            "optimizer": config.training.optimizer,
            "lr": config.training.lr,
            "weight_decay": config.training.weight_decay,
            "precision": config.training.precision,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "lr_schedule": config.training.lr_schedule,
            "warmup_epochs": config.training.warmup_epochs,
            "epochs": config.training.epochs,
            "ema_decay": config.training.ema_decay,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def make_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    schedule: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        if schedule == "constant":
            return 1.0
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class CheckpointManager:
    def __init__(self, output_dir: Path, keep: int):
        self.output_dir = output_dir
        self.keep = keep
        output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.output_dir / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.output_dir / "best.pt"

    def save(self, state: dict, epoch: int, is_best: bool) -> Path:
        epoch_path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(state, epoch_path)
        torch.save(state, self.latest_path)
        if is_best:
            torch.save(state, self.best_path)
        checkpoints = sorted(self.output_dir.glob("checkpoint_epoch_*.pt"))
        for old_path in checkpoints[:-self.keep]:
            old_path.unlink()
        return epoch_path

    def save_best(self, state: dict) -> None:
        torch.save(state, self.best_path)

    def load_for_resume(self, mode: str, signature: str, device: torch.device) -> dict | None:
        if mode == "never":
            return None
        if not self.latest_path.is_file():
            if mode == "required":
                raise FileNotFoundError(f"Resume required but checkpoint is missing: {self.latest_path}")
            return None
        state = torch.load(self.latest_path, map_location=device, weights_only=False)
        if state.get("signature") != signature:
            raise ValueError("Checkpoint signature does not match this experiment configuration")
        return state
