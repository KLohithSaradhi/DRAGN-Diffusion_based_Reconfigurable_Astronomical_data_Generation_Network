"""Checkpoint-driven, YAML-only base and LoRA comparison inference."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from dragn.config import ExperimentConfig, InferenceConfig
from dragn.images import save_image_grid
from dragn.models import (
    build_dit,
    inject_lora,
    load_adapter_state_dict,
    set_adapter_scale,
)
from dragn.training.common import file_sha256
from dragn.training.train_generative import _generate, _load_autoencoder


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_checkpoint(path: Path, expected_task: str) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{expected_task.capitalize()} checkpoint not found: {path}")
    digest = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 2 or checkpoint.get("task") != expected_task:
        raise ValueError(f"Not a DRAGN v2 {expected_task} checkpoint: {path}")
    return checkpoint, digest


def _base_state(checkpoint: dict, weights: str) -> dict[str, torch.Tensor]:
    if weights == "raw":
        return checkpoint["model_state"]
    if checkpoint.get("ema_state") is None:
        raise ValueError("EMA base weights requested, but the base checkpoint has no EMA state")
    return checkpoint["ema_state"]["model"]


def _validate_sampler(config: InferenceConfig, base_config: ExperimentConfig) -> None:
    assert base_config.objective is not None
    expected = "ancestral" if base_config.objective.type == "ddpm" else "euler"
    if config.sampling.method != expected:
        raise ValueError(
            f"Base objective {base_config.objective.type!r} requires sampling method {expected!r}"
        )
    if (
        base_config.objective.type == "ddpm"
        and config.sampling.steps != base_config.objective.timesteps
    ):
        raise ValueError(
            "Ancestral DDPM inference requires sampling.steps to equal the trained timesteps"
        )


def _runtime_config(
    config: InferenceConfig,
    base_config: ExperimentConfig,
) -> ExperimentConfig:
    autoencoder = base_config.autoencoder.model_copy(
        update={"checkpoint": config.inference.autoencoder_checkpoint}
    )
    return base_config.model_copy(
        update={"autoencoder": autoencoder, "sampling": config.sampling}
    )


def _check_adapter_compatibility(
    inference: InferenceConfig,
    runtime: ExperimentConfig,
    adapter_config: ExperimentConfig,
    checkpoint: dict,
    base_hash: str,
    ae_hash: str,
) -> None:
    if adapter_config.task != "lora" or adapter_config.lora is None:
        raise ValueError("Adapter checkpoint contains an invalid embedded configuration")
    problems = []
    if checkpoint.get("base_checkpoint_hash") != base_hash:
        problems.append("base checkpoint hash")
    if checkpoint.get("ae_checkpoint_hash") != ae_hash:
        problems.append("autoencoder checkpoint hash")
    if adapter_config.model != runtime.model:
        problems.append("DiT architecture")
    if adapter_config.objective != runtime.objective:
        problems.append("objective")
    if adapter_config.lora.base_weights != inference.inference.base_weights:
        problems.append("base weight variant (raw/EMA)")
    if problems:
        raise ValueError("Incompatible LoRA adapter: " + ", ".join(problems))


def run_inference(config: InferenceConfig) -> Path:
    """Generate base/adapter rows with identical stochastic sampler inputs."""
    if config.task != "inference":
        raise ValueError("run_inference requires task: inference")
    _set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.experiment.output_dir / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_checkpoint, base_hash = _read_checkpoint(
        config.inference.base_checkpoint, "base"
    )
    base_config = ExperimentConfig.model_validate(base_checkpoint["config"])
    if base_config.task != "base" or base_config.model is None or base_config.objective is None:
        raise ValueError("Base checkpoint contains an invalid embedded configuration")
    _validate_sampler(config, base_config)
    runtime = _runtime_config(config, base_config)
    autoencoder, ae_hash = _load_autoencoder(runtime, device)
    recorded_ae_hash = base_checkpoint.get("ae_checkpoint_hash")
    if recorded_ae_hash is None:
        recorded_path = base_config.autoencoder.checkpoint
        if recorded_path is None or not recorded_path.is_file():
            raise ValueError(
                "Base checkpoint predates embedded AE hashes and its recorded AE path "
                "cannot be verified"
            )
        recorded_ae_hash = file_sha256(recorded_path)
    if recorded_ae_hash != ae_hash:
        raise ValueError("Autoencoder checkpoint does not match the base DiT checkpoint")

    latent_size = base_config.data.image_size // autoencoder.downsample_factor
    frozen_state = _base_state(base_checkpoint, config.inference.base_weights)

    def fresh_base():
        model = build_dit(
            base_config.model, autoencoder.latent_channels, latent_size
        ).to(device)
        model.load_state_dict(frozen_state, strict=True)
        model.requires_grad_(False).eval()
        return model

    variants: list[tuple[str, torch.Tensor]] = []
    model = fresh_base()
    base_images = _generate(runtime, model, autoencoder, latent_size, device).cpu()
    variants.append(("base", base_images))
    save_image_grid(base_images, output_dir / "00_base.png", config.sampling.num_samples)
    del model

    adapter_metadata = []
    for index, adapter in enumerate(config.inference.adapters, start=1):
        checkpoint, adapter_hash = _read_checkpoint(adapter.checkpoint, "lora")
        adapter_config = ExperimentConfig.model_validate(checkpoint["config"])
        _check_adapter_compatibility(
            config, runtime, adapter_config, checkpoint, base_hash, ae_hash
        )
        assert adapter_config.lora is not None
        model = fresh_base()
        inject_lora(model, adapter_config.lora)
        if adapter.weights == "raw":
            state = checkpoint["adapter_state"]
        else:
            if checkpoint.get("ema_adapter_state") is None:
                raise ValueError(
                    f"Adapter {adapter.name!r} requests EMA weights, but none were saved"
                )
            state = checkpoint["ema_adapter_state"]["adapter"]
        load_adapter_state_dict(model, state)
        set_adapter_scale(model, adapter.scale)
        images = _generate(runtime, model, autoencoder, latent_size, device).cpu()
        variants.append((adapter.name, images))
        filename = f"{index:02d}_{adapter.name}.png"
        save_image_grid(images, output_dir / filename, config.sampling.num_samples)
        adapter_metadata.append({
            "row": index,
            "name": adapter.name,
            "checkpoint": str(adapter.checkpoint),
            "checkpoint_sha256": adapter_hash,
            "weights": adapter.weights,
            "scale": adapter.scale,
            "preset": adapter_config.lora.preset,
            "output": filename,
        })
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison = torch.cat([images for _, images in variants], dim=0)
    comparison_path = output_dir / "comparison.png"
    save_image_grid(comparison, comparison_path, config.sampling.num_samples)
    metadata = {
        "schema_version": 2,
        "task": "inference",
        "row_order": [name for name, _ in variants],
        "base": {
            "row": 0,
            "checkpoint": str(config.inference.base_checkpoint),
            "checkpoint_sha256": base_hash,
            "weights": config.inference.base_weights,
            "output": "00_base.png",
        },
        "autoencoder": {
            "checkpoint": str(config.inference.autoencoder_checkpoint),
            "checkpoint_sha256": ae_hash,
            "latent_scale": float(autoencoder.latent_scale.item()),
        },
        "objective": base_config.objective.model_dump(mode="json"),
        "sampling": config.sampling.model_dump(mode="json"),
        "adapters": adapter_metadata,
        "resolved_config": config.model_dump(mode="json"),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"device={device} variants={len(variants)} identical_noise_seed={config.sampling.seed} "
        f"comparison={comparison_path} metadata={metadata_path}",
        flush=True,
    )
    return comparison_path
