"""End-to-end generative training and validation for DRAGN v2."""

from __future__ import annotations

import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from dragn.config import ExperimentConfig
from dragn.data import create_loaders
from dragn.images import save_image_grid
from dragn.models import AutoencoderKL, DiT, build_autoencoder, build_dit


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _load_autoencoder(config: ExperimentConfig, device: torch.device) -> AutoencoderKL:
    checkpoint_path = config.autoencoder.checkpoint
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")
    autoencoder = build_autoencoder(config.autoencoder, config.data.channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != 2 or checkpoint.get("task") != "autoencoder":
        raise ValueError(f"Not a DRAGN v2 autoencoder checkpoint: {checkpoint_path}")
    autoencoder.load_state_dict(checkpoint["model_state"], strict=True)
    autoencoder.requires_grad_(False)
    autoencoder.eval()
    return autoencoder


def _flow_batch(model: DiT, latents: Tensor, source_std: float) -> tuple[Tensor, Tensor]:
    source = torch.randn_like(latents) * source_std
    time = torch.rand(latents.shape[0], device=latents.device)
    broadcast_time = time.reshape(-1, 1, 1, 1)
    path = (1.0 - broadcast_time) * source + broadcast_time * latents
    target_velocity = latents - source
    return model(path, time), target_velocity


@torch.no_grad()
def _sample_flow(
    model: DiT,
    autoencoder: AutoencoderKL,
    count: int,
    latent_size: int,
    source_std: float,
    steps: int,
    device: torch.device,
) -> Tensor:
    model.eval()
    latents = torch.randn(count, model.in_channels, latent_size, latent_size, device=device) * source_std
    step_size = 1.0 / steps
    for index in range(steps):
        time = torch.full((count,), index / steps, device=device)
        latents = latents + step_size * model(latents, time)
    return autoencoder.decode_from_diffusion(latents)


def train_generative(config: ExperimentConfig) -> Path:
    if config.task != "base" or config.model is None or config.objective is None or config.sampling is None:
        raise ValueError("Generative training requires a complete task: base configuration")
    if config.objective.type != "flow":
        raise NotImplementedError("The authentic Step 3 pipeline currently supports flow; DDPM is Step 4")
    _set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.experiment.output_dir / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = create_loaders(config.data, config.experiment.seed)
    autoencoder = _load_autoencoder(config, device)
    latent_size = config.data.image_size // autoencoder.downsample_factor
    model = build_dit(config.model, autoencoder.latent_channels, latent_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    natural_total = config.training.epochs * len(train_loader)
    total_steps = min(natural_total, config.training.max_steps or natural_total)
    if total_steps == 0:
        raise RuntimeError("Training loader produced no batches")
    print(
        f"device={device} batch_size={config.data.batch_size} "
        f"train_images={len(train_loader.dataset)} val_images={len(validation_loader.dataset)}",
        flush=True,
    )
    print(
        f"image_shape=({config.data.channels},{config.data.image_size},{config.data.image_size}) "
        f"latent_shape=({autoencoder.latent_channels},{latent_size},{latent_size}) "
        f"tokens={(latent_size // config.model.patch_size) ** 2} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}",
        flush=True,
    )

    global_step = 0
    model.train()
    for _ in range(config.training.epochs):
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                latents = autoencoder.encode_for_diffusion(images, sample_posterior=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.training.precision):
                prediction, target = _flow_batch(model, latents, config.objective.source_std)
                loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            if config.training.grad_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            else:
                gradient_norm = torch.nn.utils.get_total_norm(
                    [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                )
            optimizer.step()
            global_step += 1
            if global_step == 1 or global_step % config.training.log_every == 0 or global_step == total_steps:
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
                print(
                    f"step={global_step}/{total_steps} loss={loss.item():.6f} "
                    f"grad_norm={float(gradient_norm):.6f} gpu_mb={allocated:.1f} peak_gpu_mb={peak:.1f}",
                    flush=True,
                )
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    model.eval()
    validation_losses: list[float] = []
    fixed_images = None
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(validation_loader):
            images = images.to(device, non_blocking=True)
            latents = autoencoder.encode_for_diffusion(images, sample_posterior=False)
            prediction, target = _flow_batch(model, latents, config.objective.source_std)
            validation_losses.append(torch.nn.functional.mse_loss(prediction, target).item())
            if fixed_images is None:
                fixed_images = images[:config.sampling.num_samples]
            if config.training.validation_batches and batch_index + 1 >= config.training.validation_batches:
                break
    validation_loss = sum(validation_losses) / len(validation_losses)
    print(f"validation_flow_loss={validation_loss:.6f}", flush=True)

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        "schema_version": 2,
        "task": "base",
        "objective": config.objective.model_dump(mode="json"),
        "step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.model_dump(mode="json"),
        "validation_loss": validation_loss,
    }, checkpoint_path)
    reloaded = build_dit(config.model, autoencoder.latent_channels, latent_size).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state"])
    reloaded.eval()
    generated = _sample_flow(
        reloaded,
        autoencoder,
        config.sampling.num_samples,
        latent_size,
        config.objective.source_std,
        config.sampling.steps,
        device,
    )
    save_image_grid(generated, output_dir / "generated.png", columns=max(1, round(math.sqrt(len(generated)))))
    if fixed_images is not None:
        with torch.no_grad():
            reconstruction = autoencoder.decode_from_diffusion(
                autoencoder.encode_for_diffusion(fixed_images, sample_posterior=False)
            )
        save_image_grid(
            torch.cat((fixed_images, reconstruction), dim=0),
            output_dir / "validation_reconstructions.png",
            columns=len(fixed_images),
        )
    print("checkpoint_reload=successful", flush=True)
    print(f"checkpoint={checkpoint_path}", flush=True)
    print(f"generated={output_dir / 'generated.png'}", flush=True)
    print(f"validation_reconstructions={output_dir / 'validation_reconstructions.png'}", flush=True)
    return checkpoint_path
