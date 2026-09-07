"""Runnable AutoencoderKL training stage."""

from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from dragn.config import ExperimentConfig
from dragn.data import create_loaders
from dragn.images import save_image_grid
from dragn.models import build_autoencoder
from dragn.training.autoencoder import AutoencoderKLLoss, LatentScaleEstimator


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def train_autoencoder(config: ExperimentConfig) -> Path:
    if config.task != "autoencoder":
        raise ValueError("train_autoencoder requires task: autoencoder")
    _set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.experiment.output_dir / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader = create_loaders(config.data, config.experiment.seed)
    model = build_autoencoder(config.autoencoder, config.data.channels).to(device)
    loss_config = config.autoencoder.loss
    criterion = AutoencoderKLLoss(
        reconstruction=loss_config.reconstruction,
        reconstruction_weight=loss_config.reconstruction_weight,
        kl_weight=loss_config.kl_weight,
        kl_warmup_fraction=loss_config.kl_warmup_fraction,
        perceptual_weight=loss_config.perceptual_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    natural_total = config.training.epochs * len(train_loader)
    total_steps = min(natural_total, config.training.max_steps or natural_total)
    if total_steps == 0:
        raise RuntimeError("Training loader produced no batches")
    print(f"device={device} train_images={len(train_loader.dataset)} val_images={len(validation_loader.dataset)}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,} total_steps={total_steps}")
    global_step = 0
    model.train()
    for _ in range(config.training.epochs):
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, config.training.precision):
                output = model(images, sample_posterior=True)
                losses = criterion(output, images, global_step, total_steps)
            losses.total.backward()
            if config.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            global_step += 1
            if global_step == 1 or global_step % config.training.log_every == 0:
                print(f"step={global_step}/{total_steps} total={losses.total.item():.6f} recon={losses.reconstruction.item():.6f} kl={losses.kl.item():.3f} kl_weight={losses.kl_weight:.2e}")
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    model.eval()
    estimator = LatentScaleEstimator()
    validation_losses: list[float] = []
    fixed_images = None
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(validation_loader):
            images = images.to(device, non_blocking=True)
            posterior = model.encode(images)
            estimator.update(posterior.sample())
            reconstruction = model.decode(posterior.mode())
            validation_losses.append(torch.nn.functional.l1_loss(reconstruction, images).item())
            if fixed_images is None:
                fixed_images = images
            if config.training.validation_batches and batch_index + 1 >= config.training.validation_batches:
                break
    latent_std = estimator.standard_deviation()
    model.set_latent_scale(latent_std)
    validation_l1 = sum(validation_losses) / len(validation_losses)
    print(f"validation_l1={validation_l1:.6f} latent_std={latent_std.item():.6f} latent_scale={model.latent_scale.item():.6f}")
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        "schema_version": 2,
        "task": "autoencoder",
        "step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.model_dump(mode="json"),
        "validation_l1": validation_l1,
    }, checkpoint_path)

    reloaded = build_autoencoder(config.autoencoder, config.data.channels).to(device)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(saved["model_state"])
    reloaded.eval()
    if fixed_images is not None:
        with torch.no_grad():
            scaled = reloaded.encode_for_diffusion(fixed_images, sample_posterior=False)
            reconstructions = reloaded.decode_from_diffusion(scaled)
        save_image_grid(torch.cat((fixed_images, reconstructions), dim=0), output_dir / "reconstructions.png", columns=fixed_images.shape[0])
    print(f"checkpoint={checkpoint_path}")
    print(f"reconstructions={output_dir / 'reconstructions.png'}")
    return checkpoint_path
