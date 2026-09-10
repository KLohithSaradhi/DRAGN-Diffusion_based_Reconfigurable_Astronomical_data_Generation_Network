"""Production single-GPU AutoencoderKL trainer."""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from dragn.config import ExperimentConfig
from dragn.data import create_loaders
from dragn.images import save_image_grid
from dragn.models import build_autoencoder
from dragn.training.autoencoder import AutoencoderKLLoss, LatentScaleEstimator
from dragn.training.common import (
    CheckpointManager,
    ExponentialMovingAverage,
    capture_rng_state,
    experiment_signature,
    file_sha256,
    make_epoch_scheduler,
    restore_rng_state,
    wandb_run_id,
)


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


@torch.no_grad()
def _validate(config, model, loader, device):
    model.eval()
    estimator = LatentScaleEstimator()
    losses: list[float] = []
    fixed_images = None
    for batch_index, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        posterior = model.encode(images)
        estimator.update(posterior.sample())
        reconstruction = model.decode(posterior.mode())
        losses.append(torch.nn.functional.l1_loss(reconstruction, images).item())
        if fixed_images is None:
            fixed_images = images
        if config.training.validation_batches and batch_index + 1 >= config.training.validation_batches:
            break
    return sum(losses) / len(losses), estimator.standard_deviation(), fixed_images


def train_autoencoder(config: ExperimentConfig) -> Path:
    if config.task != "autoencoder":
        raise ValueError("train_autoencoder requires task: autoencoder")
    _set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.experiment.output_dir / config.experiment.name
    train_loader, validation_loader = create_loaders(config.data, config.experiment.seed)
    model = build_autoencoder(config.autoencoder, config.data.channels).to(device)
    ema = (
        ExponentialMovingAverage(model, config.training.ema_decay)
        if config.training.ema_decay is not None
        else None
    )
    loss_config = config.autoencoder.loss
    criterion = AutoencoderKLLoss(
        reconstruction=loss_config.reconstruction,
        reconstruction_weight=loss_config.reconstruction_weight,
        kl_weight=loss_config.kl_weight,
        kl_warmup_fraction=loss_config.kl_warmup_fraction,
        perceptual_weight=loss_config.perceptual_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = make_epoch_scheduler(
        optimizer, config.training.epochs, config.training.warmup_epochs, config.training.lr_schedule
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and config.training.precision == "fp16",
    )
    signature = experiment_signature(config)
    data_manifest_hash = (
        file_sha256(config.data.manifest) if config.data.manifest is not None else None
    )
    checkpoints = CheckpointManager(output_dir, config.training.keep_epoch_checkpoints)
    resume = checkpoints.load_for_resume(config.experiment.resume, signature, device)
    start_epoch, global_step, best_validation = 1, 0, float("inf")
    if resume is not None:
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        if ema is not None:
            ema.load_state_dict(resume["ema_state"])
        start_epoch = resume["epoch"] + 1
        global_step = resume["global_step"]
        best_validation = resume["best_validation"]
        restore_rng_state(resume["rng_state"])
        train_loader.generator.set_state(
            resume["loader_generator_state"].detach().cpu().to(torch.uint8)
        )
        print(f"resumed={checkpoints.latest_path} next_epoch={start_epoch}", flush=True)

    wandb_run = None
    if config.logging.wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.logging.project,
            name=config.experiment.name,
            id=wandb_run_id(config, signature),
            resume="allow",
            config=config.model_dump(mode="json"),
            dir=str(output_dir / "wandb"),
        )
        wandb_run.summary["model/parameters"] = sum(
            parameter.numel() for parameter in model.parameters()
        )

    print(
        f"device={device} precision={config.training.precision} batch_size={config.data.batch_size} "
        f"accumulation={config.training.gradient_accumulation_steps} "
        f"train_images={len(train_loader.dataset)} val_images={len(validation_loader.dataset)}",
        flush=True,
    )
    stopped_early = False
    for epoch in range(start_epoch, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.monotonic()
        epoch_loss = 0.0
        seen_images = 0
        pending = 0
        group_size = config.training.gradient_accumulation_steps
        for batch_index, (images, _) in enumerate(train_loader, start=1):
            if pending == 0:
                group_size = min(
                    config.training.gradient_accumulation_steps,
                    len(train_loader) - batch_index + 1,
                )
            images = images.to(device, non_blocking=True)
            with _autocast(device, config.training.precision):
                output = model(images, sample_posterior=True)
                losses = criterion(output, images, global_step, max(config.training.epochs * len(train_loader), 1))
                scaled_loss = losses.total / group_size
            scaler.scale(scaled_loss).backward()
            pending += 1
            epoch_loss += losses.total.item() * images.shape[0]
            seen_images += images.shape[0]
            if pending < group_size:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip) \
                if config.training.grad_clip is not None else torch.nn.utils.get_total_norm(
                    [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            global_step += 1
            if ema is not None:
                ema.update(model)
            if global_step == 1 or global_step % config.training.log_every == 0:
                elapsed = max(time.monotonic() - epoch_start, 1e-6)
                peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
                print(
                    f"epoch={epoch}/{config.training.epochs} step={global_step} "
                    f"total={losses.total.item():.6f} recon={losses.reconstruction.item():.6f} "
                    f"kl={losses.kl.item():.3f} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"grad_norm={float(gradient_norm):.4f} images_per_second={seen_images / elapsed:.2f} "
                    f"peak_gpu_mb={peak:.1f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/total_loss": losses.total.item(),
                        "train/reconstruction_loss": losses.reconstruction.item(),
                        "train/kl": losses.kl.item(),
                        "train/kl_weight": losses.kl_weight,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": float(gradient_norm),
                        "train/images_per_second": seen_images / elapsed,
                        "system/peak_gpu_mb": peak,
                        "epoch": epoch,
                    }, step=global_step)
            if config.training.max_steps and global_step >= config.training.max_steps:
                stopped_early = True
                break

        scheduler.step()
        evaluation_model = ema.model if ema is not None else model
        validation_loss = None
        fixed_images = None
        if epoch % config.training.validate_every == 0 or stopped_early or epoch == config.training.epochs:
            validation_loss, latent_std, fixed_images = _validate(
                config, evaluation_model, validation_loader, device
            )
            evaluation_model.set_latent_scale(latent_std)
            model.set_latent_scale(latent_std)
            print(
                f"epoch={epoch} train_loss={epoch_loss / max(seen_images, 1):.6f} "
                f"validation_l1={validation_loss:.6f} latent_scale={model.latent_scale.item():.6f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "train/epoch_loss": epoch_loss / max(seen_images, 1),
                    "validation/l1": validation_loss,
                    "validation/latent_scale": float(model.latent_scale.item()),
                    "epoch": epoch,
                }, step=global_step)
        is_best = validation_loss is not None and validation_loss < best_validation
        if is_best:
            best_validation = validation_loss
        state = {
            "schema_version": 2,
            "task": "autoencoder",
            "signature": signature,
            "data_manifest_hash": data_manifest_hash,
            "epoch": epoch,
            "global_step": global_step,
            "best_validation": best_validation,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "loader_generator_state": train_loader.generator.get_state(),
            "config": config.model_dump(mode="json"),
        }
        if is_best:
            checkpoints.save_best(state)
        checkpoints.save_latest(state)
        if epoch % config.training.save_every == 0 or stopped_early or epoch == config.training.epochs:
            saved_path = checkpoints.save(state, epoch, is_best)
            print(f"checkpoint={saved_path}", flush=True)
        if fixed_images is not None and (
            epoch % config.training.sample_every == 0 or stopped_early or epoch == config.training.epochs
        ):
            reconstruction = evaluation_model.decode_from_diffusion(
                evaluation_model.encode_for_diffusion(fixed_images, sample_posterior=False)
            )
            reconstruction_path = output_dir / f"reconstructions_epoch_{epoch:04d}.png"
            save_image_grid(
                torch.cat((fixed_images, reconstruction), dim=0),
                reconstruction_path,
                columns=len(fixed_images),
            )
            if wandb_run is not None:
                wandb_run.log({
                    "validation/reconstructions": wandb.Image(str(reconstruction_path)),
                    "epoch": epoch,
                }, step=global_step)
        if stopped_early:
            break
    if wandb_run is not None:
        wandb_run.finish()
    return checkpoints.latest_path
