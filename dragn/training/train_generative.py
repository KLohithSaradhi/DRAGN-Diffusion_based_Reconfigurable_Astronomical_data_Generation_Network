"""Production single-GPU generative trainer for DRAGN v2."""

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
from dragn.models import AutoencoderKL, build_autoencoder, build_dit
from dragn.objectives import DiffusionSchedule, ddpm_prediction_and_target, flow_prediction_and_target
from dragn.samplers import sample_ddpm, sample_flow, seeded_generator
from dragn.training.common import (
    CheckpointManager,
    ExponentialMovingAverage,
    capture_rng_state,
    experiment_signature,
    file_sha256,
    make_epoch_scheduler,
    restore_rng_state,
)
from dragn.training.inference_bundle import ensure_inference_bundle


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


def _load_autoencoder(config: ExperimentConfig, device: torch.device) -> tuple[AutoencoderKL, str]:
    checkpoint_path = config.autoencoder.checkpoint
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")
    checkpoint_hash = file_sha256(checkpoint_path)
    autoencoder = build_autoencoder(config.autoencoder, config.data.channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 2 or checkpoint.get("task") != "autoencoder":
        raise ValueError(f"Not a DRAGN v2 autoencoder checkpoint: {checkpoint_path}")
    autoencoder_state = (
        checkpoint["ema_state"]["model"]
        if checkpoint.get("ema_state") is not None
        else checkpoint["model_state"]
    )
    autoencoder.load_state_dict(autoencoder_state, strict=True)
    autoencoder.requires_grad_(False)
    autoencoder.eval()
    return autoencoder, checkpoint_hash


def _objective_batch(config, model, latents, schedule, generator=None):
    if config.objective.type == "flow":
        return flow_prediction_and_target(model, latents, config.objective.source_std, generator)
    return ddpm_prediction_and_target(model, latents, schedule, generator)


@torch.no_grad()
def _validate(config, model, autoencoder, loader, schedule, device) -> tuple[float, torch.Tensor | None]:
    model.eval()
    losses: list[float] = []
    fixed_images = None
    for batch_index, (images, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        latents = autoencoder.encode_for_diffusion(images, sample_posterior=False)
        generator = seeded_generator(device, config.sampling.seed + batch_index)
        prediction, target = _objective_batch(config, model, latents, schedule, generator)
        losses.append(torch.nn.functional.mse_loss(prediction, target).item())
        if fixed_images is None:
            fixed_images = images[:config.sampling.num_samples]
        if config.training.validation_batches and batch_index + 1 >= config.training.validation_batches:
            break
    return sum(losses) / len(losses), fixed_images


@torch.no_grad()
def _generate(config, model, autoencoder, latent_size, device) -> torch.Tensor:
    model.eval()
    shape = (config.sampling.num_samples, autoencoder.latent_channels, latent_size, latent_size)
    generator = seeded_generator(device, config.sampling.seed)
    if config.objective.type == "flow":
        latents = sample_flow(
            model, shape, config.objective.source_std, config.sampling.steps, generator
        )
    else:
        schedule = DiffusionSchedule.create(
            config.objective.timesteps, config.objective.schedule, device
        )
        latents = sample_ddpm(model, shape, schedule, generator)
    return autoencoder.decode_from_diffusion(latents)


def train_generative(
    config: ExperimentConfig,
    experiment_directory: Path | None = None,
) -> Path:
    if config.task != "base" or config.model is None or config.objective is None or config.sampling is None:
        raise ValueError("Generative training requires a complete task: base configuration")
    _set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.experiment.output_dir / config.experiment.name
    train_loader, validation_loader = create_loaders(config.data, config.experiment.seed)
    autoencoder, ae_hash = _load_autoencoder(config, device)
    latent_size = config.data.image_size // autoencoder.downsample_factor
    model = build_dit(config.model, autoencoder.latent_channels, latent_size).to(device)
    ema = (
        ExponentialMovingAverage(model, config.training.ema_decay)
        if config.training.ema_decay is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = make_epoch_scheduler(
        optimizer,
        config.training.epochs,
        config.training.warmup_epochs,
        config.training.lr_schedule,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=device.type == "cuda" and config.training.precision == "fp16",
    )
    diffusion_schedule = (
        DiffusionSchedule.create(config.objective.timesteps, config.objective.schedule, device)
        if config.objective.type == "ddpm"
        else None
    )
    signature = experiment_signature(config, ae_hash)
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
        if experiment_directory is not None:
            inference_yaml, inference_sbatch = ensure_inference_bundle(
                config, checkpoints.latest_path, experiment_directory
            )
            print(
                f"inference_yaml={inference_yaml} inference_sbatch={inference_sbatch}",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if config.logging.wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.logging.project,
            name=config.experiment.name,
            config=config.model_dump(mode="json"),
            dir=str(output_dir / "wandb"),
        )
    print(
        f"device={device} precision={config.training.precision} batch_size={config.data.batch_size} "
        f"accumulation={config.training.gradient_accumulation_steps} "
        f"train_images={len(train_loader.dataset)} val_images={len(validation_loader.dataset)}",
        flush=True,
    )
    print(
        f"latent_shape=({autoencoder.latent_channels},{latent_size},{latent_size}) "
        f"tokens={(latent_size // config.model.patch_size) ** 2} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}",
        flush=True,
    )

    stopped_early = False
    for epoch in range(start_epoch, config.training.epochs + 1):
        model.train()
        autoencoder.eval()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.monotonic()
        seen_images = 0
        running_loss = 0.0
        pending_batches = 0
        group_size = config.training.gradient_accumulation_steps
        for batch_index, (images, _) in enumerate(train_loader, start=1):
            if pending_batches == 0:
                group_size = min(
                    config.training.gradient_accumulation_steps,
                    len(train_loader) - batch_index + 1,
                )
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                latents = autoencoder.encode_for_diffusion(images, sample_posterior=True)
            with _autocast(device, config.training.precision):
                prediction, target = _objective_batch(
                    config, model, latents, diffusion_schedule
                )
                batch_loss = torch.nn.functional.mse_loss(prediction, target)
                scaled_loss = batch_loss / group_size
            scaler.scale(scaled_loss).backward()
            pending_batches += 1
            running_loss += batch_loss.item() * images.shape[0]
            seen_images += images.shape[0]
            should_step = (
                pending_batches == group_size
            )
            if not should_step:
                continue
            scaler.unscale_(optimizer)
            if config.training.grad_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            else:
                gradient_norm = torch.nn.utils.get_total_norm(
                    [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending_batches = 0
            global_step += 1
            if ema is not None:
                ema.update(model)
            if global_step == 1 or global_step % config.training.log_every == 0:
                elapsed = max(time.monotonic() - epoch_start, 1e-6)
                peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
                metrics = {
                    "train/loss": batch_loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(gradient_norm),
                    "train/images_per_second": seen_images / elapsed,
                    "system/peak_gpu_mb": peak,
                    "epoch": epoch,
                }
                print(
                    f"epoch={epoch}/{config.training.epochs} step={global_step} "
                    f"loss={batch_loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"grad_norm={float(gradient_norm):.4f} images_per_second={seen_images / elapsed:.2f} "
                    f"peak_gpu_mb={peak:.1f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)
            if config.training.max_steps and global_step >= config.training.max_steps:
                stopped_early = True
                break

        scheduler.step()
        epoch_loss = running_loss / max(seen_images, 1)
        validation_loss = None
        fixed_images = None
        evaluation_model = ema.model if ema is not None else model
        if epoch % config.training.validate_every == 0 or stopped_early or epoch == config.training.epochs:
            validation_loss, fixed_images = _validate(
                config, evaluation_model, autoencoder, validation_loader, diffusion_schedule, device
            )
            print(
                f"epoch={epoch} train_loss={epoch_loss:.6f} "
                f"validation_{config.objective.type}_loss={validation_loss:.6f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"train/epoch_loss": epoch_loss, "validation/loss": validation_loss, "epoch": epoch},
                    step=global_step,
                )

        is_best = validation_loss is not None and validation_loss < best_validation
        if is_best:
            best_validation = validation_loss
        state = {
            "schema_version": 2,
            "task": "base",
            "signature": signature,
            "objective": config.objective.model_dump(mode="json"),
            "ae_checkpoint_hash": ae_hash,
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
        if epoch % config.training.save_every == 0 or stopped_early or epoch == config.training.epochs:
            saved_path = checkpoints.save(state, epoch, is_best)
            print(f"checkpoint={saved_path}", flush=True)
            if experiment_directory is not None:
                inference_yaml, inference_sbatch = ensure_inference_bundle(
                    config, checkpoints.latest_path, experiment_directory
                )
                print(
                    f"inference_yaml={inference_yaml} inference_sbatch={inference_sbatch}",
                    flush=True,
                )
        if epoch % config.training.sample_every == 0 or stopped_early or epoch == config.training.epochs:
            generated = _generate(config, evaluation_model, autoencoder, latent_size, device)
            save_image_grid(
                generated,
                output_dir / f"generated_epoch_{epoch:04d}.png",
                columns=max(1, round(len(generated) ** 0.5)),
            )
            if fixed_images is not None:
                reconstruction = autoencoder.decode_from_diffusion(
                    autoencoder.encode_for_diffusion(fixed_images, sample_posterior=False)
                )
                save_image_grid(
                    torch.cat((fixed_images, reconstruction), dim=0),
                    output_dir / f"validation_reconstructions_epoch_{epoch:04d}.png",
                    columns=len(fixed_images),
                )
        if stopped_early:
            break
    if wandb_run is not None:
        wandb_run.finish()
    return checkpoints.latest_path
