"""Authentic single-GPU LoRA specialization for a frozen DRAGN DiT."""

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
from dragn.models import (
    AdapterEMA,
    adapter_parameters,
    adapter_state_dict,
    build_dit,
    inject_lora,
    load_adapter_state_dict,
    set_adapter_scale,
)
from dragn.objectives import DiffusionSchedule
from dragn.training.common import (
    CheckpointManager,
    capture_rng_state,
    experiment_signature,
    file_sha256,
    make_epoch_scheduler,
    restore_rng_state,
)
from dragn.training.train_generative import (
    _generate,
    _load_autoencoder,
    _objective_batch,
    _validate,
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


def _load_frozen_base(config: ExperimentConfig, device: torch.device, latent_size: int):
    assert config.lora is not None and config.model is not None and config.objective is not None
    path = config.lora.base_checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Base DiT checkpoint not found: {path}")
    checkpoint_hash = file_sha256(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != 2 or checkpoint.get("task") != "base":
        raise ValueError(f"Not a DRAGN v2 base DiT checkpoint: {path}")
    saved = ExperimentConfig.model_validate(checkpoint["config"])
    mismatches = []
    if saved.model != config.model:
        mismatches.append("model")
    if saved.objective != config.objective:
        mismatches.append("objective")
    if saved.autoencoder.model_dump(exclude={"checkpoint"}) != config.autoencoder.model_dump(exclude={"checkpoint"}):
        mismatches.append("autoencoder architecture")
    if (saved.data.image_size, saved.data.channels) != (config.data.image_size, config.data.channels):
        mismatches.append("image geometry")
    if mismatches:
        raise ValueError("LoRA/base checkpoint mismatch: " + ", ".join(mismatches))

    model = build_dit(config.model, config.autoencoder.latent_channels, latent_size).to(device)
    if config.lora.base_weights == "ema":
        if checkpoint.get("ema_state") is None:
            raise ValueError("base_weights=ema requested, but the base checkpoint has no EMA state")
        state = checkpoint["ema_state"]["model"]
    else:
        state = checkpoint["model_state"]
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model, checkpoint_hash, checkpoint


def train_lora(config: ExperimentConfig) -> Path:
    if (
        config.task != "lora" or config.model is None or config.objective is None
        or config.sampling is None or config.lora is None or config.data.filter is None
    ):
        raise ValueError("LoRA training requires a complete task: lora configuration")
    _set_seed(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.experiment.output_dir / config.experiment.name
    train_loader, validation_loader = create_loaders(config.data, config.experiment.seed)
    autoencoder, ae_hash = _load_autoencoder(config, device)
    latent_size = config.data.image_size // autoencoder.downsample_factor
    model, base_hash, base_checkpoint = _load_frozen_base(config, device, latent_size)
    replaced = inject_lora(model, config.lora)
    trainable = list(adapter_parameters(model))
    if not trainable:
        raise RuntimeError("The selected LoRA preset produced no trainable parameters")
    ema = AdapterEMA(model, config.training.ema_decay) if config.training.ema_decay else None
    optimizer = torch.optim.AdamW(
        trainable, lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = make_epoch_scheduler(
        optimizer, config.training.epochs, config.training.warmup_epochs,
        config.training.lr_schedule,
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda" and config.training.precision == "fp16"
    )
    diffusion_schedule = (
        DiffusionSchedule.create(config.objective.timesteps, config.objective.schedule, device)
        if config.objective.type == "ddpm" else None
    )
    signature = experiment_signature(config, ae_hash, base_hash)
    checkpoints = CheckpointManager(output_dir, config.training.keep_epoch_checkpoints)
    resume = checkpoints.load_for_resume(config.experiment.resume, signature, device)
    start_epoch, global_step, best_validation = 1, 0, float("inf")
    if resume is not None:
        load_adapter_state_dict(model, resume["adapter_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        if ema is not None:
            ema.load_state_dict(resume["ema_adapter_state"])
        start_epoch = resume["epoch"] + 1
        global_step = resume["global_step"]
        best_validation = resume["best_validation"]
        restore_rng_state(resume["rng_state"])
        train_loader.generator.set_state(resume["loader_generator_state"])
        print(f"resumed={checkpoints.latest_path} next_epoch={start_epoch}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if config.logging.wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.logging.project, name=config.experiment.name,
            config=config.model_dump(mode="json"), dir=str(output_dir / "wandb"),
        )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    print(
        f"device={device} preset={config.lora.preset} base_weights={config.lora.base_weights} "
        f"joint_filter={config.data.filter.instrument}/{config.data.filter.class_name} "
        f"batch_size={config.data.batch_size} train_images={len(train_loader.dataset)} "
        f"val_images={len(validation_loader.dataset)}",
        flush=True,
    )
    print(
        f"adapted_modules={len(replaced)} trainable_parameters={trainable_parameters:,} "
        f"total_parameters={total_parameters:,} trainable_fraction={trainable_parameters / total_parameters:.6f}",
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
            if config.training.max_steps and global_step >= config.training.max_steps:
                stopped_early = True
                break
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
            if pending_batches != group_size:
                continue
            scaler.unscale_(optimizer)
            if config.training.grad_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config.training.grad_clip)
            else:
                gradient_norm = torch.nn.utils.get_total_norm(
                    [parameter.grad for parameter in trainable if parameter.grad is not None]
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
                print(
                    f"epoch={epoch}/{config.training.epochs} step={global_step} "
                    f"loss={batch_loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"grad_norm={float(gradient_norm):.6e} images_per_second={seen_images / elapsed:.2f} "
                    f"peak_gpu_mb={peak:.1f}", flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": batch_loss.item(), "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": float(gradient_norm), "epoch": epoch,
                    }, step=global_step)

        scheduler.step()
        epoch_loss = running_loss / max(seen_images, 1)
        validation_loss = None
        fixed_images = None
        should_validate = (
            epoch % config.training.validate_every == 0 or stopped_early
            or epoch == config.training.epochs
        )
        should_sample = (
            epoch % config.training.sample_every == 0 or stopped_early
            or epoch == config.training.epochs
        )
        evaluation_context = ema.apply(model) if ema is not None else nullcontext(model)
        with evaluation_context:
            if should_validate:
                validation_loss, fixed_images = _validate(
                    config, model, autoencoder, validation_loader, diffusion_schedule, device
                )
                print(
                    f"epoch={epoch} train_loss={epoch_loss:.6f} "
                    f"validation_{config.objective.type}_loss={validation_loss:.6f}", flush=True,
                )
            if should_sample:
                set_adapter_scale(model, config.lora.inference_scale)
                generated = _generate(config, model, autoencoder, latent_size, device)
                set_adapter_scale(model, 1.0)
                save_image_grid(
                    generated, output_dir / f"generated_epoch_{epoch:04d}.png",
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

        is_best = validation_loss is not None and validation_loss < best_validation
        if is_best:
            best_validation = validation_loss
        state = {
            "schema_version": 2,
            "task": "lora",
            "signature": signature,
            "objective": config.objective.model_dump(mode="json"),
            "preset": config.lora.preset,
            "targets": list(config.lora.targets),
            "base_checkpoint_hash": base_hash,
            "base_checkpoint_task": base_checkpoint["task"],
            "ae_checkpoint_hash": ae_hash,
            "epoch": epoch,
            "global_step": global_step,
            "best_validation": best_validation,
            "adapter_state": adapter_state_dict(model),
            "ema_adapter_state": ema.state_dict() if ema is not None else None,
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
        if stopped_early:
            break
    if wandb_run is not None:
        wandb_run.finish()
    return checkpoints.latest_path
