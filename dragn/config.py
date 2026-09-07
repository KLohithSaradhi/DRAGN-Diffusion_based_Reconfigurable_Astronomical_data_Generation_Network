"""Strict, YAML-first configuration models for DRAGN v2 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentSection(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    output_dir: Path = Path("./results")
    seed: int = Field(default=42, ge=0)
    resume: Literal["never", "auto", "required"] = "auto"


class DataFilter(StrictModel):
    instrument: str = Field(min_length=1)
    class_name: str = Field(min_length=1)


class DataSection(StrictModel):
    dataset: Literal["astro", "mnist"]
    root_dir: Path
    layout: Literal["flat_class", "instrument_class"]
    image_size: int = Field(gt=0)
    channels: Literal[1, 3]
    transform: str = Field(min_length=1)
    batch_size: int = Field(gt=0)
    num_workers: int = Field(default=4, ge=0)
    validation_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    filter: DataFilter | None = None


class AutoencoderLoss(StrictModel):
    reconstruction: Literal["l1", "mse"] = "l1"
    reconstruction_weight: float = Field(default=1.0, gt=0)
    kl_weight: float = Field(default=1e-6, ge=0)
    kl_warmup_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    perceptual_weight: float = Field(default=0.0, ge=0)


class AutoencoderSection(StrictModel):
    enabled: bool = True
    architecture: Literal["autoencoder_kl"] = "autoencoder_kl"
    checkpoint: Path | None = None
    latent_channels: int = Field(default=16, gt=0)
    downsample_factor: int = Field(default=8, gt=0)
    base_channels: int = Field(default=64, gt=0)
    channel_multipliers: list[int] = Field(default_factory=lambda: [1, 2, 4, 4], min_length=2)
    num_res_blocks: int = Field(default=2, gt=0)
    loss: AutoencoderLoss = Field(default_factory=AutoencoderLoss)

    @model_validator(mode="after")
    def validate_downsampling(self) -> "AutoencoderSection":
        if self.downsample_factor & (self.downsample_factor - 1):
            raise ValueError("autoencoder.downsample_factor must be a power of two")
        expected_stages = self.downsample_factor.bit_length()
        if len(self.channel_multipliers) != expected_stages:
            raise ValueError(
                "autoencoder.channel_multipliers must contain one entry for the input "
                "stage plus one per downsampling stage"
            )
        return self


class DiTSection(StrictModel):
    architecture: Literal["dit"] = "dit"
    patch_size: int = Field(default=2, gt=0)
    hidden_size: int = Field(default=256, gt=0)
    depth: int = Field(default=6, gt=0)
    num_heads: int = Field(default=8, gt=0)
    mlp_ratio: float = Field(default=4.0, gt=0)
    positional_embedding: Literal["sincos_2d"] = "sincos_2d"
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_attention_width(self) -> "DiTSection":
        if self.hidden_size % self.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if self.hidden_size % 4:
            raise ValueError("model.hidden_size must be divisible by 4 for 2D sin-cos positions")
        return self


class ObjectiveSection(StrictModel):
    type: Literal["ddpm", "flow"]
    prediction: Literal["epsilon", "velocity"]
    timesteps: int = Field(default=1000, gt=0)
    schedule: Literal["linear", "cosine"] = "cosine"
    source_std: float = Field(default=1.0, gt=0)
    time_distribution: Literal["uniform"] = "uniform"

    @model_validator(mode="after")
    def validate_parameterization(self) -> "ObjectiveSection":
        expected = "epsilon" if self.type == "ddpm" else "velocity"
        if self.prediction != expected:
            raise ValueError(f"objective.type={self.type!r} requires prediction={expected!r}")
        return self


class LoRASection(StrictModel):
    enabled: bool = True
    base_checkpoint: Path
    rank: int = Field(default=16, gt=0)
    alpha: float = Field(default=16.0, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    targets: list[Literal["attention.qkv", "attention.proj", "mlp.fc1", "mlp.fc2"]] = \
        Field(default_factory=lambda: ["attention.qkv", "attention.proj", "mlp.fc1", "mlp.fc2"], min_length=1)


class TrainingSection(StrictModel):
    epochs: int = Field(gt=0)
    optimizer: Literal["adamw"] = "adamw"
    lr: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    grad_clip: float | None = Field(default=1.0, gt=0)
    ema_decay: float | None = Field(default=0.9999, gt=0, lt=1)
    save_every: int = Field(default=10, gt=0)
    sample_every: int = Field(default=10, gt=0)
    log_every: int = Field(default=10, gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    validation_batches: int | None = Field(default=None, gt=0)
    smoke_test: bool = False


class SamplingSection(StrictModel):
    method: Literal["ancestral", "euler"]
    steps: int = Field(default=50, gt=0)


class LoggingSection(StrictModel):
    wandb: bool = False
    project: str = "dragn"


class ExperimentConfig(StrictModel):
    schema_version: Literal[2]
    task: Literal["autoencoder", "base", "lora"]
    experiment: ExperimentSection
    data: DataSection
    autoencoder: AutoencoderSection
    model: DiTSection | None = None
    objective: ObjectiveSection | None = None
    lora: LoRASection | None = None
    training: TrainingSection
    sampling: SamplingSection | None = None
    logging: LoggingSection = Field(default_factory=LoggingSection)

    @model_validator(mode="after")
    def validate_task_contract(self) -> "ExperimentConfig":
        if self.data.image_size % self.autoencoder.downsample_factor:
            raise ValueError("data.image_size must be divisible by autoencoder.downsample_factor")

        if self.task == "autoencoder":
            if self.model is not None or self.objective is not None or self.lora is not None:
                raise ValueError("autoencoder tasks cannot define model, objective, or lora sections")
            if self.sampling is not None:
                raise ValueError("autoencoder tasks cannot define a sampling section")
            return self

        if self.model is None or self.objective is None or self.sampling is None:
            raise ValueError("base and lora tasks require model, objective, and sampling sections")
        if self.autoencoder.enabled and self.autoencoder.checkpoint is None:
            raise ValueError("generative tasks with the autoencoder enabled require its checkpoint")

        model_input_size = (
            self.data.image_size // self.autoencoder.downsample_factor
            if self.autoencoder.enabled
            else self.data.image_size
        )
        if model_input_size % self.model.patch_size:
            raise ValueError("DiT input size must be divisible by model.patch_size")

        expected_sampler = "ancestral" if self.objective.type == "ddpm" else "euler"
        if self.sampling.method != expected_sampler:
            raise ValueError(
                f"objective.type={self.objective.type!r} requires sampling.method={expected_sampler!r}"
            )

        if self.task == "base":
            if self.lora is not None:
                raise ValueError("base tasks cannot define a lora section")
            if self.data.filter is not None:
                raise ValueError("base DiT training must be unconditional and use unfiltered data")
        else:
            if self.lora is None or not self.lora.enabled:
                raise ValueError("lora tasks require an enabled lora section")
            if self.data.filter is None:
                raise ValueError("joint LoRA tasks require both instrument and class_name filters")

        return self


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and strictly validate a v2 experiment YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return ExperimentConfig.model_validate(payload)
