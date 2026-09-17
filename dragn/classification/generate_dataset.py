"""Resolve LoRA experiment YAMLs and export a labeled synthetic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dragn.config import ExperimentConfig, SamplingSection, load_config

from .export_synthetic import ExportConfig, Variant, export_synthetic


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentSource(StrictModel):
    experiment_config: Path
    checkpoint: Path | None = None
    weights: Literal["ema", "raw"] = "ema"
    scale: float = Field(default=1.0, ge=0)


class DatasetGenerationConfig(StrictModel):
    schema_version: Literal[1]
    task: Literal["generate_classification_data"]
    real_manifest: Path
    output_dir: Path
    manifest: Path
    ratio: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=16, gt=0)
    sampling: SamplingSection
    sources: list[ExperimentSource] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_configs(self) -> "DatasetGenerationConfig":
        paths = [source.experiment_config for source in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("sources must not repeat an experiment_config")
        return self


def _checkpoint_for(config: ExperimentConfig, source: ExperimentSource) -> Path:
    if source.checkpoint is not None:
        return source.checkpoint
    return config.experiment.output_dir / config.experiment.name / "best.pt"


def resolve_export_config(config: DatasetGenerationConfig) -> ExportConfig:
    variants: list[Variant] = []
    autoencoder_checkpoint = None
    base_checkpoint = None
    base_weights = None
    combinations = set()
    for source in config.sources:
        loaded = load_config(source.experiment_config)
        if not isinstance(loaded, ExperimentConfig) or loaded.task != "lora":
            raise ValueError(f"Source must be a task: lora experiment: {source.experiment_config}")
        if loaded.lora is None or loaded.data.filter is None or loaded.autoencoder.checkpoint is None:
            raise ValueError(f"Incomplete LoRA experiment configuration: {source.experiment_config}")
        instrument = loaded.data.filter.instrument.upper()
        class_name = loaded.data.filter.class_name.lower()
        combination = (instrument, class_name)
        if combination in combinations:
            raise ValueError(f"Multiple sources resolve to {instrument}/{class_name}")
        combinations.add(combination)
        if instrument not in {"SDSS", "SUBARU"}:
            raise ValueError(f"Unsupported instrument {instrument!r} in {source.experiment_config}")
        if class_name not in {"lens", "spiral", "ring", "companion", "smooth"}:
            raise ValueError(f"Unsupported object {class_name!r} in {source.experiment_config}")
        current_ae = loaded.autoencoder.checkpoint
        current_base = loaded.lora.base_checkpoint
        current_base_weights = loaded.lora.base_weights
        if autoencoder_checkpoint is None:
            autoencoder_checkpoint = current_ae
            base_checkpoint = current_base
            base_weights = current_base_weights
        elif (
            current_ae != autoencoder_checkpoint
            or current_base != base_checkpoint
            or current_base_weights != base_weights
        ):
            raise ValueError("All source experiments must share one autoencoder, base checkpoint, and base weight variant")
        variants.append(Variant(
            name=f"{instrument.lower()}_{class_name}",
            instrument=instrument,
            class_name=class_name,
            experiment_config=source.experiment_config,
            checkpoint=_checkpoint_for(loaded, source),
            weights=source.weights,
            scale=source.scale,
        ))
    assert autoencoder_checkpoint is not None and base_checkpoint is not None and base_weights is not None
    return ExportConfig(
        schema_version=1,
        task="synthetic_export",
        output_dir=config.output_dir,
        manifest=config.manifest,
        real_manifest=config.real_manifest,
        autoencoder_checkpoint=autoencoder_checkpoint,
        base_checkpoint=base_checkpoint,
        base_weights=base_weights,
        ratio=config.ratio,
        batch_size=config.batch_size,
        sampling=config.sampling,
        variants=variants,
    )


def load_generation_config(path: Path) -> DatasetGenerationConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return DatasetGenerationConfig.model_validate(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_generation_config(args.config)
    manifest = export_synthetic(resolve_export_config(config))
    print(f"synthetic_manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
