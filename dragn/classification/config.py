"""Strict configuration for the DRAGN classification benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentSection(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    output_dir: Path = Path("./results/classification")
    seed: int = Field(default=42, ge=0)


class ClassificationDataSection(StrictModel):
    manifest: Path
    synthetic_manifest: Path | None = None
    instrument_filter: Literal["SDSS", "SUBARU"] | None = None
    image_size: int = Field(default=224, gt=0)
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)
    augment: bool = True


class ResNetSection(StrictModel):
    architecture: Literal["resnet18"] = "resnet18"
    weights: Literal["imagenet"] = "imagenet"
    fine_tune: Literal["all"] = "all"


class TrainingSection(StrictModel):
    epochs: int = Field(default=50, gt=0)
    optimizer: Literal["adamw"] = "adamw"
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    early_stopping_patience: int = Field(default=8, gt=0)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"


class LoggingSection(StrictModel):
    wandb: bool = False
    project: str = "dragn-classification"


class ClassificationConfig(StrictModel):
    schema_version: Literal[1]
    task: Literal["classification"]
    target: Literal["object", "instrument"]
    experiment: ExperimentSection
    data: ClassificationDataSection
    model: ResNetSection = Field(default_factory=ResNetSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)

    @model_validator(mode="after")
    def validate_task_scope(self) -> "ClassificationConfig":
        if self.target == "object" and self.data.instrument_filter is None:
            raise ValueError("object classification requires data.instrument_filter")
        if self.target == "instrument" and self.data.instrument_filter is not None:
            raise ValueError("instrument classification must use all instruments")
        return self


def load_classification_config(path: str | Path) -> ClassificationConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Classification YAML must contain a mapping")
    return ClassificationConfig.model_validate(payload)
