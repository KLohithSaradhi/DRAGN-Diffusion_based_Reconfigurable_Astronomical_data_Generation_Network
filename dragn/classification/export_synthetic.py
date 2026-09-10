"""Export labeled, provenance-tracked DRAGN samples as individual images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Literal

import torch
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from dragn.config import ExperimentConfig, SamplingSection
from dragn.models import build_dit, inject_lora, load_adapter_state_dict, set_adapter_scale
from dragn.training.common import file_sha256
from dragn.training.inference import _base_state, _read_checkpoint
from dragn.training.train_generative import _generate, _load_autoencoder

from .data import INSTRUMENTS, OBJECTS, read_manifest


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Variant(StrictModel):
    name: str = Field(min_length=1)
    instrument: Literal["SDSS", "SUBARU"]
    class_name: Literal["lens", "spiral"]
    checkpoint: Path
    weights: Literal["ema", "raw"] = "ema"
    scale: float = Field(default=1.0, ge=0)


class ExportConfig(StrictModel):
    schema_version: Literal[1]
    task: Literal["synthetic_export"]
    output_dir: Path
    manifest: Path
    real_manifest: Path
    autoencoder_checkpoint: Path
    base_checkpoint: Path
    base_weights: Literal["ema", "raw"] = "ema"
    ratio: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=16, gt=0)
    sampling: SamplingSection
    variants: list[Variant] = Field(min_length=4, max_length=4)


def _save_individual(images: torch.Tensor, directory: Path, start: int) -> list[Path]:
    images = images.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).byte()
    paths = []
    directory.mkdir(parents=True, exist_ok=True)
    for offset, tensor in enumerate(images):
        array = tensor.permute(1, 2, 0).numpy()
        if array.shape[2] == 1:
            array = array[:, :, 0]
        path = directory / f"sample_{start + offset:07d}.png"
        Image.fromarray(array).save(path)
        paths.append(path.resolve())
    return paths


def export_synthetic(config: ExportConfig) -> Path:
    combinations = {(variant.instrument, variant.class_name) for variant in config.variants}
    expected = {(instrument, class_name) for instrument in INSTRUMENTS for class_name in OBJECTS}
    if combinations != expected:
        raise ValueError(f"variants must cover each instrument/object combination exactly once: {sorted(expected)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_checkpoint, base_hash = _read_checkpoint(config.base_checkpoint, "base")
    base_config = ExperimentConfig.model_validate(base_checkpoint["config"])
    if base_config.model is None or base_config.objective is None:
        raise ValueError("Base checkpoint has no generative model configuration")
    expected_sampler = "ancestral" if base_config.objective.type == "ddpm" else "euler"
    if config.sampling.method != expected_sampler:
        raise ValueError(f"Base objective requires sampling method {expected_sampler}")
    runtime = base_config.model_copy(update={
        "autoencoder": base_config.autoencoder.model_copy(update={"checkpoint": config.autoencoder_checkpoint}),
        "sampling": config.sampling,
    })
    autoencoder, ae_hash = _load_autoencoder(runtime, device)
    if base_checkpoint.get("ae_checkpoint_hash") != ae_hash:
        raise ValueError("Export autoencoder does not match the base checkpoint")
    real_manifest_hash = file_sha256(config.real_manifest)
    if base_checkpoint.get("data_manifest_hash") != real_manifest_hash:
        raise ValueError("Base checkpoint was not trained from the selected real split manifest")
    ae_checkpoint, _ = _read_checkpoint(config.autoencoder_checkpoint, "autoencoder")
    if ae_checkpoint.get("data_manifest_hash") != real_manifest_hash:
        raise ValueError("Autoencoder was not trained from the selected real split manifest")
    if base_config.data.manifest_split != "train":
        raise ValueError("Leakage-safe export requires a base model trained with manifest_split: train")
    latent_size = base_config.data.image_size // autoencoder.downsample_factor
    frozen_state = _base_state(base_checkpoint, config.base_weights)
    real_records = [record for record in read_manifest(config.real_manifest, "real") if record.split == "train"]
    rows = []
    provenance = {"base_checkpoint": str(config.base_checkpoint), "base_sha256": base_hash,
                  "autoencoder_checkpoint": str(config.autoencoder_checkpoint), "autoencoder_sha256": ae_hash,
                  "ratio": config.ratio, "variants": []}
    for variant_index, variant in enumerate(config.variants):
        real_count = sum(
            row.instrument == variant.instrument and row.class_name == variant.class_name
            for row in real_records
        )
        sample_count = round(real_count * config.ratio)
        if sample_count < 1:
            raise RuntimeError(f"No real training examples found for {variant.instrument}/{variant.class_name}")
        adapter_checkpoint, adapter_hash = _read_checkpoint(variant.checkpoint, "lora")
        adapter_config = ExperimentConfig.model_validate(adapter_checkpoint["config"])
        if adapter_config.lora is None or adapter_config.data.filter is None:
            raise ValueError(f"Invalid LoRA checkpoint: {variant.checkpoint}")
        if (adapter_config.data.filter.instrument.upper(), adapter_config.data.filter.class_name.lower()) != (
            variant.instrument, variant.class_name
        ):
            raise ValueError(f"Adapter label does not match export variant {variant.name}")
        if adapter_checkpoint.get("base_checkpoint_hash") != base_hash or adapter_checkpoint.get("ae_checkpoint_hash") != ae_hash:
            raise ValueError(f"Adapter {variant.name} was not trained from the selected base and autoencoder")
        if adapter_checkpoint.get("data_manifest_hash") != real_manifest_hash:
            raise ValueError(f"Adapter {variant.name} was not trained from the selected real split manifest")
        if adapter_config.model != base_config.model or adapter_config.objective != base_config.objective:
            raise ValueError(f"Adapter {variant.name} architecture/objective mismatch")
        if adapter_config.lora.base_weights != config.base_weights:
            raise ValueError(f"Adapter {variant.name} uses a different base weight variant")
        model = build_dit(base_config.model, autoencoder.latent_channels, latent_size).to(device)
        model.load_state_dict(frozen_state, strict=True)
        model.requires_grad_(False).eval()
        inject_lora(model, adapter_config.lora)
        state = adapter_checkpoint["adapter_state"] if variant.weights == "raw" else (
            adapter_checkpoint.get("ema_adapter_state") or {}
        ).get("adapter")
        if state is None:
            raise ValueError(f"Adapter {variant.name} has no requested EMA weights")
        load_adapter_state_dict(model, state)
        set_adapter_scale(model, variant.scale)
        variant_dir = config.output_dir / variant.instrument / variant.class_name
        generated = 0
        while generated < sample_count:
            current = min(config.batch_size, sample_count - generated)
            sampling = config.sampling.model_copy(update={
                "num_samples": current,
                "seed": config.sampling.seed + variant_index * 1_000_000 + generated,
            })
            images = _generate(runtime.model_copy(update={"sampling": sampling}), model,
                               autoencoder, latent_size, device)
            for path in _save_individual(images, variant_dir, generated):
                rows.append({"path": path, "instrument": variant.instrument,
                             "class_name": variant.class_name, "split": "train", "source": "synthetic"})
            generated += current
        provenance["variants"].append({"name": variant.name, "instrument": variant.instrument,
                                       "class_name": variant.class_name, "samples": sample_count,
                                       "checkpoint": str(variant.checkpoint), "checkpoint_sha256": adapter_hash,
                                       "weights": variant.weights, "scale": variant.scale})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    config.manifest.parent.mkdir(parents=True, exist_ok=True)
    with config.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "instrument", "class_name", "split", "source"])
        writer.writeheader()
        writer.writerows(rows)
    (config.manifest.parent / "synthetic_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return config.manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    path = export_synthetic(ExportConfig.model_validate(payload))
    print(f"synthetic_manifest={path}", flush=True)


if __name__ == "__main__":
    main()
