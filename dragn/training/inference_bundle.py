"""Generate inference artifacts beside a base or LoRA training YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from dragn.config import ExperimentConfig, InferenceConfig


INFERENCE_SBATCH = """#!/bin/bash
#SBATCH --job-name=dragn_inference
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate /home1/lkanduku/miniconda3/envs/torch/
set -euo pipefail

DRAGN_EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
DRAGN_CONFIG_PATH="$DRAGN_EXPERIMENT_DIR/inference.yaml"
DRAGN_REPO_ROOT="$(git -C "$DRAGN_EXPERIMENT_DIR" rev-parse --show-toplevel)"

if [[ ! -f "$DRAGN_REPO_ROOT/train.py" ]]; then
    echo "ERROR: train.py not found at $DRAGN_REPO_ROOT/train.py" >&2
    exit 1
fi
if [[ ! -f "$DRAGN_CONFIG_PATH" ]]; then
    echo "ERROR: inference.yaml not found in $DRAGN_EXPERIMENT_DIR" >&2
    exit 1
fi

cd "$DRAGN_REPO_ROOT"
python -c "import numpy, PIL, pydantic, torch, yaml"
srun --unbuffered python -u "$DRAGN_REPO_ROOT/train.py" --config "$DRAGN_CONFIG_PATH"
"""


def _inference_payload(config: ExperimentConfig, latest_checkpoint: Path) -> dict:
    if config.task not in {"base", "lora"} or config.sampling is None:
        raise ValueError("Inference bundles can only be generated for base or LoRA training")
    inference = {
        "autoencoder_checkpoint": config.autoencoder.checkpoint,
        "base_checkpoint": latest_checkpoint,
        "base_weights": "ema" if config.training.ema_decay is not None else "raw",
        "adapters": [],
    }
    if config.task == "lora":
        assert config.lora is not None
        inference["base_checkpoint"] = config.lora.base_checkpoint
        inference["base_weights"] = config.lora.base_weights
        inference["adapters"] = [{
            "name": config.lora.preset,
            "checkpoint": latest_checkpoint,
            "weights": "ema" if config.training.ema_decay is not None else "raw",
            "scale": config.lora.inference_scale,
        }]
    output_dir = config.experiment.output_dir / config.experiment.name
    payload = {
        "schema_version": 2,
        "task": "inference",
        "experiment": {
            "name": "inference",
            "output_dir": output_dir,
            "seed": config.experiment.seed,
            "resume": "never",
        },
        "inference": inference,
        "sampling": config.sampling,
    }
    return InferenceConfig.model_validate(payload).model_dump(mode="json")


def ensure_inference_bundle(
    config: ExperimentConfig,
    latest_checkpoint: Path,
    experiment_directory: Path,
) -> tuple[Path, Path]:
    """Create the deterministic bundle once; latest.pt keeps it current."""
    experiment_directory.mkdir(parents=True, exist_ok=True)
    logs_directory = experiment_directory / "logs"
    logs_directory.mkdir(parents=True, exist_ok=True)
    yaml_path = experiment_directory / "inference.yaml"
    sbatch_path = experiment_directory / "inference.sbatch"
    if not yaml_path.exists():
        payload = _inference_payload(config, latest_checkpoint)
        yaml_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    if not sbatch_path.exists():
        sbatch_path.write_text(INFERENCE_SBATCH, encoding="utf-8")
        sbatch_path.chmod(0o755)
    return yaml_path, sbatch_path
