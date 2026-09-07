"""Single YAML entry point for DRAGN v2 experiments."""

import argparse
from pathlib import Path

from dragn.config import load_config
from dragn.training.train_autoencoder import train_autoencoder
from dragn.training.train_generative import train_generative
from dragn.training.train_lora import train_lora
from dragn.training.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    experiment_directory = Path(args.config).resolve().parent
    if config.task == "inference":
        run_inference(config)
        return
    if config.task == "autoencoder":
        train_autoencoder(config)
        return
    if config.task == "base":
        train_generative(config, experiment_directory)
        return
    if config.task == "lora":
        train_lora(config, experiment_directory)
        return
    raise ValueError(f"Unsupported task: {config.task!r}")


if __name__ == "__main__":
    main()
