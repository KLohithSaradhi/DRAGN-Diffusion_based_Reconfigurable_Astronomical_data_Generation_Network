"""Single YAML entry point for DRAGN v2 experiments."""

import argparse

from dragn.config import load_config
from dragn.training.train_autoencoder import train_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.task == "autoencoder":
        train_autoencoder(config)
        return
    raise NotImplementedError(f"task={config.task!r} is scheduled for a later DRAGN v2 step")


if __name__ == "__main__":
    main()
