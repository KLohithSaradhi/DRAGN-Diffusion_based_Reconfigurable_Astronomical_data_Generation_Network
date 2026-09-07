"""Minimal, deterministic image loading for DRAGN v2."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from dragn.config import DataSection


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageDataset(Dataset[tuple[Tensor, dict[str, str]]]):
    def __init__(self, config: DataSection):
        if config.transform != "astro_standard":
            raise NotImplementedError(
                f"transform={config.transform!r} is not implemented in the Step 2 data path"
            )
        self.config = config
        self.samples = self._discover()
        if not self.samples:
            raise RuntimeError(f"No matching images found under {config.root_dir}")

    def _discover(self) -> list[tuple[Path, str, str]]:
        root = self.config.root_dir.expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        samples: list[tuple[Path, str, str]] = []
        if self.config.layout == "instrument_class":
            for instrument_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                for class_dir in sorted(path for path in instrument_dir.iterdir() if path.is_dir()):
                    for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                            samples.append((image_path, instrument_dir.name, class_dir.name))
        else:
            for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                    if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append((image_path, "", class_dir.name))
        if self.config.filter is not None:
            requested = self.config.filter
            samples = [
                sample for sample in samples
                if sample[1] == requested.instrument and sample[2] == requested.class_name
            ]
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, str]]:
        path, instrument, class_name = self.samples[index]
        mode = "L" if self.config.channels == 1 else "RGB"
        with Image.open(path) as image:
            image = image.convert(mode)
            image = image.resize((self.config.image_size, self.config.image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32).copy()
        if array.ndim == 2:
            array = array[:, :, None]
        tensor = torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0
        return tensor, {"path": str(path), "instrument": instrument, "class_name": class_name}


def create_loaders(config: DataSection, seed: int) -> tuple[DataLoader, DataLoader]:
    dataset = ImageDataset(config)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * config.validation_fraction))
    if validation_count >= len(indices):
        raise RuntimeError("Dataset needs at least two images for a train/validation split")
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(Subset(dataset, training_indices), shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(Subset(dataset, validation_indices), shuffle=False, **common)
    return train_loader, validation_loader
