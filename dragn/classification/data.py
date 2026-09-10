"""Manifest-driven real and synthetic image loading."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import ClassificationConfig


INSTRUMENTS = ("SDSS", "SUBARU")
OBJECTS = ("lens", "spiral")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    instrument: str
    class_name: str
    split: str
    source: str


def discover_real_images(root: Path) -> list[ImageRecord]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Real dataset root does not exist: {root}")
    records: list[ImageRecord] = []
    for instrument_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        instrument = instrument_dir.name.upper()
        if instrument not in INSTRUMENTS:
            continue
        for class_dir in sorted(path for path in instrument_dir.iterdir() if path.is_dir()):
            class_name = class_dir.name.lower()
            if class_name not in OBJECTS:
                continue
            for path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(ImageRecord(path.resolve(), instrument, class_name, "", "real"))
    if not records:
        raise RuntimeError(f"No SDSS/SUBARU lens/spiral images found under {root}")
    return records


def write_split_manifest(
    root: Path,
    output: Path,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Path:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and sum to less than one")
    grouped: dict[tuple[str, str], list[ImageRecord]] = {}
    for record in discover_real_images(root):
        grouped.setdefault((record.instrument, record.class_name), []).append(record)
    rows: list[ImageRecord] = []
    for group_index, (key, records) in enumerate(sorted(grouped.items())):
        random.Random(seed + group_index).shuffle(records)
        count = len(records)
        if count < 3:
            raise RuntimeError(f"Stratum {key} needs at least three images, found {count}")
        validation_count = max(1, round(count * validation_fraction))
        test_count = max(1, round(count * test_fraction))
        if validation_count + test_count >= count:
            validation_count = test_count = 1
        for index, record in enumerate(records):
            split = "test" if index < test_count else "validation" if index < test_count + validation_count else "train"
            rows.append(ImageRecord(record.path, record.instrument, record.class_name, split, "real"))
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "instrument", "class_name", "split", "source"])
        writer.writeheader()
        for row in sorted(rows, key=lambda value: str(value.path)):
            writer.writerow(row.__dict__)
    return output


def read_manifest(path: Path, source_default: str) -> list[ImageRecord]:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = Path(row["path"]).expanduser()
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            records.append(ImageRecord(
                image_path, row["instrument"].upper(), row["class_name"].lower(),
                row.get("split", "train"), row.get("source", source_default),
            ))
    return records


class ClassificationDataset(Dataset):
    def __init__(self, records: list[ImageRecord], target: str, transform):
        self.records = records
        self.target = target
        self.transform = transform
        self.classes = list(OBJECTS if target == "object" else INSTRUMENTS)
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        label_name = record.class_name if self.target == "object" else record.instrument
        return tensor, self.class_to_index[label_name], index


def _transforms(image_size: int, augment: bool):
    operations = [transforms.Resize((image_size, image_size))]
    if augment:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(operations)


def create_loaders(config: ClassificationConfig):
    real = read_manifest(config.data.manifest, "real")
    if config.data.instrument_filter:
        real = [row for row in real if row.instrument == config.data.instrument_filter]
    by_split = {split: [row for row in real if row.split == split] for split in ("train", "validation", "test")}
    if config.data.synthetic_manifest is not None:
        synthetic = read_manifest(config.data.synthetic_manifest, "synthetic")
        if config.data.instrument_filter:
            synthetic = [row for row in synthetic if row.instrument == config.data.instrument_filter]
        synthetic = [row for row in synthetic if row.split == "train"]
        real_training = by_split["train"]
        strata = {(row.instrument, row.class_name) for row in real_training}
        for stratum in strata:
            real_count = sum((row.instrument, row.class_name) == stratum for row in real_training)
            synthetic_count = sum((row.instrument, row.class_name) == stratum for row in synthetic)
            if synthetic_count != real_count:
                raise RuntimeError(
                    f"Synthetic 1:1 requirement failed for {stratum}: "
                    f"real={real_count}, synthetic={synthetic_count}"
                )
        by_split["train"].extend(synthetic)
    if any(not by_split[split] for split in by_split):
        raise RuntimeError("Every real train/validation/test partition must contain images")
    train_dataset = ClassificationDataset(
        by_split["train"], config.target, _transforms(config.data.image_size, config.data.augment)
    )
    eval_transform = _transforms(config.data.image_size, False)
    validation_dataset = ClassificationDataset(by_split["validation"], config.target, eval_transform)
    test_dataset = ClassificationDataset(by_split["test"], config.target, eval_transform)
    generator = torch.Generator().manual_seed(config.experiment.seed)
    common = dict(batch_size=config.data.batch_size, num_workers=config.data.num_workers,
                  pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(validation_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )
