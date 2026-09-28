"""Manifest-driven real and synthetic image loading."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dragn.manifest import (
    file_sha256,
    normalized_pixel_sha256,
    validate_real_manifest,
    validate_synthetic_manifest,
)

from .config import ClassificationConfig


INSTRUMENTS = ("SDSS", "SUBARU")
OBJECTS = ("lens", "spiral", "ring", "companion", "smooth")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    instrument: str
    class_name: str
    split: str
    source: str
    group_id: str = ""
    content_sha256: str = ""
    pixel_sha256: str = ""


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
        raise RuntimeError(f"No supported SDSS/SUBARU object images found under {root}")
    return records


def write_split_manifest(
    root: Path,
    output: Path,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    group_map: Path | None = None,
    group_id_mode: str = "stem",
) -> Path:
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and sum to less than one")
    if group_id_mode not in {"stem", "path"}:
        raise ValueError("group_id_mode must be 'stem' or 'path'")
    records = discover_real_images(root)
    supplied_groups: dict[str, str] = {}
    if group_map is not None:
        with group_map.expanduser().open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                supplied_path = Path(row["path"]).expanduser()
                if not supplied_path.is_absolute():
                    supplied_path = group_map.parent / supplied_path
                supplied_groups[str(supplied_path.resolve())] = row["group_id"]
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    group_owner: dict[str, int] = {}
    content_owner: dict[str, int] = {}
    pixel_owner: dict[str, int] = {}
    metadata = []
    for index, record in enumerate(records):
        resolved = str(record.path.resolve())
        if group_map is not None:
            if resolved not in supplied_groups:
                raise ValueError(f"Group map has no group_id for {resolved}")
            preliminary_group = supplied_groups[resolved]
        elif group_id_mode == "stem":
            preliminary_group = record.path.stem
        else:
            preliminary_group = str(record.path.resolve())
        content_hash = file_sha256(record.path)
        pixel_hash = normalized_pixel_sha256(record.path)
        for key, owners in (
            (preliminary_group, group_owner),
            (content_hash, content_owner),
            (pixel_hash, pixel_owner),
        ):
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
        metadata.append((preliminary_group, content_hash, pixel_hash))

    members: dict[int, list[int]] = {}
    for index in range(len(records)):
        members.setdefault(find(index), []).append(index)
    stable_group_ids = {}
    for root_index, indices in members.items():
        labels = sorted({metadata[index][0] for index in indices})
        stable_group_ids[root_index] = hashlib.sha256(
            "\n".join(labels).encode("utf-8")
        ).hexdigest()[:24]

    strata = sorted({(record.instrument, record.class_name) for record in records})
    target_counts = {}
    for stratum in strata:
        count = sum((record.instrument, record.class_name) == stratum for record in records)
        target_counts[stratum] = {
            "test": max(1, round(count * test_fraction)),
            "validation": max(1, round(count * validation_fraction)),
        }
    best_assignment = None
    best_score = float("inf")
    for attempt in range(512):
        assignment = {}
        for root_index, group_id in stable_group_ids.items():
            digest = hashlib.sha256(
                f"{seed}:{attempt}:{group_id}".encode("utf-8")
            ).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
            assignment[root_index] = (
                "test" if value < test_fraction
                else "validation" if value < test_fraction + validation_fraction
                else "train"
            )
        counts = {(stratum, split): 0 for stratum in strata for split in ("train", "validation", "test")}
        for index, record in enumerate(records):
            counts[((record.instrument, record.class_name), assignment[find(index)])] += 1
        if any(counts[(stratum, split)] == 0 for stratum in strata for split in ("train", "validation", "test")):
            continue
        score = sum(
            abs(counts[(stratum, split)] - target_counts[stratum][split])
            for stratum in strata for split in ("test", "validation")
        )
        if score < best_score:
            best_assignment, best_score = assignment, score
    if best_assignment is None:
        raise RuntimeError(
            "Could not create non-empty group-isolated train/validation/test partitions; "
            "provide more independent groups or a corrected group map"
        )
    rows = []
    for index, record in enumerate(records):
        preliminary_group, content_hash, pixel_hash = metadata[index]
        rows.append(ImageRecord(
            record.path, record.instrument, record.class_name,
            best_assignment[find(index)], "real", stable_group_ids[find(index)],
            content_hash, pixel_hash,
        ))
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "path", "instrument", "class_name", "split", "source",
            "group_id", "content_sha256", "pixel_sha256",
        ])
        writer.writeheader()
        for row in sorted(rows, key=lambda value: str(value.path)):
            writer.writerow(row.__dict__)
    validate_real_manifest(output)
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
                row.get("group_id", ""), row.get("content_sha256", ""), row.get("pixel_sha256", ""),
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
    validate_real_manifest(config.data.manifest)
    real = read_manifest(config.data.manifest, "real")
    if config.data.instrument_filter:
        real = [row for row in real if row.instrument == config.data.instrument_filter]
    by_split = {split: [row for row in real if row.split == split] for split in ("train", "validation", "test")}
    expected_labels = set(OBJECTS if config.target == "object" else INSTRUMENTS)
    for split, records in by_split.items():
        observed_labels = {
            row.class_name if config.target == "object" else row.instrument
            for row in records
        }
        if observed_labels != expected_labels:
            raise RuntimeError(
                f"Real {split} split labels do not match the benchmark; "
                f"expected={sorted(expected_labels)}, observed={sorted(observed_labels)}"
            )
    if config.data.synthetic_manifest is not None:
        validate_synthetic_manifest(config.data.synthetic_manifest)
        if config.data.require_synthetic_provenance:
            provenance_path = config.data.synthetic_manifest.parent / "synthetic_provenance.json"
            if not provenance_path.is_file():
                raise FileNotFoundError(
                    f"Leakage-safe augmented training requires provenance: {provenance_path}"
                )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            expected_real_hash = file_sha256(config.data.manifest)
            expected_synthetic_hash = file_sha256(config.data.synthetic_manifest)
            if provenance.get("real_manifest_sha256") != expected_real_hash:
                raise ValueError("Synthetic data provenance references a different real split manifest")
            if provenance.get("synthetic_manifest_sha256") != expected_synthetic_hash:
                raise ValueError("Synthetic manifest has changed since its provenance was recorded")
            if not provenance.get("variants") or any(
                not variant.get("checkpoint_sha256") for variant in provenance["variants"]
            ):
                raise ValueError("Synthetic provenance is missing generator checkpoint hashes")
        synthetic = read_manifest(config.data.synthetic_manifest, "synthetic")
        if config.data.instrument_filter:
            synthetic = [row for row in synthetic if row.instrument == config.data.instrument_filter]
        synthetic = [row for row in synthetic if row.split == "train"]
        held_out_content = {
            row.content_sha256 for row in real if row.split != "train"
        }
        held_out_pixels = {
            row.pixel_sha256 for row in real if row.split != "train"
        }
        for row in synthetic:
            if file_sha256(row.path) in held_out_content:
                raise ValueError(f"Synthetic image duplicates held-out real file content: {row.path}")
            if normalized_pixel_sha256(row.path) in held_out_pixels:
                raise ValueError(f"Synthetic image duplicates held-out normalized pixels: {row.path}")
        real_training = by_split["train"]
        strata = {(row.instrument, row.class_name) for row in real_training}
        synthetic_strata = {(row.instrument, row.class_name) for row in synthetic}
        if synthetic_strata != strata:
            raise RuntimeError(
                f"Synthetic strata do not match real training strata; "
                f"expected={sorted(strata)}, observed={sorted(synthetic_strata)}"
            )
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
