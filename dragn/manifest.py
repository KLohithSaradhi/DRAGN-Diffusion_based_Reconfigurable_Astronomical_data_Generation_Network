"""Validation and hashing for leakage-safe dataset manifests."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image


REAL_SPLITS = {"train", "validation", "test"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_pixel_sha256(path: Path) -> str:
    with Image.open(path) as image:
        normalized = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        return hashlib.sha256(normalized.tobytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    return rows


def validate_real_manifest(path: Path) -> list[dict[str, str]]:
    """Reject path, content, or physical-object groups spanning real-data splits."""
    rows = read_rows(path)
    required = {
        "path", "instrument", "class_name", "split", "source",
        "group_id", "content_sha256", "pixel_sha256",
    }
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(
            "Leakage-safe manifest is missing columns: " + ", ".join(sorted(missing_columns))
        )
    seen_paths: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    content_splits: dict[str, set[str]] = {}
    pixel_splits: dict[str, set[str]] = {}
    manifest_parent = path.expanduser().parent
    for row_number, row in enumerate(rows, start=2):
        if row["source"] != "real":
            raise ValueError(f"Real manifest row {row_number} has source={row['source']!r}")
        if row["split"] not in REAL_SPLITS:
            raise ValueError(f"Real manifest row {row_number} has invalid split={row['split']!r}")
        for field in ("path", "instrument", "class_name", "group_id", "content_sha256", "pixel_sha256"):
            if not row[field]:
                raise ValueError(f"Real manifest row {row_number} has an empty {field}")
        row_path = Path(row["path"]).expanduser()
        if not row_path.is_absolute():
            row_path = manifest_parent / row_path
        resolved = str(row_path.resolve())
        if resolved in seen_paths:
            raise ValueError(f"Real manifest repeats image path: {resolved}")
        seen_paths.add(resolved)
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
        content_splits.setdefault(row["content_sha256"], set()).add(row["split"])
        pixel_splits.setdefault(row["pixel_sha256"], set()).add(row["split"])
    for label, mapping in (
        ("group_id", group_splits),
        ("content_sha256", content_splits),
        ("pixel_sha256", pixel_splits),
    ):
        leaked = {key: value for key, value in mapping.items() if len(value) > 1}
        if leaked:
            example, splits = next(iter(leaked.items()))
            raise ValueError(
                f"Leakage detected: {label}={example!r} occurs across splits {sorted(splits)}"
            )
    for row in rows:
        image_path = Path(row["path"]).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Real manifest references a missing image: {image_path}")
        if file_sha256(image_path) != row["content_sha256"]:
            raise ValueError(f"Real image content changed after splitting: {image_path}")
        if normalized_pixel_sha256(image_path) != row["pixel_sha256"]:
            raise ValueError(f"Real image pixels changed after splitting: {image_path}")
    return rows


def validate_synthetic_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_rows(path)
    required = {"path", "instrument", "class_name", "split", "source"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(
            "Synthetic manifest is missing columns: " + ", ".join(sorted(missing_columns))
        )
    seen_paths = set()
    manifest_parent = path.expanduser().parent
    for row_number, row in enumerate(rows, start=2):
        if row["source"] != "synthetic" or row["split"] != "train":
            raise ValueError(
                f"Synthetic manifest row {row_number} must have source=synthetic and split=train"
            )
        row_path = Path(row["path"]).expanduser()
        if not row_path.is_absolute():
            row_path = manifest_parent / row_path
        resolved = str(row_path.resolve())
        if resolved in seen_paths:
            raise ValueError(f"Synthetic manifest repeats image path: {resolved}")
        if not Path(resolved).is_file():
            raise FileNotFoundError(f"Synthetic manifest references a missing image: {resolved}")
        seen_paths.add(resolved)
    return rows
