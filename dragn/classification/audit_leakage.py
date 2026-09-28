"""Audit real splits and optional synthetic provenance before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dragn.manifest import file_sha256, validate_real_manifest, validate_synthetic_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-manifest", required=True, type=Path)
    parser.add_argument("--synthetic-manifest", type=Path)
    args = parser.parse_args()
    real_rows = validate_real_manifest(args.real_manifest)
    report = {
        "real_manifest": str(args.real_manifest),
        "real_manifest_sha256": file_sha256(args.real_manifest),
        "real_counts": dict(sorted(Counter(
            f"{row['split']}:{row['instrument']}:{row['class_name']}" for row in real_rows
        ).items())),
        "unique_groups": len({row["group_id"] for row in real_rows}),
        "status": "pass",
    }
    if args.synthetic_manifest is not None:
        synthetic_rows = validate_synthetic_manifest(args.synthetic_manifest)
        provenance_path = args.synthetic_manifest.parent / "synthetic_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Synthetic provenance does not exist: {provenance_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("real_manifest_sha256") != report["real_manifest_sha256"]:
            raise ValueError("Synthetic provenance was generated from a different real manifest")
        synthetic_hash = file_sha256(args.synthetic_manifest)
        if provenance.get("synthetic_manifest_sha256") != synthetic_hash:
            raise ValueError("Synthetic manifest hash does not match its provenance")
        report.update({
            "synthetic_manifest": str(args.synthetic_manifest),
            "synthetic_manifest_sha256": synthetic_hash,
            "synthetic_counts": dict(sorted(Counter(
                f"{row['instrument']}:{row['class_name']}" for row in synthetic_rows
            ).items())),
        })
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
