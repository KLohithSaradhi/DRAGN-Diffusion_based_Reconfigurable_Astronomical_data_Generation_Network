"""Create the one immutable real-data split used by the benchmark."""

import argparse
from pathlib import Path

from .data import write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = write_split_manifest(args.root, args.output, args.seed)
    print(f"manifest={path}", flush=True)


if __name__ == "__main__":
    main()
