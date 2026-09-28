"""Create the one immutable real-data split used by the benchmark."""

import argparse
from pathlib import Path

from .data import write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-map", type=Path,
        help="Optional CSV with path,group_id columns; use one group_id per physical object",
    )
    parser.add_argument(
        "--group-id-mode", choices=("stem", "path"), default="stem",
        help="Fallback grouping: shared filename stems stay in one split; path disables physical grouping",
    )
    args = parser.parse_args()
    path = write_split_manifest(
        args.root, args.output, args.seed,
        group_map=args.group_map, group_id_mode=args.group_id_mode,
    )
    print(f"manifest={path}", flush=True)


if __name__ == "__main__":
    main()
