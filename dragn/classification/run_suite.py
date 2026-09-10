"""Run all classification YAMLs for shared seeds and aggregate results."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from .config import load_classification_config
from .train import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, default=Path("experiments/classification"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    args = parser.parse_args()
    rows = []
    for config_path in sorted(args.experiments.glob("*/config.yaml")):
        config = load_classification_config(config_path)
        for seed in args.seeds:
            seeded = config.model_copy(update={
                "experiment": config.experiment.model_copy(update={"seed": seed})
            })
            metrics_path = train_classifier(seeded)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({"experiment": config.experiment.name, "seed": seed,
                         "target": config.target, "augmented": config.data.synthetic_manifest is not None,
                         **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")}})
    output = args.experiments / "results.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["experiment"], []).append(row)
    summary_rows = []
    for experiment, values in sorted(grouped.items()):
        result = {"experiment": experiment, "runs": len(values)}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc"):
            samples = [float(row[metric]) for row in values if row[metric] is not None]
            result[f"{metric}_mean"] = statistics.mean(samples) if samples else None
            result[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        summary_rows.append(result)
    baselines = {
        "object_sdss_dragn": "object_sdss_real",
        "object_subaru_dragn": "object_subaru_real",
        "instrument_dragn": "instrument_real",
    }
    indexed = {row["experiment"]: row for row in summary_rows}
    for augmented, baseline in baselines.items():
        if augmented in indexed and baseline in indexed:
            indexed[augmented]["macro_f1_delta_vs_real"] = (
                indexed[augmented]["macro_f1_mean"] - indexed[baseline]["macro_f1_mean"]
            )
    paired_rows = []
    row_index = {(row["experiment"], row["seed"]): row for row in rows}
    for augmented, baseline in baselines.items():
        deltas = []
        for seed in args.seeds:
            if (augmented, seed) not in row_index or (baseline, seed) not in row_index:
                continue
            delta = row_index[(augmented, seed)]["macro_f1"] - row_index[(baseline, seed)]["macro_f1"]
            deltas.append(delta)
            paired_rows.append({"augmented": augmented, "baseline": baseline, "seed": seed,
                                "macro_f1_delta": delta})
        if deltas:
            indexed[augmented]["paired_delta_std"] = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    summary = args.experiments / "summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row})
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    paired = args.experiments / "paired_deltas.csv"
    with paired.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["augmented", "baseline", "seed", "macro_f1_delta"])
        writer.writeheader()
        writer.writerows(paired_rows)
    print(f"results={output} summary={summary} paired={paired}", flush=True)


if __name__ == "__main__":
    main()
