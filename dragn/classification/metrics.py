"""Dependency-light binary classification metrics and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def classification_metrics(targets: torch.Tensor, predictions: torch.Tensor, probabilities: torch.Tensor) -> dict:
    matrix = torch.zeros((2, 2), dtype=torch.long)
    for target, prediction in zip(targets.long(), predictions.long()):
        matrix[target, prediction] += 1
    per_class = []
    for index in range(2):
        tp = matrix[index, index].item()
        fp = matrix[:, index].sum().item() - tp
        fn = matrix[index, :].sum().item() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum())})
    accuracy = matrix.diag().sum().item() / max(matrix.sum().item(), 1)
    positive_count = int((targets == 1).sum())
    negative_count = int((targets == 0).sum())
    roc_auc = None
    if positive_count and negative_count:
        order = torch.argsort(probabilities)
        sorted_scores = probabilities[order]
        ranks = torch.arange(1, len(probabilities) + 1, dtype=torch.float64)
        start = 0
        while start < len(sorted_scores):
            end = start + 1
            while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[start:end] = ranks[start:end].mean()
            start = end
        target_in_rank_order = targets[order]
        positive_rank_sum = ranks[target_in_rank_order == 1].sum()
        roc_auc = float(
            (positive_rank_sum - positive_count * (positive_count + 1) / 2)
            / (positive_count * negative_count)
        )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": sum(item["recall"] for item in per_class) / 2,
        "macro_f1": sum(item["f1"] for item in per_class) / 2,
        "roc_auc": roc_auc,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
