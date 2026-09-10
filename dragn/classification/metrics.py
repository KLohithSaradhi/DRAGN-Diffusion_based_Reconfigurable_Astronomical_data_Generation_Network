"""Dependency-light binary classification metrics and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _binary_auc(binary_targets: torch.Tensor, scores: torch.Tensor) -> float | None:
    positive_count = int((binary_targets == 1).sum())
    negative_count = int((binary_targets == 0).sum())
    if not positive_count or not negative_count:
        return None
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    positive_rank_sum = ranks[binary_targets[order] == 1].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def classification_metrics(targets: torch.Tensor, predictions: torch.Tensor, probabilities: torch.Tensor) -> dict:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [samples, classes]")
    class_count = probabilities.shape[1]
    matrix = torch.zeros((class_count, class_count), dtype=torch.long)
    for target, prediction in zip(targets.long(), predictions.long()):
        matrix[target, prediction] += 1
    per_class = []
    for index in range(class_count):
        tp = matrix[index, index].item()
        fp = matrix[:, index].sum().item() - tp
        fn = matrix[index, :].sum().item() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index].sum())})
    accuracy = matrix.diag().sum().item() / max(matrix.sum().item(), 1)
    class_auc = [
        _binary_auc((targets == index).long(), probabilities[:, index])
        for index in range(class_count)
    ]
    valid_auc = [value for value in class_auc if value is not None]
    roc_auc = sum(valid_auc) / len(valid_auc) if valid_auc else None
    return {
        "accuracy": accuracy,
        "balanced_accuracy": sum(item["recall"] for item in per_class) / class_count,
        "macro_f1": sum(item["f1"] for item in per_class) / class_count,
        "roc_auc": roc_auc,
        "per_class_roc_auc": class_auc,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
