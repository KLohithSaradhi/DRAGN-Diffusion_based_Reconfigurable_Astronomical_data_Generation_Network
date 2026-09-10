"""Train and test one leakage-safe ResNet18 benchmark experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import ClassificationConfig, load_classification_config
from .data import create_loaders
from .metrics import classification_metrics, save_metrics
from .model import build_classifier
from dragn.training.common import file_sha256


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _run_epoch(model, loader, criterion, device, precision, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets, predictions, probabilities = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            probability = logits.softmax(dim=1)[:, 1]
            total_loss += loss.item() * labels.numel()
            targets.append(labels.detach().cpu())
            predictions.append(logits.argmax(dim=1).detach().cpu())
            probabilities.append(probability.detach().float().cpu())
    target = torch.cat(targets)
    prediction = torch.cat(predictions)
    probability = torch.cat(probabilities)
    metrics = classification_metrics(target, prediction, probability)
    metrics["loss"] = total_loss / len(target)
    return metrics, target, prediction, probability


def train_classifier(config: ClassificationConfig) -> Path:
    _seed_everything(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.experiment.output_dir / config.experiment.name / f"seed_{config.experiment.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader = create_loaders(config)
    model = build_classifier(config.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr,
                                  weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.training.epochs)
    scaler = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda" and config.training.precision == "fp16"
    )
    resolved = config.model_dump(mode="json")
    (output_dir / "config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    shutil.copy2(config.data.manifest, output_dir / "real_manifest.csv")
    input_provenance = {"real_manifest_sha256": file_sha256(config.data.manifest)}
    if config.data.synthetic_manifest is not None:
        shutil.copy2(config.data.synthetic_manifest, output_dir / "synthetic_manifest.csv")
        input_provenance["synthetic_manifest_sha256"] = file_sha256(config.data.synthetic_manifest)
        synthetic_provenance = config.data.synthetic_manifest.parent / "synthetic_provenance.json"
        if synthetic_provenance.is_file():
            shutil.copy2(synthetic_provenance, output_dir / "synthetic_provenance.json")
    (output_dir / "input_provenance.json").write_text(
        json.dumps(input_provenance, indent=2), encoding="utf-8"
    )
    wandb_run = None
    if config.logging.wandb:
        import wandb
        wandb_run = wandb.init(project=config.logging.project, name=f"{config.experiment.name}-seed-{config.experiment.seed}", config=resolved)
    best_f1, stale_epochs = -1.0, 0
    history = []
    best_path = output_dir / "best.pt"
    for epoch in range(1, config.training.epochs + 1):
        train_metrics, *_ = _run_epoch(model, train_loader, criterion, device,
                                        config.training.precision, optimizer, scaler)
        validation_metrics, *_ = _run_epoch(model, validation_loader, criterion, device,
                                             config.training.precision)
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{key}": value for key, value in train_metrics.items() if isinstance(value, (int, float))},
               **{f"validation_{key}": value for key, value in validation_metrics.items() if isinstance(value, (int, float))}}
        history.append(row)
        print(f"epoch={epoch} train_macro_f1={train_metrics['macro_f1']:.4f} validation_macro_f1={validation_metrics['macro_f1']:.4f}", flush=True)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": resolved, "epoch": epoch,
                        "validation_metrics": validation_metrics}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.training.early_stopping_patience:
                break
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    test_metrics, targets, predictions, probabilities = _run_epoch(
        model, test_loader, criterion, device, config.training.precision
    )
    class_names = list(test_loader.dataset.classes)
    test_metrics["class_names"] = class_names
    test_metrics["best_epoch"] = checkpoint["epoch"]
    test_metrics["real_train_images"] = sum(row.source == "real" for row in train_loader.dataset.records)
    test_metrics["synthetic_train_images"] = sum(row.source != "real" for row in train_loader.dataset.records)
    save_metrics(test_metrics, output_dir / "test_metrics.json")
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for name, row in zip(class_names, test_metrics["confusion_matrix"]):
            writer.writerow([name, *row])
    with (output_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "target", "prediction", "positive_probability"])
        for record, target, prediction, probability in zip(
            test_loader.dataset.records, targets.tolist(), predictions.tolist(), probabilities.tolist()
        ):
            writer.writerow([record.path, class_names[target], class_names[prediction], probability])
    if wandb_run is not None:
        wandb_run.log({f"test/{key}": value for key, value in test_metrics.items() if isinstance(value, (int, float))})
        wandb_run.finish()
    print(f"test_macro_f1={test_metrics['macro_f1']:.4f} artifacts={output_dir}", flush=True)
    return output_dir / "test_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_classification_config(args.config)
    if args.seed is not None:
        config = config.model_copy(update={
            "experiment": config.experiment.model_copy(update={"seed": args.seed})
        })
    train_classifier(config)


if __name__ == "__main__":
    main()
