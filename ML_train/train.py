import argparse
import yaml
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW

# ensure local ML_train imports work when run from repo root
sys.path.append(str(Path(__file__).resolve().parent))
from data import DataFactory
from model import build_model


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        # print(targets)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / total
    acc = correct / total
    return epoch_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / total
    acc = correct / total

    try:
        from sklearn.metrics import f1_score
        import torch as _t
        preds = _t.cat(all_preds).numpy()
        targets = _t.cat(all_targets).numpy()
        f1 = f1_score(targets, preds, average='macro')
    except Exception:
        f1 = None

    return epoch_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    out_dir = Path(cfg.get('output_dir', './runs')) / cfg.get('run_name', f"run_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, num_classes, class_names = DataFactory.create_loaders(cfg)

    # Model
    model = build_model(num_classes, cfg.get('model', {}))
    device = get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    lr = float(cfg.get('training', {}).get('lr', 1e-4))
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    epochs = int(cfg.get('training', {}).get('epochs', 10))

    # Optional WandB
    use_wandb = cfg.get('logging', {}).get('wandb', False)
    if use_wandb:
        try:
            import wandb
            
            # Explicitly verifies the local credentials set by `wandb login`
            wandb.login() 
            
            wandb.init(
                project=cfg.get('logging', {}).get('project', 'ml_train'), 
                name=cfg.get('run_name')
            )
            wandb.config.update(cfg)
            
            # Standardized logging lambda
            log = lambda k, v, step=None: wandb.log({k: v}, step=step) if step is not None else wandb.log({k: v})
            
        except ImportError:
            print("Warning: 'wandb' is enabled in config but not installed. Run `pip install wandb`. Falling back to no logging.")
            use_wandb = False
            log = lambda *a, **k: None
        except Exception as e:
            print(f"Warning: Failed to initialize wandb ({e}). Falling back to no logging.")
            use_wandb = False
            log = lambda *a, **k: None
    else:
        log = lambda *a, **k: None

    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs}  Train loss: {train_loss:.4f} acc: {train_acc:.4f}  Val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1}")

        log('train/loss', train_loss, step=epoch)
        log('train/acc', train_acc, step=epoch)
        log('val/loss', val_loss, step=epoch)
        log('val/acc', val_acc, step=epoch)
        if val_f1 is not None:
            log('val/f1', val_f1, step=epoch)

        # Save best by val F1 (or val acc if F1 not available)
        metric = val_f1 if val_f1 is not None else val_acc
        if metric is not None and metric > best_val_f1:
            best_val_f1 = metric
            ckpt_path = out_dir / 'best_model.pth'
            torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'class_names': class_names}, ckpt_path)

    # final save
    torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'class_names': class_names}, out_dir / 'final_model.pth')
    print(f"Training complete. Artifacts saved to {out_dir}")


if __name__ == '__main__':
    main()