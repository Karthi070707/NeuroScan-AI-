"""
Training Script — EfficientNet-B4 Brain Tumor Classifier
=========================================================
Two-phase fine-tuning strategy (adapted from deepfake detector):

  Phase 1 (epochs 1-WARMUP_EPOCHS):
      Backbone frozen → only classification head trains
      LR = Config.WARMUP_LR (3e-4)

  Phase 2 (epochs WARMUP_EPOCHS+1 to NUM_EPOCHS):
      Full network trains with differential LR:
        - Head   : Config.LEARNING_RATE      (3e-5)
        - Backbone: LEARNING_RATE × 0.1      (3e-6)
      Scheduler: CosineAnnealingLR

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 24 --no-mixup
    python train.py --resume outputs/checkpoints/last_model.pth
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from dataset import build_dataloaders, mixup_data, mixup_criterion
from model import BrainTumorClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Logger: writes to stdout AND a log file simultaneously
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    """Tees all print() output to both terminal and a log file."""
    def __init__(self, log_path: Path):
        self._terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, msg):
        self._terminal.write(msg)
        self._terminal.flush()
        self._log.write(msg)
        self._log.flush()

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = Config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def per_class_accuracy(
    outputs : torch.Tensor,
    labels  : torch.Tensor,
    n_classes: int = Config.NUM_CLASSES,
) -> dict:
    preds = outputs.argmax(dim=1)
    result = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            result[Config.CLASS_NAMES[c]] = None
        else:
            result[Config.CLASS_NAMES[c]] = (preds[mask] == labels[mask]).float().mean().item()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model       : BrainTumorClassifier,
    loader      : torch.utils.data.DataLoader,
    criterion   : nn.Module,
    optimizer   : torch.optim.Optimizer,
    device      : str,
    use_mixup   : bool = True,
    mixup_alpha : float = Config.MIXUP_ALPHA,
    scaler      : torch.cuda.amp.GradScaler = None,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for imgs, labels, _ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp augmentation
        if use_mixup and mixup_alpha > 0:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device == "cuda" and scaler is not None)):
            logits, _ = model(imgs)
            if use_mixup and mixup_alpha > 0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Accuracy on un-mixed labels
        acc = accuracy(logits.detach(), labels)
        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(
    model     : BrainTumorClassifier,
    loader    : torch.utils.data.DataLoader,
    criterion : nn.Module,
    device    : str,
) -> tuple[float, float, dict]:
    """Evaluate on loader. Returns (avg_loss, accuracy, per_class_acc)."""
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []

    for imgs, labels, _ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss   = total_loss / len(loader)
    avg_acc    = accuracy(all_logits, all_labels)
    per_cls    = per_class_accuracy(all_logits, all_labels)
    return avg_loss, avg_acc, per_cls


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model      : BrainTumorClassifier,
    optimizer  : torch.optim.Optimizer,
    epoch      : int,
    metrics    : dict,
    path       : Path,
    is_best    : bool = False,
):
    state = {
        "epoch"             : epoch,
        "model_state_dict"  : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics"           : metrics,
        "config"            : {
            "num_classes" : Config.NUM_CLASSES,
            "class_names" : Config.CLASS_NAMES,
            "image_size"  : Config.IMAGE_SIZE,
            "use_srm"     : Config.USE_SRM,
        },
    }
    torch.save(state, path)
    if is_best:
        best_path = path.parent / "best_model.pth"
        torch.save(state, best_path)
        print(f"  [BEST] Model saved -> {best_path}")


def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[OK] Resumed from checkpoint: {path}  (epoch {ckpt['epoch']})")
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(Config.RANDOM_SEED)

    # ── Logger (writes to terminal + log file) ────────────────────────────────
    log_path = Config.OUTPUT_DIR / "train_log.txt"
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(log_path)
    print(f"Training log -> {log_path}")
    print()

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Brain Tumor Classifier — EfficientNet-B4")
    print(f"{'='*60}")
    print(f"  Device     : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"  Image size : {Config.IMAGE_SIZE}×{Config.IMAGE_SIZE}")
    print(f"  Classes    : {Config.CLASS_NAMES}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Epochs     : {args.epochs} (warm-up: {Config.WARMUP_EPOCHS})")
    print(f"  MixUp      : {'on (α=' + str(Config.MIXUP_ALPHA) + ')' if args.use_mixup else 'off'}")
    print(f"{'='*60}\n")

    # ── Output directories ────────────────────────────────────────────────────
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    Config.VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BrainTumorClassifier(
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        use_srm=Config.USE_SRM,
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    # ── Mixed precision scaler ────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch  = 0
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    no_improve   = 0

    if args.resume and Path(args.resume).exists():
        optimizer_placeholder = torch.optim.AdamW(model.parameters())
        start_epoch, ckpt_metrics = load_checkpoint(
            Path(args.resume), model, optimizer_placeholder, device
        )
        # Recover history from metrics if available
        history = ckpt_metrics.get("history", history)
        best_val_acc = max(history.get("val_acc", [0])) if history.get("val_acc") else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Head-only warm-up
    # ─────────────────────────────────────────────────────────────────────────
    if start_epoch < Config.WARMUP_EPOCHS:
        print("[Phase 1] Head-only warm-up (backbone frozen)")
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            model.get_head_params(),
            lr=Config.WARMUP_LR,
            weight_decay=Config.WEIGHT_DECAY,
        )

        for epoch in range(start_epoch, Config.WARMUP_EPOCHS):
            ep_start = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                use_mixup=args.use_mixup, scaler=scaler,
            )
            vl_loss, vl_acc, per_cls = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - ep_start

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)

            is_best = vl_acc > best_val_acc
            if is_best:
                best_val_acc = vl_acc
                no_improve   = 0
            else:
                no_improve  += 1

            print(
                f"  Epoch [{epoch+1:02d}/{Config.WARMUP_EPOCHS}]  "
                f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
                f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f}  "
                f"{'[BEST]' if is_best else ''}  [{elapsed:.0f}s]"
            )
            print(f"    Per-class: " + " | ".join(
                f"{k}={v:.3f}" if v is not None else f"{k}=N/A"
                for k, v in per_cls.items()
            ))

            save_checkpoint(
                model, optimizer, epoch + 1,
                {"history": history, "best_val_acc": best_val_acc},
                Config.CHECKPOINT_DIR / "last_model.pth",
                is_best=is_best,
            )

        print(f"\n[OK] Warm-up complete. Best val acc: {best_val_acc:.4f}\n")
        start_epoch = Config.WARMUP_EPOCHS

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Full fine-tuning with differential LR
    # ─────────────────────────────────────────────────────────────────────────
    print("[Phase 2] Full fine-tuning (differential LR)")
    model.unfreeze_backbone()

    optimizer = torch.optim.AdamW([
        {"params": model.get_backbone_params(), "lr": Config.LEARNING_RATE * Config.BACKBONE_LR_MULT},
        {"params": model.get_head_params(),     "lr": Config.LEARNING_RATE},
    ], weight_decay=Config.WEIGHT_DECAY)

    remaining_epochs = args.epochs - start_epoch
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs,
        eta_min=Config.ETA_MIN,
    )

    no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        ep_start = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=args.use_mixup, scaler=scaler,
        )
        vl_loss, vl_acc, per_cls = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - ep_start

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        is_best = vl_acc > best_val_acc
        if is_best:
            best_val_acc = vl_acc
            no_improve   = 0
        else:
            no_improve  += 1

        current_lr = optimizer.param_groups[1]["lr"]
        print(
            f"  Epoch [{epoch+1:02d}/{args.epochs}]  "
            f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f}  |  "
            f"Val: loss={vl_loss:.4f} acc={vl_acc:.4f}  "
            f"LR={current_lr:.2e}  {'[BEST]' if is_best else f'[no-improve: {no_improve}/{Config.EARLY_STOP_PATIENCE}]'}  "
            f"[{elapsed:.0f}s]"
        )
        print(f"    Per-class: " + " | ".join(
            f"{k}={v:.3f}" if v is not None else f"{k}=N/A"
            for k, v in per_cls.items()
        ))

        save_checkpoint(
            model, optimizer, epoch + 1,
            {"history": history, "best_val_acc": best_val_acc},
            Config.CHECKPOINT_DIR / "last_model.pth",
            is_best=is_best,
        )

        # Early stopping
        if no_improve >= Config.EARLY_STOP_PATIENCE:
            print(f"\n[STOP] Early stopping at epoch {epoch+1}  "
                  f"(no improvement for {Config.EARLY_STOP_PATIENCE} epochs)")
            break

    # ─────────────────────────────────────────────────────────────────────────
    # Save training history
    # ─────────────────────────────────────────────────────────────────────────
    history_path = Config.LOG_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[OK] Training history saved -> {history_path}")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Best checkpoint   : {Config.CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"{'='*60}\n")
    if hasattr(sys.stdout, 'close'):
        sys.stdout.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B4 Brain Tumor Classifier"
    )
    parser.add_argument("--epochs",      type=int,   default=Config.NUM_EPOCHS,
                        help=f"Total epochs (default: {Config.NUM_EPOCHS})")
    parser.add_argument("--batch-size",  type=int,   default=Config.BATCH_SIZE,
                        help=f"Batch size (default: {Config.BATCH_SIZE})")
    parser.add_argument("--num-workers", type=int,   default=Config.NUM_WORKERS,
                        help=f"DataLoader workers (default: {Config.NUM_WORKERS})")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-mixup",    action="store_true",
                        help="Disable MixUp augmentation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.use_mixup = not args.no_mixup
    train(args)
