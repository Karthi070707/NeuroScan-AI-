"""
Evaluate — Brain Tumor Classifier on Test Set
=============================================
Produces:
  - Overall accuracy, per-class accuracy
  - Confusion matrix heatmap (PNG)
  - ROC curves (one-vs-rest, PNG)
  - Full classification report (JSON + console)

Usage:
    python evaluate.py
    python evaluate.py --checkpoint outputs/checkpoints/best_model.pth
    python evaluate.py --checkpoint best_model.pth --save-dir outputs/eval
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config
from model import BrainTumorClassifier
from dataset import BrainMRIDataset, get_val_transforms


# ── Inference on full test set ────────────────────────────────────────────────

@torch.no_grad()
def run_test(model, test_loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for imgs, labels, _ in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits, _ = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return (np.concatenate(all_probs, axis=0),
            np.array(all_preds),
            np.array(all_labels))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(probs, preds, labels, class_names):
    n = len(labels)
    overall_acc = (preds == labels).mean()

    per_class = {}
    for c, name in enumerate(class_names):
        mask = labels == c
        if mask.sum() == 0:
            per_class[name] = {"accuracy": None, "precision": None,
                                "recall": None, "f1": None, "support": 0}
            continue
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        acc  = (preds[mask] == labels[mask]).mean()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        per_class[name] = {
            "accuracy":  float(acc),
            "precision": float(prec),
            "recall":    float(rec),
            "f1":        float(f1),
            "support":   int(mask.sum()),
        }

    # Macro averages
    vals = [v for v in per_class.values() if v["f1"] is not None]
    macro_f1   = np.mean([v["f1"]   for v in vals])
    macro_prec = np.mean([v["precision"] for v in vals])
    macro_rec  = np.mean([v["recall"]    for v in vals])

    return {
        "overall_accuracy"  : float(overall_acc),
        "macro_precision"   : float(macro_prec),
        "macro_recall"      : float(macro_rec),
        "macro_f1"          : float(macro_f1),
        "per_class"         : per_class,
        "n_samples"         : n,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, class_names, save_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ImportError:
        print("  ⚠️  seaborn/sklearn missing — skipping confusion matrix plot")
        return

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#1e293b")

    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="#334155",
        cbar_kws={"label": "% of true class"},
    )
    ax.set_xlabel("Predicted", color="#e2e8f0", fontsize=12)
    ax.set_ylabel("True",      color="#e2e8f0", fontsize=12)
    ax.set_title("Confusion Matrix (% of true class)",
                 color="#f1f5f9", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#94a3b8")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0f1e")
    print(f"  💾 Confusion matrix → {save_path}")
    plt.close()


def plot_roc_curves(labels, probs, class_names, save_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
    except ImportError:
        print("  ⚠️  sklearn/matplotlib missing — skipping ROC plot")
        return

    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#1e293b")

    colors = list(Config.CLASS_COLORS.values())
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, c], probs[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[c], lw=2,
                label=f"{class_names[c]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="#e2e8f0")
    ax.set_ylabel("True Positive Rate",  color="#e2e8f0")
    ax.set_title("ROC Curves (One-vs-Rest)", color="#f1f5f9",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", facecolor="#1e293b", labelcolor="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0f1e")
    print(f"  💾 ROC curves → {save_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classifier")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS)
    parser.add_argument("--save-dir",   default=None,
                        help="Directory for plots/JSON (default: outputs/eval)")
    args = parser.parse_args()

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.checkpoint or str(Config.CHECKPOINT_DIR / "best_model.pth")
    save_dir  = Path(args.save_dir) if args.save_dir else Config.OUTPUT_DIR / "eval"
    save_dir.mkdir(parents=True, exist_ok=True)

    if not Path(ckpt_path).exists():
        print(f"❌  Checkpoint not found: {ckpt_path}"); sys.exit(1)

    # Load model
    model = BrainTumorClassifier(num_classes=Config.NUM_CLASSES,
                                  pretrained=False, use_srm=Config.USE_SRM).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"\n✅ Model loaded: {ckpt_path}")

    # Test DataLoader
    test_ds = BrainMRIDataset(Config.TEST_DIR, transform=get_val_transforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    print(f"   Test samples: {len(test_ds)}")

    # Run inference
    print("\n⏳ Running inference on test set…")
    probs, preds, labels = run_test(model, test_loader, device)

    # Metrics
    metrics = compute_metrics(probs, preds, labels, Config.CLASS_NAMES)

    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  Overall Accuracy : {metrics['overall_accuracy']*100:.2f}%")
    print(f"  Macro Precision  : {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall     : {metrics['macro_recall']*100:.2f}%")
    print(f"  Macro F1         : {metrics['macro_f1']*100:.2f}%")
    print(f"\n  Per-Class Results:")
    for name, m in metrics["per_class"].items():
        if m["accuracy"] is None:
            print(f"    {name:<14} — no samples")
        else:
            print(f"    {name:<14}  acc={m['accuracy']*100:.1f}%  "
                  f"p={m['precision']*100:.1f}%  r={m['recall']*100:.1f}%  "
                  f"f1={m['f1']*100:.1f}%  n={m['support']}")
    print(f"{'='*55}\n")

    # Save JSON report
    report_path = save_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  💾 Report → {report_path}")

    # Plots
    plot_confusion_matrix(labels, preds, Config.CLASS_NAMES,
                          str(save_dir / "confusion_matrix.png"))
    plot_roc_curves(labels, probs, Config.CLASS_NAMES,
                    str(save_dir / "roc_curves.png"))

    print(f"\n  All outputs → {save_dir}\n")


if __name__ == "__main__":
    main()
