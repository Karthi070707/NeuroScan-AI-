"""
Inference — EfficientNet-B4 Brain Tumor Classifier with Grad-CAM
================================================================
Usage:
    python inference.py --source mri.jpg                  # predict only
    python inference.py --source mri.jpg --visualise      # + Grad-CAM overlay
    python inference.py --source mri.jpg --all-classes    # 5-panel all-class CAM
    python inference.py --source Testing/ --output results.csv
    python inference.py --source mri.jpg --checkpoint path/to/best_model.pth
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import Config
from model import BrainTumorClassifier


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str) -> BrainTumorClassifier:
    model = BrainTumorClassifier(num_classes=Config.NUM_CLASSES,
                                  pretrained=False, use_srm=Config.USE_SRM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    best_acc = ckpt.get("metrics", {}).get("best_val_acc")
    print(f"\n✅ Loaded: {ckpt_path}  |  epoch={ckpt.get('epoch','?')}"
          + (f"  best_val_acc={best_acc:.4f}" if best_acc else ""))
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img: Image.Image, sz: int = Config.IMAGE_SIZE) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMG_MEAN, Config.IMG_STD),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)


# ── Core prediction ───────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, tensor: torch.Tensor, device: str):
    tensor = tensor.to(device)
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        logits, emb = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    return int(np.argmax(probs)), probs, emb.squeeze(0).cpu()


def predict_tta(model, img_pil: Image.Image, device: str):
    """Average original + horizontal flip predictions."""
    _, p1, _ = predict(model, preprocess(img_pil), device)
    _, p2, _ = predict(model, preprocess(img_pil.transpose(Image.FLIP_LEFT_RIGHT)), device)
    avg = [(a + b) / 2 for a, b in zip(p1, p2)]
    return int(np.argmax(avg)), avg


# ── Grad-CAM visualisation ────────────────────────────────────────────────────

def visualise_gradcam(model, img_pil: Image.Image, pred_class: int,
                      probs, save_path=None, device="cpu"):
    try:
        from gradcam_utils import BrainTumorGradCAM
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cam = BrainTumorGradCAM(model, device=device)
        overlay = cam.overlay(img_pil, target_class=pred_class)
        img_np  = np.array(img_pil.convert("RGB"))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor("#0a0f1e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1e293b"); ax.axis("off")
        ax1.imshow(img_np)
        ax1.set_title("Original MRI", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax2.imshow(overlay)
        ax2.set_title(
            f"Grad-CAM → {Config.CLASS_LABELS[pred_class]} ({max(probs)*100:.1f}%)",
            color=Config.CLASS_COLORS[pred_class], fontsize=12, fontweight="bold",
        )
        bar = "  ".join(f"{Config.CLASS_NAMES[i]}: {probs[i]*100:.1f}%" for i in range(4))
        fig.text(0.5, 0.01, bar, ha="center", color="#94a3b8", fontsize=9)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0f1e")
            print(f"  💾 Grad-CAM → {save_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"  ⚠️  Grad-CAM failed: {e}")


def visualise_all_classes(model, img_pil, pred_class, probs, save_path=None, device="cpu"):
    try:
        from gradcam_utils import BrainTumorGradCAM, generate_all_class_cams
        cam = BrainTumorGradCAM(model, device=device)
        generate_all_class_cams(cam, img_pil, probs, pred_class, save_path)
    except Exception as e:
        print(f"  ⚠️  All-class CAM failed: {e}")


# ── Single image ──────────────────────────────────────────────────────────────

def predict_image(model, img_path, device, use_tta=True,
                  visualise=False, all_classes=False, save_dir=None):
    img_pil = Image.open(img_path).convert("RGB")
    if use_tta:
        pred_class, probs = predict_tta(model, img_pil, device)
    else:
        pred_class, probs, _ = predict(model, preprocess(img_pil), device)

    label, conf = Config.CLASS_LABELS[pred_class], max(probs)
    print(f"\n  File: {Path(img_path).name}")
    print(f"  Prediction: {label}  ({conf*100:.2f}%)"
          + ("  ✅" if conf >= Config.CONFIDENCE_THRESHOLD else "  ⚠️ low confidence"))
    for i, name in enumerate(Config.CLASS_NAMES):
        bar = "█" * int(probs[i] * 30) + "░" * (30 - int(probs[i] * 30))
        print(f"    {name:<12} [{bar}] {probs[i]*100:5.1f}%" + (" ◀" if i == pred_class else ""))

    Config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
    stem    = Path(img_path).stem
    base    = Path(save_dir) if save_dir else Config.VIZ_DIR
    if visualise:
        visualise_gradcam(model, img_pil, pred_class, probs,
                          str(base / f"{stem}_gradcam.png"), device)
    if all_classes:
        visualise_all_classes(model, img_pil, pred_class, probs,
                              str(base / f"{stem}_all_classes.png"), device)
    return pred_class, probs, label


# ── Batch directory ───────────────────────────────────────────────────────────

def predict_directory(model, dir_path, device, use_tta=True):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    files = sorted(p for p in Path(dir_path).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        print(f"❌  No images in: {dir_path}"); return []

    print(f"\n  Batch: {len(files)} images\n")
    results = []
    for fp in files:
        img_pil = Image.open(fp).convert("RGB")
        if use_tta:
            pc, probs = predict_tta(model, img_pil, device)
        else:
            pc, probs, _ = predict(model, preprocess(img_pil), device)
        label, conf = Config.CLASS_LABELS[pc], max(probs)
        print(f"  {fp.name:<40} {label:<18} {conf*100:6.2f}%")
        results.append((str(fp), label, Config.CLASS_NAMES[pc], conf, *probs))
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Classifier — Inference")
    parser.add_argument("--source",      required=True)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--output",      default=None, help="Save CSV for batch")
    parser.add_argument("--visualise",   action="store_true")
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--no-tta",      action="store_true")
    parser.add_argument("--save-dir",    default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = args.checkpoint or str(Config.CHECKPOINT_DIR / "best_model.pth")
    if not Path(ckpt).exists():
        print(f"❌  Checkpoint not found: {ckpt}\n    Run:  python train.py"); sys.exit(1)

    model  = load_model(ckpt, device)
    source = Path(args.source)
    use_tta = not args.no_tta

    if source.is_file():
        pc, probs, label = predict_image(
            model, str(source), device, use_tta,
            args.visualise, args.all_classes, args.save_dir,
        )
        print(f"\n{'='*50}\n  RESULT: {label}  ({max(probs)*100:.1f}%)\n{'='*50}\n")
    elif source.is_dir():
        results = predict_directory(model, str(source), device, use_tta)
        if args.output and results:
            with open(args.output, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["path", "label", "class_name", "confidence", *Config.CLASS_NAMES])
                csv.writer(f).writerows(results)
            print(f"\n  💾 Results → {args.output}")
    else:
        print(f"❌  Not found: {source}"); sys.exit(1)


if __name__ == "__main__":
    main()
