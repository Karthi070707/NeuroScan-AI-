"""
Grad-CAM Explainability Utilities — Brain Tumor Classifier
==========================================================
Provides:
  - BrainTumorGradCAM   : wrapper around pytorch-grad-cam
  - overlay_heatmap()   : blends CAM onto MRI image
  - generate_all_class_cams() : 4-panel view, one CAM per class
  - save_gradcam_overlay()    : save PNG to disk

Requires:  pip install grad-cam
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Optional imports (graceful degradation if grad-cam not installed)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pytorch_grad_cam import (
        GradCAM,
        GradCAMPlusPlus,
        EigenCAM,
    )
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False
    warnings.warn(
        "pytorch-grad-cam not installed. Install with:  pip install grad-cam\n"
        "Grad-CAM overlays will be unavailable.",
        RuntimeWarning, stacklevel=2,
    )

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_cam(
    img_pil   : Image.Image,
    image_size: int = Config.IMAGE_SIZE,
    device    : str = "cpu",
) -> torch.Tensor:
    """Convert PIL image → normalized tensor on device. Returns (1, 3, H, W)."""
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMG_MEAN, Config.IMG_STD),
    ])
    return tf(img_pil.convert("RGB")).unsqueeze(0).to(device)


def load_image_rgb(path: str | Path) -> Image.Image:
    """Load image as RGB PIL."""
    return Image.open(str(path)).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    img_np   : np.ndarray,       # (H, W, 3) uint8 RGB
    cam_np   : np.ndarray,       # (H', W') float32 [0, 1]
    alpha    : float = Config.GRADCAM_ALPHA,
    colormap : int   = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend Grad-CAM heatmap onto original image.

    Args:
        img_np   : RGB uint8 array (H, W, 3)
        cam_np   : Grad-CAM float32 array, any spatial size
        alpha    : heatmap contribution [0, 1]
        colormap : OpenCV colormap constant

    Returns:
        overlay  : RGB uint8 blend (H, W, 3)
    """
    h, w = img_np.shape[:2]
    # Resize CAM to image size
    cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)
    # Normalize to [0, 255]
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    cam_uint8 = (cam_norm * 255).astype(np.uint8)
    # Apply colormap
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    # Blend
    overlay = (img_np.astype(np.float32) * (1 - alpha) +
               heatmap_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return overlay


def cam_to_contour_overlay(
    img_np    : np.ndarray,
    cam_np    : np.ndarray,
    threshold : float = 0.5,
    color     : tuple = (0, 255, 0),
    thickness : int   = 2,
) -> np.ndarray:
    """
    Draw contour of high-activation region on image (tumor localization).

    Args:
        img_np    : RGB uint8 (H, W, 3)
        cam_np    : Grad-CAM float32
        threshold : activation threshold for contour mask
        color     : contour BGR color
        thickness : contour line thickness

    Returns:
        annotated : RGB uint8 with contour drawn
    """
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam_np, (w, h))
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    mask = (cam_norm >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = img_np.copy()
    # Convert to BGR for cv2
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.drawContours(annotated_bgr, contours, -1, color, thickness)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# BrainTumorGradCAM
# ─────────────────────────────────────────────────────────────────────────────

class BrainTumorGradCAM:
    """
    Grad-CAM wrapper for BrainTumorClassifier.

    Usage:
        cam = BrainTumorGradCAM(model, device)
        heatmap = cam.generate(img_pil, target_class=0)   # 0=glioma
        overlay = cam.overlay(img_pil, target_class=0)
    """

    def __init__(
        self,
        model   : torch.nn.Module,
        device  : str = "cpu",
        method  : str = "gradcam",    # "gradcam" | "gradcam++" | "eigencam"
    ):
        if not HAS_GRADCAM:
            raise RuntimeError(
                "pytorch-grad-cam required. Install with:  pip install grad-cam"
            )
        self.model  = model
        self.device = device
        self.method = method

        target_layer = [model.get_target_layer()]
        CAM_MAP = {
            "gradcam":   GradCAM,
            "gradcam++": GradCAMPlusPlus,
            "eigencam":  EigenCAM,
        }
        cam_cls = CAM_MAP.get(method, GradCAM)
        self.cam_obj = cam_cls(model=model, target_layers=target_layer)

    def generate(
        self,
        img_pil      : Image.Image,
        target_class : Optional[int] = None,   # None → use predicted class
    ) -> np.ndarray:
        """
        Compute Grad-CAM for img_pil.

        Returns:
            cam_map : (H', W') float32 [0, 1]   (spatial CAM from backbone)
        """
        tensor  = preprocess_for_cam(img_pil, device=self.device)
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        cam_map = self.cam_obj(input_tensor=tensor, targets=targets)[0]
        return cam_map.astype(np.float32)

    def overlay(
        self,
        img_pil      : Image.Image,
        target_class : Optional[int] = None,
        alpha        : float = Config.GRADCAM_ALPHA,
    ) -> np.ndarray:
        """Generate Grad-CAM and blend onto original image. Returns RGB uint8."""
        img_np  = np.array(img_pil.convert("RGB"))
        cam_map = self.generate(img_pil, target_class)
        return overlay_heatmap(img_np, cam_map, alpha=alpha)

    def localization_contour(
        self,
        img_pil      : Image.Image,
        target_class : Optional[int] = None,
        threshold    : float = 0.5,
    ) -> np.ndarray:
        """Overlay tumor localization contour on image. Returns RGB uint8."""
        img_np  = np.array(img_pil.convert("RGB"))
        cam_map = self.generate(img_pil, target_class)
        return cam_to_contour_overlay(img_np, cam_map, threshold=threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-class 4-panel visualization
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_class_cams(
    cam_obj    : "BrainTumorGradCAM",
    img_pil    : Image.Image,
    probs      : list[float],
    pred_class : int,
    save_path  : Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Generate 5-panel Grad-CAM visualization:
      [Original | glioma | meningioma | notumor | pituitary]

    Panels are titled with class name + probability.
    Predicted class panel is highlighted with a border.

    Returns the figure as RGB numpy array, or None if matplotlib not available.
    """
    if not HAS_MPL:
        print("[WARN] matplotlib not installed — skipping multi-class CAM plot.")
        return None

    class_names = Config.CLASS_NAMES
    n_classes   = len(class_names)
    img_np      = np.array(img_pil.convert("RGB"))

    fig, axes = plt.subplots(1, n_classes + 1, figsize=(5 * (n_classes + 1), 5))
    fig.patch.set_facecolor("#0a0f1e")

    # Panel 0: original
    axes[0].imshow(img_np)
    axes[0].set_title("Original MRI", color="#e2e8f0", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    for cls_idx in range(n_classes):
        ax = axes[cls_idx + 1]
        overlay = cam_obj.overlay(img_pil, target_class=cls_idx)
        ax.imshow(overlay)

        prob_pct = probs[cls_idx] * 100
        label_name = Config.CLASS_LABELS[cls_idx]
        color = Config.CLASS_COLORS[cls_idx]

        title = f"{label_name}\n{prob_pct:.1f}%"
        ax.set_title(title, color=color, fontsize=12, fontweight="bold")
        ax.axis("off")

        # Highlight predicted class with a colored border
        if cls_idx == pred_class:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    fig.suptitle(
        f"Grad-CAM Analysis  —  Predicted: {Config.CLASS_LABELS[pred_class]}  "
        f"({probs[pred_class]*100:.1f}%)",
        color="#f1f5f9", fontsize=14, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0f1e")
        print(f"[GradCAM] Saved → {save_path}")

    # Render to numpy
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    w_px, h_px = fig.canvas.get_width_height()
    result = np.frombuffer(buf, dtype=np.uint8).reshape(h_px, w_px, 3)
    plt.close(fig)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Convenience save function
# ─────────────────────────────────────────────────────────────────────────────

def save_gradcam_overlay(
    img_pil    : Image.Image,
    cam_map    : np.ndarray,
    save_path  : str | Path,
    alpha      : float = Config.GRADCAM_ALPHA,
):
    """Save a Grad-CAM overlay PNG to disk."""
    img_np  = np.array(img_pil.convert("RGB"))
    overlay = overlay_heatmap(img_np, cam_map, alpha=alpha)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), overlay_bgr)
    print(f"[GradCAM] Overlay saved → {save_path}")


if __name__ == "__main__":
    print("gradcam_utils: HAS_GRADCAM =", HAS_GRADCAM, "| HAS_MPL =", HAS_MPL)
