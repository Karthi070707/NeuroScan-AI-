"""
FastAPI Dashboard Backend — Brain Tumor Classifier
==================================================
Endpoints:
  POST /predict        → JSON prediction + probabilities + confidence
  POST /gradcam        → Base64 Grad-CAM overlay image
  POST /all-classes    → Base64 5-panel all-class CAM
  POST /batch          → Batch prediction JSON array
  GET  /classes        → Class metadata (names, colors, labels)
  GET  /health         → Server health check

Usage:
    cd brain_tumor_classifier/dashboard
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
# Dashboard lives in brain_tumor_classifier/dashboard/ — add parent to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from model import BrainTumorClassifier


# ─────────────────────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Brain Tumor Classifier API",
    description="Explainable AI for brain tumor MRI classification (EfficientNet-B4 + Grad-CAM)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Model singleton
# ─────────────────────────────────────────────────────────────────────────────

_MODEL   : Optional[BrainTumorClassifier] = None
_DEVICE  : str = "cuda" if torch.cuda.is_available() else "cpu"
_CAM_OBJ = None


def get_model() -> BrainTumorClassifier:
    global _MODEL
    if _MODEL is None:
        ckpt_path = Config.CHECKPOINT_DIR / "best_model.pth"
        if not ckpt_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint not found: {ckpt_path}. "
                       "Train the model first:  python train.py",
            )
        _MODEL = BrainTumorClassifier(
            num_classes=Config.NUM_CLASSES,
            pretrained=False,
            use_srm=Config.USE_SRM,
        ).to(_DEVICE)
        ckpt = torch.load(str(ckpt_path), map_location=_DEVICE, weights_only=False)
        _MODEL.load_state_dict(ckpt["model_state_dict"])
        _MODEL.eval()
        print(f"✅ Model loaded: {ckpt_path}  |  device={_DEVICE}")
    return _MODEL


def get_cam():
    global _CAM_OBJ
    if _CAM_OBJ is None:
        try:
            from gradcam_utils import BrainTumorGradCAM
            _CAM_OBJ = BrainTumorGradCAM(get_model(), device=_DEVICE)
        except Exception as e:
            print(f"[WARN] Grad-CAM init failed: {e}")
            _CAM_OBJ = None
    return _CAM_OBJ


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(img: Image.Image) -> torch.Tensor:
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMG_MEAN, Config.IMG_STD),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def run_predict(img: Image.Image):
    model = get_model()
    t1 = preprocess(img).to(_DEVICE)
    t2 = preprocess(img.transpose(Image.FLIP_LEFT_RIGHT)).to(_DEVICE)
    with torch.amp.autocast("cuda", enabled=(_DEVICE == "cuda")):
        l1, _ = model(t1)
        l2, _ = model(t2)
    p1 = torch.softmax(l1, dim=1).squeeze(0).cpu().tolist()
    p2 = torch.softmax(l2, dim=1).squeeze(0).cpu().tolist()
    probs = [(a + b) / 2 for a, b in zip(p1, p2)]
    pred  = int(np.argmax(probs))
    return pred, probs


def pil_to_base64(img_arr: np.ndarray) -> str:
    """Convert RGB numpy array to base64 PNG string."""
    pil = Image.fromarray(img_arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health():
    ckpt_ok = (Config.CHECKPOINT_DIR / "best_model.pth").exists()
    return {
        "status"  : "ok",
        "device"  : _DEVICE,
        "model_ready": ckpt_ok,
        "classes" : Config.CLASS_NAMES,
    }


@app.get("/classes")
async def get_classes():
    return {
        "classes": [
            {
                "index" : i,
                "name"  : Config.CLASS_NAMES[i],
                "label" : Config.CLASS_LABELS[i],
                "color" : Config.CLASS_COLORS[i],
            }
            for i in range(Config.NUM_CLASSES)
        ]
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Classify uploaded MRI image. Returns class, probabilities, confidence."""
    img  = await read_image(file)
    pred, probs = run_predict(img)
    conf = max(probs)
    return {
        "predicted_class" : Config.CLASS_NAMES[pred],
        "predicted_label" : Config.CLASS_LABELS[pred],
        "predicted_index" : pred,
        "confidence"      : round(conf, 4),
        "is_confident"    : conf >= Config.CONFIDENCE_THRESHOLD,
        "probabilities"   : {
            Config.CLASS_NAMES[i]: round(probs[i], 4)
            for i in range(Config.NUM_CLASSES)
        },
        "color"           : Config.CLASS_COLORS[pred],
    }


@app.post("/gradcam")
async def gradcam_endpoint(
    file        : UploadFile = File(...),
    target_class: Optional[int] = None,
):
    """
    Generate Grad-CAM overlay for predicted (or specified) class.
    Returns base64-encoded PNG overlay.
    """
    img  = await read_image(file)
    pred, probs = run_predict(img)
    tc   = target_class if target_class is not None else pred

    cam = get_cam()
    if cam is None:
        raise HTTPException(status_code=503,
                            detail="Grad-CAM unavailable. Install:  pip install grad-cam")

    overlay = cam.overlay(img, target_class=tc)
    img_b64  = pil_to_base64(overlay)
    orig_b64 = pil_to_base64(np.array(img))

    return {
        "original_b64"    : orig_b64,
        "overlay_b64"     : img_b64,
        "predicted_class" : Config.CLASS_NAMES[pred],
        "predicted_label" : Config.CLASS_LABELS[pred],
        "target_class"    : Config.CLASS_NAMES[tc],
        "confidence"      : round(max(probs), 4),
        "probabilities"   : {
            Config.CLASS_NAMES[i]: round(probs[i], 4)
            for i in range(Config.NUM_CLASSES)
        },
    }


@app.post("/all-classes")
async def all_classes_endpoint(file: UploadFile = File(...)):
    """
    Generate Grad-CAM overlays for ALL 4 classes.
    Returns array of {class_name, label, probability, overlay_b64}.
    """
    img  = await read_image(file)
    pred, probs = run_predict(img)

    cam = get_cam()
    if cam is None:
        raise HTTPException(status_code=503,
                            detail="Grad-CAM unavailable. Install:  pip install grad-cam")

    class_overlays = []
    for c in range(Config.NUM_CLASSES):
        overlay = cam.overlay(img, target_class=c)
        class_overlays.append({
            "class_index"  : c,
            "class_name"   : Config.CLASS_NAMES[c],
            "class_label"  : Config.CLASS_LABELS[c],
            "probability"  : round(probs[c], 4),
            "color"        : Config.CLASS_COLORS[c],
            "is_predicted" : c == pred,
            "overlay_b64"  : pil_to_base64(overlay),
        })

    return {
        "original_b64"  : pil_to_base64(np.array(img)),
        "predicted_index": pred,
        "predicted_label": Config.CLASS_LABELS[pred],
        "confidence"    : round(max(probs), 4),
        "class_overlays": class_overlays,
    }


@app.post("/batch")
async def batch_endpoint(files: list[UploadFile] = File(...)):
    """Batch predict multiple uploaded MRI images."""
    results = []
    for f in files:
        try:
            img  = await read_image(f)
            pred, probs = run_predict(img)
            conf = max(probs)
            results.append({
                "filename"       : f.filename,
                "predicted_class": Config.CLASS_NAMES[pred],
                "predicted_label": Config.CLASS_LABELS[pred],
                "confidence"     : round(conf, 4),
                "is_confident"   : conf >= Config.CONFIDENCE_THRESHOLD,
                "probabilities"  : {
                    Config.CLASS_NAMES[i]: round(probs[i], 4)
                    for i in range(Config.NUM_CLASSES)
                },
                "color"          : Config.CLASS_COLORS[pred],
                "error"          : None,
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return {"results": results, "total": len(results)}
