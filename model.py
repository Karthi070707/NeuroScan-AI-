"""
BrainTumorClassifier — EfficientNet (B0 on CPU / B4 on GPU)
=============================================================
Pretrained ImageNet backbone fine-tuned for 4-class brain tumor MRI classification.

Backbone auto-selection:
  GPU available  →  efficientnet_b4  (1792 features, ~18.5M params, best accuracy)
  CPU only       →  efficientnet_b0  (1280 features, ~5.3M params,  ~3x faster)

Architecture Overview
---------------------
  Input MRI (224×224 RGB)
       ↓  [optional SRM high-pass filter → 6-channel input]
  EfficientNet-B0/B4 backbone (pretrained, timm)
       ↓  feat_dim-channel spatial feature map
  Adaptive Average Pool → (B, feat_dim)
       ↓
  Dropout(0.4) → Linear(feat_dim→512) → GELU → Dropout(0.3) → Linear(512→4)
       ↓
  4-class logits ── CrossEntropyLoss during training
       ↓ softmax at inference
  Class probabilities [glioma, meningioma, notumor, pituitary]

Grad-CAM
--------
  Target: last MBConv block (backbone.blocks[-1]) — same for B0 and B4.

Adapted from:  EfficientNetB4Detector (deepfake detection)
Changes:       num_classes 1→4, binary→4-class head, SRM optional, auto B0/B4
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required.  Install with:  pip install timm") from exc


# ─────────────────────────────────────────────────────────────────────────────
# SRM High-Pass Filter (optional — designed for pixel-level artifact detection)
# ─────────────────────────────────────────────────────────────────────────────

class SRMConv2d(nn.Module):
    """
    Spatial Rich Model (SRM) Laplacian High-Pass Filter.
    Extracts high-frequency texture residuals.
    Fixed (non-trainable) convolution weights.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            padding=1, bias=False, groups=in_channels
        )
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1],
        ], dtype=torch.float32) / 8.0
        kernel = kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# BrainTumorClassifier
# ─────────────────────────────────────────────────────────────────────────────

class BrainTumorClassifier(nn.Module):
    """
    EfficientNet-B4 backbone + two-layer classification head for 4-class MRI.

    Forward returns:
        logits    : (B, 4)   raw class logits (CrossEntropyLoss-compatible)
        embedding : (B, 512) pre-classifier feature vector (for SHAP / analysis)
    """

    def __init__(
        self,
        num_classes : int   = 4,
        pretrained  : bool  = True,
        drop_rate   : float = 0.4,
        use_srm     : bool  = False,
        backbone_name: str  = None,   # None = auto (B0 on CPU, B4 on GPU)
    ):
        super().__init__()
        self.use_srm     = use_srm
        self.num_classes = num_classes
        in_chans         = 6 if use_srm else 3

        # ── SRM layer ───────────────────────────────────────────────────────
        if use_srm:
            self.srm_layer = SRMConv2d(in_channels=3)

        # ── Backbone auto-selection ──────────────────────────────────────────
        # B4 on GPU (best accuracy), B0 on CPU (3x faster, suitable for training)
        if backbone_name is None:
            import torch
            backbone_name = "efficientnet_b4" if torch.cuda.is_available() else "efficientnet_b0"
        self.backbone_name = backbone_name
        print(f"  Backbone: {backbone_name}  ({'GPU' if 'b4' in backbone_name else 'CPU-optimized'})")

        # global_pool='' keeps the spatial feature map for Grad-CAM hooks
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
        )
        feat_dim = self.backbone.num_features   # 1280 for B0, 1792 for B4

        # ── Global Average Pool ─────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classification Head ─────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),              # head[0]
            nn.Linear(feat_dim, 512),           # head[1]
            nn.GELU(),                          # head[2]
            nn.Dropout(drop_rate * 0.75),       # head[3]
            nn.Linear(512, num_classes),        # head[4]
        )
        self._init_head()

    # ── Weight initialisation ────────────────────────────────────────────────

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Fine-tuning helpers ──────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze backbone; only classification head trains (warm-up phase)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze full network for end-to-end fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_backbone_params(self):
        return list(self.backbone.parameters())

    def get_head_params(self):
        return list(self.gap.parameters()) + list(self.head.parameters())

    # ── Grad-CAM target ──────────────────────────────────────────────────────

    def get_target_layer(self):
        """
        Last MBConv block of EfficientNet-B4.
        Pass to GradCAM(model, target_layers=[model.get_target_layer()])
        """
        return self.backbone.blocks[-1]

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        if self.use_srm:
            srm_x = self.srm_layer(x)
            x = torch.cat([x, srm_x], dim=1)

        # (B, 1792, H', W') — spatial feature map (kept for Grad-CAM)
        features  = self.backbone.forward_features(x)
        # (B, 1792)
        pooled    = self.gap(features).flatten(1)

        # Head forward (split for embedding access)
        drop1     = self.head[0](pooled)
        lin1      = self.head[1](drop1)
        act       = self.head[2](lin1)
        embedding = act                             # (B, 512)
        drop2     = self.head[3](embedding)
        logits    = self.head[4](drop2)             # (B, 4)

        return logits, embedding

    def predict_proba(self, x: torch.Tensor):
        """Return softmax probabilities — convenience method for inference."""
        with torch.no_grad():
            logits, emb = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs, emb


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading EfficientNet-B4 pretrained weights (requires internet)…")
    model  = BrainTumorClassifier(pretrained=True, use_srm=False)
    dummy  = torch.randn(2, 3, 224, 224)
    logits, emb = model(dummy)
    print(f"Logits    : {logits.shape}")    # (2, 4)
    print(f"Embedding : {emb.shape}")       # (2, 512)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total/1e6:.1f}M")
    print(f"Grad-CAM target: {model.get_target_layer().__class__.__name__}")
