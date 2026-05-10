"""
BrainMRIDataset — Dataset & DataLoader Utilities
=================================================
Handles loading, splitting, and augmenting the Brain MRI Tumor dataset.

Dataset structure expected:
    Brain MRI Tumor/
        Training/
            glioma/       *.jpg
            meningioma/   *.jpg
            notumor/      *.jpg
            pituitary/    *.jpg
        Testing/
            glioma/       *.jpg
            ...
"""

import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int = Config.IMAGE_SIZE) -> transforms.Compose:
    """Heavy augmentation for training — includes geometric + photometric."""
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.1,
            hue=0.02,
        ),
        transforms.RandomGrayscale(p=0.05),  # MRI images often near-grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),  # random occlusion
    ])


def get_val_transforms(image_size: int = Config.IMAGE_SIZE) -> transforms.Compose:
    """Minimal transforms for validation and test — no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
    ])


def get_tta_transforms(image_size: int = Config.IMAGE_SIZE) -> list:
    """Test-Time Augmentation: list of transforms to average over."""
    base = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
    ])
    flipped = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
    ])
    return [base, flipped]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BrainMRIDataset(Dataset):
    """
    Brain MRI tumor classification dataset.

    Loads images from class-named subdirectories.
    Class index is determined by sorted folder name → Config.CLASS_NAMES ordering.

    Returns:
        image   : torch.Tensor (3, H, W) — normalized
        label   : int          — class index
        path    : str          — absolute path to source image (for traceability)
    """

    def __init__(
        self,
        root_dir  : str | Path,
        transform : Optional[Callable] = None,
        class_names: list = Config.CLASS_NAMES,
    ):
        self.root_dir    = Path(root_dir)
        self.transform   = transform
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[WARN] Class directory not found: {class_dir}")
                continue
            label = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root_dir}")
        print(f"[Dataset] Loaded {len(self.samples)} images from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)

    def class_counts(self) -> dict:
        """Return per-class sample counts."""
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Train/Val Split
# ─────────────────────────────────────────────────────────────────────────────

def split_train_val(
    dataset   : BrainMRIDataset,
    val_split : float = Config.VAL_SPLIT,
    seed      : int   = Config.RANDOM_SEED,
) -> Tuple[Subset, Subset]:
    """
    Stratified train/val split — preserves class balance.
    Returns (train_subset, val_subset).
    """
    rng = random.Random(seed)

    # Group indices by class
    class_indices: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for label, indices in sorted(class_indices.items()):
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = int(len(shuffled) * val_split)
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ─────────────────────────────────────────────────────────────────────────────
# MixUp Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x     : torch.Tensor,
    y     : torch.Tensor,
    alpha : float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp: blends pairs of training samples.
    Returns: mixed_x, y_a, y_b, lam
    Loss = lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss for MixUp."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    batch_size  : int  = Config.BATCH_SIZE,
    num_workers : int  = Config.NUM_WORKERS,
    val_split   : float = Config.VAL_SPLIT,
    seed        : int  = Config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Full training dataset with train transforms
    train_full = BrainMRIDataset(Config.TRAIN_DIR, transform=None)
    train_sub, val_sub = split_train_val(train_full, val_split, seed)

    # Apply correct transforms to each subset
    train_sub.dataset.transform = get_train_transforms()
    # For val, we need a separate dataset object with val transforms
    val_dataset  = BrainMRIDataset(Config.TRAIN_DIR, transform=get_val_transforms())
    val_sub2 = Subset(val_dataset, val_sub.indices)

    test_dataset = BrainMRIDataset(Config.TEST_DIR, transform=get_val_transforms())

    # Count info
    print(f"\n[DataLoader] Train: {len(train_sub)} | Val: {len(val_sub)} | Test: {len(test_dataset)}")
    print(f"[DataLoader] Train class dist: {train_full.class_counts()}")

    _pin = torch.cuda.is_available()   # only pin memory if GPU is present
    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=_pin, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_sub2, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_l, val_l, test_l = build_dataloaders(batch_size=4, num_workers=0)
    imgs, labels, paths = next(iter(train_l))
    print(f"Batch shape  : {imgs.shape}")
    print(f"Labels       : {labels}")
    print(f"Sample path  : {paths[0]}")
