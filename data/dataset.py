"""
PyTorch Dataset and DataLoader utilities for the MDLF framework.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from data.preprocessing import COGNITIVE_FEATURES, LIFESTYLE_FEATURES


# ---------------------------------------------------------------------------
# Augmentation transforms (MRI slices treated as 2-D images)
# ---------------------------------------------------------------------------

def get_mri_transform(train: bool = True) -> transforms.Compose:
    """Return torchvision transforms for MRI slices."""
    base = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if train:
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        return transforms.Compose(augment + base)
    return transforms.Compose(base)


def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to a tensor (training augmentation)."""
    return tensor + torch.randn_like(tensor) * std


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class AlzheimerMultimodalDataset(Dataset):
    """
    Multimodal dataset for Alzheimer's detection combining:
      - Structural MRI slices  → (30, C, H, W) or single slice strategy
      - Cognitive scores        → (8,)
      - Lifestyle time-series   → (12, 8)

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Must contain columns: SubjectID, LabelInt, MRIPath, LifestylePath,
        and all COGNITIVE_FEATURES.
    cognitive_array : np.ndarray of shape (N, 8), pre-processed
    lifestyle_array : np.ndarray of shape (N, 12, 8), pre-processed
    train : bool
        If True, apply data augmentation to MRI slices.
    slice_aggregate : str
        'mean' – average all 30 slice features, 'single' – use the central
        axial slice only, 'random' – pick one slice at random during training.
    """

    LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        cognitive_array: np.ndarray,
        lifestyle_array: np.ndarray,
        train: bool = True,
        slice_aggregate: str = "single",
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.cog = torch.tensor(cognitive_array, dtype=torch.float32)
        self.life = torch.tensor(lifestyle_array, dtype=torch.float32)
        self.train = train
        self.slice_aggregate = slice_aggregate
        self.mri_transform = get_mri_transform(train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        label = int(row["LabelInt"])

        # ---- MRI ----
        mri_tensor = self._load_mri(row["MRIPath"])

        # ---- Cognitive ----
        cog_tensor = self.cog[idx]

        # ---- Lifestyle ----
        life_tensor = self.life[idx]  # (12, 8)

        return {
            "mri": mri_tensor,
            "cognitive": cog_tensor,
            "lifestyle": life_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": row["SubjectID"],
        }

    def _load_mri(self, mri_path: str) -> torch.Tensor:
        """
        Load preprocessed MRI slices from a .npy file of shape (30, H, W, 3)
        or a .nii.gz file that will be processed on-the-fly.
        Returns a (3, 224, 224) tensor after slice selection.
        """
        if mri_path.endswith(".npy"):
            slices = np.load(mri_path)  # (30, H, W, 3) or (30, H, W)
        elif mri_path.endswith((".nii", ".nii.gz")):
            from data.preprocessing import preprocess_mri_volume
            slices = preprocess_mri_volume(mri_path)  # (30, 224, 224, 3)
        else:
            raise ValueError(f"Unsupported MRI format: {mri_path}")

        # Handle grayscale
        if slices.ndim == 3:
            slices = np.stack([slices] * 3, axis=-1)

        # Slice selection strategy
        n_slices = slices.shape[0]
        if self.slice_aggregate == "single":
            # Use central axial slice (index 5 of the 10 axial slices)
            slice_idx = 5
        elif self.slice_aggregate == "random" and self.train:
            slice_idx = np.random.randint(0, n_slices)
        else:
            slice_idx = n_slices // 2

        img = slices[slice_idx]  # (H, W, 3)

        if img.dtype != np.uint8:
            vmin, vmax = img.min(), img.max()
            if vmax - vmin > 1e-8:
                img = ((img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        pil_img = Image.fromarray(img)
        tensor = self.mri_transform(pil_img)  # (3, 224, 224)

        if self.train:
            tensor = add_gaussian_noise(tensor, std=0.01)

        return tensor


# ---------------------------------------------------------------------------
# Dataset split utility
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split ensuring class proportions are
    maintained across all three splits (as described in the paper).
    No subject appears in more than one split.
    """
    from sklearn.model_selection import train_test_split

    test_ratio = 1.0 - train_ratio - val_ratio
    assert abs(test_ratio) > 1e-8, "train + val ratios must be < 1.0"

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df["LabelInt"], random_state=random_seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - relative_val),
        stratify=temp_df["LabelInt"], random_state=random_seed,
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cog_train: np.ndarray,
    cog_val: np.ndarray,
    cog_test: np.ndarray,
    life_train: np.ndarray,
    life_val: np.ndarray,
    life_test: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders."""

    train_ds = AlzheimerMultimodalDataset(train_df, cog_train, life_train, train=True)
    val_ds = AlzheimerMultimodalDataset(val_df, cog_val, life_val, train=False)
    test_ds = AlzheimerMultimodalDataset(test_df, cog_test, life_test, train=False)

    # Class-balanced sampler for training
    labels = train_df["LabelInt"].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
    }
    return loaders
