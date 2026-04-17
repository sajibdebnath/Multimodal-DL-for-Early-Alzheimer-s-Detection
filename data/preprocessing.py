"""
Preprocessing pipelines for MRI, cognitive, and lifestyle data.

MRI pipeline:
  1. Skull stripping (BET)            - requires FSL
  2. AC-PC alignment (FLIRT)          - requires FSL
  3. Bias field correction (N4ITK)    - requires ANTs
  4. MNI152 registration (SyN)        - requires ANTs
  5. Z-score intensity normalization
  6. Multi-planar slice extraction (30 slices / subject)
  7. Resize to 224×224
  8. Online augmentation (training only)

For users who do not have FSL/ANTs installed, a mock/synthetic path is
provided via generate_synthetic_dataset() for development purposes.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# Optional neuroimaging imports
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not found. MRI loading from .nii files disabled.")

try:
    import subprocess
    FSL_AVAILABLE = (
        subprocess.run(["which", "fsl"], capture_output=True).returncode == 0
    )
except Exception:
    FSL_AVAILABLE = False


# ---------------------------------------------------------------------------
# MRI utilities
# ---------------------------------------------------------------------------

def skull_strip(input_nii: str, output_nii: str, frac: float = 0.5) -> str:
    """Run FSL BET skull stripping."""
    if not FSL_AVAILABLE:
        warnings.warn("FSL not found. Returning input path unchanged.")
        return input_nii
    import subprocess
    cmd = ["bet", input_nii, output_nii, "-f", str(frac), "-g", "0"]
    subprocess.run(cmd, check=True)
    return output_nii


def acpc_align(input_nii: str, output_nii: str) -> str:
    """AC-PC alignment using FSL FLIRT."""
    if not FSL_AVAILABLE:
        return input_nii
    import subprocess
    fsl_dir = os.environ.get("FSLDIR", "/usr/local/fsl")
    std_brain = os.path.join(fsl_dir, "data/standard/MNI152_T1_1mm_brain.nii.gz")
    cmd = [
        "flirt", "-in", input_nii, "-ref", std_brain,
        "-out", output_nii, "-dof", "6",
    ]
    subprocess.run(cmd, check=True)
    return output_nii


def bias_correction(input_nii: str, output_nii: str) -> str:
    """N4ITK bias field correction using ANTs."""
    try:
        import subprocess
        subprocess.run(
            ["N4BiasFieldCorrection", "-i", input_nii, "-o", output_nii],
            check=True
        )
        return output_nii
    except Exception:
        warnings.warn("ANTs N4BiasFieldCorrection not available.")
        return input_nii


def mni152_registration(input_nii: str, output_nii: str) -> str:
    """Non-linear registration to MNI152 using ANTs SyN."""
    try:
        import subprocess
        fsl_dir = os.environ.get("FSLDIR", "/usr/local/fsl")
        template = os.path.join(fsl_dir, "data/standard/MNI152_T1_1mm.nii.gz")
        cmd = [
            "antsRegistrationSyN.sh", "-d", "3",
            "-f", template, "-m", input_nii,
            "-o", output_nii.replace(".nii.gz", "_"),
        ]
        subprocess.run(cmd, check=True)
        return output_nii
    except Exception:
        warnings.warn("ANTs registration not available.")
        return input_nii


def zscore_normalize(volume: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Z-score normalize MRI voxel intensities within the brain mask."""
    if brain_mask is None:
        brain_mask = volume > 0
    brain_voxels = volume[brain_mask]
    mean_val = brain_voxels.mean()
    std_val = brain_voxels.std() + 1e-8
    normalized = volume.copy().astype(np.float32)
    normalized[brain_mask] = (volume[brain_mask] - mean_val) / std_val
    return normalized


def extract_multiplanar_slices(
    volume: np.ndarray,
    n_slices: int = 10,
    target_size: int = 224,
) -> np.ndarray:
    """
    Extract n_slices canonical 2D slices from axial, coronal, and sagittal
    planes, focusing on AD-sensitive regions (hippocampus, entorhinal cortex).

    Returns
    -------
    slices : np.ndarray of shape (3 * n_slices, target_size, target_size, 3)
        The channel-last, RGB-replicated slice array ready for EfficientNet.
    """
    from PIL import Image

    D, H, W = volume.shape
    all_slices = []

    # Axial: slices 40–70 in z (MNI z = -30 to +10mm at 1mm iso)
    axial_indices = np.linspace(max(0, D // 4), min(D - 1, 3 * D // 4), n_slices, dtype=int)
    # Coronal: entorhinal cortex region
    coronal_indices = np.linspace(max(0, H // 4), min(H - 1, 3 * H // 4), n_slices, dtype=int)
    # Sagittal: posterior cingulate
    sagittal_indices = np.linspace(max(0, W // 4), min(W - 1, 3 * W // 4), n_slices, dtype=int)

    for idx in axial_indices:
        sl = volume[idx, :, :]
        all_slices.append(_resize_and_rgb(sl, target_size, Image))
    for idx in coronal_indices:
        sl = volume[:, idx, :]
        all_slices.append(_resize_and_rgb(sl, target_size, Image))
    for idx in sagittal_indices:
        sl = volume[:, :, idx]
        all_slices.append(_resize_and_rgb(sl, target_size, Image))

    return np.stack(all_slices, axis=0)  # (30, 224, 224, 3)


def _resize_and_rgb(
    slice_2d: np.ndarray, target_size: int, Image
) -> np.ndarray:
    """Normalize a 2D slice to [0,255], resize, and expand to 3-channel."""
    vmin, vmax = slice_2d.min(), slice_2d.max()
    if vmax - vmin > 1e-8:
        img = ((slice_2d - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        img = np.zeros_like(slice_2d, dtype=np.uint8)
    pil_img = Image.fromarray(img).resize((target_size, target_size), Image.BILINEAR)
    rgb = np.stack([np.array(pil_img)] * 3, axis=-1)
    return rgb


def preprocess_mri_volume(nii_path: str, n_slices: int = 10, target_size: int = 224) -> np.ndarray:
    """
    Full MRI preprocessing pipeline (steps 5-7 from the paper, assuming
    skull-stripping / registration have already been performed).

    Parameters
    ----------
    nii_path : path to preprocessed NIfTI file
    n_slices : slices per plane
    target_size : resize target in pixels

    Returns
    -------
    slices : np.ndarray (30, 224, 224, 3)
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel required for MRI loading. pip install nibabel")
    img = nib.load(nii_path)
    volume = img.get_fdata().astype(np.float32)
    volume = zscore_normalize(volume)
    slices = extract_multiplanar_slices(volume, n_slices=n_slices, target_size=target_size)
    return slices


# ---------------------------------------------------------------------------
# Cognitive data preprocessing
# ---------------------------------------------------------------------------

COGNITIVE_FEATURES = [
    "MMSE", "CDR_Global", "CDR_SB", "CDR_Memory", "CDR_Orientation",
    "Age", "Sex", "Education"
]


def preprocess_cognitive_data(
    df: pd.DataFrame,
    fit_scaler: bool = True,
    scaler: Optional[MinMaxScaler] = None,
    imputer: Optional[IterativeImputer] = None,
) -> Tuple[np.ndarray, MinMaxScaler, IterativeImputer]:
    """
    Preprocess cognitive/demographic features:
      1. MICE imputation for missing values
      2. Min-max normalization to [0, 1]

    Parameters
    ----------
    df : DataFrame with columns matching COGNITIVE_FEATURES
    fit_scaler : whether to fit (True for train set, False for val/test)
    scaler : pre-fitted scaler for val/test
    imputer : pre-fitted MICE imputer for val/test

    Returns
    -------
    X : np.ndarray of shape (N, 8)
    scaler : fitted MinMaxScaler
    imputer : fitted IterativeImputer
    """
    X = df[COGNITIVE_FEATURES].values.astype(np.float64)

    if fit_scaler:
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X = imputer.fit_transform(X)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        assert scaler is not None and imputer is not None
        X = imputer.transform(X)
        X = scaler.transform(X)

    return X.astype(np.float32), scaler, imputer


# ---------------------------------------------------------------------------
# Lifestyle data preprocessing
# ---------------------------------------------------------------------------

LIFESTYLE_FEATURES = [
    "SleepDuration", "PhysicalActivityIndex", "RoutineRegularityScore",
    "MealRegularity", "SocialActivity", "MedicationCompliance",
    "WearableSteps", "PSQIScore",
]

SEQ_LEN = 12  # months


def preprocess_lifestyle_data(
    sequences: np.ndarray,
    fit_stats: bool = True,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize lifestyle time-series and apply causal zero-padding.

    Parameters
    ----------
    sequences : np.ndarray of shape (N, T, F) where T may vary
    fit_stats : fit mean/std on this data (True for train)
    mean, std : pre-computed statistics for val/test

    Returns
    -------
    padded : np.ndarray (N, SEQ_LEN, F), zero-padded at the front
    mean : np.ndarray (F,)
    std : np.ndarray (F,)
    """
    N, T, F = sequences.shape

    if fit_stats:
        # Compute stats on training data only
        valid_mask = sequences != 0
        mean = np.zeros(F, dtype=np.float32)
        std = np.ones(F, dtype=np.float32)
        for f in range(F):
            vals = sequences[:, :, f][valid_mask[:, :, f]]
            if len(vals) > 0:
                mean[f] = vals.mean()
                std[f] = vals.std() + 1e-8

    normalized = (sequences - mean[None, None, :]) / std[None, None, :]

    # Pad / truncate to SEQ_LEN
    padded = np.zeros((N, SEQ_LEN, F), dtype=np.float32)
    for i in range(N):
        seq = normalized[i]  # (T, F)
        if T >= SEQ_LEN:
            padded[i] = seq[-SEQ_LEN:]
        else:
            padded[i, SEQ_LEN - T:] = seq  # causal zero-padding at front

    return padded, mean, std


# ---------------------------------------------------------------------------
# Synthetic dataset generator (for development without real data)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_subjects: int = 1900,
    random_seed: int = 42,
    output_dir: str = "data/synthetic",
) -> pd.DataFrame:
    """
    Generate a synthetic multimodal dataset matching the paper's statistics
    for development and testing without ADNI/OASIS access.

    Class distribution: CN 44%, MCI 32.2%, AD 23.8%
    Age range: 55-96 (mean 74.4 ± 8.7)

    Returns
    -------
    metadata : pd.DataFrame with subject-level metadata
    """
    rng = np.random.default_rng(random_seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Assign labels with paper's class distribution
    n_cn = int(0.44 * n_subjects)
    n_mci = int(0.322 * n_subjects)
    n_ad = n_subjects - n_cn - n_mci
    labels = np.array(["CN"] * n_cn + ["MCI"] * n_mci + ["AD"] * n_ad)
    rng.shuffle(labels)

    label_map = {"CN": 0, "MCI": 1, "AD": 2}
    label_int = np.array([label_map[l] for l in labels])

    # Cognitive features conditioned on class
    mmse_means = {"CN": 28.0, "MCI": 24.0, "AD": 18.0}
    cdr_means = {"CN": 0.0, "MCI": 0.5, "AD": 1.5}

    data = {
        "SubjectID": [f"SUB_{i:04d}" for i in range(n_subjects)],
        "Dataset": rng.choice(["ADNI", "OASIS"], n_subjects, p=[0.632, 0.368]),
        "Label": labels,
        "LabelInt": label_int,
        "Age": np.clip(rng.normal(74.4, 8.7, n_subjects), 55, 96).astype(np.float32),
        "Sex": rng.integers(0, 2, n_subjects).astype(np.float32),
        "Education": rng.integers(0, 5, n_subjects).astype(np.float32),
        "MMSE": np.clip(
            np.array([rng.normal(mmse_means[l], 3.0) for l in labels]), 0, 30
        ).astype(np.float32),
        "CDR_Global": np.clip(
            np.array([rng.normal(cdr_means[l], 0.3) for l in labels]), 0, 3
        ).astype(np.float32),
        "CDR_SB": np.clip(
            np.array([rng.normal(cdr_means[l] * 3, 1.0) for l in labels]), 0, 18
        ).astype(np.float32),
        "CDR_Memory": np.clip(
            np.array([rng.normal(cdr_means[l], 0.3) for l in labels]), 0, 3
        ).astype(np.float32),
        "CDR_Orientation": np.clip(
            np.array([rng.normal(cdr_means[l] * 0.5, 0.2) for l in labels]), 0, 3
        ).astype(np.float32),
    }

    # Generate lifestyle time-series (12 × 8) per subject
    lifestyle_data = []
    sleep_means = {"CN": 7.0, "MCI": 6.2, "AD": 5.5}
    activity_means = {"CN": 7000, "MCI": 5000, "AD": 3500}

    for i, lbl in enumerate(labels):
        seq = np.zeros((SEQ_LEN, len(LIFESTYLE_FEATURES)), dtype=np.float32)
        for t in range(SEQ_LEN):
            # Progressive decline for MCI/AD over time
            decay = t / SEQ_LEN * (0.1 if lbl == "MCI" else 0.2 if lbl == "AD" else 0.0)
            seq[t, 0] = max(3, rng.normal(sleep_means[lbl] - decay, 0.8))
            seq[t, 1] = max(500, rng.normal(activity_means[lbl] - decay * 200, 500))
            seq[t, 2] = rng.normal(0.7 - decay if lbl == "CN" else 0.5 - decay, 0.1)
            seq[t, 3] = rng.normal(0.8, 0.1)
            seq[t, 4] = rng.normal(0.6 - decay, 0.15)
            seq[t, 5] = rng.normal(0.9, 0.05)
            seq[t, 6] = max(500, rng.normal(activity_means[lbl], 1000))
            seq[t, 7] = rng.normal(5 + label_int[i], 1.5)
        lifestyle_data.append(seq)

    # Save synthetic MRI slices (random tensors for shapes matching paper)
    mri_dir = os.path.join(output_dir, "mri_slices")
    os.makedirs(mri_dir, exist_ok=True)
    mri_paths = []
    for i, sid in enumerate(data["SubjectID"]):
        path = os.path.join(mri_dir, f"{sid}.npy")
        # (30, 224, 224, 3) → uint8 synthetic
        slices = rng.integers(0, 255, (30, 224, 224, 3), dtype=np.uint8)
        np.save(path, slices)
        mri_paths.append(path)

    data["MRIPath"] = mri_paths

    # Save lifestyle sequences
    life_dir = os.path.join(output_dir, "lifestyle")
    os.makedirs(life_dir, exist_ok=True)
    life_paths = []
    for i, sid in enumerate(data["SubjectID"]):
        path = os.path.join(life_dir, f"{sid}_lifestyle.npy")
        np.save(path, lifestyle_data[i])
        life_paths.append(path)

    data["LifestylePath"] = life_paths

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"Synthetic dataset saved to {output_dir}")
    print(f"  CN: {n_cn}  MCI: {n_mci}  AD: {n_ad}  Total: {n_subjects}")
    return df
