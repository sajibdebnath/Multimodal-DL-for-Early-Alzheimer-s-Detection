# MDLF: Multimodal Deep Learning Framework for Early Alzheimer's Disease Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PeerJ Computer Science](https://img.shields.io/badge/Journal-PeerJ%20Computer%20Science-informational)](https://peerj.com/computer-science/)

> **Paper:** *A Multimodal Deep Learning Framework for Early Alzheimer's Disease Detection Using MRI, Cognitive Scores, and Lifestyle Data*
> Sajib Debnath, Md. Uzzal Mia, Arindam Kishor Biswas, Lipika Pal, Md. Sarwar Hosain, Tetsuya Shimamura
> **Journal:** PeerJ Computer Science (AI Application Article)
> **GitHub:** [sajibdebnath/Multimodal-DL-for-Early-Alzheimer-s-Detection](https://github.com/sajibdebnath/Multimodal-DL-for-Early-Alzheimer-s-Detection/tree/main)

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Information](#dataset-information)
3. [Code Information](#code-information)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Usage Instructions](#usage-instructions)
8. [Methodology](#methodology)
9. [Key Results](#key-results)
10. [Citations](#citations)
11. [License & Contribution Guidelines](#license--contribution-guidelines)
12. [Acknowledgements](#acknowledgements)

---

## Overview

This repository provides a **production-ready, modular implementation** of the **Multimodal Deep Learning Framework (MDLF)** proposed in the paper above. The MDLF jointly integrates three complementary data modalities to classify patients into three diagnostic categories: **Cognitively Normal (CN)**, **Mild Cognitive Impairment (MCI)**, and **Alzheimer's Disease (AD)**.

| Modality | Encoder | Output Dim |
|---|---|---|
| Structural MRI | EfficientNet-B4 (pretrained ImageNet) | 512-d |
| Cognitive Scores (MMSE, CDR, …) | 3-layer MLP | 128-d |
| Lifestyle Time-Series (sleep, activity, …) | BiLSTM + Transformer | 256-d |
| **Fused** | Cross-modal attention | 512-d |

---

## Dataset Information

### Third-Party Datasets Used

This study uses two publicly available neuroimaging datasets. **Both require free registration and acceptance of a Data Use Agreement before access is granted.** Please comply with the terms of access for each dataset independently.

#### 1. ADNI — Alzheimer's Disease Neuroimaging Initiative
- **URL:** [https://adni.loni.usc.edu](https://adni.loni.usc.edu)
- **License / Access:** Data Use Agreement (LONI / USC) — free registration required
- **What to download:** T1-weighted MRI scans, MMSE/CDR clinical assessments, and (for ADNI-3) wearable accelerometer data from the Wearables Substudy
- **Citation:** ADNI data collection and sharing was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012).

#### 2. OASIS — Open Access Series of Imaging Studies
- **URL:** [https://www.oasis-brains.org](https://www.oasis-brains.org)
- **License / Access:** CC-BY 4.0 (OASIS-1/2); Data Use Agreement required for OASIS-3
- **What to download:** T1-weighted MRI, clinical assessment data (MMSE, CDR)
- **Citation:** LaMontagne et al. (2019). OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease. *medRxiv*.

> **Note:** This study uses a combined cohort of N = 1,900 subjects drawn from both ADNI and OASIS datasets.

### Expected Data Layout

After downloading, organize your data as follows:

```
data/
├── raw/
│   ├── ADNI/                     # downloaded ADNI data
│   └── OASIS/                    # downloaded OASIS data
└── processed/
    ├── metadata.csv              # one row per subject (see columns below)
    ├── mri_slices/               # SUB_0001.npy  shape (30, 224, 224, 3)
    └── lifestyle/                # SUB_0001_lifestyle.npy  shape (12, 8)
```

**`metadata.csv` required columns:**

| Column | Description |
|---|---|
| `SubjectID` | Unique subject identifier |
| `LabelInt` | 0 = CN, 1 = MCI, 2 = AD |
| `MRIPath` | Path to `.npy` or `.nii.gz` MRI file |
| `LifestylePath` | Path to `(12, 8)` lifestyle `.npy` file |
| `MMSE` | Mini-Mental State Examination score |
| `CDR_Global` | CDR global score |
| `CDR_SB` | CDR Sum of Boxes |
| `CDR_Memory` | CDR Memory domain |
| `CDR_Orientation` | CDR Orientation domain |
| `Age` | Age in years |
| `Sex` | 0 = male, 1 = female |
| `Education` | Ordinal 0–4 |

### Class Distribution (N = 1,900)

| Class | Label | Count | Proportion |
|---|---|---|---|
| Cognitively Normal | CN | 836 | 44.0% |
| Mild Cognitive Impairment | MCI | 612 | 32.2% |
| Alzheimer's Disease | AD | 452 | 23.8% |

---

## Code Information

### Source Code Repository

The complete implementation is available on GitHub:

> **GitHub:** [https://github.com/sajibdebnath/Multimodal-DL-for-Early-Alzheimer-s-Detection](https://github.com/sajibdebnath/Multimodal-DL-for-Early-Alzheimer-s-Detection/tree/main)

The repository includes:
- Full model architecture (MRI branch, Cognitive branch, Lifestyle branch, Cross-modal Attention Fusion)
- Training and evaluation pipelines
- Ablation study script
- Preprocessing pipelines for all three modalities
- SHAP-based interpretability
- Visualization utilities (ROC, PR curves, confusion matrix, attention maps)
- Synthetic data generator (no real data needed for a quick test)

### Use of AI Tools in Code Preparation

Parts of this code were drafted with the assistance of AI writing tools (e.g., GitHub Copilot) for boilerplate generation and grammar correction. All model logic, experimental design, and results were implemented and verified by the authors.

---

## Repository Structure

```
mdlf_alzheimer/
├── Figures/
│   └── 15 figures '.png'         # All Figures (from paper)
├── configs/
│   └── default_config.yaml       # All hyperparameters (from paper Table 3)
├── data/
│   ├── preprocessing.py          # MRI / cognitive / lifestyle pipelines
│   └── dataset.py                # PyTorch Dataset, DataLoaders, augmentation
├── models/
│   ├── mri_branch.py             # EfficientNet-B4 encoder (§MRI Branch)
│   ├── cognitive_branch.py       # 3-layer MLP encoder (§Cognitive Branch)
│   ├── lifestyle_branch.py       # BiLSTM + Transformer (§Lifestyle Branch)
│   ├── fusion.py                 # Cross-modal attention fusion (Eq. 4–7)
│   └── mdlf.py                   # Full model + Algorithm 1 training loop
├── training/
│   ├── loss.py                   # Class-weighted cross-entropy (Eq. 9–10)
│   └── trainer.py                # End-to-end trainer with early stopping
├── evaluation/
│   ├── metrics.py                # Accuracy, AUC, F1, CI, McNemar's test
│   └── interpretability.py       # SHAP + attention weight visualization
├── scripts/
│   ├── train.py                  # Main training entry point
│   ├── evaluate.py               # Standalone evaluation + SHAP
│   └── ablation.py               # Reproduces Table 5 ablation study
├── utils/
│   ├── helpers.py                # Seeding, config loading, formatting
│   └── visualization.py          # Training curves, ROC, PR, confusion matrix
├── requirements.txt
├── README.md
└── setup.py
```

---

## Requirements

### Software Dependencies

| Package | Minimum Version | Purpose |
|---|---|---|
| Python | 3.9+ | Runtime |
| PyTorch | 2.0+ | Deep learning framework |
| torchvision | 0.15+ | Image transforms and pretrained models |
| timm | 0.9+ | EfficientNet-B4 pretrained weights |
| scikit-learn | 1.2+ | Metrics, cross-validation |
| numpy | 1.23+ | Array operations |
| pandas | 1.5+ | Metadata handling |
| matplotlib | 3.6+ | Plotting |
| seaborn | 0.12+ | Visualization |
| PyYAML | 6.0+ | Config loading |
| tqdm | 4.64+ | Progress bars |
| scipy | 1.10+ | Statistical tests (McNemar) |

Install all required packages via:

```bash
pip install -r requirements.txt
```

### Optional Extras

```bash
pip install shap>=0.42.1          # SHAP interpretability
pip install optuna>=3.2.0         # Hyperparameter optimisation
pip install lifelines>=0.27.0     # Kaplan-Meier survival curves
```

### Neuroimaging Preprocessing (Raw NIfTI Only)

If processing raw `.nii`/`.nii.gz` files (skull-stripping, registration), you additionally need:

- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (v6.0+)
- [ANTs](https://github.com/ANTsX/ANTs) (v2.3+)

> Pre-processed `.npy` slice arrays skip this requirement entirely.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/sajibdebnath/Multimodal-DL-for-Early-Alzheimer-s-Detection.git
cd Multimodal-DL-for-Early-Alzheimer-s-Detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install as a package
pip install -e .
```

---

## Usage Instructions

### Quick Start — Synthetic Data (No Real Data Needed)

To verify the complete pipeline without ADNI/OASIS access:

```bash
python scripts/train.py \
    --config configs/default_config.yaml \
    --synthetic
```

This generates 1,900 synthetic subjects matching the paper's class distribution (CN 44% / MCI 32.2% / AD 23.8%), trains the full MDLF, and saves results to `results/` and checkpoints to `checkpoints/`.

### Training with Real Data

```bash
# With real preprocessed data
python scripts/train.py \
    --config configs/default_config.yaml \
    --data_root data/processed

# Resume from checkpoint
python scripts/train.py \
    --config configs/default_config.yaml \
    --data_root data/processed \
    --resume checkpoints/best_model.pt
```

### Evaluation

```bash
# Evaluate a saved checkpoint and generate all figures
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --synthetic

# Also run SHAP analysis (slower, requires shap package)
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/default_config.yaml \
    --synthetic \
    --run_shap
```

Output figures are saved to `results/figures/`:

| File | Description | Paper Figure |
|---|---|---|
| `training_curves.png` | Accuracy & loss over epochs | Fig. 14 |
| `confusion_matrix.png` | Normalized confusion matrix | Fig. 12 |
| `roc_curves_mdlf.png` | Per-class and macro-average ROC | Fig. 10 |
| `pr_curves_mdlf.png` | Precision-recall curves | Fig. 11 |
| `attention_weights_by_class.png` | Mean attention weights per class | Fig. 7 |
| `shap_feature_importance.png` | SHAP feature importance | Fig. 9 |

### Ablation Study

Reproduce **Table 5** from the paper:

```bash
python scripts/ablation.py \
    --config configs/default_config.yaml \
    --synthetic \
    --epochs 30
```

---

## Methodology

### Model Architecture

The MDLF processes three modalities in parallel through dedicated encoders, then fuses them using a cross-modal attention mechanism:

```
Input
  ├── MRI slices (B, 3, 224, 224)
  │     └─ EfficientNet-B4 (blocks 5-7 fine-tuned) + GAP + Linear(1792→512)
  │                                          → F_mri ∈ R^512
  ├── Cognitive scores (B, 8)
  │     └─ MLP [8→256→128→64] + Linear(64→128)
  │                                          → F_cog ∈ R^128
  └── Lifestyle sequences (B, 12, 8)
        └─ Positional Encoding + BiLSTM(128×2=256) + [CLS] +
           Transformer×2 (4 heads, ff=512, GEGLU, pre-norm)
                                               → F_life ∈ R^256

Cross-Modal Attention Fusion
  ├── Linear projections: F_mri/cog/life → R^512
  ├── Shared 2-layer attention: e_i = w_a^T · tanh(W_a · F̃_i)
  ├── α_i = softmax(e_i)   [Σ α_i = 1]
  └── F_fused = LayerNorm(W · Σ α_i·F̃_i)    → R^512

Classification Head
  └── Linear(512→256) + ReLU + Dropout(0.4) + Linear(256→3) + Softmax
                                               → P(CN, MCI, AD)
```

Total parameters: **~24.3 million** (17.6M EfficientNet-B4 + 6.7M other).

### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Main learning rate | 1×10⁻⁴ |
| EfficientNet-B4 backbone lr | 1×10⁻⁵ |
| LR schedule | Cosine annealing with warm restarts (T₀=10, T_mult=2) |
| Weight decay | 1×10⁻⁵ |
| Max epochs | 100 |
| Early stopping patience | 15 |
| Class loss weighting | Inverse frequency |

### Data Preprocessing Steps

1. **MRI:** Skull-stripping (FSL BET), affine registration to MNI152 (ANTs), slice extraction (30 axial slices per subject), resize to 224×224, RGB normalization (ImageNet mean/std).
2. **Cognitive Scores:** Z-score normalization; missing values imputed via median per diagnostic group.
3. **Lifestyle Sequences:** Min-max normalization per feature channel; zero-padding for missing timesteps.

---

## Key Results (from the Paper, ADNI + OASIS combined, N = 1,900)

| Metric | Value |
|---|---|
| **Accuracy** | **92.6%** |
| Macro-average AUC | 0.973 |
| Macro F1 | 91.5% |
| Macro Specificity | 95.3% |
| 95% CI | [90.3%, 94.8%] |

### Ablation Study Results (Table 5)

| Configuration | Acc (%) | AUC |
|---|---|---|
| MRI Only (EfficientNet-B4) | 88.1 | 0.940 |
| Cognitive Only (MLP) | 74.1 | 0.821 |
| Lifestyle (BiLSTM+Transformer) | 71.6 | 0.798 |
| MRI + Cog (Bimodal) | 86.2 | 0.930 |
| All + Early Fusion | 90.1 | 0.955 |
| All + Late Fusion | 90.8 | 0.961 |
| **Proposed MDLF** | **92.6** | **0.973** |

---

## Citations

If you use this code or the MDLF framework in your research, please cite:

```bibtex
@article{debnath2026mdlf,
  title   = {A Multimodal Deep Learning Framework for Early Alzheimer's Disease
             Detection Using MRI, Cognitive Scores, and Lifestyle Data},
  author  = {Debnath, Sajib and Mia, Md. Uzzal and Biswas, Arindam Kishor and
             Pal, Lipika and Hosain, Md. Sarwar and Shimamura, Tetsuya},
  journal = {PeerJ Computer Science},
  year    = {2026}
}
```

Please also cite the datasets used:

**ADNI:**
```
Data used in preparation of this article were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators
within the ADNI contributed to the design and implementation of ADNI and/or provided
data but did not participate in analysis or writing of this report.
```

**OASIS:**
```bibtex
@article{lamontagne2019oasis,
  title   = {OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset
             for Normal Aging and Alzheimer Disease},
  author  = {LaMontagne, Pamela J and others},
  journal = {medRxiv},
  year    = {2019},
  doi     = {10.1101/2019.12.13.19014902}
}
```

---

## License & Contribution Guidelines

### License

This repository is released under the **MIT License**. See [LICENSE](LICENSE) for details.

The ADNI and OASIS datasets are subject to their own data use agreements — please comply with the terms of access for each dataset independently.

### Contribution Guidelines

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request describing your changes.

Please open an Issue first for major changes to discuss your proposal.

---

## Acknowledgements

- [ADNI](https://adni.loni.usc.edu) — Alzheimer's Disease Neuroimaging Initiative
- [OASIS](https://www.oasis-brains.org) — Open Access Series of Imaging Studies
- [timm](https://github.com/huggingface/pytorch-image-models) — EfficientNet-B4 implementation
- [SHAP](https://github.com/slundberg/shap) — Shapley Additive Explanations
- AI writing tools (GitHub Copilot) were used for boilerplate code assistance and grammar correction during manuscript preparation.
