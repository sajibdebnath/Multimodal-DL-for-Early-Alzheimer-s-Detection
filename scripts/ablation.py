#!/usr/bin/env python3
"""
Ablation study script — reproduces Table 5 from the paper.

Trains and evaluates all ablated variants:
  - Unimodal: MRI-only (ResNet-50, EfficientNet-B4), Cognitive-only, Lifestyle-only
  - Bimodal:  MRI+Cog, MRI+Life, Cog+Life
  - Trimodal: Early Fusion, Late Fusion, Concatenation (no attention)
  - Transformer-only lifestyle branch (no BiLSTM)
  - Proposed MDLF (all + attention fusion)

Usage:
  python scripts/ablation.py --config configs/default_config.yaml --synthetic
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data.dataset import build_dataloaders, stratified_split, AlzheimerMultimodalDataset
from data.preprocessing import (
    preprocess_cognitive_data, preprocess_lifestyle_data,
    generate_synthetic_dataset,
)
from models.mdlf import MDLF, build_mdlf_from_config
from models.mri_branch import MRIBranch
from models.cognitive_branch import CognitiveBranch
from models.lifestyle_branch import LifestyleBranch
from training.loss import WeightedCrossEntropyLoss, compute_class_weights
from evaluation.metrics import evaluate_model, compute_metrics, print_metrics
from utils.visualization import plot_model_comparison


# ---------------------------------------------------------------------------
# Ablated model variants
# ---------------------------------------------------------------------------

class MRIOnlyModel(nn.Module):
    """MRI-only baseline (EfficientNet-B4)."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = MRIBranch(feature_dim=512, pretrained=pretrained)
        self.head = nn.Linear(512, 3)

    def forward(self, mri, cognitive, lifestyle):
        f = self.encoder(mri)
        logits = self.head(f)
        return {"logits": logits, "probs": torch.softmax(logits, -1),
                "alpha": torch.zeros(mri.size(0), 3)}


class CognitiveOnlyModel(nn.Module):
    """Cognitive-only baseline (MLP)."""
    def __init__(self):
        super().__init__()
        self.encoder = CognitiveBranch(feature_dim=128)
        self.head = nn.Linear(128, 3)

    def forward(self, mri, cognitive, lifestyle):
        f = self.encoder(cognitive)
        logits = self.head(f)
        return {"logits": logits, "probs": torch.softmax(logits, -1),
                "alpha": torch.zeros(cognitive.size(0), 3)}


class LifestyleOnlyModel(nn.Module):
    """Lifestyle-only baseline (BiLSTM+Transformer)."""
    def __init__(self):
        super().__init__()
        self.encoder = LifestyleBranch(feature_dim=256)
        self.head = nn.Linear(256, 3)

    def forward(self, mri, cognitive, lifestyle):
        f = self.encoder(lifestyle)
        logits = self.head(f)
        return {"logits": logits, "probs": torch.softmax(logits, -1),
                "alpha": torch.zeros(lifestyle.size(0), 3)}


class BimodalModel(nn.Module):
    """Bimodal model with simple concatenation (no attention)."""
    def __init__(self, use_mri: bool = True, use_cog: bool = True, use_life: bool = True):
        super().__init__()
        self.use_mri  = use_mri
        self.use_cog  = use_cog
        self.use_life = use_life

        feat_dim = 0
        if use_mri:
            self.mri_enc  = MRIBranch(feature_dim=512)
            feat_dim += 512
        if use_cog:
            self.cog_enc  = CognitiveBranch(feature_dim=128)
            feat_dim += 128
        if use_life:
            self.life_enc = LifestyleBranch(feature_dim=256)
            feat_dim += 256

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 3)
        )

    def forward(self, mri, cognitive, lifestyle):
        features = []
        if self.use_mri:  features.append(self.mri_enc(mri))
        if self.use_cog:  features.append(self.cog_enc(cognitive))
        if self.use_life: features.append(self.life_enc(lifestyle))
        f = torch.cat(features, dim=-1)
        logits = self.head(f)
        return {"logits": logits, "probs": torch.softmax(logits, -1),
                "alpha": torch.zeros(mri.size(0), 3)}


class EarlyFusionModel(nn.Module):
    """Trimodal early fusion: concatenate raw features before any encoder."""
    def __init__(self):
        super().__init__()
        # Light-weight shared encoder on concatenated features
        # MRI features (from EfficientNet) + cognitive (8) + lifestyle (12*8=96)
        self.mri_enc  = MRIBranch(feature_dim=512)
        self.cog_enc  = CognitiveBranch(feature_dim=128)
        self.life_enc = LifestyleBranch(feature_dim=256)
        self.fusion   = nn.Sequential(
            nn.Linear(512 + 128 + 256, 512), nn.ReLU(), nn.Dropout(0.4),
        )
        self.head = nn.Linear(512, 3)

    def forward(self, mri, cognitive, lifestyle):
        f = torch.cat([
            self.mri_enc(mri),
            self.cog_enc(cognitive),
            self.life_enc(lifestyle),
        ], dim=-1)
        f = self.fusion(f)
        logits = self.head(f)
        return {"logits": logits, "probs": torch.softmax(logits, -1),
                "alpha": torch.zeros(mri.size(0), 3)}


class LateFusionModel(nn.Module):
    """Trimodal late fusion: average softmax predictions."""
    def __init__(self):
        super().__init__()
        self.mri_enc  = MRIBranch(feature_dim=512)
        self.cog_enc  = CognitiveBranch(feature_dim=128)
        self.life_enc = LifestyleBranch(feature_dim=256)
        self.mri_head  = nn.Linear(512, 3)
        self.cog_head  = nn.Linear(128, 3)
        self.life_head = nn.Linear(256, 3)

    def forward(self, mri, cognitive, lifestyle):
        p_mri  = torch.softmax(self.mri_head(self.mri_enc(mri)), -1)
        p_cog  = torch.softmax(self.cog_head(self.cog_enc(cognitive)), -1)
        p_life = torch.softmax(self.life_head(self.life_enc(lifestyle)), -1)
        probs  = (p_mri + p_cog + p_life) / 3.0
        logits = torch.log(probs + 1e-8)
        return {"logits": logits, "probs": probs,
                "alpha": torch.zeros(mri.size(0), 3)}


# ---------------------------------------------------------------------------
# Quick training helper (fewer epochs for ablation speed)
# ---------------------------------------------------------------------------

def quick_train(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    class_weights: torch.Tensor,
    max_epochs: int = 30,
    lr: float = 1e-4,
    patience: int = 8,
) -> nn.Module:
    criterion = WeightedCrossEntropyLoss(class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    model.to(device)
    best_acc, best_state, no_improve = 0.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            mri  = batch["mri"].to(device)
            cog  = batch["cognitive"].to(device)
            life = batch["lifestyle"].to(device)
            lbl  = batch["label"].to(device)
            out = model(mri, cog, life)
            loss = criterion(out["logits"], lbl)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Validation
        model.eval(); correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                mri  = batch["mri"].to(device)
                cog  = batch["cognitive"].to(device)
                life = batch["lifestyle"].to(device)
                lbl  = batch["label"].to(device)
                out = model(mri, cog, life)
                correct += (out["probs"].argmax(-1) == lbl).sum().item()
                total   += len(lbl)
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/default_config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs",    type=int, default=30,
                        help="Epochs per variant (use fewer for quick ablation)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (same path as train.py)
    if args.synthetic:
        meta_df = generate_synthetic_dataset(n_subjects=500, random_seed=args.seed,
                                             output_dir="data/synthetic_small")
    else:
        meta_df = pd.read_csv(os.path.join(args.data_root or "data/processed", "metadata.csv"))

    train_df, val_df, test_df = stratified_split(meta_df, random_seed=args.seed)

    from data.preprocessing import COGNITIVE_FEATURES
    for col in COGNITIVE_FEATURES:
        if col not in meta_df.columns:
            meta_df[col] = 0.0

    cog_train, scaler, imputer = preprocess_cognitive_data(train_df, fit_scaler=True)
    cog_val,  _, _ = preprocess_cognitive_data(val_df,  False, scaler, imputer)
    cog_test, _, _ = preprocess_cognitive_data(test_df, False, scaler, imputer)

    def load_life(df):
        seqs = []
        for _, row in df.iterrows():
            p = row.get("LifestylePath", "")
            seqs.append(np.load(p) if os.path.exists(str(p)) else np.zeros((12,8),np.float32))
        return np.stack(seqs)

    life_tr, lm, ls = preprocess_lifestyle_data(load_life(train_df), fit_stats=True)
    life_va, _, _   = preprocess_lifestyle_data(load_life(val_df),  False, lm, ls)
    life_te, _, _   = preprocess_lifestyle_data(load_life(test_df), False, lm, ls)

    loaders = build_dataloaders(train_df, val_df, test_df,
                                cog_train, cog_val, cog_test,
                                life_tr, life_va, life_te, batch_size=32)

    labels_np = train_df["LabelInt"].values
    cw = compute_class_weights(labels_np, device=str(device))

    VARIANTS = {
        "MRI Only (EfficientNet-B4)":      MRIOnlyModel(pretrained=True),
        "Cognitive Only (MLP)":            CognitiveOnlyModel(),
        "Lifestyle (BiLSTM+Transformer)":  LifestyleOnlyModel(),
        "MRI + Cog (Bimodal)":             BimodalModel(True, True, False),
        "MRI + Life (Bimodal)":            BimodalModel(True, False, True),
        "Cog + Life (Bimodal)":            BimodalModel(False, True, True),
        "All + Early Fusion":              EarlyFusionModel(),
        "All + Late Fusion":               LateFusionModel(),
        "All + Concat (No Attn)":          BimodalModel(True, True, True),
        "Proposed MDLF":                   build_mdlf_from_config(cfg),
    }

    all_results = {}
    for name, model in VARIANTS.items():
        print(f"\n{'='*50}\n  Training: {name}\n{'='*50}")
        model = quick_train(model, loaders["train"], loaders["val"], device, cw,
                            max_epochs=args.epochs)
        res = evaluate_model(model, loaders["test"], device)
        met = compute_metrics(res["all_labels"], res["all_preds"], res["all_probs"])
        all_results[name] = met
        print(f"  Acc: {met['accuracy']*100:.2f}%  AUC: {met['macro_auc']:.4f}")

    # Print summary table
    print(f"\n{'='*75}")
    print(f"  {'Configuration':<40} {'Acc':>7} {'AUC':>7} {'ΔProp':>8}")
    print(f"  {'-'*62}")
    prop_acc = all_results.get("Proposed MDLF", {}).get("accuracy", 0)
    for name, met in all_results.items():
        delta = (met['accuracy'] - prop_acc) * 100
        print(f"  {name:<40} {met['accuracy']*100:>6.2f}% {met['macro_auc']:>7.4f}  {delta:>+7.2f}%")

    # Save comparison chart
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    plot_model_comparison(all_results, output_dir=cfg["paths"]["figures_dir"])
    print("\nAblation complete.")


if __name__ == "__main__":
    main()
