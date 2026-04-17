#!/usr/bin/env python3
"""
Main training script for the MDLF Alzheimer's Detection Framework.

Usage:
  python scripts/train.py --config configs/default_config.yaml
  python scripts/train.py --config configs/default_config.yaml --synthetic  # no real data needed
  python scripts/train.py --config configs/default_config.yaml --data_root /path/to/data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml

from data.dataset import build_dataloaders, stratified_split
from data.preprocessing import (
    preprocess_cognitive_data,
    preprocess_lifestyle_data,
    generate_synthetic_dataset,
)
from models.mdlf import build_mdlf_from_config
from training.trainer import Trainer
from evaluation.metrics import evaluate_model, compute_metrics, print_metrics
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MDLF for Alzheimer's Detection")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate and use synthetic dataset (no real data required)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory containing metadata.csv, MRI, and lifestyle files")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Load or generate data
    # ------------------------------------------------------------------ #
    import pandas as pd

    if args.synthetic:
        print("Generating synthetic dataset...")
        metadata_df = generate_synthetic_dataset(
            n_subjects=1900,
            random_seed=args.seed,
            output_dir=os.path.join(cfg["data"]["processed_root"], "synthetic"),
        )
        data_root = os.path.join(cfg["data"]["processed_root"], "synthetic")
    else:
        data_root = args.data_root or cfg["data"].get("processed_root", "data/processed")
        csv_path = os.path.join(data_root, "metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"metadata.csv not found at {csv_path}. "
                "Use --synthetic or preprocess your data first."
            )
        metadata_df = pd.read_csv(csv_path)

    print(f"Loaded {len(metadata_df)} subjects.")
    print(metadata_df["Label"].value_counts().to_string())

    # ------------------------------------------------------------------ #
    # Split
    # ------------------------------------------------------------------ #
    train_df, val_df, test_df = stratified_split(
        metadata_df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=args.seed,
    )
    print(f"Split: train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")

    # ------------------------------------------------------------------ #
    # Preprocess cognitive features
    # ------------------------------------------------------------------ #
    from data.preprocessing import COGNITIVE_FEATURES
    for col in COGNITIVE_FEATURES:
        if col not in metadata_df.columns:
            metadata_df[col] = 0.0  # fallback for synthetic

    cog_train, scaler, imputer = preprocess_cognitive_data(
        train_df, fit_scaler=True
    )
    cog_val, _, _  = preprocess_cognitive_data(val_df,  fit_scaler=False, scaler=scaler, imputer=imputer)
    cog_test, _, _ = preprocess_cognitive_data(test_df, fit_scaler=False, scaler=scaler, imputer=imputer)

    # ------------------------------------------------------------------ #
    # Preprocess lifestyle sequences
    # ------------------------------------------------------------------ #
    def load_lifestyle_sequences(df: pd.DataFrame) -> np.ndarray:
        seqs = []
        for _, row in df.iterrows():
            if os.path.exists(row["LifestylePath"]):
                seq = np.load(row["LifestylePath"])
            else:
                seq = np.zeros((12, 8), dtype=np.float32)
            seqs.append(seq)
        return np.stack(seqs, axis=0)

    life_train_raw = load_lifestyle_sequences(train_df)
    life_val_raw   = load_lifestyle_sequences(val_df)
    life_test_raw  = load_lifestyle_sequences(test_df)

    life_train, life_mean, life_std = preprocess_lifestyle_data(life_train_raw, fit_stats=True)
    life_val,  _, _ = preprocess_lifestyle_data(life_val_raw,  fit_stats=False, mean=life_mean, std=life_std)
    life_test, _, _ = preprocess_lifestyle_data(life_test_raw, fit_stats=False, mean=life_mean, std=life_std)

    # ------------------------------------------------------------------ #
    # DataLoaders
    # ------------------------------------------------------------------ #
    loaders = build_dataloaders(
        train_df, val_df, test_df,
        cog_train, cog_val, cog_test,
        life_train, life_val, life_test,
        batch_size=cfg["training"]["batch_size"],
        num_workers=min(4, os.cpu_count() or 1),
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = build_mdlf_from_config(cfg)
    param_counts = model.count_parameters()
    print(f"\nModel parameters:")
    for k, v in param_counts.items():
        print(f"  {k:20s}: {v:,}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {args.resume}")

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        cfg=cfg,
        device=device,
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        log_dir=cfg["paths"]["logs_dir"],
    )
    history = trainer.fit()

    # ------------------------------------------------------------------ #
    # Evaluate on test set
    # ------------------------------------------------------------------ #
    print("\nEvaluating on held-out test set...")
    results = evaluate_model(model, loaders["test"], device, return_attention=True)
    metrics = compute_metrics(
        results["all_labels"], results["all_preds"], results["all_probs"]
    )
    print_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Save figures
    # ------------------------------------------------------------------ #
    fig_dir = cfg["paths"]["figures_dir"]
    plot_training_curves(history, output_dir=fig_dir)
    plot_confusion_matrix(
        results["all_labels"], results["all_preds"],
        title="MDLF Confusion Matrix (Test Set)",
        output_path=os.path.join(fig_dir, "confusion_matrix_mdlf.png"),
    )
    plot_roc_curves(results["all_labels"], results["all_probs"],
                    model_name="MDLF", output_dir=fig_dir)
    plot_pr_curves(results["all_labels"], results["all_probs"],
                   model_name="MDLF", output_dir=fig_dir)

    if "all_alpha" in results:
        from utils.visualization import plt
        from evaluation.interpretability import plot_attention_weights
        plot_attention_weights(results["all_alpha"], results["all_labels"],
                               output_dir=fig_dir)

    # Save metrics
    import json
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    serializable = {k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix", "per_class", "classification_report", "ci_95")}
    serializable["ci_low"]  = float(metrics["ci_95"][0])
    serializable["ci_high"] = float(metrics["ci_95"][1])
    with open(os.path.join(cfg["paths"]["results_dir"], "test_metrics.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nMetrics saved to {cfg['paths']['results_dir']}/test_metrics.json")


if __name__ == "__main__":
    main()
